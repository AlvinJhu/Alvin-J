"""
台股風險模型 v3：不猜漲跌，改預測「風險」
v2 證實方向（漲/跌）沒有樣本外預測力；但波動度具群聚性，是少數可預測的量（Engle, 1982; Corsi, 2009）。
1. 波動度預測：未來 20 日實現波動，HAR-RV（日/週/月）＋ Parkinson 高低幅度＋ VIX ＋ 費半（對數線性迴歸）
   基準：EWMA（RiskMetrics λ=0.94）、過去 20 日波動
2. 大跌機率：未來 20 日內最低收盤較今日跌超過 7%（樣本基準率約 10%）
3. 部位：持股 = min(1, 目標波動 / 預測波動)，可再依大跌機率減碼；調整帶避免頻繁交易
   參數只用 2016–2022 的樣本外預測選定，2023–2026 完全保留驗證
輸出：reports/risk_vol_forecast.csv、risk_crash_auc.csv、risk_strategy.csv、risk_today.json、risk_equity.png
"""
import os, sys, json, itertools
import numpy as np, pandas as pd
sys.argv = [a for a in sys.argv if a != "--yf"] + []
import model as M
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.metrics import roc_auc_score, brier_score_loss
from xgboost import XGBClassifier

R = M.R; H = 20; ANN = np.sqrt(252)
SPLIT = pd.Timestamp("2023-01-01")  # 之前：選參數；之後：保留驗證
DD = -0.07

def build(df):
    c = df.close; lr = np.log(c).diff()
    X = pd.DataFrame(index=df.index)
    X["rv1"] = np.log(lr.abs() * ANN + 1e-4)
    X["rv5"] = np.log(lr.rolling(5).std() * ANN)
    X["rv22"] = np.log(lr.rolling(22).std() * ANN)
    X["rv66"] = np.log(lr.rolling(66).std() * ANN)
    pk = np.sqrt((np.log(df.high / df.low) ** 2).rolling(5).mean() / (4 * np.log(2))) * ANN  # Parkinson (1980)
    X["park5"] = np.log(pk)
    X["neg5"] = lr.clip(upper=0).rolling(5).sum()  # 槓桿效果：下跌後波動升高
    X["vix"] = np.log(df.vix)
    if "sox" in df:
        sl = np.log(df.sox).diff(); X["sox_rv22"] = np.log(sl.rolling(22).std() * ANN); X["sox_20"] = df.sox.pct_change(20)
    X["dd60"] = c / c.rolling(60).max() - 1
    X["bias240"] = c / c.rolling(240).mean() - 1
    if "margin" in df: X["margin_20"] = df.margin.pct_change(20).shift(1)
    if "foreign" in df: X["fgn_20"] = (df.foreign.rolling(20).sum() / df.turn.rolling(20).sum()).shift(1)
    fut = pd.concat([lr.shift(-k) for k in range(1, H + 1)], axis=1)
    y_vol = np.log(fut.std(axis=1) * ANN).where(fut.notna().all(axis=1))
    mn = pd.concat([c.shift(-k) for k in range(1, H + 1)], axis=1).min(axis=1) / c - 1
    y_dd = (mn < DD).astype(float).where(fut.notna().all(axis=1))
    ew = np.log(np.sqrt((lr ** 2).ewm(alpha=0.06).mean()) * ANN)
    return X.replace([np.inf, -np.inf], np.nan), y_vol, y_dd, ew, lr

def wf(X, y, fit_pred, min_train=500, step=63, h=H):
    out = pd.Series(np.nan, index=X.index); start = min_train
    while start < len(X):
        end = min(start + step, len(X)); tr = np.arange(max(0, start - h)); tr = tr[y.iloc[tr].notna().values]
        out.iloc[start:end] = fit_pred(X.iloc[tr], y.iloc[tr], X.iloc[start:end]); start = end
    return out

def ols(Xt, yt, Xs):
    """對數波動 OLS；加 0.5×殘差變異做對數常態偏誤修正，避免低估變異（QLIKE 對低估懲罰重）。"""
    m = LinearRegression().fit(Xt, yt); s2 = np.var(yt - m.predict(Xt))
    return m.predict(Xs) + 0.5 * s2
def logit(Xt, yt, Xs): return make_pipeline(StandardScaler(), LogisticRegression(C=0.1, max_iter=2000)).fit(Xt, yt).predict_proba(Xs)[:, 1]
def xgb(Xt, yt, Xs):
    return XGBClassifier(n_estimators=300, max_depth=2, learning_rate=0.03, subsample=0.8, colsample_bytree=0.7,
                         min_child_weight=30, reg_lambda=5.0, verbosity=0).fit(Xt, yt).predict_proba(Xs)[:, 1]

def r2(y, p, base): return 1 - ((y - p) ** 2).sum() / ((y - base) ** 2).sum()
def qlike(y, p):  # Patton (2011)：對變異數預測的穩健損失
    v, f = np.exp(2 * y), np.exp(2 * p); return float(np.mean(v / f - np.log(v / f) - 1))

def strat(w, r, cost=0.001):
    return w * r - w.diff().abs().fillna(w.iloc[0]) * cost

def weights(vol_fc, p_dd, target, cut, thr, band):
    raw = np.minimum(1.0, target / np.asarray(vol_fc)) * np.where(p_dd > thr, cut, 1.0)
    w = np.empty(len(raw)); cur = raw[0]
    for i, x in enumerate(raw):  # 調整帶：差距超過 band 才調整
        if abs(x - cur) > band: cur = x
        w[i] = cur
    return pd.Series(w, index=vol_fc.index)

def main():
    df = M.load(); X, y_vol, y_dd, ew, lr = build(df)
    X = X.dropna(); y_vol, y_dd, ew = y_vol.reindex(X.index), y_dd.reindex(X.index), ew.reindex(X.index)
    print(f"樣本 {X.index[0].date()} ~ {X.index[-1].date()}，{len(X)} 天，大跌（20 日內 {DD:.0%}）基準率 {y_dd.mean():.3f}")
    har = ["rv1", "rv5", "rv22"]
    sets = {"HAR": har, "HAR+幅度+VIX": har + ["rv66", "park5", "neg5", "vix"],
            "HAR+幅度+VIX+費半籌碼": [c for c in X if c not in ("dd60", "bias240")]}

    # 1) 波動度預測
    vf, rows = {}, []
    for n, cols in sets.items(): vf[n] = wf(X[cols], y_vol, ols)
    ok = y_vol.notna() & vf["HAR"].notna()
    base = {"過去20日波動": X.rv22, "EWMA(0.94)": ew}
    for n, p in {**base, **vf}.items():
        for per, m in (("2018–2022", ok & (X.index < SPLIT)), ("2023–2026", ok & (X.index >= SPLIT)), ("全期", ok)):
            yy, pp = y_vol[m], p[m]
            rows.append({"模型": n, "期間": per, "天數": int(m.sum()), "R2(對數波動)": r2(yy, pp, yy.mean()),
                         "相關係數": float(np.corrcoef(yy, pp)[0, 1]), "QLIKE": qlike(yy.values, pp.values),
                         "MAE(年化波動點)": float(np.mean(np.abs(np.exp(yy) - np.exp(pp))))})
    vt = pd.DataFrame(rows); print(vt.round(3).to_string(index=False))
    vt.to_csv(os.path.join(R, "risk_vol_forecast.csv"), index=False, encoding="utf-8-sig")

    # 2) 大跌機率
    crash_cols = sets["HAR+幅度+VIX+費半籌碼"] + ["dd60", "bias240"]
    pc = {"logit": wf(X[crash_cols], y_dd, logit), "xgb": wf(X[crash_cols], y_dd, xgb)}
    pc["ens"] = (pc["logit"] + pc["xgb"]) / 2
    pc["只用預測波動(基準)"] = vf["HAR+幅度+VIX"]
    rows = []
    for n, p in pc.items():
        for per, m0 in (("2018–2022", X.index < SPLIT), ("2023–2026", X.index >= SPLIT), ("全期", X.index == X.index)):
            m = m0 & p.notna() & y_dd.notna(); yy, pp = y_dd[m], p[m]
            lo, hi = M.block_auc_ci(yy.values, pp.values, 20)
            top = pp >= pp.quantile(0.8)
            rows.append({"模型": n, "期間": per, "天數": int(m.sum()), "基準率": yy.mean(), "AUC": roc_auc_score(yy, pp),
                         "AUC_95%下界": lo, "AUC_95%上界": hi,
                         "最高20%機率日的大跌率": yy[top].mean(), "其餘80%的大跌率": yy[~top].mean()})
    ct = pd.DataFrame(rows); print(ct.round(3).to_string(index=False))
    ct.to_csv(os.path.join(R, "risk_crash_auc.csv"), index=False, encoding="utf-8-sig")

    # 3) 部位規則：只用 2018–2022 選參數
    vol_fc = np.exp(vf["HAR+幅度+VIX"]); p_dd = pc["ens"]
    r = (df.close.shift(-1) / df.close - 1).reindex(X.index)
    test = vol_fc.dropna().index.intersection(p_dd.dropna().index); test = test[r[test].notna()]
    vol_fc, p_dd, r = vol_fc[test], p_dd[test], r[test]
    ins, oos = test < SPLIT, test >= SPLIT
    grid = []
    for tg, cut, thr, band in itertools.product([0.12, 0.15, 0.18, 0.22], [1.0, 0.5, 0.25], [0.2, 0.3, 0.4], [0.0, 0.1, 0.2]):
        if cut == 1.0 and thr != 0.2: continue
        w = weights(vol_fc, p_dd.values, tg, cut, thr, band); s = strat(w, r)
        st = M.stats(s[ins]); grid.append({"target": tg, "cut": cut, "thr": thr, "band": band, **st})
    grid = pd.DataFrame(grid).sort_values("Sharpe", ascending=False)
    best = grid.iloc[0]; print("2018–2022 選出參數：", best[["target", "cut", "thr", "band", "Sharpe"]].to_dict())
    grid.to_csv(os.path.join(R, "risk_param_grid_insample.csv"), index=False, encoding="utf-8-sig")

    vol20 = (lr.rolling(20).std() * ANN).reindex(test)
    cands = {
        "v3 風險模型（選定參數）": weights(vol_fc, p_dd.values, best.target, best.cut, best.thr, best.band),
        "v3 只用預測波動（不看大跌機率）": weights(vol_fc, p_dd.values, best.target, 1.0, 1.0, best.band),
        "v2 波動目標（過去20日波動）": pd.Series(M.exposure(np.full(len(test), .6), vol20.values), index=test),
        "買進持有": pd.Series(1.0, index=test),
    }
    rows = []
    for n, w in cands.items():
        s = strat(w, r) if n != "買進持有" else r
        for per, m in (("2018–2022（選參數）", ins), ("2023–2026（保留驗證）", oos), ("全期", ins | oos)):
            rows.append({"策略": n, "期間": per, **M.stats(s[m]), "平均持股": w[m].mean(),
                         "年換手": w[m].diff().abs().sum() / (m.sum() / 252)})
    sp = pd.DataFrame(rows); print(sp.round(3).to_string(index=False))
    sp.to_csv(os.path.join(R, "risk_strategy.csv"), index=False, encoding="utf-8-sig")

    # 4) 今日判斷：全部資料重訓
    last = X.index[-1]; fit_v = y_vol.notna(); fit_d = y_dd.notna()
    cols_v = sets["HAR+幅度+VIX"]
    v_today = float(np.exp(ols(X.loc[fit_v, cols_v], y_vol[fit_v], X.loc[[last], cols_v])[0]))
    pd_today = float((logit(X.loc[fit_d, crash_cols], y_dd[fit_d], X.loc[[last], crash_cols])[0] +
                      xgb(X.loc[fit_d, crash_cols], y_dd[fit_d], X.loc[[last], crash_cols])[0]) / 2)
    w_today = float(min(1.0, best.target / v_today) * (best.cut if pd_today > best.thr else 1.0))
    c = float(df.close.iloc[-1]); s20 = v_today / ANN * np.sqrt(H)
    hist = pd.Series(np.exp(y_vol.dropna()))
    today = {"date": str(last.date()), "close": c,
             "pred_vol_20d_ann": round(v_today, 4), "vol_percentile_since_2016": round(float((hist < v_today).mean()), 3),
             "p_drawdown_7pct_within_20d": round(pd_today, 3), "base_rate": round(float(y_dd.mean()), 3),
             "range_20d_68": [round(c * np.exp(-s20)), round(c * np.exp(s20))],
             "range_20d_95": [round(c * np.exp(-1.96 * s20)), round(c * np.exp(1.96 * s20))],
             "VaR95_1d_pct": round(-1.645 * v_today / ANN, 4),
             "suggest_equity": round(w_today, 3), "params": best[["target", "cut", "thr", "band"]].to_dict()}
    json.dump(today, open(os.path.join(R, "risk_today.json"), "w", encoding="utf-8"), ensure_ascii=False, indent=2)
    print(json.dumps(today, ensure_ascii=False))

    try:
        import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
        fig, ax = plt.subplots(3, 1, figsize=(11, 9), sharex=True, gridspec_kw={"height_ratios": [3, 1.3, 1]})
        for n, w in cands.items():
            s = strat(w, r) if n != "買進持有" else r
            ax[0].plot((1 + s).cumprod(), label={"v3 風險模型（選定參數）": "v3 risk model", "v3 只用預測波動（不看大跌機率）": "v3 vol-forecast only",
                                                 "v2 波動目標（過去20日波動）": "v2 vol target", "買進持有": "Buy & Hold"}[n])
        ax[0].axvline(SPLIT, color="k", ls="--", lw=.8); ax[0].text(SPLIT, ax[0].get_ylim()[1] * .9, " holdout", fontsize=8)
        ax[0].set_yscale("log"); ax[0].legend(fontsize=8); ax[0].set_title("TAIEX risk model: walk-forward, params chosen on 2018-2022 only")
        ax[1].plot(np.exp(y_vol[test]), lw=.7, label="realized 20d vol"); ax[1].plot(vol_fc, lw=.7, label="forecast"); ax[1].legend(fontsize=8)
        ax[2].fill_between(test, cands["v3 風險模型（選定參數）"].values, step="post", alpha=.5); ax[2].set_ylabel("weight")
        fig.tight_layout(); fig.savefig(os.path.join(R, "risk_equity.png"), dpi=110)
    except Exception as e:
        print("圖檔略過：", e)

if __name__ == "__main__":
    main()
