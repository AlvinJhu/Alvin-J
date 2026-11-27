"""
多頭參考指標回測（2026-11-23）：只回測、不改正式模型
四個指標（門檻事先寫死，不做參數搜尋，避免資料探勘偏誤）：
1. 市場寬度：前百大（2015 年底名單，還原股價）站上 200 日線比例；全市場月資料（站上 10 個月線）當穩健性檢查
   訊號：過去 60 日寬度曾低於 30%，且目前比 60 日低點回升 20 個百分點以上
2. 融資減肥：融資餘額 20 日變化 < -10%，且加權指數距 60 日高點跌超過 10%
3. 外資回補：外資連續 3 日現貨買超，且外資台指期淨部位 5 日內增加（空單減少）
4. 估值：加權（市值）盈餘殖利率 E/P 落在 2015 起擴張視窗的最便宜 20%（至少 36 個月歷史）
所有籌碼資料延後 1 日（盤後公布），月資料延後 1 日。
評估：未來 20/60 日報酬（訊號日 vs 全部）、方向 AUC（方向事先決定，等同樣本外）、各年代、跌深日子內的 AUC；
疊加 v3.2：警訊期間若出現多頭訊號就提早結束減碼（回補），比較 Sharpe／最大回撤。
輸出：reports/bull_*.csv、bull_today.json
"""
import os, sys, json
import numpy as np, pandas as pd
sys.argv = sys.argv[:1]
import model as M
import risk_model as RM
from sklearn.metrics import roc_auc_score

R, EXT = M.R, os.path.join(M.D, "ext")
PERIODS = [("2016–2019", "2016-01-01", "2019-12-31"), ("2020–2022", "2020-01-01", "2022-12-31"),
           ("2023–2026", "2023-01-01", "2026-12-31"), ("全期", "2000-01-01", "2026-12-31")]

def breadth_daily(index):
    d = pd.read_csv(os.path.join(EXT, "stk_daily_top100.csv"), dtype={"code": str}, parse_dates=["date"])
    px = d.pivot_table(index="date", columns="code", values="close_adj").sort_index()
    ma = px.rolling(200, min_periods=200).mean()
    valid = ma.notna() & px.notna()
    br = ((px > ma) & valid).sum(axis=1) / valid.sum(axis=1)
    br[valid.sum(axis=1) < 50] = np.nan
    return br.reindex(index)

def monthly_market(index):
    m = pd.read_csv(os.path.join(EXT, "stk_monthly.csv"), dtype={"code": str}, parse_dates=["date"])
    m = m[m.code.str.fullmatch(r"\d{4}") & ~m.code.str.startswith("00")]  # 排除 ETF
    m["cap"] = m.close * m.shares
    px = m.pivot_table(index="date", columns="code", values="close").sort_index()
    ma = px.rolling(10, min_periods=10).mean(); valid = ma.notna() & px.notna()
    br_m = ((px > ma) & valid).sum(axis=1) / valid.sum(axis=1); br_m[valid.sum(axis=1) < 300] = np.nan
    v = m[m.cap > 0]
    e = v[v.pe > 0].assign(earn=lambda x: x.cap / x.pe).groupby("date").earn.sum()
    ep = e / v[v.pe.notna()].groupby("date").cap.sum()  # 有本益比資料者（含虧損公司以 0 盈餘計？證交所虧損不列本益比，此處只計有列者）
    dy = (v.cap * v.dy / 100).groupby(v.date).sum() / v[v.dy.notna()].groupby("date").cap.sum()
    out = pd.DataFrame({"br_m": br_m, "ep": ep, "dy": dy})
    # 擴張視窗百分位（只用當時已知歷史），至少 36 個月
    for k in ("ep", "dy"):
        out[k + "_pct"] = [np.nan if i < 35 else (out[k].iloc[:i + 1] <= out[k].iloc[i]).mean() for i in range(len(out))]
    # 月底資料當天收盤後才知道：延後 1 個交易日生效，日資料向前填補
    out = out.reindex(out.index.union(index)).ffill().reindex(index).shift(1)
    return out

def build(df):
    c = df.close; f = pd.DataFrame(index=df.index)
    f["dd60"] = c / c.rolling(60).max() - 1
    br = breadth_daily(df.index); f["breadth"] = br
    f["breadth_rebound"] = br - br.rolling(60).min()
    f["sig_breadth"] = (br.rolling(60).min() < 0.30) & (f.breadth_rebound >= 0.20)
    mg = df.margin.shift(1); f["margin_20"] = mg.pct_change(20)
    f["sig_margin"] = (f.margin_20 < -0.10) & (f.dd60 < -0.10)
    fg = (df.foreign / df.turn).shift(1); f["fgn_5"] = (df.foreign.rolling(5).sum() / df.turn.rolling(5).sum()).shift(1)
    import futures_eval as FE
    fu = FE.load_futures(df.index)
    f["fut_fgn_chg5"] = (fu.fut_fgn_net.diff(5) / fu.oi).shift(1)
    f["fgn_combo"] = f.fgn_5.rank(pct=True) + f.fut_fgn_chg5.rank(pct=True)  # 連續分數（僅供 AUC）
    f["sig_foreign"] = (fg > 0).rolling(3).sum().eq(3) & (f.fut_fgn_chg5 > 0)
    f["sig_foreign"] = f.sig_foreign.astype(object).where(f.fut_fgn_chg5.notna()); f["fgn_combo"] = f.fgn_combo.where(f.fut_fgn_chg5.notna())
    mm = monthly_market(df.index)
    for k in mm: f[k] = mm[k]
    f["sig_value"] = (f.ep_pct >= 0.80).where(f.ep_pct.notna())
    sigs = ["sig_breadth", "sig_margin", "sig_foreign", "sig_value"]
    f["sig_any"] = f[sigs].fillna(False).astype(bool).any(axis=1)
    for k in sigs + ["sig_any"]: f[k] = f[k].astype(object).where(f.dd60.notna())
    for h in (20, 60): f[f"ret{h}"] = c.shift(-h) / c - 1
    return f

SCORES = {"市場寬度（回升幅度）": "breadth_rebound", "市場寬度（水準，低=多）": "-breadth", "全市場寬度月資料（低=多）": "-br_m",
          "融資 20 日減幅": "-margin_20", "外資現貨＋期貨回補": "fgn_combo", "盈餘殖利率百分位（便宜=多）": "ep_pct",
          "殖利率百分位（便宜=多）": "dy_pct"}
SIGS = {"寬度回升": "sig_breadth", "融資減肥": "sig_margin", "外資回補": "sig_foreign", "估值便宜": "sig_value", "任一訊號": "sig_any"}

def get(f, k): return -f[k[1:]] if k.startswith("-") else f[k]

def onsets(s, gap=20):
    """獨立事件：訊號開啟日，且距上次事件至少 gap 個交易日"""
    on = s.fillna(False).astype(bool); idx = np.where(on & ~on.shift(1, fill_value=False))[0]; out, last = [], -10**9
    for i in idx:
        if i - last >= gap: out.append(i); last = i
    return out

def eval_auc(f):
    rows = []
    for n, k in SCORES.items():
        for h in (20, 60):
            for subset in ("全部日子", "跌深日（距60日高點<-10%）"):
                for pn, a, b in PERIODS:
                    s, y = get(f, k), f[f"ret{h}"]
                    m = s.notna() & y.notna() & (f.index >= a) & (f.index <= b)
                    if subset != "全部日子": m &= f.dd60 < -0.10
                    if m.sum() < 60: continue
                    yy = (y[m] > 0).astype(int)
                    if yy.nunique() < 2: continue
                    auc = roc_auc_score(yy, s[m]); lo, hi = M.block_auc_ci(yy.values, s[m].values, h)
                    rows.append({"指標": n, "期間": pn, "標的": f"{h}日漲跌", "樣本": subset, "天數": int(m.sum()),
                                 "上漲率": yy.mean(), "AUC": auc, "95%下界": lo, "95%上界": hi,
                                 "IC(Spearman)": s[m].corr(y[m], method="spearman")})
    return pd.DataFrame(rows)

def eval_events(f):
    rows = []
    for n, k in SIGS.items():
        for pn, a, b in PERIODS:
            m = (f.index >= a) & (f.index <= b)
            sub = f[m]; on = sub[k].fillna(False).astype(bool)
            if sub[k].notna().sum() < 60: continue
            ev = onsets(sub[k]); r = {"訊號": n, "期間": pn, "訊號日數": int(on.sum()), "獨立事件數": len(ev)}
            for h in (20, 60):
                y = sub[f"ret{h}"]; base = y[sub[k].notna()]
                e = y.iloc[ev].dropna()
                r.update({f"事件後{h}日平均報酬": e.mean() if len(e) else np.nan, f"事件後{h}日上漲比例": (e > 0).mean() if len(e) else np.nan,
                          f"全部日{h}日平均報酬": base.mean(), f"全部日{h}日上漲比例": (base > 0).mean(),
                          f"事件後{h}日最差": e.min() if len(e) else np.nan})
            rows.append(r)
    return pd.DataFrame(rows)

def warn_with_exit(p_dd, thr, hold, sig):
    """v3.2 警訊持續，但警訊期間若多頭訊號出現且當天大跌機率未超過門檻，就提早結束減碼"""
    flag = np.asarray(p_dd) > thr; sig = np.asarray(sig, bool); act = np.zeros(len(flag), bool); last = -10**9
    for i, fl in enumerate(flag):
        if fl: last = i
        elif sig[i] and i - last < hold: last = -10**9
        act[i] = i - last < hold
    return act

def weights_custom(vol_fc, act, target, cut, band, boost=None):
    raw = np.minimum(1.0, target / np.asarray(vol_fc)) * np.where(act, cut, 1.0)
    if boost is not None: raw = np.where(np.asarray(boost, bool) & ~act, 1.0, raw)
    w = np.empty(len(raw)); cur = raw[0]
    for i, x in enumerate(raw):
        if abs(x - cur) > band: cur = x
        w[i] = cur
    return pd.Series(w, index=vol_fc.index)

def v32_inputs(df):
    """重現 risk_model.py v3.2 的樣本外預測波動與大跌機率（同因子、同 walk-forward）"""
    X, y_vol, y_dd, ew, lr = RM.build(df)
    import futures_eval as FE
    X["opt_fgn_bias"] = FE.fut_features(df, FE.load_futures(df.index))["opt_fgn_bias"]
    X = X.dropna(); y_vol, y_dd = y_vol.reindex(X.index), y_dd.reindex(X.index)
    har = ["rv1", "rv5", "rv22"]; volc = har + ["rv66", "park5", "neg5", "vix"]
    crash = [c for c in X if c not in ("dd60", "bias240", "opt_fgn_bias")] + ["dd60", "bias240", "opt_fgn_bias"]
    vol_fc = np.exp(RM.wf(X[volc], y_vol, RM.ols))
    p_dd = (RM.wf(X[crash], y_dd, RM.logit) + RM.wf(X[crash], y_dd, RM.xgb)) / 2
    return vol_fc, p_dd

def overlay(df, f):
    vol_fc, p_dd = v32_inputs(df)
    r = (df.close.shift(-1) / df.close - 1)
    test = vol_fc.dropna().index.intersection(p_dd.dropna().index); test = test[r.reindex(test).notna()]
    vol_fc, p_dd, r = vol_fc[test], p_dd[test], r[test]
    T, CUT, THR, BAND, HOLD = 0.22, 0.5, 0.25, 0.2, 10
    base_act = RM.warn_active(p_dd.values, THR, HOLD)
    cands = {"v3.2 現行": weights_custom(vol_fc, base_act, T, CUT, BAND), "買進持有": pd.Series(1.0, index=test)}
    acts = {"v3.2 現行": base_act, "買進持有": np.zeros(len(test), bool)}
    for n, k in SIGS.items():
        s = f[k].reindex(test).fillna(False).astype(bool).values
        acts[f"v3.2＋{n}提早回補"] = warn_with_exit(p_dd.values, THR, HOLD, s)
        cands[f"v3.2＋{n}提早回補"] = weights_custom(vol_fc, acts[f"v3.2＋{n}提早回補"], T, CUT, BAND)
        s20 = pd.Series(s, index=test).rolling(20, min_periods=1).max().astype(bool).values  # 訊號後 20 日持股拉滿
        cands[f"v3.2＋{n}後20日滿持股"] = weights_custom(vol_fc, base_act, T, CUT, BAND, boost=s20); acts[f"v3.2＋{n}後20日滿持股"] = base_act
    rows = []
    split = pd.Timestamp("2023-01-01")
    for n, w in cands.items():
        s = RM.strat(w, r) if n != "買進持有" else r
        for pn, m in (("2018–2022", test < split), ("2023–2026", test >= split), ("全期", test == test)):
            rows.append({"策略": n, "期間": pn, **M.stats(s[m]), "平均持股": w[m].mean(),
                         "警訊減碼天數": int(acts[n][np.asarray(m)].sum())})
    return pd.DataFrame(rows), test

def main():
    df = M.load(); f = build(df)
    f = f[f.index >= "2016-01-01"]
    auc = eval_auc(f); ev = eval_events(f)
    pd.set_option("display.width", 250); pd.set_option("display.max_columns", 30)
    print(auc[auc.樣本 == "全部日子"].round(3).to_string(index=False))
    print(auc[auc.樣本 != "全部日子"].round(3).to_string(index=False))
    print(ev.round(3).to_string(index=False))
    auc.to_csv(os.path.join(R, "bull_auc.csv"), index=False, encoding="utf-8-sig")
    ev.to_csv(os.path.join(R, "bull_events.csv"), index=False, encoding="utf-8-sig")
    corr = f[["dd60", "breadth", "margin_20", "fgn_combo", "ep_pct", "br_m"]].corr(method="spearman")
    print(corr.round(2))
    ov, test = overlay(df, f)
    print(ov.round(3).to_string(index=False))
    ov.to_csv(os.path.join(R, "bull_overlay.csv"), index=False, encoding="utf-8-sig")
    last = f.index[-1]; L = f.loc[last]
    today = {"date": str(last.date()), "breadth_200d_top100": round(float(L.breadth), 3),
             "breadth_60d_low": round(float(f.breadth.rolling(60).min().iloc[-1]), 3),
             "breadth_10m_all_market": round(float(L.br_m), 3), "margin_20d_chg": round(float(L.margin_20), 4),
             "dd60": round(float(L.dd60), 4), "fgn_5d_net_over_turnover": round(float(L.fgn_5), 4),
             "fut_fgn_chg5_over_oi": round(float(L.fut_fgn_chg5), 4), "ep": round(float(L.ep), 4),
             "ep_pct_since2015": round(float(L.ep_pct), 3), "dy": round(float(L.dy), 4), "dy_pct_since2015": round(float(L.dy_pct), 3),
             "signals": {n: bool(L[k]) if pd.notna(L[k]) else None for n, k in SIGS.items()}}
    json.dump(today, open(os.path.join(R, "bull_today.json"), "w", encoding="utf-8"), ensure_ascii=False, indent=2)
    print(json.dumps(today, ensure_ascii=False))

if __name__ == "__main__":
    main()
