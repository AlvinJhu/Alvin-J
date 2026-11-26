"""
比特幣恐慌貪婪指數（Crypto Fear & Greed，alternative.me，2018-02 起）與比特幣本身報酬/波動，
能否改善台股 20 日大跌（≥7%）預警？做法比照 pc_eval2.py / opt_ablation.py。

資料（data/ext/）：crypto_fng.csv（日期=發布日，UTC 00:00 發布，台股開盤前已知）
                 btc_usdt.csv（Binance BTCUSDT 日收盤，UTC 00:00 收；台股 d 日只用 d-1 日收盤）
時點：台股交易日 d 使用 FNG(d) 與 BTC 收盤(d-1)，皆在台北 08:00 前已知，無前視。

A：v3.2 LIVE 因子（含外資選擇權偏向）
B：A + 恐慌指數（fng、fng 20 日變化）
C：A + 比特幣（5/20 日報酬、22 日波動、距 60 日高點）
D：A + 兩者
指標：單一因子固定方向 AUC（區塊 bootstrap CI）、walk-forward 樣本外 AUC 與配對差異 CI、v3.2 參數 Sharpe/MDD
輸出：reports/btc_fng_single.csv、btc_fng_auc.csv、btc_fng_yearly.csv、btc_fng_strategy.csv
"""
import os, sys
sys.argv = sys.argv[:1]
import numpy as np, pandas as pd
import risk_model as RM, model as M, futures_eval as FE
from sklearn.metrics import roc_auc_score

EXT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data", "ext")

def crypto_features(index):
    f = pd.read_csv(os.path.join(EXT, "crypto_fng.csv"), parse_dates=["date"]).set_index("date").fng
    b = pd.read_csv(os.path.join(EXT, "btc_usdt.csv"), parse_dates=["date"]).set_index("date").close
    cal = pd.date_range(min(f.index[0], b.index[0]), max(f.index[-1], b.index[-1]) + pd.Timedelta(days=1))
    f = f.reindex(cal).ffill(limit=3); b = b.reindex(cal).ffill(limit=3)
    bl = np.log(b).diff()
    C = pd.DataFrame(index=cal)
    C["fng"] = f / 100
    C["fng_chg20"] = (f - f.shift(20)) / 100
    C["btc_ret5"] = b.pct_change(5).shift(1)          # d-1 收盤
    C["btc_ret20"] = b.pct_change(20).shift(1)
    C["btc_rv22"] = np.log(bl.rolling(22).std() * np.sqrt(365)).shift(1)
    C["btc_dd60"] = (b / b.rolling(60).max() - 1).shift(1)
    return C.reindex(index)

def ens_pred(X, y, cols):
    return (RM.wf(X[cols], y, RM.logit) + RM.wf(X[cols], y, RM.xgb)) / 2

def paired_ci(y, a, b, block=20, n=1000, seed=0):
    y, a, b = map(np.asarray, (y, a, b)); T = len(y); rng = np.random.default_rng(seed); d = []
    for _ in range(n):
        st = rng.integers(0, T, int(np.ceil(T / block))); ix = (st[:, None] + np.arange(block)).ravel()[:T] % T
        if 0 < y[ix].mean() < 1: d.append(roc_auc_score(y[ix], a[ix]) - roc_auc_score(y[ix], b[ix]))
    d = np.array(d); return np.percentile(d, [2.5, 97.5]), float((d <= 0).mean())

df = M.load(); X0, y_vol0, y_dd0, ew, lr = RM.build(df)
X0["opt_fgn_bias"] = FE.fut_features(df, FE.load_futures(df.index))["opt_fgn_bias"]
CR = crypto_features(df.index); FNG = ["fng", "fng_chg20"]; BTC = ["btc_ret5", "btc_ret20", "btc_rv22", "btc_dd60"]
X0 = X0.join(CR)
live = [c for c in X0 if c not in ("dd60", "bias240", "opt_fgn_bias", *FNG, *BTC)] + ["dd60", "bias240", "opt_fgn_bias"]
vol_cols = ["rv1", "rv5", "rv22", "rv66", "park5", "neg5", "vix"]
XA = X0[live + FNG + BTC].dropna(); yA = y_dd0.reindex(XA.index); vA = y_vol0.reindex(XA.index)
print("樣本：", XA.index[0].date(), "~", XA.index[-1].date(), len(XA), "天；LIVE 因子：", live)

# 1) 單一因子固定方向 AUC（不經模型；>0.5 代表值越高越易大跌）
ok = yA.notna(); rows = []
for c in FNG + BTC:
    for per, m in (("全期", ok), ("2023–2026 保留期", ok & (XA.index >= RM.SPLIT))):
        a = roc_auc_score(yA[m], XA.loc[m, c]); lo, hi = M.block_auc_ci(yA[m].values, XA.loc[m, c].values, 20)
        rows.append({"因子": c, "期間": per, "天數": int(m.sum()), "AUC(值高→大跌)": a, "95%下界": lo, "95%上界": hi})
sg = pd.DataFrame(rows); print(sg.round(3).to_string(index=False))
sg.to_csv(os.path.join(RM.R, "btc_fng_single.csv"), index=False, encoding="utf-8-sig")

# 2) walk-forward 樣本外 AUC
vol = np.exp(RM.wf(XA[vol_cols], vA, RM.ols))
runs = {"A v3.2": live, "B +恐慌指數": live + FNG, "C +比特幣": live + BTC, "D +兩者": live + FNG + BTC}
runs = {k: (ens_pred(XA, yA, cols), vol) for k, cols in runs.items()}
common = runs["A v3.2"][0].dropna().index
common = common[yA.reindex(common).notna().values]; y = yA.reindex(common)
pers = {"2020–2022": common < RM.SPLIT, "2023–2026 保留期": common >= RM.SPLIT, "全期": np.ones(len(common), bool)}
rows = []
for per, m in pers.items():
    pa = runs["A v3.2"][0].reindex(common)[m]
    for k, (p, _) in runs.items():
        pp = p.reindex(common)[m]; lo, hi = M.block_auc_ci(y[m].values, pp.values, 20)
        r = {"期間": per, "模型": k, "天數": int(m.sum()), "AUC": roc_auc_score(y[m], pp), "95%下界": lo, "95%上界": hi}
        if k != "A v3.2":
            (dlo, dhi), pv = paired_ci(y[m].values, pp.values, pa.values)
            r.update({"此版減A": r["AUC"] - roc_auc_score(y[m], pa), "差_95%下界": dlo, "差_95%上界": dhi, "差<=0機率": pv})
        rows.append(r)
auc = pd.DataFrame(rows); print(auc.round(3).to_string(index=False))
auc.to_csv(os.path.join(RM.R, "btc_fng_auc.csv"), index=False, encoding="utf-8-sig")

yr = []
for Y in sorted(set(common.year)):
    m = common.year == Y
    if 0 < y[m].mean() < 1:
        yr.append({"年": Y, "天數": int(m.sum()), "大跌基準率": y[m].mean(),
                   **{k: roc_auc_score(y[m], p.reindex(common)[m]) for k, (p, _) in runs.items()}})
yr = pd.DataFrame(yr); print(yr.round(3).to_string(index=False))
yr.to_csv(os.path.join(RM.R, "btc_fng_yearly.csv"), index=False, encoding="utf-8-sig")

# 3) 策略（v3.2 參數：目標波動 22%、大跌機率 >25% 減碼至 1/2、維持 10 日、調整帶 0.2）
r = (df.close.shift(-1) / df.close - 1).reindex(common); cm = common[r.notna().values]; r = r[cm]
rows = []
for k, (p, v) in runs.items():
    w = RM.weights(v.reindex(cm), p.reindex(cm).values, 0.22, 0.5, 0.25, 0.2, 10); s = RM.strat(w, r)
    warn = RM.warn_active(p.reindex(cm).values, 0.25, 10)
    for per, m in (("2020–2022", cm < RM.SPLIT), ("2023–2026 保留期", cm >= RM.SPLIT), ("全期", np.ones(len(cm), bool))):
        rows.append({"模型": k, "期間": per, **M.stats(s[m]), "警訊日比例": warn[m].mean()})
for per, m in (("2020–2022", cm < RM.SPLIT), ("2023–2026 保留期", cm >= RM.SPLIT), ("全期", np.ones(len(cm), bool))):
    rows.append({"模型": "買進持有", "期間": per, **M.stats(r[m])})
st = pd.DataFrame(rows); print(st.round(3).to_string(index=False))
st.to_csv(os.path.join(RM.R, "btc_fng_strategy.csv"), index=False, encoding="utf-8-sig")
print("最新：", CR.dropna().iloc[-1].round(3).to_dict())
