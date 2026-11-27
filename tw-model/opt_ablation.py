"""
外資選擇權偏向（opt_fgn_bias）消融回測：v3.2 預設（含此因子）vs 拿掉此因子
A：含 opt（樣本自 2018 起，受選擇權資料限制）
B：同樣本拿掉 opt（純看因子貢獻）
C：拿掉 opt、樣本自 2016 起（實際替代方案：多 2 年訓練資料）
指標：配對區塊 bootstrap AUC 差異（20 日區塊）、分年度 AUC、v3.2 固定參數 Sharpe/MDD
輸出：reports/opt_ablation_auc.csv、opt_ablation_yearly.csv、opt_ablation_strategy.csv
"""
import os, sys
sys.argv = sys.argv[:1]
import numpy as np, pandas as pd
import risk_model as RM, model as M, futures_eval as FE
from sklearn.metrics import roc_auc_score

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
base_cols = [c for c in X0 if c not in ("dd60", "bias240", "opt_fgn_bias")] + ["dd60", "bias240"]
vol_cols = ["rv1", "rv5", "rv22", "rv66", "park5", "neg5", "vix"]
runs = {}
XA = X0.dropna(); yA = y_dd0.reindex(XA.index); vA = y_vol0.reindex(XA.index)
runs["A 含外資選擇權"] = (ens_pred(XA, yA, base_cols + ["opt_fgn_bias"]), np.exp(RM.wf(XA[vol_cols], vA, RM.ols)))
runs["B 同樣本拿掉"] = (ens_pred(XA, yA, base_cols), runs["A 含外資選擇權"][1])
XC = X0.drop(columns="opt_fgn_bias"); XC = XC[XC.index >= "2016-01-01"].dropna()
yC = y_dd0.reindex(XC.index); vC = y_vol0.reindex(XC.index)
runs["C 拿掉、2016 起訓練"] = (ens_pred(XC, yC, base_cols), np.exp(RM.wf(XC[vol_cols], vC, RM.ols)))

common = runs["A 含外資選擇權"][0].dropna().index
for k in runs: common = common.intersection(runs[k][0].dropna().index)
common = common[y_dd0.reindex(common).notna().values]
y = y_dd0.reindex(common)
pers = {"2020–2022": common < RM.SPLIT, "2023–2026 保留期": common >= RM.SPLIT, "全期": np.ones(len(common), bool)}
rows = []
for per, m in pers.items():
    for k, (p, _) in runs.items():
        pp = p.reindex(common)[m]; lo, hi = M.block_auc_ci(y[m].values, pp.values, 20)
        r = {"期間": per, "模型": k, "天數": int(m.sum()), "AUC": roc_auc_score(y[m], pp), "95%下界": lo, "95%上界": hi}
        if k != "A 含外資選擇權":
            (dlo, dhi), pv = paired_ci(y[m].values, runs["A 含外資選擇權"][0].reindex(common)[m].values, pp.values)
            r.update({"A減此版_AUC差": r["AUC"] * 0 + roc_auc_score(y[m], runs["A 含外資選擇權"][0].reindex(common)[m]) - r["AUC"],
                      "差_95%下界": dlo, "差_95%上界": dhi, "差<=0機率": pv})
        rows.append(r)
auc = pd.DataFrame(rows); print(auc.round(3).to_string(index=False))
auc.to_csv(os.path.join(RM.R, "opt_ablation_auc.csv"), index=False, encoding="utf-8-sig")

yr = []
for Y in sorted(set(common.year)):
    m = common.year == Y
    if 0 < y[m].mean() < 1:
        yr.append({"年": Y, "天數": int(m.sum()), "大跌基準率": y[m].mean(),
                   **{k: roc_auc_score(y[m], p.reindex(common)[m]) for k, (p, _) in runs.items()}})
yr = pd.DataFrame(yr); print(yr.round(3).to_string(index=False))
yr.to_csv(os.path.join(RM.R, "opt_ablation_yearly.csv"), index=False, encoding="utf-8-sig")

r = (df.close.shift(-1) / df.close - 1).reindex(common); okr = r.notna().values; cm = common[okr]; r = r[cm]
rows = []
for cut in (0.5, 0.25):
    for k, (p, v) in runs.items():
        w = RM.weights(v.reindex(cm), p.reindex(cm).values, 0.22, cut, 0.25, 0.2, 10); s = RM.strat(w, r)
        warn = RM.warn_active(p.reindex(cm).values, 0.25, 10)
        for per, m in (("2020–2022", cm < RM.SPLIT), ("2023–2026 保留期", cm >= RM.SPLIT), ("全期", np.ones(len(cm), bool))):
            rows.append({"減碼": cut, "模型": k, "期間": per, **M.stats(s[m]), "警訊日比例": warn[m].mean()})
    for per, m in (("2020–2022", cm < RM.SPLIT), ("2023–2026 保留期", cm >= RM.SPLIT), ("全期", np.ones(len(cm), bool))):
        if cut == 0.5: rows.append({"減碼": "-", "模型": "買進持有", "期間": per, **M.stats(r[m])})
st = pd.DataFrame(rows); print(st.round(3).to_string(index=False))
st.to_csv(os.path.join(RM.R, "opt_ablation_strategy.csv"), index=False, encoding="utf-8-sig")
