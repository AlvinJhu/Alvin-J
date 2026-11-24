import sys, os; HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE); os.chdir(HERE)
"""台指選擇權全市場 P/C Ratio 再驗證（2026-11-24）：
1) 不擬合、固定方向的原始 AUC（低 P/C=過熱→大跌；高 P/C=恐慌→大漲，即坊間「反向指標」說法）
2) 加進 v3.2 大跌模型的增量 AUC（walk-forward）
3) 對 v3.2 持股規則（0.22 目標、>25% 減碼 1/2、維持 10 日）Sharpe/MDD 的影響
4) 對 20 日大漲（最高收盤 >+7%）與 20 日漲跌方向的預測力
資料：data/ext/opt_txo_daily_agg.csv（FinMind TaiwanOptionDaily 全履約價彙總，fetch_local.py --optdaily）。"""
SRC = "taifex" if "--taifex" in sys.argv else "finmind"  # 須在 import risk_model 前讀（它會清掉 argv）
import numpy as np, pandas as pd, model as M, risk_model as RM, futures_eval as FE
from sklearn.metrics import roc_auc_score
df = M.load(); X, yv, ydd, ew, lr = RM.build(df)
X["opt_fgn_bias"] = FE.fut_features(df, FE.load_futures(df.index))["opt_fgn_bias"]
if SRC == "taifex":  # 期交所官方 P/C Ratio（fetch_taifex_pc.py）
    t = pd.read_csv("data/ext/taifex_pcratio.csv", parse_dates=["date"]).set_index("date").reindex(df.index)
    o = pd.DataFrame({"oi_P": t.put_oi, "oi_C": t.call_oi, "vol_P": t.put_vol, "vol_C": t.call_vol})
else:
    o = pd.read_csv("data/ext/opt_txo_daily_agg.csv", parse_dates=["date"]).set_index("date").reindex(df.index)
P = pd.DataFrame(index=df.index)
P["pc_oi"] = np.log(o.oi_P / o.oi_C); P["pc_vol"] = np.log(o.vol_P / o.vol_C); P["pc_vol5"] = P.pc_vol.rolling(5).mean()
P["pc_oi_z60"] = (P.pc_oi - P.pc_oi.rolling(60).mean()) / P.pc_oi.rolling(60).std()
P = P.shift(1)  # 用前一日已公布值，避免偷看
c = df.close; fut = pd.concat([c.shift(-k) for k in range(1, 21)], axis=1); full = fut.notna().all(axis=1)
y_up = ((fut.max(axis=1) / c - 1) > 0.07).astype(float).where(full)
y_dir = ((c.shift(-20) / c - 1) > 0).astype(float).where(full)
A = pd.concat([X, P], axis=1).dropna(); SPL = RM.SPLIT
per = (("2018–22", A.index < SPL), ("2023–26", A.index >= SPL), ("全期", A.index == A.index))
print("樣本", A.index[0].date(), A.index[-1].date(), len(A))
rows = []
for tn, y, sign in (("大跌(20日內-7%)", ydd, -1), ("大漲(20日內+7%)", y_up, 1), ("20日上漲", y_dir, 1)):
    y = y.reindex(A.index)
    for f in P.columns:
        for pn, m in per:
            mm = m & y.notna(); lo, hi = M.block_auc_ci(y[mm].values, sign * A.loc[mm, f].values, 20)
            rows.append({"標的": tn, "指標": f, "期間": pn, "AUC": roc_auc_score(y[mm], sign * A.loc[mm, f]), "下界": lo, "上界": hi})
T = pd.DataFrame(rows); print(T.round(3).to_string(index=False))
y = ydd.reindex(A.index); base = list(X.columns); prob = {}
for n, cols in (("v3.2", base), ("v3.2+PC", base + list(P.columns))):
    prob[n] = (RM.wf(A[cols], y, RM.logit) + RM.wf(A[cols], y, RM.xgb)) / 2
    for pn, m in per:
        mm = m & y.notna() & prob[n].notna(); lo, hi = M.block_auc_ci(y[mm].values, prob[n][mm].values, 20)
        print(n, pn, f"AUC {roc_auc_score(y[mm], prob[n][mm]):.3f} [{lo:.2f},{hi:.2f}]")
vc = ["rv1", "rv5", "rv22", "rv66", "park5", "neg5", "vix"]
vol = np.exp(RM.wf(A[vc], yv.reindex(A.index), RM.ols))
r = (c.shift(-1) / c - 1).reindex(A.index)
t = vol.dropna().index.intersection(prob["v3.2+PC"].dropna().index); t = t[r[t].notna()]
S = []
for n in prob:
    w = RM.weights(vol[t], prob[n][t].values, 0.22, 0.5, 0.25, 0.2, 10); s = RM.strat(w, r[t])
    for pn, m in (("2018–22", t < SPL), ("2023–26", t >= SPL), ("全期", t == t)):
        S.append({"策略": n, "期間": pn, **M.stats(s[m]), "平均持股": w[m].mean()})
for pn, m in (("2018–22", t < SPL), ("2023–26", t >= SPL), ("全期", t == t)):
    S.append({"策略": "買進持有", "期間": pn, **M.stats(r[t][m]), "平均持股": 1.0})
S = pd.DataFrame(S); print(S.round(3).to_string(index=False))
q = pd.qcut(A.pc_oi, 5)
G = pd.DataFrame({"q": q, "大跌": ydd.reindex(A.index), "大漲": y_up.reindex(A.index), "20日報酬": (c.shift(-20) / c - 1).reindex(A.index)})
print(G.groupby("q", observed=True).agg(["mean", "size"]).round(3).iloc[:, [0, 1, 2, 4]])
print("P/C 未平倉比(原始倍數) 分位：", np.exp(A.pc_oi.shift(-1)).describe().round(2).to_dict())
os.makedirs("reports", exist_ok=True)
T.to_csv(f"reports/pc_eval2_{SRC}_auc.csv", index=False, encoding="utf-8-sig"); S.to_csv(f"reports/pc_eval2_{SRC}_strategy.csv", index=False, encoding="utf-8-sig")
