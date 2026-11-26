"""
台指選擇權波動率指數（期交所 TAIWAN VIX）對 v3.2 大跌預警的增量測試。做法比照 tej_eval.py / btc_fng_eval.py。
重點：v3.2 的訊號主要來自波動（HAR 已實現波動＋美國 VIX），所以分兩層檢查台指 VIX 有沒有「波動以外」的資訊：
  1) 波動預測：HAR+幅度+美VIX 加上台指 VIX，保留期 R2/QLIKE 是否改善（QLIKE 差異區塊 bootstrap CI）
  2) 大跌機率：v3.2 因子加上台指 VIX 特徵，walk-forward 樣本外 AUC 與配對差異 CI；另測「台指 VIX 取代美 VIX」
  3) 策略：v3.2 參數（目標 22%、減碼 1/2、門檻 25%、持有 10 日）Sharpe/MDD
時點：台指 VIX 日盤收在 13:45，晚於現貨 13:30 收盤，但仍在當日收盤後、預測 d+1..d+20 之前已知。
資料：data/ext/twvix_daily.csv（fetch_twvix.py，一次性 GitHub Actions 抓期交所官方分鐘資料）
輸出：reports/twvix_vol.csv、twvix_single.csv、twvix_auc.csv、twvix_yearly.csv、twvix_strategy.csv
"""
import os, sys; HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE)
sys.argv = sys.argv[:1]
import numpy as np, pandas as pd
import risk_model as RM, model as M, futures_eval as FE
from sklearn.metrics import roc_auc_score

EXT = os.path.join(HERE, "data", "ext"); ANN = RM.ANN
def boot_idx(T, block, rng):
    st = rng.integers(0, T, int(np.ceil(T / block))); return (st[:, None] + np.arange(block)).ravel()[:T] % T

def paired_ci(y, a, b, block=20, n=1000, seed=0):
    y, a, b = map(np.asarray, (y, a, b)); rng = np.random.default_rng(seed); d = []
    for _ in range(n):
        ix = boot_idx(len(y), block, rng)
        if 0 < y[ix].mean() < 1: d.append(roc_auc_score(y[ix], a[ix]) - roc_auc_score(y[ix], b[ix]))
    d = np.array(d); return np.percentile(d, [2.5, 97.5]), float((d <= 0).mean())

def ql(y, p): v, f = np.exp(2 * y), np.exp(2 * p); return v / f - np.log(v / f) - 1
def ql_ci(y, a, b, block=20, n=1000, seed=0):  # 新版減舊版的平均 QLIKE（負＝改善）
    d = ql(np.asarray(y), np.asarray(a)) - ql(np.asarray(y), np.asarray(b)); rng = np.random.default_rng(seed)
    bs = [d[boot_idx(len(d), block, rng)].mean() for _ in range(n)]
    return float(d.mean()), np.percentile(bs, [2.5, 97.5])

def ens_pred(X, y, cols): return (RM.wf(X[cols], y, RM.logit) + RM.wf(X[cols], y, RM.xgb)) / 2

tv = pd.read_csv(os.path.join(EXT, "twvix_daily.csv"), parse_dates=["date"]).set_index("date").sort_index()
df = M.load(); X0, y_vol0, y_dd0, ew, lr = RM.build(df)
X0["opt_fgn_bias"] = FE.fut_features(df, FE.load_futures(df.index))["opt_fgn_bias"]
live = list(X0.columns)
t = tv.reindex(df.index)
miss = t.close.isna().mean(); t = t.ffill(limit=3)
T = pd.DataFrame(index=df.index)
T["twvix"] = np.log(t.close)                               # 水準
T["twvix_vrp"] = np.log(t.close / 100) - X0.rv22           # 隱含 − 已實現（對數比）：選擇權買方願付的保險溢價
T["twvix_chg5"] = T.twvix.diff(5)                          # 5 日變化（恐慌升溫）
T["twvix_z60"] = (T.twvix - T.twvix.rolling(60).mean()) / T.twvix.rolling(60).std()
T["twvix_us"] = T.twvix - X0.vix                           # 台 VIX 相對美 VIX（台股特有風險）
T["twvix_rng"] = np.log(t.high / t.low).rolling(5).mean()  # 日內台 VIX 振幅
newc = list(T.columns)
U = pd.DataFrame(index=df.index)  # 對照組：同樣做法套在美國 VIX，檢查增量是否為台股特有
U["usvix_vrp"] = X0.vix - np.log(100) - X0.rv22; U["usvix_chg5"] = X0.vix.diff(5)
U["usvix_z60"] = (X0.vix - X0.vix.rolling(60).mean()) / X0.vix.rolling(60).std()
XA = X0.join(T).join(U)[live + newc + list(U.columns)].dropna(); yA = y_dd0.reindex(XA.index); vA = y_vol0.reindex(XA.index)
print(f"台指 VIX {tv.index.min().date()}~{tv.index.max().date()}，交易日缺值 {miss:.1%}；樣本 {XA.index[0].date()}~{XA.index[-1].date()}，{len(XA)} 天")
print("相關係數：", XA[["rv22", "vix"] + newc].corr().loc[["twvix"]].round(2).to_dict("records"))

# 1) 波動預測
volb = ["rv1", "rv5", "rv22", "rv66", "park5", "neg5", "vix"]
vsets = {"A HAR+幅度+美VIX（LIVE）": volb, "B +台VIX": volb + ["twvix"], "C 台VIX取代美VIX": [c for c in volb if c != "vix"] + ["twvix"],
         "D +台VIX全部特徵": volb + newc}
vf = {k: RM.wf(XA[c], vA, RM.ols) for k, c in vsets.items()}
ok = vA.notna() & vf["A HAR+幅度+美VIX（LIVE）"].notna(); rows = []
for per, m in (("保留期前", ok & (XA.index < RM.SPLIT)), ("2023–2026 保留期", ok & (XA.index >= RM.SPLIT)), ("全期", ok)):
    yy = vA[m]; pa = vf["A HAR+幅度+美VIX（LIVE）"][m]
    for k, p in vf.items():
        r = {"期間": per, "模型": k, "天數": int(m.sum()), "R2": RM.r2(yy, p[m], yy.mean()), "QLIKE": RM.qlike(yy.values, p[m].values)}
        if not k.startswith("A"):
            d, (lo, hi) = ql_ci(yy.values, p[m].values, pa.values); r.update({"QLIKE差(負=改善)": d, "差_95%下界": lo, "差_95%上界": hi})
        rows.append(r)
vt = pd.DataFrame(rows); print(vt.round(4).to_string(index=False))
vt.to_csv(os.path.join(RM.R, "twvix_vol.csv"), index=False, encoding="utf-8-sig")

# 2a) 單一因子
okd = yA.notna(); rows = []
for c in ["rv22", "vix"] + newc + list(U.columns):
    for per, m in (("全期", okd), ("2023–2026 保留期", okd & (XA.index >= RM.SPLIT))):
        lo, hi = M.block_auc_ci(yA[m].values, XA.loc[m, c].values, 20)
        rows.append({"因子": c, "期間": per, "天數": int(m.sum()), "AUC(值高→大跌)": roc_auc_score(yA[m], XA.loc[m, c]), "95%下界": lo, "95%上界": hi})
sg = pd.DataFrame(rows); print(sg.round(3).to_string(index=False))
sg.to_csv(os.path.join(RM.R, "twvix_single.csv"), index=False, encoding="utf-8-sig")

# 2b) 大跌模型
runs = {"A v3.2": live, "+台VIX水準": live + ["twvix"], "+台VIX溢價/變化": live + ["twvix_vrp", "twvix_chg5", "twvix_z60"],
        "+台VIX全部": live + newc, "對照：+美VIX溢價/變化": live + list(U.columns), "台VIX取代美VIX": [c for c in live if c != "vix"] + ["twvix"]}
P = {k: ens_pred(XA, yA, c) for k, c in runs.items()}
P["只用預測波動(LIVE)"] = vf["A HAR+幅度+美VIX（LIVE）"]; P["只用台VIX水準"] = XA.twvix
common = P["A v3.2"].dropna().index; common = common[yA.reindex(common).notna().values]; y = yA.reindex(common)
pers = {"保留期前": common < RM.SPLIT, "2023–2026 保留期": common >= RM.SPLIT, "全期": np.ones(len(common), bool)}
rows = []
for per, m in pers.items():
    pa = P["A v3.2"].reindex(common)[m]
    for k, p in P.items():
        pp = p.reindex(common)[m]; lo, hi = M.block_auc_ci(y[m].values, pp.values, 20)
        r = {"期間": per, "模型": k, "天數": int(m.sum()), "基準率": y[m].mean(), "AUC": roc_auc_score(y[m], pp), "95%下界": lo, "95%上界": hi}
        if k != "A v3.2":
            (dlo, dhi), pv = paired_ci(y[m].values, pp.values, pa.values)
            r.update({"此版減A": r["AUC"] - roc_auc_score(y[m], pa), "差_95%下界": dlo, "差_95%上界": dhi, "差<=0機率": pv})
        rows.append(r)
auc = pd.DataFrame(rows); print(auc.round(3).to_string(index=False))
auc.to_csv(os.path.join(RM.R, "twvix_auc.csv"), index=False, encoding="utf-8-sig")

yr = []
for Y in sorted(set(common.year)):
    m = common.year == Y
    if 0 < y[m].mean() < 1:
        yr.append({"年": Y, "天數": int(m.sum()), "大跌基準率": y[m].mean(), **{k: roc_auc_score(y[m], P[k].reindex(common)[m]) for k in runs}})
yr = pd.DataFrame(yr); print(yr.round(3).to_string(index=False))
yr.to_csv(os.path.join(RM.R, "twvix_yearly.csv"), index=False, encoding="utf-8-sig")

# 3) 策略（v3.2 固定參數）
r = (df.close.shift(-1) / df.close - 1).reindex(common); cm = common[r.notna().values]; r = r[cm]
volA, volB = np.exp(vf["A HAR+幅度+美VIX（LIVE）"]), np.exp(vf["B +台VIX"])
strats = {"A v3.2": (P["A v3.2"], volA), "+台VIX全部（只改大跌）": (P["+台VIX全部"], volA),
          "+台VIX水準（只改大跌）": (P["+台VIX水準"], volA), "+台VIX（大跌＋波動都改）": (P["+台VIX全部"], volB),
          "台VIX取代美VIX": (P["台VIX取代美VIX"], np.exp(vf["C 台VIX取代美VIX"]))}
rows = []; P3 = (("保留期前", cm < RM.SPLIT), ("2023–2026 保留期", cm >= RM.SPLIT), ("全期", np.ones(len(cm), bool)))
for k, (p, v) in strats.items():
    w = RM.weights(v.reindex(cm), p.reindex(cm).values, 0.22, 0.5, 0.25, 0.2, 10); s = RM.strat(w, r)
    warn = RM.warn_active(p.reindex(cm).values, 0.25, 10)
    for per, m in P3: rows.append({"模型": k, "期間": per, **M.stats(s[m]), "警訊日比例": warn[m].mean(), "平均持股": w[m].mean()})
for per, m in P3: rows.append({"模型": "買進持有", "期間": per, **M.stats(r[m])})
st = pd.DataFrame(rows); print(st.round(3).to_string(index=False))
st.to_csv(os.path.join(RM.R, "twvix_strategy.csv"), index=False, encoding="utf-8-sig")
print("最新台指 VIX：", T.dropna().iloc[-1].round(3).to_dict(), "日期", T.dropna().index[-1].date())
