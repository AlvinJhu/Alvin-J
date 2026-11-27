"""
選擇權／期貨籌碼因子對 v3.2 大跌預警的增量測試（做法比照 twvix_eval.py）。
這次測還沒測過的三類：
  1) 外資台指期淨部位（口數）：水準、5/20 日變化、「淨空單 > 7 萬口」門檻；另附占全市場 OI 比例版
  2) 近月 P/C 未平倉比（只算最近月合約）。全部合約的 P/C OI 已在 pc_eval.py / PR #11 測過，這裡只當對照
  3) 近月最大未平倉履約價（支撐／壓力）：Call 牆、Put 牆離現貨距離、區間寬度、現貨跌破 Put 牆
時點：籌碼一律延後 1 日（與 futures_eval.py 相同的保守設定）。
資料：data/ext/fut_tx_inst.csv、fut_tx_daily.csv（法人 2018-06 起）、opt_txo_daily_agg.csv（2016 起）
輸出：reports/optchip_single.csv、optchip_auc.csv、optchip_yearly.csv、optchip_strategy.csv、optchip_fgn70k.csv
"""
import os, sys; HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE)
sys.argv = sys.argv[:1]
import numpy as np, pandas as pd
import risk_model as RM, model as M, futures_eval as FE
from sklearn.metrics import roc_auc_score

EXT = os.path.join(HERE, "data", "ext")
def boot_idx(T, block, rng):
    st = rng.integers(0, T, int(np.ceil(T / block))); return (st[:, None] + np.arange(block)).ravel()[:T] % T

def paired_ci(y, a, b, block=20, n=1000, seed=0):
    y, a, b = map(np.asarray, (y, a, b)); rng = np.random.default_rng(seed); d = []
    for _ in range(n):
        ix = boot_idx(len(y), block, rng)
        if 0 < y[ix].mean() < 1: d.append(roc_auc_score(y[ix], a[ix]) - roc_auc_score(y[ix], b[ix]))
    d = np.array(d); return np.percentile(d, [2.5, 97.5]), float((d <= 0).mean())

def ens_pred(X, y, cols): return (RM.wf(X[cols], y, RM.logit) + RM.wf(X[cols], y, RM.xgb)) / 2

df = M.load(); X0, y_vol0, y_dd0, ew, lr = RM.build(df)
fu = FE.load_futures(df.index)
X0["opt_fgn_bias"] = FE.fut_features(df, fu)["opt_fgn_bias"]
live = list(X0.columns)
c = df.close
o = pd.read_csv(os.path.join(EXT, "opt_txo_daily_agg.csv"), parse_dates=["date"]).set_index("date").reindex(df.index)

F = pd.DataFrame(index=df.index)
net = fu.fut_fgn_net
F["fgn_net_k"] = -net / 1e4                                   # 外資淨空單（萬口，正＝淨空）
F["fgn_short70k"] = (net < -70000).astype(float).where(net.notna())
F["fgn_chg5_k"] = -net.diff(5) / 1e4; F["fgn_chg20_k"] = -net.diff(20) / 1e4   # 空單增加為正
F["fgn_net_oi"] = -net / fu.oi                                # 占全市場 OI 比例（跨年代較可比）
F["nearpc_oi"] = np.log(o.near_oi_P / o.near_oi_C)            # 近月 P/C 未平倉（新）
F["pc_oi_all"] = np.log(o.oi_P / o.oi_C)                      # 全部合約 P/C 未平倉（已測，對照）
F["callwall_dist"] = o.maxoi_strike_C / c - 1                 # 壓力離現貨（正＝在上方）
F["putwall_dist"] = c / o.maxoi_strike_P - 1                  # 現貨離支撐（負＝已跌破）
F["wall_width"] = (o.maxoi_strike_C - o.maxoi_strike_P) / c   # 市場認定的區間寬度
F["below_putwall"] = (c < o.maxoi_strike_P).astype(float).where(o.maxoi_strike_P.notna())
F = F.shift(FE.LAG).replace([np.inf, -np.inf], np.nan)
groups = {"外資期貨口數": ["fgn_net_k", "fgn_chg5_k", "fgn_chg20_k", "fgn_short70k"],
          "外資期貨占OI": ["fgn_net_oi"],
          "近月P/C OI": ["nearpc_oi"],
          "最大OI支撐壓力": ["callwall_dist", "putwall_dist", "wall_width", "below_putwall"]}
newc = [x for g in groups.values() for x in g]
XA = X0.join(F)[live + newc + ["pc_oi_all"]].dropna(); yA = y_dd0.reindex(XA.index)
print(f"樣本 {XA.index[0].date()}~{XA.index[-1].date()}，{len(XA)} 天；淨空>7萬口天數 {int(XA.fgn_short70k.sum())}（全在 {sorted(set(XA.index[XA.fgn_short70k > 0].year))}）")
print("與 v3.2 主因子相關：\n", XA[["rv22", "vix", "dd60", "opt_fgn_bias"] + newc].corr().loc[newc, ["rv22", "vix", "dd60", "opt_fgn_bias"]].round(2).to_string())

# 1) 單一因子（原始值，值高→大跌）
okd = yA.notna(); rows = []
for col in ["rv22", "opt_fgn_bias"] + newc + ["pc_oi_all"]:
    for per, m in (("全期", okd), ("保留期前", okd & (XA.index < RM.SPLIT)), ("2023–2026 保留期", okd & (XA.index >= RM.SPLIT))):
        if XA.loc[m, col].nunique() < 2: continue
        lo, hi = M.block_auc_ci(yA[m].values, XA.loc[m, col].values, 20)
        rows.append({"因子": col, "期間": per, "天數": int(m.sum()), "AUC(值高→大跌)": roc_auc_score(yA[m], XA.loc[m, col]), "95%下界": lo, "95%上界": hi})
sg = pd.DataFrame(rows); print(sg.round(3).to_string(index=False))
sg.to_csv(os.path.join(RM.R, "optchip_single.csv"), index=False, encoding="utf-8-sig")

# 2) 大跌模型增量
runs = {"A v3.2": live}
for g, cols in groups.items(): runs[f"+{g}"] = live + cols
runs["+全部新因子"] = live + newc
runs["對照：+全部合約P/C OI"] = live + ["pc_oi_all"]
P = {k: ens_pred(XA, yA, cols) for k, cols in runs.items()}
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
auc.to_csv(os.path.join(RM.R, "optchip_auc.csv"), index=False, encoding="utf-8-sig")

yr = []
for Y in sorted(set(common.year)):
    m = common.year == Y
    if 0 < y[m].mean() < 1:
        yr.append({"年": Y, "天數": int(m.sum()), "大跌基準率": y[m].mean(), **{k: roc_auc_score(y[m], P[k].reindex(common)[m]) for k in runs}})
yr = pd.DataFrame(yr); print(yr.round(3).to_string(index=False))
yr.to_csv(os.path.join(RM.R, "optchip_yearly.csv"), index=False, encoding="utf-8-sig")

# 3) 策略（v3.2 固定參數：目標 22%、減碼 1/2、門檻 25%、持有 10 日；波動預測不變）
volb = ["rv1", "rv5", "rv22", "rv66", "park5", "neg5", "vix"]
vol = np.exp(RM.wf(XA[volb], y_vol0.reindex(XA.index), RM.ols))
r = (df.close.shift(-1) / df.close - 1).reindex(common); cm = common[r.notna().values]; r = r[cm]
def run(p): return RM.strat(RM.weights(vol.reindex(cm), p.reindex(cm).values, 0.22, 0.5, 0.25, 0.2, 10), r)
def sharpe(x): return x.mean() / x.std() * np.sqrt(252)
P3 = (("保留期前", cm < RM.SPLIT), ("2023–2026 保留期", cm >= RM.SPLIT), ("全期", np.ones(len(cm), bool)))
rows = []; sA = run(P["A v3.2"])
for k, p in P.items():
    s = run(p); w = RM.weights(vol.reindex(cm), p.reindex(cm).values, 0.22, 0.5, 0.25, 0.2, 10)
    warn = RM.warn_active(p.reindex(cm).values, 0.25, 10)
    for per, m in P3:
        row = {"模型": k, "期間": per, **M.stats(s[m]), "警訊日比例": warn[m].mean(), "平均持股": w[m].mean()}
        if k != "A v3.2":
            a_, b_ = s[m].values, sA[m].values; rng = np.random.default_rng(0); bs = []
            for _ in range(1000):
                ix = boot_idx(len(a_), 20, rng); bs.append(sharpe(pd.Series(a_[ix])) - sharpe(pd.Series(b_[ix])))
            lo, hi = np.percentile(bs, [2.5, 97.5])
            row.update({"Sharpe差vsA": sharpe(s[m]) - sharpe(sA[m]), "差_95%下界": lo, "差_95%上界": hi})
        rows.append(row)
for per, m in P3: rows.append({"模型": "買進持有", "期間": per, **M.stats(r[m])})
st = pd.DataFrame(rows); print(st.round(3).to_string(index=False))
st.to_csv(os.path.join(RM.R, "optchip_strategy.csv"), index=False, encoding="utf-8-sig")

# 4) 7 萬口門檻：歷史上只出現在 2026，模型無從學習，改做事件描述
E = pd.DataFrame({"淨空>7萬": XA.fgn_short70k, "20日內跌7%": yA, "v3.2機率": P["A v3.2"]}).dropna()
E["年"] = E.index.year
ev = E[E.index.year == 2026].groupby("淨空>7萬").agg(天數=("20日內跌7%", "size"), 大跌比例=("20日內跌7%", "mean"), v3_2平均機率=("v3.2機率", "mean"))
fwd = (c.shift(-20) / c - 1).reindex(XA.index)
ev["之後20日平均報酬"] = [fwd[(XA.index.year == 2026) & (XA.fgn_short70k == g)].mean() for g in ev.index]
print("2026 年淨空單 7 萬口門檻事件：\n", ev.round(3).to_string())
ev.to_csv(os.path.join(RM.R, "optchip_fgn70k.csv"), encoding="utf-8-sig")
last = F.dropna(how="all").iloc[-1]; print("最新（已延後 1 日）：", F.dropna(how="all").index[-1].date(), last.round(3).to_dict())
