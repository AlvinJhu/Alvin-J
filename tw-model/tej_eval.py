"""
TEJ 全市場個股資料（2015–2026，含下市櫃；FCU 學校授權）彙總成市場層級指標，
測試能否改善台股 20 日大跌（≥7%）預警。做法比照 btc_fng_eval.py / pc_eval2.py。

授權：原始 TEJ 個股檔不放 repo，只放每日/每週全市場彙總（data/ext/tej_*.csv）。
資料：
  tej_tdcc_weekly_agg.csv  集保股權分散（每週，資料日=集保結算日，多為週五）
  tej_inst_daily_agg.csv   個股三大法人買賣超彙總
  tej_daily_breadth.csv    個股日資料彙總（注意/處置/漲跌停/寬度/估值），使用者電腦上產生
時點：集保在結算日當晚至下週初才公布 → 保守假設結算日 +3 個日曆日（下週一）起可用，
      --tdcc-lag 7 做敏感度。日資料以當日收盤後已知，模型本來就以 d 日收盤預測 d+1..d+20。
指標：單一因子固定方向 AUC（區塊 bootstrap CI）、walk-forward 樣本外 AUC 與配對差異 CI、v3.2 參數 Sharpe/MDD
輸出：reports/tej_single.csv、tej_auc.csv、tej_yearly.csv、tej_strategy.csv
"""
import os, sys, argparse
ap = argparse.ArgumentParser(); ap.add_argument("--tdcc-lag", type=int, default=3)
ap.add_argument("--tag", default="tej"); args = ap.parse_args(); sys.argv = sys.argv[:1]
import numpy as np, pandas as pd
import risk_model as RM, model as M, futures_eval as FE
from sklearn.metrics import roc_auc_score

EXT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data", "ext")
def rd(n): 
    p = os.path.join(EXT, n); return pd.read_csv(p, parse_dates=["date"]).set_index("date").sort_index() if os.path.exists(p) else None

def tdcc_features(index, lag):
    w = rd("tej_tdcc_weekly_agg.csv"); w = w[w.n > 1500]
    F = pd.DataFrame(index=w.index)
    F["tdcc_dbig4"] = w.d_big_mean.rolling(4).sum()         # 千張大戶持股比 4 週變化（個股平均）
    F["tdcc_bigsw4"] = w.big_sw.diff(4)                     # 千張大戶持股比（股數加權）4 週變化
    F["tdcc_bigdown4"] = w.frac_big_down.rolling(4).mean()  # 大戶比下降的個股比例
    F["tdcc_dsmall4"] = w.d_small_mean.rolling(4).sum()      # 50 張以下散戶持股比 4 週變化
    F["tdcc_holders4"] = w.d_holders_med.rolling(4).sum()   # 股東人數 4 週成長（散戶進場）
    F["tdcc_holders13"] = w.d_holders_med.rolling(13).sum()
    F.index = F.index + pd.Timedelta(days=lag)
    return F.reindex(index.union(F.index)).ffill(limit=10).reindex(index)

def inst_features(index):
    a = rd("tej_inst_daily_agg.csv"); a = a[a.n > 1500]
    F = pd.DataFrame(index=a.index)
    br = (a.fb - a.fs) / a.n; tbr = (a.tb - a.ts) / a.n
    F["inst_fbreadth5"] = br.rolling(5).mean()      # 外資買超家數 − 賣超家數（佔比）
    F["inst_fbreadth20"] = br.rolling(20).mean()
    F["inst_fown_chg20"] = a.f_own.diff(20)         # 外資持股比 20 日變化
    F["inst_trust20"] = a.t_net.rolling(20).sum() / a.f_base  # 投信 20 日買超 / 市值
    F["inst_tbreadth20"] = tbr.rolling(20).mean()
    return F.reindex(index)

def breadth_features(index):
    b = rd("tej_daily_breadth.csv")
    if b is None: return None
    b = b[b.all_n > 1200]; n = b.all_n; F = pd.DataFrame(index=b.index)
    F["br_attn5"] = (b.all_n_attn / n).rolling(5).mean()          # 注意股佔比
    F["br_attn_z"] = (F.br_attn5 - F.br_attn5.rolling(250).mean()) / F.br_attn5.rolling(250).std()
    F["br_disp5"] = (b.all_n_disp / n).rolling(5).mean()          # 處置股佔比
    F["br_disp_z"] = (F.br_disp5 - F.br_disp5.rolling(250).mean()) / F.br_disp5.rolling(250).std()
    F["br_lu5"] = (b.all_n_limit_up / n).rolling(5).mean()        # 漲停家數佔比
    F["br_ld5"] = (b.all_n_limit_down / n).rolling(5).mean()
    F["br_ma200"] = b.all_n_above_ma200 / b.all_n_ma200_valid     # 站上 200 日線比例
    F["br_ma60"] = b.all_n_above_ma60 / b.all_n_ma60_valid
    F["br_hl20"] = ((b.all_n_high_252 - b.all_n_low_252) / n).rolling(20).mean()  # 新高−新低
    F["br_xsd20"] = b.all_ret_xsd.rolling(20).mean()              # 個股報酬離散度
    F["br_turn20"] = np.log(b.all_turn_med.rolling(20).mean() / b.all_turn_med.rolling(250).mean())  # 週轉率熱度
    for c in ("pe_agg", "pb_agg"):                                 # 估值 5 年滾動分位
        s = b["all_" + c]; F["val_" + c] = s.rolling(1250, min_periods=500).rank(pct=True)
    return F.reindex(index)

def ens_pred(X, y, cols): return (RM.wf(X[cols], y, RM.logit) + RM.wf(X[cols], y, RM.xgb)) / 2

def paired_ci(y, a, b, block=20, n=1000, seed=0):
    y, a, b = map(np.asarray, (y, a, b)); T = len(y); rng = np.random.default_rng(seed); d = []
    for _ in range(n):
        st = rng.integers(0, T, int(np.ceil(T / block))); ix = (st[:, None] + np.arange(block)).ravel()[:T] % T
        if 0 < y[ix].mean() < 1: d.append(roc_auc_score(y[ix], a[ix]) - roc_auc_score(y[ix], b[ix]))
    d = np.array(d); return np.percentile(d, [2.5, 97.5]), float((d <= 0).mean())

df = M.load(); X0, y_vol0, y_dd0, ew, lr = RM.build(df)
X0["opt_fgn_bias"] = FE.fut_features(df, FE.load_futures(df.index))["opt_fgn_bias"]
live = [c for c in X0.columns]
groups = {"集保": tdcc_features(df.index, args.tdcc_lag), "法人個股": inst_features(df.index)}
B = breadth_features(df.index)
if B is not None:
    groups["注意處置"] = B[["br_attn5", "br_attn_z", "br_disp5", "br_disp_z"]]
    groups["漲跌停寬度"] = B[["br_lu5", "br_ld5", "br_ma200", "br_ma60", "br_hl20", "br_xsd20", "br_turn20"]]
    groups["估值分位"] = B[["val_pe_agg", "val_pb_agg"]]
allf = pd.concat(groups.values(), axis=1); X0 = X0.join(allf)
newc = list(allf.columns)
XA = X0[live + newc].dropna(); yA = y_dd0.reindex(XA.index); vA = y_vol0.reindex(XA.index)
print("樣本：", XA.index[0].date(), "~", XA.index[-1].date(), len(XA), "天")

ok = yA.notna(); rows = []
for g, G in groups.items():
    for c in G.columns:
        for per, m in (("全期", ok), ("2023–2026 保留期", ok & (XA.index >= RM.SPLIT))):
            a = roc_auc_score(yA[m], XA.loc[m, c]); lo, hi = M.block_auc_ci(yA[m].values, XA.loc[m, c].values, 20)
            rows.append({"組": g, "因子": c, "期間": per, "天數": int(m.sum()), "AUC(值高→大跌)": a, "95%下界": lo, "95%上界": hi})
sg = pd.DataFrame(rows); print(sg.round(3).to_string(index=False))
sg.to_csv(os.path.join(RM.R, f"{args.tag}_single.csv"), index=False, encoding="utf-8-sig")

vol_cols = ["rv1", "rv5", "rv22", "rv66", "park5", "neg5", "vix"]
vol = np.exp(RM.wf(XA[vol_cols], vA, RM.ols))
runs = {"A v3.2": live, **{f"+{g}": live + list(G.columns) for g, G in groups.items()}, "+全部": live + newc}
runs = {k: (ens_pred(XA, yA, cols), vol) for k, cols in runs.items()}
common = runs["A v3.2"][0].dropna().index
common = common[yA.reindex(common).notna().values]; y = yA.reindex(common)
pers = {"保留期前": common < RM.SPLIT, "2023–2026 保留期": common >= RM.SPLIT, "全期": np.ones(len(common), bool)}
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
auc.to_csv(os.path.join(RM.R, f"{args.tag}_auc.csv"), index=False, encoding="utf-8-sig")

yr = []
for Y in sorted(set(common.year)):
    m = common.year == Y
    if 0 < y[m].mean() < 1:
        yr.append({"年": Y, "天數": int(m.sum()), "大跌基準率": y[m].mean(),
                   **{k: roc_auc_score(y[m], p.reindex(common)[m]) for k, (p, _) in runs.items()}})
yr = pd.DataFrame(yr); print(yr.round(3).to_string(index=False))
yr.to_csv(os.path.join(RM.R, f"{args.tag}_yearly.csv"), index=False, encoding="utf-8-sig")

r = (df.close.shift(-1) / df.close - 1).reindex(common); cm = common[r.notna().values]; r = r[cm]
rows = []; P3 = (("保留期前", cm < RM.SPLIT), ("2023–2026 保留期", cm >= RM.SPLIT), ("全期", np.ones(len(cm), bool)))
for k, (p, v) in runs.items():
    w = RM.weights(v.reindex(cm), p.reindex(cm).values, 0.22, 0.5, 0.25, 0.2, 10); s = RM.strat(w, r)
    warn = RM.warn_active(p.reindex(cm).values, 0.25, 10)
    for per, m in P3: rows.append({"模型": k, "期間": per, **M.stats(s[m]), "警訊日比例": warn[m].mean()})
for per, m in P3: rows.append({"模型": "買進持有", "期間": per, **M.stats(r[m])})
st = pd.DataFrame(rows); print(st.round(3).to_string(index=False))
st.to_csv(os.path.join(RM.R, f"{args.tag}_strategy.csv"), index=False, encoding="utf-8-sig")
print("最新：", allf.dropna(how="all").iloc[-1].round(3).to_dict())
