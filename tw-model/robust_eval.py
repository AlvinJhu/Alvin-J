"""
穩健參數選擇：用「各年代中表現最差的一段」選 v3 部位參數，目標降低保費、保留防跌
資料與預測同 long_eval.py（1999 起、不含籌碼因子、walk-forward）。
規則（事先固定）：
  網格：減碼比例 cut ∈ {1/4, 1/3, 1/2, 2/3}、警訊維持 hold ∈ {1, 5, 10} 日、門檻 thr ∈ {0.25, 0.30, 0.40}、
        目標波動 target ∈ {0.18, 0.22, 0.30}、調整帶固定 0.2 → 108 組，另加「不減碼只做波動目標」
  選參數期：2001–2014，切成 2001–2007、2008–2014 兩段
  主準則 MaxMin：兩段 Sharpe 的較小值最大
  輔準則（同列）：兩段 Calmar（CAGR / |最大回撤|）較小值最大
  驗證期：2015–2022、2023–2026（選參數時完全未用）
  「保費」= 買進持有 CAGR − 策略 CAGR；「防跌」= 歷次空頭（回落 > 20%）策略平均少跌的百分點
  另做走動式重選：每年初用之前所有年代（每 7 年一段）的 MaxMin 重選，檢驗規則本身而非單一參數
輸出：reports/robust_grid.csv、robust_strategy.csv、robust_bear.csv、robust_walkforward_params.csv
"""
import os, sys, itertools, warnings
import numpy as np, pandas as pd
sys.argv = sys.argv[:1]; warnings.filterwarnings("ignore")
import model as M, risk_model as RM, cta_model as CT, long_eval as LE

R = M.R; BAND = 0.2
SEL = [("2001–2007", "2001-01-01", "2008-01-01"), ("2008–2014", "2008-01-01", "2015-01-01")]
TEST = [("2015–2022", "2015-01-01", "2023-01-01"), ("2023–2026", "2023-01-01", "2100-01-01")]
GRID = list(itertools.product([0.25, 1 / 3, 0.5, 2 / 3], [1, 5, 10], [0.25, 0.30, 0.40], [0.18, 0.22, 0.30]))

def preds():
    df = LE.load_long(); c = df.close
    X, y_vol, y_dd, _, _ = RM.build(df)
    X = X[[k for k in X if k not in ("margin_20", "fgn_20")]].dropna(); y_vol, y_dd = y_vol.reindex(X.index), y_dd.reindex(X.index)
    v = np.exp(RM.wf(X[CT.HARX], y_vol, RM.ols)); p = (RM.wf(X, y_dd, RM.logit) + RM.wf(X, y_dd, RM.xgb)) / 2
    r = (c.shift(-1) / c - 1).reindex(X.index)
    idx = v.dropna().index.intersection(p.dropna().index); idx = idx[r[idx].notna()]
    return v[idx], p[idx], r[idx]

def per(s, a, b):
    x = s[(s.index >= a) & (s.index < b)]; st = M.stats(x)
    st["Calmar"] = st["CAGR"] / abs(st["最大回撤"]) if st["最大回撤"] < 0 else np.nan
    return st

def bears(rr):
    bh = (1 + rr).cumprod(); dd = bh / bh.cummax() - 1; ev, on = [], False
    for d, x in dd.items():
        if not on and x < -0.20: on = True; pk = bh[:d].idxmax()
        if on and x == 0: ev.append((pk, bh[pk:d].idxmin())); on = False
    if on: ev.append((pk, bh[pk:].idxmin()))
    return ev

def seg(s, p0, p1): return float((1 + s[(s.index > p0) & (s.index <= p1)]).prod() - 1)

def main():
    v, p, rr = preds(); pv = p.values
    series = {}
    for cut, hold, thr, tg in GRID:
        series[(cut, hold, thr, tg)] = RM.strat(RM.weights(v, pv, tg, cut, thr, BAND, hold), rr)
    for tg in (0.18, 0.22, 0.30):
        series[(1.0, 1, 1.0, tg)] = RM.strat(RM.weights(v, pv, tg, 1.0, 1.0, BAND, 1), rr)
    bh = rr; ev = bears(rr)
    rows = []
    for k, s in series.items():
        row = {"cut": round(k[0], 3), "hold": k[1], "thr": k[2], "target": k[3]}
        for n, a, b in SEL + TEST + [("全期", "1900-01-01", "2100-01-01")]:
            st = per(s, a, b); row[f"{n} Sharpe"] = st["Sharpe"]; row[f"{n} Calmar"] = st["Calmar"]
            row[f"{n} CAGR"] = st["CAGR"]; row[f"{n} 回撤"] = st["最大回撤"]
        row["選參數 MaxMin Sharpe"] = min(row["2001–2007 Sharpe"], row["2008–2014 Sharpe"])
        row["選參數 MaxMin Calmar"] = min(row["2001–2007 Calmar"], row["2008–2014 Calmar"])
        sel_ev = [e for e in ev if e[1] < pd.Timestamp("2015-01-01")]
        row["2001–14 空頭平均少跌(百分點)"] = float(np.mean([seg(s, *e) - seg(bh, *e) for e in sel_ev])) * 100
        row["全期空頭平均少跌(百分點)"] = float(np.mean([seg(s, *e) - seg(bh, *e) for e in ev])) * 100
        rows.append(row)
    G = pd.DataFrame(rows)
    G["全期保費(CAGR 差)"] = per(bh, "1900-01-01", "2100-01-01")["CAGR"] - G["全期 CAGR"]
    G.sort_values("選參數 MaxMin Sharpe", ascending=False).to_csv(os.path.join(R, "robust_grid.csv"), index=False, encoding="utf-8-sig")
    pick_s = G.loc[G["選參數 MaxMin Sharpe"].idxmax()]; pick_c = G.loc[G["選參數 MaxMin Calmar"].idxmax()]
    def find(r_):
        for k in series:
            if abs(k[0] - r_.cut) < 1e-3 and k[1] == r_.hold and k[2] == r_.thr and k[3] == r_.target: return k
    cands = {f"MaxMin Sharpe 選出（cut {pick_s.cut:.2f}、hold {int(pick_s.hold)}、thr {pick_s.thr}、目標 {pick_s.target}）": series[find(pick_s)],
             f"MaxMin Calmar 選出（cut {pick_c.cut:.2f}、hold {int(pick_c.hold)}、thr {pick_c.thr}、目標 {pick_c.target}）": series[find(pick_c)],
             "v3.1 現行（cut 0.25、hold 10、thr 0.25、目標 0.22）": series[(0.25, 10, 0.25, 0.22)],
             "減碼 1/2、hold 10（其餘同 v3.1）": series[(0.5, 10, 0.25, 0.22)],
             "減碼 1/2、hold 5（其餘同 v3.1）": series[(0.5, 5, 0.25, 0.22)],
             "只做波動目標 22%": series[(1.0, 1, 1.0, 0.22)], "買進持有": bh}

    # 走動式重選：每年初，用之前所有完整 7 年段的最差段 Sharpe 選參數，套用該年
    years = range(2008, int(rr.index[-1].year) + 1); wf_ret, wf_par = [], []
    for y in years:
        segs = [(str(a), str(a + 7)) for a in range(2001, y - 6, 7)]
        segs = [(f"{a}-01-01", f"{b}-01-01") for a, b in segs if int(b) <= y]
        if not segs: segs = [("2001-01-01", f"{y}-01-01")]
        best, bk = -9, None
        for k in GRID:
            s = series[k]; m_ = min(per(s, a, b)["Sharpe"] for a, b in segs)
            if m_ > best: best, bk = m_, k
        wf_par.append({"年": y, "cut": round(bk[0], 3), "hold": bk[1], "thr": bk[2], "target": bk[3], "最差段 Sharpe": best})
        s = series[bk]; wf_ret.append(s[(s.index >= f"{y}-01-01") & (s.index < f"{y+1}-01-01")])
    cands["走動式 MaxMin 每年重選"] = pd.concat(wf_ret)
    pd.DataFrame(wf_par).to_csv(os.path.join(R, "robust_walkforward_params.csv"), index=False, encoding="utf-8-sig")
    print(pd.DataFrame(wf_par).to_string(index=False))

    rows, base = [], cands["v3.1 現行（cut 0.25、hold 10、thr 0.25、目標 0.22）"]
    for n, s in cands.items():
        for pn, a, b in SEL + TEST + [("2015–2026（驗證）", "2015-01-01", "2100-01-01"), ("2008–2026", "2008-01-01", "2100-01-01"), ("全期", "1900-01-01", "2100-01-01")]:
            if n.startswith("走動式") and a < "2008-01-01" and pn != "全期": continue
            m = (s.index >= a) & (s.index < b)
            if not m.any(): continue
            st = per(s, a, b); x, y_ = s[m], bh.reindex(s.index)[m]
            (lo, hi), _ = CT.boot_diff(x.values, y_.values) if n != "買進持有" else ((np.nan, np.nan), None)
            rows.append({"策略": n, "期間": pn, **st, "保費(買進持有 CAGR − 策略)": per(bh, max(a, str(s.index[0].date())), b)["CAGR"] - st["CAGR"],
                         "Sharpe差 vs 買進持有 95%下界": lo, "95%上界": hi})
    T = pd.DataFrame(rows); T.to_csv(os.path.join(R, "robust_strategy.csv"), index=False, encoding="utf-8-sig")
    with pd.option_context("display.width", 280):
        print(T[["策略", "期間", "CAGR", "Sharpe", "最大回撤", "保費(買進持有 CAGR − 策略)", "Sharpe差 vs 買進持有 95%下界", "95%上界"]].round(3).to_string(index=False))
    rows = []
    for p0, p1 in ev:
        row = {"高點": str(p0.date()), "低點": str(p1.date())}
        for n, s in cands.items(): row[n] = seg(s, p0, p1) if s.index[0] <= p0 else np.nan
        rows.append(row)
    B = pd.DataFrame(rows); B.to_csv(os.path.join(R, "robust_bear.csv"), index=False, encoding="utf-8-sig")
    with pd.option_context("display.width", 280): print(B.round(3).to_string(index=False))

if __name__ == "__main__":
    main()
