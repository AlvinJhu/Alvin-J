"""
布林通道（20 日、±2σ）因子對 20 日內跌 7% 的預警力；與 v3.1 大跌模型同樣本比較（Bollinger, 2001）。
因子：%B、%B 5 日變化、帶寬、帶寬 120 日分位、近 5 日收在下軌下方天數、近 5 日收在上軌上方天數。
部位規則固定用 v3.1 選定參數（目標波動 22%、門檻 25%、減碼至 1/4、維持 10 日、調整帶 0.2），不重新選參數。
"""
import sys, os
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE); os.chdir(HERE)
sys.argv = sys.argv[:1]
import numpy as np, pandas as pd, model as M, risk_model as RM, futures_eval as FE

def bb(c, n=20, k=2):
    ma, sd = c.rolling(n).mean(), c.rolling(n).std()
    up, lo = ma + k * sd, ma - k * sd
    f = pd.DataFrame(index=c.index)
    f["bb_pctb"] = (c - lo) / (up - lo)
    f["bb_pctb_chg5"] = f.bb_pctb.diff(5)
    f["bb_width"] = np.log((up - lo) / ma)
    f["bb_width_pct120"] = f.bb_width.rolling(120).rank(pct=True)
    f["bb_below5"] = (c < lo).astype(float).rolling(5).sum()
    f["bb_above5"] = (c > up).astype(float).rolling(5).sum()
    return f

def main():
    df = M.load(); X, yv, ydd, ew, lr = RM.build(df)
    X["opt_fgn_bias"] = FE.fut_features(df, FE.load_futures(df.index))["opt_fgn_bias"]
    B = bb(df.close)
    A = pd.concat([X, B], axis=1).dropna(); y = ydd.reindex(A.index); yv = yv.reindex(A.index); SPL = RM.SPLIT
    print("樣本", A.index[0].date(), A.index[-1].date(), len(A))
    pers = (("2018–2022", A.index < SPL), ("2023–2026", A.index >= SPL), ("全期", A.index == A.index))
    rows = []
    def add(name, p):
        r = {"模型": name}
        for per, m in pers:
            s = M.score("dd", "x", p[m], y[m], 20); r[per] = s["AUC"]; r[per + "下界"] = s["AUC_95%下界"]; r[per + "上界"] = s["AUC_95%上界"]
        rows.append(r)
    for c in B.columns: add("單獨 " + c, RM.wf(A[[c]], y, RM.logit))
    base = list(X.columns); P = {}
    for name, cols in (("v3.1", base), ("v3.1＋布林", base + list(B.columns))):
        P[name] = (RM.wf(A[cols], y, RM.logit) + RM.wf(A[cols], y, RM.xgb)) / 2; add(name, P[name])
    T = pd.DataFrame(rows); print(T.round(3).to_string(index=False))
    T.to_csv(os.path.join(M.R, "bb_auc.csv"), index=False, encoding="utf-8-sig")
    print("bb_width 與 rv22 相關：", round(np.corrcoef(A.bb_width, A.rv22)[0, 1], 3))

    vf = np.exp(RM.wf(A[["rv1", "rv5", "rv22", "rv66", "park5", "neg5", "vix"]], yv, RM.ols))
    r = (df.close.shift(-1) / df.close - 1).reindex(A.index)
    idx = P["v3.1"].dropna().index; idx = idx[r[idx].notna()]
    rows = []
    for name in P:
        w = RM.weights(vf[idx], P[name][idx].values, 0.22, 0.25, 0.25, 0.2, 10); s = RM.strat(w, r[idx])
        for per, m in (("2018–2022", idx < SPL), ("2023–2026", idx >= SPL)):
            rows.append({"策略": name, "期間": per, **M.stats(s[m]), "平均持股": w[m].mean()})
    for per, m in (("2018–2022", idx < SPL), ("2023–2026", idx >= SPL)):
        rows.append({"策略": "買進持有", "期間": per, **M.stats(r[idx][m]), "平均持股": 1.0})
    S = pd.DataFrame(rows); print(S.round(3).to_string(index=False))
    S.to_csv(os.path.join(M.R, "bb_strategy.csv"), index=False, encoding="utf-8-sig")
    print(pd.DataFrame({"pctB": A.bb_pctb, "width_pct": A.bb_width_pct120, "p_v31": P["v3.1"], "p_bb": P["v3.1＋布林"]}).loc["2026-10-20":].round(3).to_string())

if __name__ == "__main__":
    main()
