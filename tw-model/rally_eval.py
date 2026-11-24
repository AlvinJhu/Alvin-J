"""
大漲通知模型：未來 20 日內最高收盤較今日漲超過 7% 的機率；與大跌機率並列，檢驗兩者能否分開。
因子同 v3.1 大跌模型；walk-forward、embargo 20 日；2018–2022 選參數、2023–2026 保留驗證。
輸出：reports/rally_auc.csv、rally_strategy.csv、rally_today.json
"""
import os, sys, json, itertools
ARGS = sys.argv[1:]; sys.argv = sys.argv[:1]
import numpy as np, pandas as pd
import model as M, risk_model as RM, futures_eval as FE

UP = 0.07

def main():
    df = M.load(); X, yv, ydd, ew, lr = RM.build(df)
    X["opt_fgn_bias"] = FE.fut_features(df, FE.load_futures(df.index))["opt_fgn_bias"]
    X = X.dropna(); c = df.close
    fut = pd.concat([c.shift(-k) for k in range(1, 21)], axis=1)
    full = fut.notna().all(axis=1)
    mx, mn = fut.max(axis=1) / c - 1, fut.min(axis=1) / c - 1
    y_up = (mx > UP).astype(float).where(full).reindex(X.index)
    y_dd = (mn < RM.DD).astype(float).where(full).reindex(X.index)
    y_clean = ((mx > UP) & (mn > -0.05)).astype(float).where(full).reindex(X.index)  # 乾淨上漲：漲 7% 且未先跌 5%
    yv = yv.reindex(X.index)
    cols = list(X.columns); vc = ["rv1", "rv5", "rv22", "rv66", "park5", "neg5", "vix"]
    print(f"樣本 {X.index[0].date()}~{X.index[-1].date()}；大漲基準率 {y_up.mean():.3f}，大跌 {y_dd.mean():.3f}，乾淨上漲 {y_clean.mean():.3f}")
    P = {}
    for k, y in (("大漲", y_up), ("大跌", y_dd), ("乾淨上漲", y_clean)):
        P[k] = (RM.wf(X[cols], y, RM.logit) + RM.wf(X[cols], y, RM.xgb)) / 2
    P["預測波動"] = RM.wf(X[vc], yv, RM.ols)
    SPL = RM.SPLIT; rows = []
    for k, y in (("大漲", y_up), ("大跌", y_dd), ("乾淨上漲", y_clean)):
        for pn in (k, "預測波動"):
            p = P[pn]
            for per, m0 in (("2018–2022", X.index < SPL), ("2023–2026", X.index >= SPL), ("全期", X.index == X.index)):
                m = m0 & p.notna() & y.notna(); yy, pp = y[m], p[m]; lo, hi = M.block_auc_ci(yy.values, pp.values, 20)
                top = pp >= pp.quantile(.8)
                rows.append({"標的": k, "用哪個分數排序": pn + "機率" if pn != "預測波動" else "預測波動", "期間": per, "基準率": yy.mean(),
                             "AUC": M.roc_auc_score(yy, pp), "下界": lo, "上界": hi, "最高20%日發生率": yy[top].mean(), "其餘": yy[~top].mean()})
    A = pd.DataFrame(rows); print(A.round(3).to_string(index=False))
    A.to_csv(os.path.join(M.R, "rally_auc.csv"), index=False, encoding="utf-8-sig")
    ok = P["大漲"].notna() & P["大跌"].notna()
    print("大漲機率與大跌機率相關：", round(np.corrcoef(P["大漲"][ok], P["大跌"][ok])[0, 1], 3),
          "；大漲機率與預測波動相關：", round(np.corrcoef(P["大漲"][ok], P["預測波動"][ok])[0, 1], 3))

    # 部位：大跌警訊（含持續 10 日）→ 減碼；無警訊且「乾淨上漲」機率高 → 回補到滿檔（不受波動目標上限）
    r = (c.shift(-1) / c - 1).reindex(X.index); idx = P["大跌"].dropna().index.intersection(P["乾淨上漲"].dropna().index)
    idx = idx[r[idx].notna()]; vol = np.exp(P["預測波動"][idx]); pdd, pup, rr = P["大跌"][idx], P["乾淨上漲"][idx], r[idx]
    ins, oos = idx < SPL, idx >= SPL
    def wts(tg, thr, hold, up_thr, band=0.2, cut=0.25):
        warn = RM.warn_active(pdd.values, thr, hold)
        base = np.minimum(1, tg / vol.values)
        raw = np.where(warn, base * cut, np.where(pup.values > up_thr, 1.0, base))
        w = np.empty(len(raw)); cur = raw[0]
        for i, x in enumerate(raw):
            if abs(x - cur) > band: cur = x
            w[i] = cur
        return pd.Series(w, index=idx)
    grid = []
    for tg, thr, hold, ut in itertools.product([0.18, 0.22], [0.2, 0.25, 0.3], [5, 10, 20], [1.01, 0.3, 0.4, 0.5, 0.6]):
        s = RM.strat(wts(tg, thr, hold, ut), rr); st = M.stats(s[ins]); grid.append({"tg": tg, "thr": thr, "hold": hold, "up_thr": ut, **st})
    G = pd.DataFrame(grid).sort_values("Sharpe", ascending=False); b = G.iloc[0]
    print("2018–2022 選出：", b[["tg", "thr", "hold", "up_thr", "Sharpe"]].to_dict())
    nb = G[G.up_thr == 1.01].iloc[0]
    cands = {"大跌避開＋大漲回補": wts(b.tg, b.thr, int(b.hold), b.up_thr),
             "只有大跌避開（v3.1）": wts(nb.tg, nb.thr, int(nb.hold), 1.01), "買進持有": pd.Series(1.0, index=idx)}
    rows = []
    for n, w in cands.items():
        s = RM.strat(w, rr) if n != "買進持有" else rr
        for per, m in (("2018–2022（選參數）", ins), ("2023–2026（保留）", oos), ("全期", ins | oos)):
            rows.append({"策略": n, "期間": per, **M.stats(s[m]), "平均持股": w[m].mean(), "年換手": w[m].diff().abs().sum() / (m.sum() / 252)})
    S = pd.DataFrame(rows); print(S.round(3).to_string(index=False)); S.to_csv(os.path.join(M.R, "rally_strategy.csv"), index=False, encoding="utf-8-sig")
    print(pd.DataFrame({"大跌": pdd, "乾淨上漲": pup, "w": cands["大跌避開＋大漲回補"]}).loc["2026-10-15":].round(3).to_string())

if __name__ == "__main__":
    main()
