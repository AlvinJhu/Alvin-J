"""
長樣本驗證（1999 起）：v3 風險模型在 2000 網路泡沫、2008 金融海嘯、2011 歐債、2015 等空頭是否仍有效
資料：證交所加權指數日 OHLC 1999 起（fetch_twse.py monthly 1999-01）、VIX（1990 起）、費半 ^SOX（sox_long.csv，1994 起）
設計：
  - 只用可回溯的價格／波動因子（不含法人、融資、外資選擇權，這些 2016/2018 後才有）＝「v3 不含籌碼」版
  - 波動：HAR+幅度+VIX；大跌：未來 20 日內跌 > 7%，logit＋XGBoost 平均；擴張視窗 walk-forward、每 63 日重訓
  - 部位 A：v3.1 參數完全不動（目標 22%、大跌機率 > 25% 減至 1/4、持續 10 日、調整帶 0.2；原本只用 2018–2022 選出）
           → 2001–2017 是這組參數從未見過的期間，屬真正樣本外
  - 部位 B：只用 2001–2014 重新選參數（同 risk_model.py 網格），2015–2026 驗證，看參數是否穩定
  - 分年代報告；Sharpe 差 63 日區塊 bootstrap；歷次 > 20% 空頭逐段比較
輸出：reports/long_crash_auc.csv、long_strategy.csv、long_sharpe_diff.csv、long_bear_markets.csv、long_params.csv、long_equity.png
"""
import os, sys, json, itertools, warnings
import numpy as np, pandas as pd
sys.argv = sys.argv[:1]; warnings.filterwarnings("ignore")
import model as M, risk_model as RM, cta_model as CT
from sklearn.metrics import roc_auc_score

R = M.R; ANN = np.sqrt(252)
TARGET, CUT, THR, BAND, HOLD = 0.22, 0.25, 0.25, 0.2, 10
ERAS = [("2001–2007", "2001-01-01", "2008-01-01"), ("2008–2014", "2008-01-01", "2015-01-01"),
        ("2015–2022", "2015-01-01", "2023-01-01"), ("2023–2026", "2023-01-01", "2100-01-01")]

def load_long():
    df = M.load()
    p = os.path.join(M.D, "ext", "sox_long.csv")
    if os.path.exists(p):
        s = pd.read_csv(p, parse_dates=["date"]).set_index("date")["close"].dropna().sort_index()
        s.index = s.index + pd.Timedelta(days=1)
        df["sox"] = s.reindex(df.index, method="ffill")
    return df[df.index >= "1999-01-01"]

def main():
    df = load_long(); c = df.close
    X, y_vol, y_dd, ew, lr = RM.build(df)
    cols = [k for k in X if k not in ("margin_20", "fgn_20")]
    X = X[cols].dropna(); y_vol, y_dd = y_vol.reindex(X.index), y_dd.reindex(X.index)
    print(f"樣本 {X.index[0].date()} ~ {X.index[-1].date()}，{len(X)} 天；大跌基準率 {y_dd.mean():.3f}；因子 {cols}")
    vol_fc = np.exp(RM.wf(X[CT.HARX], y_vol, RM.ols))
    p_dd = (RM.wf(X, y_dd, RM.logit) + RM.wf(X, y_dd, RM.xgb)) / 2

    # 1) 大跌 AUC 分年代
    rows = []
    sel = lambda a, b: (X.index >= a) & (X.index < b)
    for n, p in (("v3 大跌模型（不含籌碼）", p_dd), ("只用預測波動", vol_fc)):
        for era, a, b in ERAS + [("全期", "1900-01-01", "2100-01-01")]:
            m = sel(a, b) & p.notna().values & y_dd.notna().values; yy, pp = y_dd[m], p[m]
            if yy.nunique() < 2: continue
            lo, hi = M.block_auc_ci(yy.values, pp.values, 20); top = pp >= pp.quantile(0.8)
            ep = int(((yy.diff() == 1) | ((yy == 1) & (yy.index == yy.index[0]))).sum())
            rows.append({"模型": n, "年代": era, "天數": int(m.sum()), "基準率": yy.mean(), "大跌段數": ep,
                         "AUC": roc_auc_score(yy, pp), "95%下界": lo, "95%上界": hi,
                         "最高20%機率日大跌率": yy[top].mean(), "其餘80%": yy[~top].mean()})
    ca = pd.DataFrame(rows); ca.to_csv(os.path.join(R, "long_crash_auc.csv"), index=False, encoding="utf-8-sig")
    print(ca.round(3).to_string(index=False))

    # 2) 部位
    r = (c.shift(-1) / c - 1).reindex(X.index)
    idx = vol_fc.dropna().index.intersection(p_dd.dropna().index); idx = idx[r[idx].notna()]
    v, pdv, rr = vol_fc[idx], p_dd[idx].values, r[idx]
    # B：2001–2014 選參數
    ins = idx < pd.Timestamp("2015-01-01"); grid = []
    for tg, cut, thr, band, hold in itertools.product([0.12, 0.15, 0.18, 0.22], [1.0, 0.5, 0.25], [0.2, 0.25, 0.3, 0.4], [0.0, 0.1, 0.2], [1, 5, 10, 20]):
        if cut == 1.0 and (thr != 0.2 or hold != 1): continue
        s = RM.strat(RM.weights(v, pdv, tg, cut, thr, band, hold), rr)
        grid.append({"target": tg, "cut": cut, "thr": thr, "band": band, "hold": hold, **M.stats(s[ins])})
    grid = pd.DataFrame(grid).sort_values("Sharpe", ascending=False); best = grid.iloc[0]
    grid.head(20).to_csv(os.path.join(R, "long_params.csv"), index=False, encoding="utf-8-sig")
    print("2001–2014 選出參數：", best[["target", "cut", "thr", "band", "hold", "Sharpe"]].to_dict())
    cands = {"A v3.1 參數（2018–22 選定，不動）": RM.weights(v, pdv, TARGET, CUT, THR, BAND, HOLD),
             "B 2001–14 重選參數": RM.weights(v, pdv, best.target, best.cut, best.thr, best.band, int(best.hold)),
             "只用預測波動（22%）": RM.weights(v, pdv, TARGET, 1.0, 1.0, BAND, 1),
             "買進持有": pd.Series(1.0, index=idx)}
    rows, rets = [], {}
    for n, w in cands.items():
        s = rr if n == "買進持有" else RM.strat(w, rr); rets[n] = s
        for era, a, b in ERAS + [("2001–2017（A 參數未見過）", "1900-01-01", "2018-01-01"), ("全期", "1900-01-01", "2100-01-01")]:
            m = (idx >= a) & (idx < b)
            rows.append({"策略": n, "期間": era, "起": str(idx[m][0].date()), **M.stats(s[m]), "平均持股": w[m].mean(),
                         "年換手": w[m].diff().abs().sum() / (m.sum() / 252)})
    st = pd.DataFrame(rows); st.to_csv(os.path.join(R, "long_strategy.csv"), index=False, encoding="utf-8-sig")
    with pd.option_context("display.width", 250): print(st.round(3).to_string(index=False))
    rows = []
    for a_, b_ in (("A v3.1 參數（2018–22 選定，不動）", "買進持有"), ("A v3.1 參數（2018–22 選定，不動）", "只用預測波動（22%）"),
                   ("B 2001–14 重選參數", "買進持有"), ("只用預測波動（22%）", "買進持有")):
        for era, a, b in ERAS + [("2001–2017（A 參數未見過）", "1900-01-01", "2018-01-01"), ("全期", "1900-01-01", "2100-01-01")]:
            m = (idx >= a) & (idx < b); (lo, hi), pw = CT.boot_diff(rets[a_][m].values, rets[b_][m].values)
            rows.append({"比較": f"{a_} − {b_}", "期間": era, "Sharpe差": CT.sharpe(rets[a_][m]) - CT.sharpe(rets[b_][m]),
                         "95%下界": lo, "95%上界": hi, "bootstrap 勝率": pw})
    sd = pd.DataFrame(rows); sd.to_csv(os.path.join(R, "long_sharpe_diff.csv"), index=False, encoding="utf-8-sig")
    print(sd.round(3).to_string(index=False))

    # 3) 歷次空頭（買進持有自高點回落 > 20%）
    bh = (1 + rr).cumprod(); dd = bh / bh.cummax() - 1; ev, on = [], False
    for d, x in dd.items():
        if not on and x < -0.20: on = True; pk = bh[:d].idxmax()
        if on and x == 0: ev.append((pk, bh[pk:d].idxmin())); on = False
    if on: ev.append((pk, bh[pk:].idxmin()))
    rows = []
    for p0, p1 in ev:
        row = {"高點": str(p0.date()), "低點": str(p1.date())}
        for n in cands:
            s = rets[n]; row[n] = float((1 + s[(s.index > p0) & (s.index <= p1)]).prod() - 1)
        w = cands["A v3.1 參數（2018–22 選定，不動）"]; ff = w[(w.index >= p0) & (w.index <= p1)]
        row["A 首次減碼日"] = str(ff[ff < ff.iloc[0] - 0.05].index[0].date()) if (ff < ff.iloc[0] - 0.05).any() else None
        rows.append(row)
    bm = pd.DataFrame(rows); bm.to_csv(os.path.join(R, "long_bear_markets.csv"), index=False, encoding="utf-8-sig")
    with pd.option_context("display.width", 250): print(bm.round(3).to_string(index=False))

    try:
        import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
        fig, ax = plt.subplots(2, 1, figsize=(11, 8), sharex=True, gridspec_kw={"height_ratios": [3, 1]})
        lab = {"A v3.1 參數（2018–22 選定，不動）": "A: v3.1 params (unseen before 2018)", "B 2001–14 重選參數": "B: params re-fit 2001-14",
               "只用預測波動（22%）": "vol forecast only", "買進持有": "Buy & Hold (price index)"}
        for n, s in rets.items(): ax[0].plot((1 + s).cumprod(), label=lab[n], lw=1.5 if n.startswith("A") else .9)
        ax[0].set_yscale("log"); ax[0].legend(fontsize=8); ax[0].set_title("TAIEX 1999-2026: v3 risk model (no chip factors), walk-forward")
        ax[1].fill_between(idx, cands["A v3.1 參數（2018–22 選定，不動）"].values, step="post", alpha=.5); ax[1].set_ylabel("weight A")
        fig.tight_layout(); fig.savefig(os.path.join(R, "long_equity.png"), dpi=110)
    except Exception as e:
        print("圖檔略過：", e)

if __name__ == "__main__":
    main()
