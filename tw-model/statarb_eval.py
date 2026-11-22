"""
統計套利（短週期均值回歸）驗證
文獻：Gatev, Goetzmann & Rouwenhorst (2006) 距離法配對交易；Avellaneda & Lee (2010) 統計套利；
      Lehmann (1990)、Lo & MacKinlay (1990) 短期反轉；Do & Faff (2010) 配對交易獲利隨時間衰退。
三個測試（規則事先固定，所有變體全列，Holm 校正）：
  A. 台指期跨月價差均值回歸（近月 vs 次月結算價價差）：同一組合約內，價差相對過去 10 日 z > 2 → 賣價差，
     z < -2 → 買價差；z 回到 0、持有 10 日或換月前一日出場。成本：每口每邊 1 檔滑價＋手續費、期交稅 ≈ 1.8 點，
     一次來回（兩腳、進出）≈ 7.2 點；另測 4、12 點
  B. 加權指數短期反轉（以台指期近月連續報酬交易）：3 日報酬 / (60 日波動×√3) < -1.5 → 做多；> 1.5 → 放空（或空手）
     持有 1 或 5 日；成本每單位換手 0.01%（台指期實際約 0.006%）與 0.05% 壓力測試
  C. 個股配對交易（距離法，需 data/ext/stk_daily_top100.csv，由 fetch_stocks.py 取得）：
     12 個月形成期挑標準化價格距離最小 20 對，6 個月交易期，價差偏離 2σ 開倉、交叉出場；每月啟動一批、重疊 6 批；
     成本每邊 0.1425%×0.3（手續費 3 折）＋賣出證交稅 0.3%，放空另計年化 1% 借券
輸出：reports/statarb_calendar.csv、statarb_reversal.csv、statarb_pairs.csv
"""
import os, sys, warnings
import numpy as np, pandas as pd
sys.argv = sys.argv[:1]; warnings.filterwarnings("ignore")
import model as M, risk_model as RM, cta_model as CT

R = M.R; SPLIT = RM.SPLIT; ANN = np.sqrt(252)

def boot_mean_p(x, block=20, n=2000, seed=0):
    """單邊 p 值：平均日損益 ≤ 0 的區塊 bootstrap 比例（中心化於 0 的檢定改用百分位比例近似）。"""
    x = np.asarray(x); T = len(x); rng = np.random.default_rng(seed); m = []
    for _ in range(n):
        st = rng.integers(0, T, int(np.ceil(T / block))); ix = (st[:, None] + np.arange(block)).ravel()[:T] % T
        m.append(x[ix].mean())
    m = np.array(m); return float((m <= 0).mean()), np.percentile(m, [2.5, 97.5])

def holm(p):
    p = np.asarray(p); o = np.argsort(p); k = len(p); adj = np.empty(k); run = 0
    for j, i in enumerate(o): run = max(run, min(1, (k - j) * p[i])); adj[i] = run
    return adj

# ---------- A. 跨月價差 ----------
def calendar(cost_pts=(4.0, 7.2, 12.0), z_in=2.0, look=10, max_hold=10, min_vol=100):
    fd = pd.read_csv(os.path.join(M.D, "ext", "fut_tx_daily.csv"), parse_dates=["date"])
    fd = fd[(fd.trading_session == "position") & fd.contract_date.astype(str).str.fullmatch(r"\d{6}") & (fd.settlement_price > 0)]
    fd["cd"] = fd.contract_date.astype(str)
    rows = []
    for d, g in fd.groupby("date"):
        g = g.sort_values("cd")
        if len(g) < 2: continue
        a, b = g.iloc[0], g.iloc[1]
        if b.volume < min_vol: continue
        rows.append({"date": d, "pair": a.cd + "/" + b.cd, "s": b.settlement_price - a.settlement_price, "near": a.settlement_price})
    sp = pd.DataFrame(rows).set_index("date")
    sp["mu"] = sp.groupby("pair").s.transform(lambda x: x.shift(1).rolling(look, min_periods=5).mean())
    sp["sd"] = sp.groupby("pair").s.transform(lambda x: x.shift(1).rolling(look, min_periods=5).std()).clip(lower=1.0)
    sp["z"] = (sp.s - sp.mu) / sp.sd
    sp["last_of_pair"] = sp.pair != sp.pair.shift(-1)
    pos = np.zeros(len(sp)); cur, held = 0, 0; trades = []
    for i in range(len(sp)):
        z, last = sp.z.iloc[i], sp.last_of_pair.iloc[i]
        if cur != 0:
            held += 1
            if (cur > 0 and z >= 0) or (cur < 0 and z <= 0) or held >= max_hold or last:
                trades[-1]["exit"] = sp.index[i]; trades[-1]["exit_s"] = sp.s.iloc[i]; cur = 0
        if cur == 0 and not last and np.isfinite(z) and abs(z) > z_in:
            cur, held = -int(np.sign(z)), 0
            trades.append({"entry": sp.index[i], "pair": sp.pair.iloc[i], "dir": cur, "entry_s": sp.s.iloc[i], "z": z})
        pos[i] = cur
    sp["pos"] = pos
    ds = sp.s.groupby(sp.pair).diff().shift(-1)  # 同組合約的次日價差變動
    sp["pnl_gross"] = (sp.pos * ds).fillna(0.0)
    tr = pd.DataFrame(trades).dropna(subset=["exit"])
    tr["gross_pts"] = tr.dir * (tr.exit_s - tr.entry_s)
    out = []
    for cp in cost_pts:
        ntr = sp.pos.diff().abs().fillna(sp.pos.abs()) / 2 * cp  # 每次進或出 = 半個來回成本
        net = sp.pnl_gross - ntr
        for per, m in (("2023 以前", sp.index < SPLIT), ("2023 起（保留期）", sp.index >= SPLIT), ("全期", sp.index == sp.index)):
            x = net[m]; t = tr[(tr.entry >= x.index[0]) & (tr.entry <= x.index[-1])]
            p, ci = boot_mean_p(x.values)
            out.append({"來回成本(點)": cp, "期間": per, "起": str(x.index[0].date()), "交易次數": len(t),
                        "每筆毛利(點)": t.gross_pts.mean(), "每筆淨利(點)": t.gross_pts.mean() - cp,
                        "勝率(淨)": float((t.gross_pts > cp).mean()) if len(t) else np.nan,
                        "年淨利(點/口)": x.sum() / (len(x) / 252), "年淨利(新台幣/口)": x.sum() / (len(x) / 252) * 200,
                        "日損益 Sharpe": x.mean() / x.std() * ANN if x.std() > 0 else np.nan, "單邊p": p,
                        "價差日變動標準差(點)": float(ds[m].std())})
    return pd.DataFrame(out), tr

# ---------- B. 短期反轉 ----------
def reversal(df):
    c = df.close; r_tx = CT.tx_continuous(df.index)
    lr = np.log(c).diff(); vol = lr.rolling(60).std()
    z = np.log(c / c.shift(3)) / (vol * np.sqrt(3))
    nxt = r_tx.shift(-1)
    idx = z.dropna().index.intersection(nxt.dropna().index)
    idx = idx[idx >= "2016-06-01"]
    out, series = [], {}
    for hold in (1, 5):
        for side in ("多空", "只做多"):
            sig = pd.Series(np.where(z < -1.5, 1.0, np.where(z > 1.5, -1.0 if side == "多空" else 0.0, 0.0)), index=z.index)
            pos = sig.replace(0, np.nan).ffill(limit=hold - 1).fillna(0.0) if hold > 1 else sig
            pos = pos.reindex(idx)
            for cost in (0.0001, 0.0005):
                s = pos * nxt[idx] - pos.diff().abs().fillna(pos.abs()) * cost
                name = f"持有{hold}日 {side} 成本{cost:.2%}"; series[name] = s
                for per, m in (("2023 以前", idx < SPLIT), ("2023 起（保留期）", idx >= SPLIT), ("全期", idx == idx)):
                    x = s[m]; p, ci = boot_mean_p(x.values)
                    out.append({"變體": name, "期間": per, "起": str(x.index[0].date()), "持倉天數比": float((pos[m] != 0).mean()),
                                "年化報酬(簡單)": x.mean() * 252, "年化波動": x.std() * ANN,
                                "Sharpe": x.mean() / x.std() * ANN if x.std() > 0 else np.nan, "單邊p": p,
                                "年換手": pos[m].diff().abs().sum() / (m.sum() / 252)})
    o = pd.DataFrame(out); full = o.期間 == "全期"
    o.loc[full, "Holm校正p"] = holm(o.loc[full, "單邊p"].values)
    return o

# ---------- C. 個股配對（距離法） ----------
def pairs(npairs=20, form=252, trade=126, k=2.0, fee=0.001425 * 0.3, tax=0.003, borrow=0.01):
    fn = os.path.join(M.D, "ext", "stk_daily_top100.csv")
    if not os.path.exists(fn): return None
    d = pd.read_csv(fn, dtype={"code": str}, parse_dates=["date"])
    P = d.pivot_table(index="date", columns="code", values="close_adj").sort_index()
    ret = P.pct_change()
    starts = list(range(form, len(P) - 21, 21))  # 每月啟動一批
    port = []  # 每批的日報酬（等權於已開倉配對，未開倉的資金報酬為 0）
    for s0 in starts:
        F = P.iloc[s0 - form:s0].dropna(axis=1)
        if F.shape[1] < 20: continue
        N = F / F.iloc[0]
        cols = N.columns; A = N.values
        dist = ((A[:, :, None] - A[:, None, :]) ** 2).sum(axis=0)
        iu = np.triu_indices(len(cols), 1); order = np.argsort(dist[iu])[:npairs]
        T = P.iloc[s0:s0 + trade]
        batch = pd.Series(0.0, index=T.index)
        for o in order:
            i, j = cols[iu[0][o]], cols[iu[1][o]]
            sd = (N[i] - N[j]).std()
            if T[[i, j]].isna().any().any() or sd == 0: continue
            ni, nj = T[i] / F[i].iloc[-1] * N[i].iloc[-1], T[j] / F[j].iloc[-1] * N[j].iloc[-1]
            spr = (ni - nj) / sd
            pos = np.zeros(len(T)); cur = 0
            for t in range(len(T) - 1):
                if cur == 0 and abs(spr.iloc[t]) > k: cur = -np.sign(spr.iloc[t])
                elif cur != 0 and np.sign(spr.iloc[t]) == cur: cur = 0  # 價差穿越 0 出場
                pos[t] = cur
            pos = pd.Series(pos, index=T.index)
            ri, rj = ret[i].reindex(T.index).shift(-1).fillna(0), ret[j].reindex(T.index).shift(-1).fillna(0)
            # cur=+1：買 i 賣 j；每腳 0.5 單位資金
            g = 0.5 * pos * (ri - rj)
            tc = pos.diff().abs().fillna(pos.abs()) * 0.5 * (2 * fee + tax)  # 兩腳各一邊，賣出腳付稅（近似：進出各半）
            bc = (pos != 0) * 0.5 * borrow / 252
            batch += (g - tc - bc) / npairs
        port.append(batch)
    if not port: return None
    allr = pd.concat(port, axis=1).mean(axis=1, skipna=True)  # 重疊批次等權
    out = []
    for per, m in (("2023 以前", allr.index < SPLIT), ("2023 起（保留期）", allr.index >= SPLIT), ("全期", allr.index == allr.index)):
        x = allr[m]; p, _ = boot_mean_p(x.values)
        out.append({"期間": per, "起": str(x.index[0].date()), **M.stats(x), "單邊p": p})
    return pd.DataFrame(out), allr

def main():
    df = M.load()
    cal, tr = calendar()
    cal.to_csv(os.path.join(R, "statarb_calendar.csv"), index=False, encoding="utf-8-sig")
    with pd.option_context("display.width", 250): print("A. 跨月價差\n", cal.round(3).to_string(index=False))
    rv = reversal(df); rv.to_csv(os.path.join(R, "statarb_reversal.csv"), index=False, encoding="utf-8-sig")
    with pd.option_context("display.width", 250): print("B. 短期反轉\n", rv.round(3).to_string(index=False))
    pr = pairs()
    if pr is None: print("C. 個股配對：缺 data/ext/stk_daily_top100.csv，先跑 fetch_stocks.py")
    else:
        g = pairs(fee=0.0, tax=0.0, borrow=0.0)
        t = pd.concat([pr[0].assign(成本="含成本"), g[0].assign(成本="零成本（毛利）")], ignore_index=True)
        t.to_csv(os.path.join(R, "statarb_pairs.csv"), index=False, encoding="utf-8-sig")
        print("C. 個股配對\n", t.round(3).to_string(index=False))

if __name__ == "__main__":
    main()
