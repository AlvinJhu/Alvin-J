"""
多因子模型（Barra 型，月頻）：橫斷面迴歸估計因子報酬、因子共變異風險預測、因子傾斜與最小變異組合
文獻：Fama & French (1993, 2015)；Carhart (1997) 動能；Asness, Frazzini & Pedersen (2019) 品質；
      Frazzini & Pedersen (2014)、Ang et al. (2006) 低波動；Fama & MacBeth (1973)；Newey & West (1987)；
      Menchero, Orr & Wang (2011) Barra 模型；Clarke, de Silva & Thorley (2006) 最小變異組合。
資料：fetch_stocks.py 的證交所月底全市場快照（含已下市股票，無存活偏誤），2015-01 起。
設計（事先固定）：
  宇宙：每月底市值前 300 大普通股（只用當月資料）
  報酬：次月價格報酬＋殖利率/12；發行股數變動 > 2%（減資、增資、配股）改用市值變動；橫斷面 1%/99% 縮尾
  因子暴露（當月底已知，橫斷面縮尾 ±3σ 後標準化）：
    規模 = log 市值；價值 = (B/P + E/P)/2；動能 = t-12 到 t-1 月累積報酬；
    品質 = ROE（= 股價淨值比/本益比，虧損視為最低）；低波 = −過去 12 月月報酬標準差
  因子報酬：每月以 √市值 加權最小平方，截距 = 市場；Newey-West t 值（3 期落後）
  風險模型：因子共變異 = 指數加權（半衰期 24 月），個別風險 = 指數加權殘差變異；
           預測市值加權宇宙與因子傾斜組合的次月波動，偏誤統計量 b = std(實現/預測)，理想 ≈ 1
  組合（月調、成本每單位單邊換手 0.19% = 手續費 3 折 0.043% + 證交稅 0.3% 的一半）：
    市值加權宇宙（基準）、多因子綜合分數前 1/5 市值加權、因子模型最小變異（只做多、單檔上限 5%）
  保留期 2023 起；Sharpe 差異 12 月區塊 bootstrap
輸出：reports/factor_returns.csv、factor_risk_bias.csv、factor_portfolios.csv
"""
import os, sys, warnings
import numpy as np, pandas as pd
sys.argv = sys.argv[:1]; warnings.filterwarnings("ignore")
import model as M, risk_model as RM

R = M.R; SPLIT = RM.SPLIT
FACT = ["規模", "價值", "動能", "品質", "低波"]
N_UNI, COST, HL = 300, 0.0019, 24

def zs(x):
    x = x.astype(float); m, s = x.mean(), x.std()
    if not s or not np.isfinite(s): return x * 0
    return ((x - m) / s).clip(-3, 3).pipe(lambda v: (v - v.mean()) / v.std())

def load():
    m = pd.read_csv(os.path.join(M.D, "ext", "stk_monthly.csv"), dtype={"code": str}, parse_dates=["date"])
    m = m[m.code.str.fullmatch(r"[1-9]\d{3}")]
    m = m[(m.close > 0) & (m.shares > 0)].copy()
    m["cap"] = m.close * m.shares
    px = m.pivot(index="date", columns="code", values="close"); sh = m.pivot(index="date", columns="code", values="shares")
    cap = px * sh; dy = m.pivot(index="date", columns="code", values="dy").fillna(0) / 100
    pr = px.shift(-1) / px - 1; cr = cap.shift(-1) / cap - 1
    chg = (sh.shift(-1) / sh - 1).abs() > 0.02
    ret = pr.where(~chg, cr) + dy / 12  # 次月報酬（t 月底 → t+1 月底）
    lo, hi = ret.quantile(0.01, axis=1), ret.quantile(0.99, axis=1)
    ret = ret.clip(lo, hi, axis=0)
    past = ret.shift(1)  # t 月實現的報酬（t-1 → t）
    mom = (1 + past.shift(1)).rolling(11, min_periods=9).apply(np.prod, raw=True) - 1  # t-12 → t-1
    vol = past.rolling(12, min_periods=9).std()
    pe = m.pivot(index="date", columns="code", values="pe"); pb = m.pivot(index="date", columns="code", values="pb")
    return dict(cap=cap, ret=ret, mom=mom, vol=vol, pe=pe, pb=pb)

def exposures(D, t):
    cap = D["cap"].loc[t].dropna(); uni = cap.nlargest(N_UNI).index
    pe, pb = D["pe"].loc[t, uni], D["pb"].loc[t, uni]
    ep = (1 / pe).where(pe > 0, 0.0); bp = (1 / pb).where(pb > 0, np.nan)
    roe = (pb / pe).where(pe > 0)
    X = pd.DataFrame(index=uni)
    X["規模"] = zs(np.log(cap[uni]))
    X["價值"] = zs((zs(bp.fillna(bp.median())) + zs(ep)) / 2)
    X["動能"] = zs(D["mom"].loc[t, uni].fillna(D["mom"].loc[t, uni].median()))
    X["品質"] = zs(roe.fillna(roe.min() if roe.notna().any() else 0))
    X["低波"] = zs(-D["vol"].loc[t, uni].fillna(D["vol"].loc[t, uni].median()))
    return X.fillna(0), cap[uni]

def nw_tstat(x, lags=3):
    x = np.asarray(x, float); mu = x.mean(); e = x - mu; T = len(x); g = e @ e / T
    for l in range(1, lags + 1): g += 2 * (1 - l / (lags + 1)) * (e[l:] @ e[:-l]) / T
    return mu / np.sqrt(g / T)

def proj(v, u):
    """投影到 {0 ≤ w ≤ u, Σw = 1}（二分法找平移量）。"""
    lo, hi = v.min() - u, v.max()
    for _ in range(60):
        tau = (lo + hi) / 2
        if np.clip(v - tau, 0, u).sum() > 1: lo = tau
        else: hi = tau
    return np.clip(v - (lo + hi) / 2, 0, u)

def minvar(Sig, cap_max=0.05, iters=800):
    """只做多、單檔上限的最小變異：投影梯度法（步長 1/最大特徵值）。"""
    n = len(Sig); w = np.full(n, 1 / n); L = np.linalg.eigvalsh(Sig)[-1] * 2
    for _ in range(iters):
        w = proj(w - 2 * Sig @ w / L, cap_max)
    return w

def sharpe_m(x): return x.mean() / x.std() * np.sqrt(12) if x.std() > 0 else np.nan

def boot_diff_m(a, b, block=12, n=2000, seed=1):
    a, b = np.asarray(a), np.asarray(b); T = len(a); rng = np.random.default_rng(seed); out = []
    for _ in range(n):
        st = rng.integers(0, T, int(np.ceil(T / block))); ix = (st[:, None] + np.arange(block)).ravel()[:T] % T
        out.append(sharpe_m(pd.Series(a[ix])) - sharpe_m(pd.Series(b[ix])))
    return np.percentile(out, [2.5, 97.5]), float(np.mean(np.array(out) > 0))

def main():
    D = load(); dates = D["cap"].index[:-1]
    fr, resid, expo = {}, {}, {}
    for t in dates:
        if D["vol"].loc[t].notna().sum() < 100: continue
        X, cap = exposures(D, t); y = D["ret"].loc[t, X.index]; ok = y.notna()
        if ok.sum() < 100: continue
        A = np.column_stack([np.ones(ok.sum()), X[ok].values]); w = np.sqrt(cap[ok].values)
        b = np.linalg.lstsq(A * w[:, None], y[ok].values * w, rcond=None)[0]
        fr[t] = pd.Series(b, index=["市場"] + FACT); resid[t] = pd.Series(y[ok].values - A @ b, index=X.index[ok]); expo[t] = (X, cap)
    F = pd.DataFrame(fr).T
    rows = []
    for per, m in (("2023 以前", F.index < SPLIT), ("2023 起（保留期）", F.index >= SPLIT), ("全期", F.index == F.index)):
        for k in F:
            x = F.loc[m, k]
            rows.append({"因子": k, "期間": per, "月數": int(m.sum()), "月均報酬": x.mean(), "年化報酬": x.mean() * 12,
                         "年化波動": x.std() * np.sqrt(12), "IR": sharpe_m(x), "NW t 值": nw_tstat(x.values), "正報酬月比": (x > 0).mean()})
    fr_t = pd.DataFrame(rows); fr_t.to_csv(os.path.join(R, "factor_returns.csv"), index=False, encoding="utf-8-sig")
    with pd.option_context("display.width", 250): print(fr_t.round(4).to_string(index=False))

    # 風險模型與組合
    lam = 0.5 ** (1 / HL); ts = list(F.index); out, bias = [], []
    W_prev = {"市值加權宇宙": None, "多因子前 1/5": None, "因子模型最小變異": None}
    for k, t in enumerate(ts):
        if k < 36: continue  # 至少 36 個月估共變異
        hist = F.iloc[:k]  # 只用 t 之前「已實現」的因子報酬（第 k 期報酬在 t+1 才實現）
        wts = lam ** np.arange(len(hist))[::-1]; wts /= wts.sum()
        mu = (hist * wts[:, None]).sum(); C = ((hist - mu).T * wts) @ (hist - mu)
        X, cap = expo[t]; Xf = np.column_stack([np.ones(len(X)), X.values])
        sp = {}
        for s in X.index:
            e = [resid[ts[j]].get(s, np.nan) for j in range(max(0, k - 36), k)]
            e = np.array([v for v in e if np.isfinite(v)])
            sp[s] = (e ** 2).mean() if len(e) >= 12 else np.nan
        spv = pd.Series(sp); spv = spv.fillna(spv.median()).values
        Sig = Xf @ C.values @ Xf.T + np.diag(spv)
        comp = X[["價值", "動能", "品質", "低波"]].mean(axis=1)
        top = comp >= comp.quantile(0.8)
        Ws = {"市值加權宇宙": cap / cap.sum(), "多因子前 1/5": (cap * top) / (cap * top).sum(),
              "因子模型最小變異": pd.Series(minvar(Sig), index=X.index)}
        y = D["ret"].loc[t, X.index]
        for n, w in Ws.items():
            w = w.reindex(X.index).fillna(0); prev = W_prev[n]
            to = w.abs().sum() if prev is None else (w - prev.reindex(w.index).fillna(0)).abs().sum() + prev[~prev.index.isin(w.index)].abs().sum()
            r = float((w * y.fillna(0)).sum()) - to * COST
            pred = float(np.sqrt(w.values @ Sig @ w.values))
            out.append({"date": t, "組合": n, "報酬": r, "換手": to, "預測月波動": pred, "因子暴露": (X.T @ w).round(3).to_dict()})
            bias.append({"date": t, "組合": n, "z": float((w * y.fillna(0)).sum()) / pred})
            # 次月實際漲跌與期初權重的漂移忽略不計（月調）
            W_prev[n] = w * (1 + y.fillna(0)) / (1 + float((w * y.fillna(0)).sum()))
    P = pd.DataFrame(out); B = pd.DataFrame(bias)
    rows = []
    for n, g in B.groupby("組合"):
        g = g.set_index("date")
        for per, m in (("2023 以前", g.index < SPLIT), ("2023 起（保留期）", g.index >= SPLIT), ("全期", g.index == g.index)):
            z = g.z[m] - g.z[m].mean()
            rows.append({"組合": n, "期間": per, "月數": int(m.sum()), "偏誤統計量 b（≈1 準確）": float(z.std())})
    bt = pd.DataFrame(rows); bt.to_csv(os.path.join(R, "factor_risk_bias.csv"), index=False, encoding="utf-8-sig")
    print(bt.round(3).to_string(index=False))
    rows = []; base = P[P.組合 == "市值加權宇宙"].set_index("date").報酬
    for n, g in P.groupby("組合"):
        g = g.set_index("date"); r = g.報酬
        for per, m in (("2023 以前", r.index < SPLIT), ("2023 起（保留期）", r.index >= SPLIT), ("全期", r.index == r.index)):
            x = r[m]; eq = (1 + x).cumprod()
            (lo, hi), pw = boot_diff_m(x.values, base[m].values) if n != "市值加權宇宙" else ((np.nan, np.nan), np.nan)
            ex = pd.DataFrame(list(g.因子暴露[m])).mean()
            rows.append({"組合": n, "期間": per, "起": str(x.index[0].date()), "月數": len(x),
                         "CAGR": eq.iloc[-1] ** (12 / len(x)) - 1, "年化波動": x.std() * np.sqrt(12), "Sharpe": sharpe_m(x),
                         "最大回撤": float((eq / eq.cummax() - 1).min()), "年換手": g.換手[m].mean() * 12,
                         "Sharpe差 vs 市值加權 95%下界": lo, "95%上界": hi, **{f"平均{k}暴露": ex.get(k) for k in FACT}})
    pt = pd.DataFrame(rows); pt.to_csv(os.path.join(R, "factor_portfolios.csv"), index=False, encoding="utf-8-sig")
    with pd.option_context("display.width", 250): print(pt.round(3).to_string(index=False))

if __name__ == "__main__":
    main()
