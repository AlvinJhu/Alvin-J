"""
風險模型比較：GARCH / GJR-GARCH / EWMA 對照 v3 HAR，並做 VaR、CVaR（ES）回測
文獻：Bollerslev (1986) GARCH；Glosten, Jagannathan & Runkle (1993) 不對稱 GARCH；
      J.P. Morgan/Reuters (1996) RiskMetrics EWMA；Kupiec (1995)、Christoffersen (1998) VaR 回測；
      Patton (2011) QLIKE；Acerbi & Szekely (2014) ES 回測。
設計（事先固定，不調參）：
  - 每 63 交易日以擴張視窗重估參數（最少 500 天），之後逐日濾波預測 → 完全樣本外
  - 波動：未來 20 日實現波動（同 risk_model.py 標的）
  - VaR/CVaR：次日報酬 95%、99%；常態 EWMA、GARCH-t、GJR-t、歷史模擬 500 日、濾波歷史模擬（FHS，GJR 標準化殘差）
  - 部位：把 v3.1 的預測波動換成 GARCH/GJR，其他（22% 目標、1/4 減碼、門檻 25%、持續 10 日、調整帶 0.2）全部沿用；
          另測「CVaR 預算」部位：w = min(1, 預算 / 預測 97.5% CVaR)，預算 = 22% 年化目標在常態下對應的 CVaR，不另選
  - 保留期 2023 起；Sharpe 差異 63 日區塊 bootstrap
輸出：reports/garch_vol_forecast.csv、garch_var_backtest.csv、garch_strategy.csv、garch_sharpe_diff.csv
"""
import os, sys, warnings
import numpy as np, pandas as pd
from scipy import stats as sps
sys.argv = sys.argv[:1]; warnings.filterwarnings("ignore")
import model as M, risk_model as RM, futures_eval as FE, cta_model as CT
from arch import arch_model

R = M.R; ANN = np.sqrt(252); H = 20; SPLIT = RM.SPLIT
TARGET, BAND, CUT, THR, HOLD = 0.22, 0.2, 0.25, 0.25, 10
HARX = CT.HARX

def t_es(q, nu):
    """標準化 t 分配（變異數 1）左尾 q 的 ES（正值）。"""
    t = sps.t.ppf(q, nu); return float(sps.t.pdf(t, nu) / q * (nu + t * t) / (nu - 1) * np.sqrt((nu - 2) / nu))

def garch_wf(r100, kind, min_train=500, step=63):
    """回傳 DataFrame：h1 變異數、20 日累積變異數、nu、標準化殘差（全部樣本外）。"""
    n = len(r100); out = pd.DataFrame(index=r100.index, columns=["v1", "v20", "nu", "z"], dtype=float)
    start = min_train
    while start < n:
        end = min(start + step, n)
        am = arch_model(r100, mean="Constant", vol="GARCH", p=1, o=1 if kind == "gjr" else 0, q=1,
                        dist="t" if kind != "garch_n" else "normal")
        res = am.fit(last_obs=r100.index[start], disp="off")
        fc = res.forecast(horizon=H, start=r100.index[start], reindex=False)  # 第 t 列 = 用 t 日收盤前資料預測 t+1..t+20
        idx = r100.index[start:end]; var = fc.variance.iloc[:len(idx)]
        out.loc[idx, "v1"] = var.iloc[:, 0].values
        out.loc[idx, "v20"] = var.sum(axis=1).values
        out.loc[idx, "nu"] = res.params.get("nu", np.nan)
        # 標準化殘差（用固定參數濾波，僅取到 t 日為止已知者）
        sres = am.fix(res.params)
        z = (r100 - res.params["mu"]) / sres.conditional_volatility
        zz = z.iloc[start:end]; out.loc[zz.index, "z"] = zz.values
        start = end
    return out

def kupiec(hit, p):
    n, x = len(hit), int(hit.sum()); ph = x / n
    if x == 0: lr = -2 * n * np.log(1 - p)
    else: lr = -2 * ((n - x) * np.log(1 - p) + x * np.log(p) - (n - x) * np.log(1 - ph) - x * np.log(ph))
    return float(1 - sps.chi2.cdf(lr, 1))

def christoffersen(hit):
    h = hit.astype(int); a, b = h[:-1], h[1:]
    n00, n01 = int(((a == 0) & (b == 0)).sum()), int(((a == 0) & (b == 1)).sum())
    n10, n11 = int(((a == 1) & (b == 0)).sum()), int(((a == 1) & (b == 1)).sum())
    p01 = n01 / max(n00 + n01, 1); p11 = n11 / max(n10 + n11, 1); p = (n01 + n11) / max(len(a), 1)
    def ll(pp, k0, k1): return (k0 * np.log(1 - pp) if k0 and pp < 1 else 0) + (k1 * np.log(pp) if k1 and pp > 0 else 0)
    lr = -2 * (ll(p, n00 + n10, n01 + n11) - ll(p01, n00, n01) - ll(p11, n10, n11))
    return float(1 - sps.chi2.cdf(lr, 1)), n11

def main():
    df = M.load(); c = df.close; lr = np.log(c).diff()
    r = (c / c.shift(1) - 1).dropna(); r100 = r * 100
    # --- 1) 預測（全部 walk-forward）
    gm = {k: garch_wf(r100, k) for k in ("garch_n", "garch_t", "gjr")}
    ew_var = (r100 ** 2).ewm(alpha=0.06, adjust=False).mean()  # RiskMetrics λ=0.94，t 日收盤已知
    X, y_vol, y_dd, _, _ = RM.build(df)
    Xv = X.dropna(); har = np.exp(RM.wf(Xv[HARX], y_vol.reindex(Xv.index), RM.ols))
    vol = {"EWMA 0.94": np.sqrt(ew_var) / 100 * ANN, "HAR+幅度+VIX（v3）": har}
    for k, lab in (("garch_n", "GARCH(1,1)-常態"), ("garch_t", "GARCH(1,1)-t"), ("gjr", "GJR-GARCH-t")):
        vol[lab] = np.sqrt(gm[k].v20.astype(float) / H) / 100 * ANN
    vol["HAR 與 GJR 平均"] = np.exp((np.log(vol["HAR+幅度+VIX（v3）"]) + np.log(vol["GJR-GARCH-t"])) / 2)

    idx = y_vol.dropna().index
    for v in vol.values(): idx = idx.intersection(v.dropna().index)
    rows = []
    for n, v in vol.items():
        for per, m in (("2023 以前", idx < SPLIT), ("2023 起（保留期）", idx >= SPLIT), ("全期", idx == idx)):
            yy, pp = y_vol[idx[m]], np.log(v[idx[m]].astype(float))
            rows.append({"模型": n, "期間": per, "天數": int(m.sum()), "R2(對數波動)": RM.r2(yy, pp, yy.mean()),
                         "QLIKE": RM.qlike(yy.values, pp.values), "MAE(年化波動點)": float(np.mean(np.abs(np.exp(yy) - np.exp(pp))))})
    vt = pd.DataFrame(rows); vt.to_csv(os.path.join(R, "garch_vol_forecast.csv"), index=False, encoding="utf-8-sig")
    print(f"波動預測共同樣本 {idx[0].date()} ~ {idx[-1].date()}"); print(vt.round(3).to_string(index=False))

    # --- 2) VaR / CVaR 回測（次日）
    nxt = r.shift(-1)  # t 日預測 t+1 日報酬
    var_rows, q_list = [], (0.05, 0.01)
    base_idx = gm["gjr"].v1.dropna().index.intersection(nxt.dropna().index)
    hs_win = 500
    for q in q_list:
        est = {}
        s_ew = np.sqrt(ew_var) / 100; zq = sps.norm.ppf(q)
        est["EWMA-常態"] = (zq * s_ew, -sps.norm.pdf(zq) / q * s_ew)
        for k, lab in (("garch_n", "GARCH-常態"), ("garch_t", "GARCH-t"), ("gjr", "GJR-t")):
            g = gm[k]; s = np.sqrt(g.v1.astype(float)) / 100; mu = 0.0
            if k == "garch_n": est[lab] = (zq * s, -sps.norm.pdf(zq) / q * s)
            else:
                nu = g.nu.astype(float); tq = sps.t.ppf(q, nu) * np.sqrt((nu - 2) / nu)
                es = pd.Series([t_es(q, v) if np.isfinite(v) else np.nan for v in nu], index=nu.index)
                est[lab] = (tq * s, -es * s)
        hv = r.rolling(hs_win).quantile(q)
        he = r.rolling(hs_win).apply(lambda x: x[x <= np.quantile(x, q)].mean(), raw=True)
        est["歷史模擬 500 日"] = (hv, he)
        z = gm["gjr"].z.astype(float); s = np.sqrt(gm["gjr"].v1.astype(float)) / 100
        zq_f = z.rolling(hs_win, min_periods=250).quantile(q)
        ze_f = z.rolling(hs_win, min_periods=250).apply(lambda x: x[x <= np.quantile(x, q)].mean(), raw=True)
        est["濾波歷史模擬（GJR）"] = (zq_f * s, ze_f * s)
        for n, (v, e) in est.items():
            ii = base_idx.intersection(v.dropna().index).intersection(e.dropna().index)
            for per, m in (("2023 以前", ii < SPLIT), ("2023 起（保留期）", ii >= SPLIT), ("全期", ii == ii)):
                j = ii[m]; y = nxt[j]; vv = v[j].astype(float); ee = e[j].astype(float); hit = (y < vv).values
                pc, n11 = christoffersen(hit)
                # Acerbi-Szekely Z2 型統計：ES 正確時 ≈ 0，> 0 代表低估尾部損失
                z2 = float(np.sum(y.values * hit / (-ee.values)) / (len(j) * q) + 1) if len(j) else np.nan
                var_rows.append({"信賴": f"{1-q:.0%}", "模型": n, "期間": per, "天數": len(j), "預期穿越率": q,
                                 "實際穿越率": hit.mean(), "Kupiec p": kupiec(hit, q), "Christoffersen 獨立 p": pc,
                                 "連續穿越次數": n11, "穿越日平均損失/預測CVaR": float((y[hit] / ee[hit]).mean()) if hit.any() else np.nan,
                                 "ES 回測 Z2（<0 低估）": z2, "平均 VaR": float(vv.mean()), "平均 CVaR": float(ee.mean())})
    vb = pd.DataFrame(var_rows); vb.to_csv(os.path.join(R, "garch_var_backtest.csv"), index=False, encoding="utf-8-sig")
    with pd.option_context("display.width", 250):
        print(vb[vb.期間 != "全期"].round(4).to_string(index=False))

    # --- 3) 部位：v3.1 共同樣本（opt_fgn_bias 2018-06 起）
    Xc = X.copy(); Xc["opt_fgn_bias"] = FE.fut_features(df, FE.load_futures(df.index))["opt_fgn_bias"]
    Xc = Xc.dropna(); yd = y_dd.reindex(Xc.index)
    p_dd = (RM.wf(Xc, yd, RM.logit) + RM.wf(Xc, yd, RM.xgb)) / 2
    vol_c = np.exp(RM.wf(Xc[HARX], y_vol.reindex(Xc.index), RM.ols))
    rr = (c.shift(-1) / c - 1)
    common = vol_c.dropna().index.intersection(p_dd.dropna().index).intersection(gm["gjr"].v20.dropna().index)
    common = common[rr.reindex(common).notna()]
    for v in vol.values(): common = common.intersection(v.dropna().index)
    rv = rr[common]; pd_c = p_dd[common].values
    es_g = -(gm["gjr"].v1.astype(float).pipe(np.sqrt) / 100 *
             pd.Series([t_es(0.025, v) for v in gm["gjr"].nu.astype(float)], index=gm["gjr"].index))
    budget = sps.norm.pdf(sps.norm.ppf(0.025)) / 0.025 * TARGET / ANN  # 22% 年化 ↔ 常態 97.5% CVaR
    cvar_sz = (budget / -es_g[common]).clip(upper=1.0)
    cands = {"v3.1 風險模型（HAR，現行）": RM.weights(vol_c[common], pd_c, TARGET, CUT, THR, BAND, HOLD),
             "v3.1 規則＋GARCH-t 波動": RM.weights(vol["GARCH(1,1)-t"][common].astype(float), pd_c, TARGET, CUT, THR, BAND, HOLD),
             "v3.1 規則＋GJR-t 波動": RM.weights(vol["GJR-GARCH-t"][common].astype(float), pd_c, TARGET, CUT, THR, BAND, HOLD),
             "v3.1 規則＋EWMA 波動": RM.weights(vol["EWMA 0.94"][common].astype(float), pd_c, TARGET, CUT, THR, BAND, HOLD),
             "v3.1 規則＋HAR/GJR 平均": RM.weights(vol["HAR 與 GJR 平均"][common].astype(float), pd_c, TARGET, CUT, THR, BAND, HOLD),
             "GJR CVaR 預算部位（無大跌警訊）": RM.weights(-es_g[common] / budget * TARGET, np.zeros(len(common)), TARGET, 1, 1, BAND, 1),
             "GJR CVaR 預算＋大跌警訊": RM.weights(-es_g[common] / budget * TARGET, pd_c, TARGET, CUT, THR, BAND, HOLD),
             "買進持有": pd.Series(1.0, index=common)}
    rows, rets = [], {}
    for n, w in cands.items():
        s = rv if n == "買進持有" else RM.strat(w, rv); rets[n] = s
        for per, m in (("2023 以前", common < SPLIT), ("2023 起（保留期）", common >= SPLIT), ("全期", common == common)):
            rows.append({"策略": n, "期間": per, "起": str(common[m][0].date()), **M.stats(s[m]), "平均持股": w[m].mean(),
                         "年換手": w[m].diff().abs().sum() / (m.sum() / 252),
                         "日報酬 1% CVaR": float(s[m][s[m] <= s[m].quantile(0.01)].mean())})
    st = pd.DataFrame(rows); st.to_csv(os.path.join(R, "garch_strategy.csv"), index=False, encoding="utf-8-sig")
    with pd.option_context("display.width", 250): print(st.round(3).to_string(index=False))
    rows = []
    base = "v3.1 風險模型（HAR，現行）"
    for n in cands:
        if n == base: continue
        for per, m in (("2023 起（保留期）", common >= SPLIT), ("全期", common == common)):
            (lo, hi), pw = CT.boot_diff(rets[n][m].values, rets[base][m].values)
            rows.append({"比較": f"{n} − v3.1", "期間": per, "Sharpe差": CT.sharpe(rets[n][m]) - CT.sharpe(rets[base][m]),
                         "95%下界": lo, "95%上界": hi, "bootstrap 勝率": pw})
    sd = pd.DataFrame(rows); sd.to_csv(os.path.join(R, "garch_sharpe_diff.csv"), index=False, encoding="utf-8-sig")
    print(sd.round(3).to_string(index=False))

    # 今日（最後一天）GJR 讀數
    g = gm["gjr"]; last = g.v1.dropna().index[-1]; s1 = np.sqrt(float(g.v1[last])) / 100; nu = float(g.nu[last])
    print({"date": str(last.date()), "GJR 次日波動(年化)": round(s1 * ANN, 4), "GJR 20日波動(年化)": round(float(vol["GJR-GARCH-t"][last]), 4),
           "nu": round(nu, 2), "VaR99_1d": round(float(sps.t.ppf(0.01, nu) * np.sqrt((nu - 2) / nu) * s1), 4),
           "CVaR99_1d": round(-t_es(0.01, nu) * s1, 4)})

if __name__ == "__main__":
    main()
