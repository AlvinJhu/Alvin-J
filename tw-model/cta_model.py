"""
CTA 模型 v4：時間序列動能（TSMOM）＋波動度目標部位，並測試與 v3.1 大跌警訊疊加
理論：Moskowitz, Ooi & Pedersen (2012) 時間序列動能；Hurst, Ooi & Pedersen (2017) 多回看期趨勢；
      Moreira & Muir (2017) 波動度管理部位。
設計（事先固定，不調參，避免多重檢定／資料探勘）：
  訊號：過去 21/63/126/252 交易日報酬的正負號，等權平均 → [-1, 1]（MOP 原版只用 252 日）
  部位：訊號 × min(1, 22% / 預測波動)；預測波動、22%、調整帶 0.2 全沿用 v3（2018–2022 選定）
  疊加：v3.1 大跌機率 > 25% → 部位 × 1/4，至少維持 10 交易日（同 risk_model.py）
  比較 5 種 CTA 變體（僅此 5 種，全數列出）：12 月多空、多期多空、多期只做多、多期只做多＋大跌警訊、
  v3.1 與多期多空各半（分散配置）
  標的：(a) 加權指數（與 v3 同一報酬序列，可直接比較）
        (b) 台指期近月（最大未平倉合約）連續報酬：可放空的真實標的，含正逆價差／除息影響
  成本：每單位換手 0.1%（同 v3）
  驗證：與 v3 相同切點，2023 年以前／2023 起（保留期）；Sharpe 差異用 63 日區塊 bootstrap 求 95% 區間
輸出：reports/cta_strategy.csv、cta_sharpe_diff.csv、cta_crisis.csv、cta_today.json、cta_equity.png
"""
import os, sys, json
import numpy as np, pandas as pd
sys.argv = sys.argv[:1]
import model as M
import risk_model as RM
import futures_eval as FE

R = M.R; ANN = np.sqrt(252); SPLIT = RM.SPLIT
LOOKBACKS = (21, 63, 126, 252)
TARGET, BAND, CUT, THR, HOLD = 0.22, 0.2, 0.25, 0.25, 10  # 全部沿用 v3.1 已選定參數
COST = 0.001
HARX = ["rv1", "rv5", "rv22", "rv66", "park5", "neg5", "vix"]  # v3「HAR+幅度+VIX」

def tsmom(c, lbs):
    return pd.concat([np.sign(c / c.shift(k) - 1) for k in lbs], axis=1).mean(axis=1)

def tx_continuous(index):
    """台指期連續報酬：t 日持有 t-1 日未平倉量最大的合約，報酬 = 該合約 t 日收盤 / t-1 日收盤 - 1。"""
    fd = pd.read_csv(os.path.join(M.D, "ext", "fut_tx_daily.csv"), parse_dates=["date"])
    fd = fd[(fd.trading_session == "position") & ~fd.contract_date.astype(str).str.contains("/") & (fd.close > 0)]
    px = fd.pivot_table(index="date", columns="contract_date", values="close")
    oi = fd.pivot_table(index="date", columns="contract_date", values="open_interest").fillna(0)
    held = oi.idxmax(axis=1).shift(1)
    ret = pd.Series(np.nan, index=px.index)
    prev = px.shift(1)
    for d, k in held.dropna().items():
        if k in px.columns and pd.notna(px.at[d, k]) and pd.notna(prev.at[d, k]):
            ret[d] = px.at[d, k] / prev.at[d, k] - 1
    return ret.reindex(index)

def sized(sig, vol_fc, long_only=False, warn=None):
    s = np.asarray(sig, float)
    if long_only: s = np.clip(s, 0, None)
    raw = s * np.minimum(1.0, TARGET / np.asarray(vol_fc))
    if warn is not None: raw = raw * np.where(warn, CUT, 1.0)
    w = np.empty(len(raw)); cur = raw[0]
    for i, x in enumerate(raw):
        if abs(x - cur) > BAND or (x == 0) != (cur == 0) or np.sign(x) != np.sign(cur): cur = x
        w[i] = cur
    return pd.Series(w, index=vol_fc.index)

def sharpe(x): return x.mean() / x.std() * ANN if x.std() > 0 else np.nan

def boot_diff(a, b, block=63, n=2000, seed=1):
    """循環區塊 bootstrap：Sharpe(a) - Sharpe(b) 的 95% 區間（a、b 同日重抽，保留相關性）。"""
    a, b = np.asarray(a), np.asarray(b); T = len(a); rng = np.random.default_rng(seed); out = []
    for _ in range(n):
        st = rng.integers(0, T, int(np.ceil(T / block)))
        ix = (st[:, None] + np.arange(block)).ravel()[:T] % T
        out.append(sharpe(pd.Series(a[ix])) - sharpe(pd.Series(b[ix])))
    return np.percentile(out, [2.5, 97.5]), float(np.mean(np.array(out) > 0))

def main():
    df = M.load(); c = df.close
    # 1) v3 波動預測（不含選擇權因子，樣本 2016 起）與 v3.1 大跌機率（含 opt_fgn_bias，2018-06 起），皆 walk-forward
    X, y_vol, y_dd, ew, lr = RM.build(df)
    Xv = X.dropna(); yv = y_vol.reindex(Xv.index)
    vol_fc = np.exp(RM.wf(Xv[HARX], yv, RM.ols))
    Xc = X.copy(); Xc["opt_fgn_bias"] = FE.fut_features(df, FE.load_futures(df.index))["opt_fgn_bias"]
    Xc = Xc.dropna(); yd = y_dd.reindex(Xc.index)
    p_dd = (RM.wf(Xc, yd, RM.logit) + RM.wf(Xc, yd, RM.xgb)) / 2  # 與 risk_model.py 相同因子
    vol_c = np.exp(RM.wf(Xc[HARX], y_vol.reindex(Xc.index), RM.ols)).dropna()  # v3.1 同樣本的波動預測

    r_idx = (c.shift(-1) / c - 1)
    r_tx = tx_continuous(df.index).shift(-1)  # t 日收盤決定部位，賺 t+1 日報酬
    sig12, sigm = tsmom(c, (252,)), tsmom(c, LOOKBACKS)

    vol_fc = vol_fc.dropna()
    full = vol_fc.index[sigm.reindex(vol_fc.index).notna() & r_idx.reindex(vol_fc.index).notna()]
    common = full.intersection(p_dd.dropna().index).intersection(vol_c.index)
    print(f"CTA 長樣本 {full[0].date()} ~ {full[-1].date()}（{len(full)} 天）；與 v3.1 共同樣本 {common[0].date()} 起（{len(common)} 天）")

    def variants(idx):
        vf = (vol_c if idx.equals(common) else vol_fc)[idx]; warn = RM.warn_active(p_dd.reindex(idx).values, THR, HOLD) if idx.equals(common) else None
        out = {"CTA-A 12月動能 多空": sized(sig12[idx], vf),
               "CTA-B 多期動能 多空": sized(sigm[idx], vf),
               "CTA-C 多期動能 只做多": sized(sigm[idx], vf, long_only=True)}
        if warn is not None:
            out["CTA-D 只做多＋大跌警訊"] = sized(sigm[idx], vf, long_only=True, warn=warn)
            out["v3.1 風險模型（現行）"] = RM.weights(vf, p_dd[idx].values, TARGET, CUT, THR, BAND, HOLD)
            out["E 各半：v3.1＋CTA-B"] = (out["v3.1 風險模型（現行）"] + out["CTA-B 多期動能 多空"]) / 2  # 分散配置，固定 50/50
        out["波動目標（無動能）"] = sized(pd.Series(1.0, index=idx), vf)
        out["買進持有"] = pd.Series(1.0, index=idx)
        return out

    rows, rets = [], {}
    for sample, idx in (("長樣本", full), ("共同樣本", common)):
        for asset, rr in (("加權指數", r_idx), ("台指期近月", r_tx)):
            r = rr.reindex(idx).fillna(0.0)
            for n, w in variants(idx).items():
                s = r if n == "買進持有" else RM.strat(w, r, COST)
                rets[(sample, asset, n)] = s
                for per, m in (("2023 以前", idx < SPLIT), ("2023 起（保留期）", idx >= SPLIT), ("全期", idx == idx)):
                    rows.append({"樣本": sample, "標的": asset, "策略": n, "期間": per,
                                 "起": str(idx[m][0].date()), **M.stats(s[m]),
                                 "平均部位": w[m].mean(), "放空天數比": (w[m] < 0).mean(),
                                 "年換手": w[m].diff().abs().sum() / (m.sum() / 252)})
    st = pd.DataFrame(rows)
    st.to_csv(os.path.join(R, "cta_strategy.csv"), index=False, encoding="utf-8-sig")
    with pd.option_context("display.width", 250):
        print(st[st.期間 != "全期"].round(3).to_string(index=False))

    # 2) 保留期 Sharpe 差異的 bootstrap 區間
    rows = []
    for asset in ("加權指數", "台指期近月"):
        for a, b in (("CTA-D 只做多＋大跌警訊", "v3.1 風險模型（現行）"), ("CTA-C 多期動能 只做多", "v3.1 風險模型（現行）"),
                     ("CTA-B 多期動能 多空", "v3.1 風險模型（現行）"), ("E 各半：v3.1＋CTA-B", "v3.1 風險模型（現行）"), ("CTA-C 多期動能 只做多", "波動目標（無動能）"),
                     ("CTA-B 多期動能 多空", "買進持有")):
            for per, sel in (("2023 起（保留期）", lambda i: i >= SPLIT), ("共同樣本全期", lambda i: i == i)):
                sa, sb = rets[("共同樣本", asset, a)], rets[("共同樣本", asset, b)]
                m = sel(sa.index); (lo, hi), pwin = boot_diff(sa[m].values, sb[m].values)
                rows.append({"標的": asset, "比較": f"{a} − {b}", "期間": per, "Sharpe差": sharpe(sa[m]) - sharpe(sb[m]),
                             "95%下界": lo, "95%上界": hi, "bootstrap 勝率": pwin})
    sd = pd.DataFrame(rows); sd.to_csv(os.path.join(R, "cta_sharpe_diff.csv"), index=False, encoding="utf-8-sig")
    print(sd.round(3).to_string(index=False))

    # 3) 大跌事件：買進持有自高點回落 > 10% 的區段，各策略同期報酬
    bh = (1 + r_idx.reindex(common).fillna(0)).cumprod(); dd = bh / bh.cummax() - 1
    ev, in_ev = [], False
    for d, v in dd.items():
        if not in_ev and v < -0.10:
            in_ev = True; peak = bh[:d].idxmax()
        if in_ev and v == 0:
            ev.append((peak, bh[peak:d].idxmin())); in_ev = False
    if in_ev: ev.append((peak, bh[peak:].idxmin()))
    rows = []
    for p0, p1 in ev:
        row = {"高點": str(p0.date()), "低點": str(p1.date())}
        for n in ("買進持有", "v3.1 風險模型（現行）", "CTA-B 多期動能 多空", "CTA-C 多期動能 只做多", "CTA-D 只做多＋大跌警訊", "E 各半：v3.1＋CTA-B"):
            s = rets[("共同樣本", "加權指數", n)]; row[n] = float((1 + s[(s.index >= p0) & (s.index < p1)]).prod() - 1)
        rows.append(row)
    cr = pd.DataFrame(rows); cr.to_csv(os.path.join(R, "cta_crisis.csv"), index=False, encoding="utf-8-sig")
    print(cr.round(3).to_string(index=False))

    # 4) 今日
    last = df.index[-1]
    sig_parts = {f"{k}d": float(np.sign(c.iloc[-1] / c.iloc[-1 - k] - 1)) for k in LOOKBACKS}
    w_c = variants(common)
    today = {"date": str(last.date()), "close": float(c.iloc[-1]), "tsmom_signs": sig_parts, "tsmom_signal": float(sigm.iloc[-1]),
             "vol_forecast_last_backtest_day": round(float(vol_c.iloc[-1]), 4),
             "weights_last_backtest_day": {k: round(float(v.iloc[-1]), 3) for k, v in w_c.items() if k != "買進持有"},
             "note": "最後回測日權重以 walk-forward 預測計算；今日即時部位以 risk_today.json 的波動與警訊為準"}
    json.dump(today, open(os.path.join(R, "cta_today.json"), "w", encoding="utf-8"), ensure_ascii=False, indent=2)
    print(json.dumps(today, ensure_ascii=False))

    try:
        import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
        lab = {"CTA-A 12月動能 多空": "A 12m L/S", "CTA-B 多期動能 多空": "B multi L/S", "CTA-C 多期動能 只做多": "C multi long-only",
               "CTA-D 只做多＋大跌警訊": "D = C + crash warning", "v3.1 風險模型（現行）": "v3.1 risk model",
               "E 各半：v3.1＋CTA-B": "E 50/50 v3.1 + B", "波動目標（無動能）": "vol target only", "買進持有": "Buy & Hold"}
        fig, ax = plt.subplots(2, 1, figsize=(11, 8), sharex=True, gridspec_kw={"height_ratios": [3, 1]})
        for n in lab:
            s = rets[("共同樣本", "加權指數", n)]; ax[0].plot((1 + s).cumprod(), label=lab[n], lw=1.6 if n.startswith(("CTA-D", "v3.1")) else .9)
        ax[0].axvline(SPLIT, color="k", ls="--", lw=.8); ax[0].set_yscale("log"); ax[0].legend(fontsize=8)
        ax[0].set_title("TAIEX: time-series momentum + vol targeting vs v3.1 (no fitted CTA params, cost 0.1%)")
        w = variants(common)
        ax[1].plot(w["CTA-B 多期動能 多空"], lw=.7, label="B weight"); ax[1].plot(w["CTA-D 只做多＋大跌警訊"], lw=.7, label="D weight")
        ax[1].axhline(0, color="k", lw=.5); ax[1].legend(fontsize=8)
        fig.tight_layout(); fig.savefig(os.path.join(R, "cta_equity.png"), dpi=110)
    except Exception as e:
        print("圖檔略過：", e)

if __name__ == "__main__":
    main()
