"""
一次性抓取（GitHub Actions；雲端被期交所擋）：台指選擇權波動率指數（TAIWAN VIX）
期交所官網只提供近 3–4 個月的官方 VIX，所以 2016 起改用期交所 TXO 每日行情（收盤最後最佳買賣價）
依期交所公告的 CBOE VIX 編製法自行重建（近月＋次近月月選、價外序列買賣中價、30 天插補），
再用近幾個月官方值驗證重建誤差。
輸出：data/ext/twvix_daily.csv（date, close=重建值, official=官方收盤值(有的話), sig_near, sig_next, T_near, T_next, F_near）
      data/ext/fetch_twvix_log.json
用法：python fetch_twvix.py [--start 2016-01] [--end 2026-11]
"""
import os, sys, io, re, json, time, datetime as dt
import numpy as np, pandas as pd, requests, warnings; warnings.filterwarnings("ignore", category=RuntimeWarning)

HERE = os.path.dirname(os.path.abspath(__file__)); EXT = os.path.join(HERE, "data", "ext")
BASE = "https://www.taifex.com.tw"; R = 0.01  # 無風險利率（台灣短率約 1–1.7%，對 VIX 影響 <0.1 點）
S = requests.Session(); S.headers.update({"User-Agent": "Mozilla/5.0 (research data download)"})

def req(method, url, **kw):
    for k in range(5):
        try:
            r = S.request(method, url, timeout=180, **kw)
            if r.status_code == 200: return r
            print("HTTP", r.status_code, url)
        except requests.RequestException as e: print("retry", url, e)
        time.sleep(5 * (k + 1))
    return None

def official():
    """近 4 個月官方每日收盤 VIX（log2data/YYYYMMnew.txt：日期、時間、指數、收盤前 1 分鐘平均）"""
    r = req("GET", BASE + "/cht/7/vixDaily3MNew"); rows = []
    links = re.findall(r"window\.open\('([^']+new\.txt)'", r.text) if r else []
    for u in links:
        t = req("GET", u if u.startswith("http") else BASE + u)
        if t is None: continue
        for line in t.content.decode("cp950", errors="ignore").splitlines():
            m = re.match(r"\s*(\d{8})\s+(\d+)\s+([\d.]+)", line)
            if m: rows.append((pd.Timestamp(m.group(1)), float(m.group(3))))
    return pd.Series(dict(rows), name="official").sort_index(), links

def third_wed(ym):
    y, m = int(ym[:4]), int(ym[4:6]); d = dt.date(y, m, 1)
    return d + dt.timedelta(days=(2 - d.weekday()) % 7 + 14)

def num(s): return pd.to_numeric(s.astype(str).str.strip().replace({"-": np.nan, "": np.nan}), errors="coerce")

def month_df(y, m):
    a = dt.date(y, m, 1); b = (a.replace(day=28) + dt.timedelta(days=4)).replace(day=1) - dt.timedelta(days=1)
    b = min(b, dt.date.today())
    r = req("POST", BASE + "/cht/3/optDataDown", data={"down_type": "1", "commodity_id": "TXO", "commodity_id2": "",
                                                       "queryStartDate": a.strftime("%Y/%m/%d"), "queryEndDate": b.strftime("%Y/%m/%d")})
    if r is None or len(r.content) < 500: return None
    d = pd.read_csv(io.StringIO(r.content.decode("cp950", errors="ignore")), dtype=str, index_col=False)
    d.columns = [c.strip() for c in d.columns]
    d = d[(d["交易時段"].str.strip() == "一般") & (d["契約"].str.strip() == "TXO")]
    d["ym"] = d["到期月份(週別)"].str.strip(); d = d[d.ym.str.fullmatch(r"\d{6}")]  # 只用月選（TAIWAN VIX 不含週選）
    out = pd.DataFrame({"date": pd.to_datetime(d["交易日期"].str.strip()), "ym": d.ym, "K": num(d["履約價"]),
                        "cp": np.where(d["買賣權"].str.contains("買"), "C", "P"),
                        "bid": num(d["最後最佳買價"]), "ask": num(d["最後最佳賣價"]), "settle": num(d["結算價"])})
    out["exp"] = pd.to_datetime(d["契約到期日"].str.strip(), format="%Y%m%d", errors="coerce") if "契約到期日" in d else pd.NaT
    miss = out.exp.isna(); out.loc[miss, "exp"] = pd.to_datetime(out.loc[miss, "ym"].map(third_wed))
    return out

def quotes(x, mode):
    """mode: mid＝收盤最後最佳買賣中價；set＝結算價；hyb＝有買賣價用中價、否則用結算價"""
    b, a, st = x.bid, x.ask, x.settle
    mid = ((b + a) / 2).where((b > 0) & (a > 0) & (a >= b))
    return {"mid": mid, "set": st.where(st > 0), "hyb": mid.fillna(st.where(st > 0))}[mode]

def sigma2(g, T, mode="hyb", tick=0.1):
    """CBOE VIX 單一到期 σ²；g：同一到期日的買賣權報價。截斷：連續兩檔價外報價缺或 ≤ 最小跳動就停止"""
    c = g[g.cp == "C"].drop_duplicates("K").set_index("K"); p = g[g.cp == "P"].drop_duplicates("K").set_index("K")
    Ks = c.index.union(p.index).sort_values()
    cq, pq = quotes(c, mode).reindex(Ks), quotes(p, mode).reindex(Ks)
    both = (cq - pq).dropna()
    if len(both) < 3: return np.nan, np.nan
    Kst = both.abs().idxmin(); F = Kst + np.exp(R * T) * both[Kst]
    K0 = Ks[Ks <= F].max() if (Ks <= F).any() else Ks.min()
    Q = {}
    for x, order in ((pq, [k for k in Ks[::-1] if k < K0]), (cq, [k for k in Ks if k > K0])):
        bad = 0
        for k in order:
            v = x.get(k, np.nan)
            if not (v > tick): bad += 1
            else: bad = 0; Q[k] = v
            if bad >= 2: break
    q0 = np.nanmean([cq.get(K0, np.nan), pq.get(K0, np.nan)])
    if np.isfinite(q0): Q[K0] = q0
    ks = np.array(sorted(Q)); qs = np.array([Q[k] for k in ks])
    if len(ks) < 5: return np.nan, F
    dK = np.gradient(ks); dK[0] = ks[1] - ks[0]; dK[-1] = ks[-1] - ks[-2]
    s2 = 2 / T * np.sum(dK / ks ** 2 * np.exp(R * T) * qs) - 1 / T * (F / K0 - 1) ** 2
    return s2, F

def day_vix(g, day, tdays, keep=None):
    exps = sorted(g.exp.unique())
    # 換月：近月剩 ≤2 個交易日就改用次近月與第 2 個次近月
    def tleft(e): return int(((tdays > day) & (tdays <= e)).sum())
    exps = [e for e in exps if pd.Timestamp(e) > day and tleft(e) > 2][:2]
    if len(exps) < 2: return None
    if keep is not None: keep.append(g[g.exp.isin(exps)])
    out = {}; Ts = []
    for e in exps:
        mins = (pd.Timestamp(e) - day).days * 1440 + (13 * 60 + 30) - (13 * 60 + 45); Ts.append(mins / 525600)
    out["T_near"], out["T_next"] = Ts
    for mode in ("hyb", "mid", "set"):
        (s1, F1), (s2, _) = [sigma2(g[g.exp == e], T, mode) for e, T in zip(exps, Ts)]
        if mode == "hyb": out.update(sig_near=s1, sig_next=s2, F_near=F1)
        T1, T2 = Ts; v = np.nan
        if np.isfinite(s1) and np.isfinite(s2) and s1 > 0 and s2 > 0:
            N1, N2, N30, N365 = T1 * 525600, T2 * 525600, 30 * 1440, 365 * 1440
            w = (T1 * s1 * (N2 - N30) / (N2 - N1) + T2 * s2 * (N30 - N1) / (N2 - N1)) * N365 / N30
            v = 100 * np.sqrt(w) if w > 0 else np.nan
        out["close" if mode == "hyb" else "close_" + mode] = v
    return out

if __name__ == "__main__":
    a = sys.argv; st = a[a.index("--start") + 1] if "--start" in a else "2016-01"; en = a[a.index("--end") + 1] if "--end" in a else dt.date.today().strftime("%Y-%m")
    log = {"start": st, "end": en, "months": {}}
    off, links = official(); log["official_links"] = links; log["official_days"] = len(off)
    rows = []; months = pd.period_range(st, en, freq="M"); keep = []
    for pm in months:
        t0 = time.time(); d = month_df(pm.year, pm.month)
        if d is None: log["months"][str(pm)] = "fail"; print(pm, "fail"); continue
        tdays = pd.DatetimeIndex(sorted(d.date.unique()))
        for day, g in d.groupby("date"):
            r = day_vix(g, day, pd.DatetimeIndex(sorted(set(tdays) | set(pd.bdate_range(day, day + pd.Timedelta(days=70))))),
                        keep if day >= off.index.min() - pd.Timedelta(days=1) else None)
            if r: rows.append({"date": day, **r})
        log["months"][str(pm)] = f"{len(tdays)} days, {len(d)} rows, {time.time()-t0:.0f}s"; print(pm, log["months"][str(pm)], flush=True)
        time.sleep(1)
    out = pd.DataFrame(rows).set_index("date").sort_index()
    out = out.join(off, how="outer")
    out.round(6).to_csv(os.path.join(EXT, "twvix_daily.csv"), encoding="utf-8")
    if keep:  # 官方值期間的近月／次近月報價（小檔），供本機調校重建法
        pd.concat(keep).to_csv(os.path.join(EXT, "twvix_txo_quotes_recent.csv.gz"), index=False, compression="gzip")
    log["validation_by_mode"] = {}
    for col in ("close", "close_mid", "close_set"):
        bb = out[[col, "official"]].dropna(); ee = bb[col] - bb.official
        if len(bb): log["validation_by_mode"][col] = {"days": len(bb), "corr": round(float(bb.corr().iloc[0, 1]), 4), "mae": round(float(ee.abs().mean()), 3),
                                                     "within1pt": round(float((ee.abs() < 1).mean()), 3), "coverage": round(len(bb) / out.official.notna().sum(), 3)}
    both = out[["close", "official"]].dropna()
    if len(both):
        e = both.close - both.official
        log["validation"] = {"days": len(both), "corr": round(float(both.corr().iloc[0, 1]), 4), "mean_err": round(float(e.mean()), 3),
                             "mae": round(float(e.abs().mean()), 3), "max_abs_err": round(float(e.abs().max()), 3),
                             "corr_chg": round(float(both.diff().corr().iloc[0, 1]), 4)}
    log["replica_days"] = int(out.close.notna().sum()); log["replica_range"] = [str(out.close.dropna().index.min().date()), str(out.close.dropna().index.max().date())] if log["replica_days"] else None
    log["tail"] = out.tail(8).round(3).reset_index().astype(str).to_dict("records")
    json.dump(log, open(os.path.join(EXT, "fetch_twvix_log.json"), "w", encoding="utf-8"), ensure_ascii=False, indent=1)
    print(json.dumps({k: v for k, v in log.items() if k != "months"}, ensure_ascii=False))
