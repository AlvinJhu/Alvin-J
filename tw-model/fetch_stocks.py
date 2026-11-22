"""
個股資料（多因子、統計套利用）。雲端連不到證交所與 Yahoo，須在本機或 GitHub Actions 執行。
1. 證交所每月最後交易日全市場快照（2015-01 起，含當時上市、後來下市的股票 → 無存活偏誤）
   MI_INDEX（收盤價、成交金額）、BWIBBU_d（本益比、股價淨值比、殖利率）、MI_QFIIS（發行股數）
   → data/ext/stk_monthly.csv
2. 配對交易標的：2015-12 月底市值前 100 大普通股（只用當時資料選），yfinance 還原權值日線
   → data/ext/stk_daily_top100.csv（長表：date, code, close_adj, close, volume）
用法：python fetch_stocks.py [--monthly-only | --daily-only]
"""
import os, sys, json, time, datetime as dt
import pandas as pd, numpy as np, requests

HERE = os.path.dirname(os.path.abspath(__file__))
D = os.path.join(HERE, "data"); EXT = os.path.join(D, "ext"); os.makedirs(EXT, exist_ok=True)
S = requests.Session(); S.headers["User-Agent"] = "Mozilla/5.0"
BASE = "https://www.twse.com.tw/rwd/zh/"
LOG = {"errors": []}

def get(url, pause=3.5):
    for k in range(6):
        try:
            r = S.get(url + f"&_={time.time_ns()}", timeout=30, allow_redirects=False)
            if r.status_code == 200:
                j = r.json(); time.sleep(pause)
                if j.get("stat", "OK") in ("OK", "ok") or j.get("tables") or j.get("data"): return j
                return None  # 該日無資料
        except Exception as e:
            LOG["errors"].append(f"{url[:90]}: {e!r}"[:200])
        time.sleep(60 * (k + 1))
    return None

def num(x):
    try: return float(str(x).replace(",", "").strip())
    except ValueError: return np.nan

def month_ends():
    ds = set()
    for l in open(os.path.join(D, "ohlc.jsonl"), encoding="utf-8"):
        for r in json.loads(l)["j"].get("data", []):
            a, b, c = r[0].split("/"); ds.add(f"{int(a)+1911}{b}{c}")
    s = pd.Series(sorted(ds)); return s.groupby(s.str[:6]).max().tolist()

def table(j, must):
    """回傳 (fields, data)：在 j 或 j['tables'] 中找出欄位含 must 全部字串的表。"""
    cands = [j] + list(j.get("tables") or [])
    for t in cands:
        f = t.get("fields") or []
        if f and all(any(m in x for x in f) for m in must): return f, t.get("data") or []
    return None, []

def col(fields, key):
    for i, f in enumerate(fields):
        if key in f: return i
    return None

def snapshot(d):
    out = {}
    j = get(BASE + f"afterTrading/MI_INDEX?date={d}&type=ALLBUT0999&response=json")
    if j:
        f, data = table(j, ["證券代號", "收盤價"])
        if f:
            ic, ip, iv = col(f, "證券代號"), col(f, "收盤價"), col(f, "成交金額")
            for r in data: out.setdefault(r[ic].strip(), {})["close"] = num(r[ip]); out[r[ic].strip()]["value"] = num(r[iv])
    j = get(BASE + f"afterTrading/BWIBBU_d?date={d}&selectType=ALL&response=json")
    if j:
        f, data = table(j, ["證券代號", "本益比"])
        if f:
            ic, pe, pb, dy = col(f, "證券代號"), col(f, "本益比"), col(f, "淨值比"), col(f, "殖利率")
            for r in data:
                x = out.setdefault(r[ic].strip(), {}); x["pe"] = num(r[pe]); x["pb"] = num(r[pb]); x["dy"] = num(r[dy])
    j = get(BASE + f"fund/MI_QFIIS?date={d}&selectType=ALLBUT0999&response=json")
    if j:
        f, data = table(j, ["證券代號", "發行股數"])
        if f:
            ic, sh = col(f, "證券代號"), col(f, "發行股數")
            for r in data: out.setdefault(r[ic].strip(), {})["shares"] = num(r[sh])
    return pd.DataFrame.from_dict(out, orient="index").rename_axis("code").reset_index().assign(date=pd.Timestamp(d))

def monthly():
    fn = os.path.join(EXT, "stk_monthly.csv")
    old = pd.read_csv(fn, dtype={"code": str}, parse_dates=["date"]) if os.path.exists(fn) else pd.DataFrame()
    have = set(old.date.dt.strftime("%Y%m%d")) if len(old) else set()
    parts = [old] if len(old) else []
    for d in month_ends():
        if d < "20150101" or d in have: continue
        s = snapshot(d); parts.append(s); print(d, len(s), flush=True)
        pd.concat(parts, ignore_index=True).to_csv(fn, index=False)  # 每月存檔，可中斷續抓
    m = pd.concat(parts, ignore_index=True)
    LOG["monthly"] = {"rows": len(m), "months": int(m.date.nunique()), "codes": int(m.code.nunique()),
                      "non_null": {c: int(m[c].notna().sum()) for c in m.columns if c not in ("code", "date")}}
    return m

def daily(m):
    m = m[m.code.str.fullmatch(r"[1-9]\d{3}")]  # 普通股（排除 ETF、權證、特別股）
    base = m[m.date == m[m.date <= "2015-12-31"].date.max()].copy()
    base["cap"] = base.close * base.shares
    top = base.dropna(subset=["cap"]).nlargest(100, "cap").code.tolist()
    LOG["top100_asof"] = str(base.date.iloc[0].date()); LOG["top100"] = top
    import yfinance as yf
    px = yf.download([f"{c}.TW" for c in top], start="2015-01-01", auto_adjust=False, progress=False, group_by="ticker", threads=True)
    rows = []
    for c in top:
        t = f"{c}.TW"
        if t not in px.columns.get_level_values(0): continue
        x = px[t].dropna(subset=["Close"])
        if x.empty: continue
        rows.append(pd.DataFrame({"date": x.index, "code": c, "close_adj": x["Adj Close"].values,
                                  "close": x["Close"].values, "volume": x["Volume"].values}))
    out = pd.concat(rows, ignore_index=True)
    out.to_csv(os.path.join(EXT, "stk_daily_top100.csv"), index=False)
    LOG["daily"] = {"rows": len(out), "codes": int(out.code.nunique()), "missing": sorted(set(top) - set(out.code)),
                    "dates": [str(out.date.min()), str(out.date.max())]}

if __name__ == "__main__":
    if "--daily-only" in sys.argv:
        m = pd.read_csv(os.path.join(EXT, "stk_monthly.csv"), dtype={"code": str}, parse_dates=["date"])
    else:
        m = monthly()
    if "--monthly-only" not in sys.argv:
        try: daily(m)
        except Exception as e: LOG["daily_error"] = repr(e)
    LOG["errors"] = LOG["errors"][:30]
    json.dump(LOG, open(os.path.join(EXT, "fetch_stocks_log.json"), "w", encoding="utf-8"), ensure_ascii=False, indent=2, default=str)
    print(json.dumps({k: v for k, v in LOG.items() if k != "top100"}, ensure_ascii=False, default=str)[:3000])
