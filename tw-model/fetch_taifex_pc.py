"""期交所官方「台指選擇權 Put/Call Ratio」（買賣權成交量比、未平倉量比，%）每日資料，2016 起逐月抓。
來源：https://www.taifex.com.tw/cht/3/pcRatio（下載 pcRatioDown CSV；失敗改解析網頁表格）。輸出 data/ext/taifex_pcratio.csv。"""
import os, io, sys, json, time, datetime as dt
import pandas as pd, requests
HERE = os.path.dirname(os.path.abspath(__file__)); EXT = os.path.join(HERE, "data", "ext"); os.makedirs(EXT, exist_ok=True)
S = requests.Session(); S.headers["User-Agent"] = "Mozilla/5.0"
COLS = ["date", "put_vol", "call_vol", "pc_vol_pct", "put_oi", "call_oi", "pc_oi_pct"]

def clean(d):
    d = d.iloc[:, :7].copy(); d.columns = COLS
    d["date"] = pd.to_datetime(d.date.astype(str).str.strip(), errors="coerce")
    for c in COLS[1:]: d[c] = pd.to_numeric(d[c].astype(str).str.replace(",", "").str.strip(), errors="coerce")
    return d.dropna(subset=["date", "pc_oi_pct"])

def month(a, b, log):
    p = {"queryStartDate": a.strftime("%Y/%m/%d"), "queryEndDate": b.strftime("%Y/%m/%d")}
    for k in range(3):
        try:
            r = S.post("https://www.taifex.com.tw/cht/3/pcRatioDown", data=p, timeout=60)
            if r.ok and b"," in r.content[:500]:
                d = clean(pd.read_csv(io.StringIO(r.content.decode("big5", errors="ignore"))))
                if len(d): return d
            r = S.post("https://www.taifex.com.tw/cht/3/pcRatio", data={**p, "down_type": ""}, timeout=60)
            for t in pd.read_html(io.StringIO(r.text)):
                if t.shape[1] >= 7:
                    d = clean(t)
                    if len(d): return d
            log.append(f"{a.date()}: empty (status {r.status_code})")
        except Exception as e:
            log.append(f"{a.date()}: {e!r}")
        time.sleep(5 * (k + 1))
    return None

def main():
    log, parts = [], []
    a = pd.Timestamp(sys.argv[1] if len(sys.argv) > 1 else "2016-01-01"); today = pd.Timestamp(dt.date.today())
    while a <= today:
        b = min(a + pd.offsets.MonthEnd(0), today); d = month(a, b, log)
        if d is not None: parts.append(d)
        print(a.date(), 0 if d is None else len(d), flush=True); a = b + pd.Timedelta(days=1); time.sleep(1.5)
    out = pd.concat(parts).drop_duplicates("date").sort_values("date") if parts else pd.DataFrame(columns=COLS)
    out.to_csv(os.path.join(EXT, "taifex_pcratio.csv"), index=False)
    info = {"rows": len(out), "dates": [str(out.date.min()), str(out.date.max())] if len(out) else [], "errors": log[:40],
            "tail": out.tail(5).astype(str).to_dict("records")}
    json.dump(info, open(os.path.join(EXT, "fetch_taifex_pc_log.json"), "w", encoding="utf-8"), ensure_ascii=False, indent=2)
    print(json.dumps(info, ensure_ascii=False)[:3000])

if __name__ == "__main__": main()
