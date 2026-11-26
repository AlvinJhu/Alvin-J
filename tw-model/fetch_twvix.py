"""
一次性抓取：期交所「臺指選擇權波動率指數」（TAIWAN VIX）官方歷史分鐘資料 → 日資料
來源：https://www.taifex.com.tw/cht/7/vixMinNew（每月一個 txt；雲端被擋，由一次性 GitHub Actions 執行）
輸出：data/ext/twvix_daily.csv（date, open, high, low, close, n_obs；close = 日盤 13:45 前最後一筆）
      data/ext/twvix_raw.tar.gz（原始檔，供查核）、data/ext/fetch_twvix_log.json
"""
import os, re, io, json, time, tarfile, datetime as dt
import requests, pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__)); EXT = os.path.join(HERE, "data", "ext")
BASE = "https://www.taifex.com.tw"
PAGES = ["/cht/7/vixMinNew", "/cht/7/vixMin", "/cht/7/vixDaily"]
UA = {"User-Agent": "Mozilla/5.0 (research data download)"}
S = requests.Session(); S.headers.update(UA)

def get(url, tries=4):
    for k in range(tries):
        try:
            r = S.get(url, timeout=60)
            if r.status_code == 200: return r
            if r.status_code == 404: return r
        except requests.RequestException as e:
            print("retry", url, e)
        time.sleep(2 * (k + 1))
    return None

log = {"pages": {}, "files": {}}
links = set()
for p in PAGES:
    r = get(BASE + p); log["pages"][p] = getattr(r, "status_code", None)
    if r is None or r.status_code != 200: continue
    for h in re.findall(r'''(?:href|src)\s*=\s*["']([^"']+)["']''', r.text):
        if re.search(r"\.(txt|zip|csv)(\?|$)", h, re.I) and re.search(r"vix", h, re.I):
            links.add(h if h.startswith("http") else BASE + ("" if h.startswith("/") else "/") + h)
log["links_from_pages"] = sorted(links)
# 已知命名規則補洞：2016-01 起每月
m = dt.date(2016, 1, 1); today = dt.date.today()
while m <= today:
    links.add(f"{BASE}/file/taifex/Dailydownload/vix/log2data/{m:%Y%m}new.txt"); m = (m.replace(day=28) + dt.timedelta(days=4)).replace(day=1)

raw = {}
for u in sorted(links):
    r = get(u); log["files"][u] = getattr(r, "status_code", None)
    if r is not None and r.status_code == 200 and len(r.content) > 200: raw[u] = r.content
    time.sleep(0.5)

def texts(name, b):
    if name.lower().endswith(".zip") or b[:2] == b"PK":
        import zipfile
        with zipfile.ZipFile(io.BytesIO(b)) as z:
            for n in z.namelist(): yield from texts(n, z.read(n))
        return
    for enc in ("utf-8", "cp950", "big5"):
        try: yield b.decode(enc); return
        except UnicodeDecodeError: pass
    yield b.decode("cp950", errors="ignore")

pat = re.compile(r"(\d{4}/?\d{2}/?\d{2})[\s,]+(\d{1,2}:?\d{2}:?\d{2}\d*)[\s,]+([\d.]+)")
rows = []
for u, b in raw.items():
    for t in texts(u, b):
        for line in t.splitlines():
            mm = pat.search(line)
            if not mm: continue
            d = mm.group(1).replace("/", ""); tm = re.sub(r"\D", "", mm.group(2))[:6].ljust(6, "0")
            rows.append((d, tm, float(mm.group(3))))
log["n_rows_parsed"] = len(rows)
if raw:
    with tarfile.open(os.path.join(EXT, "twvix_raw.tar.gz"), "w:gz") as tf:
        for u, b in raw.items():
            ti = tarfile.TarInfo(u.rsplit("/", 1)[-1]); ti.size = len(b); tf.addfile(ti, io.BytesIO(b))
    log["sample"] = {u.rsplit("/", 1)[-1]: next(texts(u, b))[:600] for u, b in list(raw.items())[:3]}
if rows:
    d = pd.DataFrame(rows, columns=["date", "time", "v"]).drop_duplicates()
    d = d[(d.v > 1) & (d.v < 200)]
    d["date"] = pd.to_datetime(d.date, format="%Y%m%d", errors="coerce"); d = d.dropna().sort_values(["date", "time"])
    day = d[(d.time >= "084500") & (d.time <= "134500")]
    g = day.groupby("date").v
    out = pd.DataFrame({"open": g.first(), "high": g.max(), "low": g.min(), "close": g.last(), "n_obs": g.size()})
    out.round(4).to_csv(os.path.join(EXT, "twvix_daily.csv"), encoding="utf-8")
    log["dates"] = [str(out.index.min().date()), str(out.index.max().date())]; log["days"] = len(out)
    log["tail"] = out.tail(5).reset_index().astype(str).to_dict("records")
json.dump(log, open(os.path.join(EXT, "fetch_twvix_log.json"), "w", encoding="utf-8"), ensure_ascii=False, indent=1)
print(json.dumps({k: v for k, v in log.items() if k != "files"}, ensure_ascii=False)[:3000])
print("files ok:", sum(1 for v in log["files"].values() if v == 200), "/", len(log["files"]))
