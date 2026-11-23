"""
Alvin-J 原創｜CC BY-NC 4.0（非商業使用，須標示出處）
抓證交所歷史資料（請遵守證交所網站使用規範，勿調高抓取頻率）
- 月檔：MI_5MINS_HIST（加權 OHLC）、FMTQIK（成交金額）
- 日檔：BFI82U（三大法人買賣超）、MI_MARGN（融資餘額）
用法：python fetch_twse.py monthly 2015-01 | python fetch_twse.py daily 20160101（抓到哪天為止）
結果累加寫入 data/*.jsonl，可中斷續抓。
"""
import json, os, sys, time, datetime as dt
import requests

D = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data")
S = requests.Session(); S.headers["User-Agent"] = "Mozilla/5.0"
BASE = "https://www.twse.com.tw/rwd/zh/"

def get(url, pause=2.0):
    for k in range(8):
        try:
            # CDN 會快取 307 挑戰頁，加隨機參數避開快取
            r = S.get(url + f"&_={time.time_ns()}", timeout=20, allow_redirects=False)
            if r.status_code == 200:
                j = r.json(); time.sleep(pause); return j
        except Exception:
            pass
        time.sleep(120 * (k + 1))
    return None

def done(fn):
    p = os.path.join(D, fn)
    if not os.path.exists(p): return set()
    return {json.loads(l)["key"] for l in open(p) if l.strip()}

def append(fn, key, j):
    with open(os.path.join(D, fn), "a") as f:
        f.write(json.dumps({"key": key, "j": j}, ensure_ascii=False) + "\n")

def monthly(start):
    y, m = map(int, start.split("-")); today = dt.date.today()
    for fn, ep in [("ohlc.jsonl", "TAIEX/MI_5MINS_HIST?date={}&response=json"),
                   ("turnover.jsonl", "afterTrading/FMTQIK?date={}&response=json")]:
        have = done(fn); yy, mm = y, m
        while (yy, mm) <= (today.year, today.month):
            key = f"{yy}{mm:02d}01"
            if key not in have or (yy, mm) == (today.year, today.month):
                j = get(BASE + ep.format(key))
                if j: append(fn, key, j)
            mm += 1
            if mm == 13: yy, mm = yy + 1, 1

def trading_days():
    ds = set()
    for l in open(os.path.join(D, "ohlc.jsonl")):
        for r in json.loads(l)["j"].get("data", []):
            a, b, c = r[0].split("/"); ds.add(f"{int(a)+1911}{b}{c}")
    return sorted(ds, reverse=True)  # 由近到遠

DAILY = {"inst": ("inst.jsonl", "fund/BFI82U?type=day&dayDate={}&response=json"),
         "margin": ("margin.jsonl", "marginTrading/MI_MARGN?date={}&selectType=MS&response=json")}

def daily(since="20160101", pause=4.0):
    """兩個端點交錯抓，由近到遠；證交所約每 3 秒 1 次才不會被 307 擋。"""
    have = {w: done(fn) for w, (fn, _) in DAILY.items()}
    for n, d in enumerate(x for x in trading_days() if x >= since):
        for w, (fn, ep) in DAILY.items():
            if d not in have[w]:
                j = get(BASE + ep.format(d), pause)
                if j: append(fn, d, j)
        if n % 50 == 0: print(d, flush=True)

if __name__ == "__main__":
    os.makedirs(D, exist_ok=True)
    if sys.argv[1] == "monthly": monthly(sys.argv[2] if len(sys.argv) > 2 else "2015-01")
    else: daily(*(sys.argv[2:3] or ["20160101"]))
