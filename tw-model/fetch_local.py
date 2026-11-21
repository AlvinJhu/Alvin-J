"""
本機抓取（雲端對證交所、Yahoo、FinMind 全部被擋）
1. 三大法人買賣超、融資餘額：FinMind 開放 API（資料源為證交所），2 次請求抓完 2016 起全部
   失敗時改用 fetch_twse.py daily（證交所逐日，約 6 小時）
2. 費半 ^SOX、美債 10 年 ^TNX：yfinance
3. 與 data/inst.jsonl、margin.jsonl（證交所原始檔，近 375 天）逐日比對，確認兩個來源一致
輸出：data/ext/inst_finmind.csv、margin_finmind.csv、sox.csv、tnx.csv
用法：python fetch_local.py [--token FinMind_token]；python fetch_local.py --futures（期貨、選擇權未平倉）
"""
import os, sys, json, time, datetime as dt
import pandas as pd, numpy as np, requests

HERE = os.path.dirname(os.path.abspath(__file__))
D = os.path.join(HERE, "data"); EXT = os.path.join(D, "ext"); os.makedirs(EXT, exist_ok=True)
START = "2016-01-01"; TODAY = dt.date.today().isoformat()
API = "https://api.finmindtrade.com/api/v4/data"

def finmind(dataset, token=None):
    p = {"dataset": dataset, "start_date": START, "end_date": TODAY}
    h = {"Authorization": f"Bearer {token}"} if token else {}
    r = requests.get(API, params=p, headers=h, timeout=120); r.raise_for_status()
    j = r.json()
    if j.get("status") != 200 or not j.get("data"):
        raise RuntimeError(f"{dataset}: {j.get('msg')}")
    return pd.DataFrame(j["data"])

def num(x):
    try: return float(str(x).replace(",", ""))
    except ValueError: return np.nan

def check_against_twse(inst, mg):
    """與證交所原始 jsonl 比對外資買賣超、融資金額。"""
    fgn, mgt = {}, {}
    for l in open(os.path.join(D, "inst.jsonl"), encoding="utf-8"):
        r = json.loads(l); rows = {x[0]: num(x[3]) for x in r["j"].get("data", [])}
        v = rows.get("外資及陸資(不含外資自營商)", rows.get("外資及陸資"))
        if v is not None: fgn[pd.Timestamp(r["key"])] = v
    for l in open(os.path.join(D, "margin.jsonl"), encoding="utf-8"):
        r = json.loads(l); tb = r["j"].get("tables") or []
        for x in (tb[0]["data"] if tb else []):
            if x[0].startswith("融資金額"): mgt[pd.Timestamp(r["key"])] = num(x[5]) * 1000  # 仟元 -> 元
    a = inst[inst.name == "Foreign_Investor"].set_index("date")
    fm = (a.buy - a.sell); fm.index = pd.to_datetime(fm.index)
    b = mg[mg.name == "MarginPurchaseMoney"].set_index("date").TodayBalance; b.index = pd.to_datetime(b.index)
    out = {}
    for k, tw, f in (("外資買賣超", pd.Series(fgn), fm), ("融資金額", pd.Series(mgt), b)):
        j = pd.concat([tw, f], axis=1, keys=["twse", "finmind"]).dropna()
        if len(j):
            ratio = (j.finmind / j.twse).replace([np.inf, -np.inf], np.nan).median()
            out[k] = {"重疊天數": len(j), "FinMind/證交所 中位比": round(float(ratio), 6),
                      "完全一致比例": round(float(np.isclose(j.finmind, j.twse * ratio, rtol=1e-4).mean()), 4)}
    return out

FUT = {  # 期交所資料（FinMind 轉載），逐年抓避免單次過大
    "fut_tx_daily": ("TaiwanFuturesDaily", "TX"),                       # 台指期各月契約 OHLC、結算價、未平倉
    "fut_tx_inst": ("TaiwanFuturesInstitutionalInvestors", "TX"),       # 三大法人台指期多空未平倉
    "opt_txo_inst": ("TaiwanOptionInstitutionalInvestors", "TXO"),     # 三大法人台指選擇權買賣權未平倉
}

def futures(tok=None):
    """--futures：抓期貨、選擇權未平倉，存 data/ext/<name>.csv，並記錄欄位供建因子。"""
    log = {}
    for name, (ds, did) in FUT.items():
        parts = []
        for y in range(2016, dt.date.today().year + 1):
            p = {"dataset": ds, "data_id": did, "start_date": f"{y}-01-01", "end_date": min(f"{y}-12-31", TODAY)}
            h = {"Authorization": f"Bearer {tok}"} if tok else {}
            for k in range(3):
                try:
                    j = requests.get(API, params=p, headers=h, timeout=120).json()
                    if j.get("status") == 200: parts.append(pd.DataFrame(j["data"])); break
                    log.setdefault(name + "_msg", []).append(f"{y}: {j.get('msg')}")
                except Exception as e:
                    log.setdefault(name + "_msg", []).append(f"{y}: {e!r}")
                time.sleep(30)
            time.sleep(2)
        if parts:
            d = pd.concat(parts, ignore_index=True); d.to_csv(os.path.join(EXT, f"{name}.csv"), index=False)
            log[name] = {"rows": len(d), "dates": [str(d.date.min()), str(d.date.max()), int(d.date.nunique())],
                         "columns": list(d.columns), "sample": d.tail(3).to_dict("records")}
    json.dump(log, open(os.path.join(EXT, "fetch_futures_log.json"), "w", encoding="utf-8"), ensure_ascii=False, indent=2, default=str)
    print(json.dumps(log, ensure_ascii=False, indent=2, default=str)[:5000])

def main():
    tok = sys.argv[sys.argv.index("--token") + 1] if "--token" in sys.argv else os.environ.get("FINMIND_TOKEN")
    log = {}
    try:
        inst = finmind("TaiwanStockTotalInstitutionalInvestors", tok)
        mg = finmind("TaiwanStockTotalMarginPurchaseShortSale", tok)
        inst.to_csv(os.path.join(EXT, "inst_finmind.csv"), index=False)
        mg.to_csv(os.path.join(EXT, "margin_finmind.csv"), index=False)
        log["finmind"] = {"inst": [inst.date.min(), inst.date.max(), int(inst.date.nunique())],
                          "margin": [mg.date.min(), mg.date.max(), int(mg.date.nunique())],
                          "inst_names": sorted(inst.name.unique().tolist()),
                          "margin_names": sorted(mg.name.unique().tolist())}
        log["check_vs_twse"] = check_against_twse(inst, mg)
    except Exception as e:
        log["finmind_error"] = repr(e)
        print("FinMind 失敗，請改跑：python fetch_twse.py daily 20160101（約 6 小時，可中斷續抓）")
    try:
        import yfinance as yf
        for k, tk in {"sox": "^SOX", "tnx": "^TNX"}.items():
            s = yf.download(tk, start="2014-01-01", progress=False, auto_adjust=False)["Close"].squeeze().dropna()
            s.rename("close").to_csv(os.path.join(EXT, f"{k}.csv"), index_label="date")
            log[k] = [str(s.index.min().date()), str(s.index.max().date()), len(s)]
    except Exception as e:
        log["yfinance_error"] = repr(e)
    json.dump(log, open(os.path.join(EXT, "fetch_log.json"), "w", encoding="utf-8"), ensure_ascii=False, indent=2, default=str)
    print(json.dumps(log, ensure_ascii=False, indent=2, default=str))

if __name__ == "__main__":
    if "--futures" in sys.argv:
        futures(sys.argv[sys.argv.index("--token") + 1] if "--token" in sys.argv else os.environ.get("FINMIND_TOKEN"))
    else:
        main()
