"""
公開版打包腳本：從正式版 tw-model/ 程式產生 public-release/tw-risk-weekly/model/，
移除私人資訊、加上 Alvin-J 原創授權標頭與 --init 首次抓取，再重新產生 dist/ 兩個 zip。

用法：python public-release/build.py --src <正式版 tw-model 資料夾>
- 只複製程式，不複製 data/、reports/（原始資料不公開）
- 正式版程式改到讓下列替換找不到位置、或殘留私人字串時，直接失敗，不會產出有問題的公開包
- zip 內檔案時間固定，內容沒變時 zip 也不變（避免每次都產生新 commit）
"""
import os, sys, re, shutil, zipfile, py_compile

HERE = os.path.dirname(os.path.abspath(__file__))
PKG_RISK = os.path.join(HERE, "tw-risk-weekly")
PKGS = {"AlvinJ-tw-risk-weekly.zip": "tw-risk-weekly", "AlvinJ-stock-health-check.zip": "stock-health-check"}
FILES = ["risk_model.py", "model.py", "futures_eval.py", "fetch_local.py", "fetch_twse.py", "weekly_chart.py"]
HEADER = "Alvin-J 原創｜CC BY-NC 4.0（非商業使用，須標示出處）\n"
# 公開包裡不可出現的字串（個人排程時間、分支、路徑、私人註解）
FORBIDDEN = ["20:43", "19:17", "claude/project-thread", "/mnt/", "使用者決定", "sunli", "icloud", "OneDrive"]


class BuildError(RuntimeError):
    pass


def sub(s, old, new, fn, regex=False):
    """替換一次；找不到就失敗（代表正式版改了，需要人工更新本腳本）。"""
    if regex:
        s2, n = re.subn(old, new, s, count=1, flags=re.S)
    else:
        n = s.count(old); s2 = s.replace(old, new, 1)
    if n == 0:
        raise BuildError(f"{fn}：找不到要替換的段落：{old[:60]!r}")
    return s2


def add_header(s, fn):
    if not s.startswith('"""\n'):
        raise BuildError(f"{fn}：檔頭不是 docstring")
    return '"""\n' + HEADER + s[4:]


FETCH_DOC = '''"""
Alvin-J 原創｜CC BY-NC 4.0（非商業使用，須標示出處）
資料抓取（本套件不附原始資料，請自行抓取；各資料來源的使用條款由使用者自行遵守）
1. 加權指數 OHLC、成交金額：證交所（fetch_twse.py）
2. 三大法人買賣超、融資餘額：FinMind 開放 API（資料源為證交所）
3. 台指期、台指選擇權法人未平倉：FinMind（資料源為期交所）
4. VIX、美元台幣、費半 ^SOX、美債 10 年 ^TNX：yfinance
輸出：data/ohlc.jsonl、turnover.jsonl、data/ext/*.csv
用法：
  python fetch_local.py --init     第一次使用：抓全部歷史資料（約 20–40 分鐘）
  python fetch_local.py --daily    之後每個交易日收盤後：只補最新資料
  可設環境變數 FINMIND_TOKEN（FinMind 免費註冊取得），未設也可用但流量較低
"""'''

INIT_FN = '''def init_all(tok=None):
    """--init：第一次使用，從零抓齊 risk_model.py 需要的全部資料。"""
    import subprocess, yfinance as yf
    print("1/4 證交所加權指數、成交金額（2015 起，逐月）…", flush=True)
    subprocess.run([sys.executable, os.path.join(HERE, "fetch_twse.py"), "monthly", "2015-01"], check=False)
    print("2/4 FinMind 三大法人、融資；yfinance 費半、美債…", flush=True)
    main()
    print("3/4 FinMind 台指期、選擇權法人未平倉（逐年）…", flush=True)
    futures(tok)
    print("4/4 yfinance VIX、美元台幣…", flush=True)
    for tk, fn, f in (("^VIX", "vix.csv", lambda d: pd.DataFrame({"DATE": d.index.strftime("%Y-%m-%d"), "OPEN": d.Open.round(4),
                                                                   "HIGH": d.High.round(4), "LOW": d.Low.round(4), "CLOSE": d.Close.round(4)})),
                      ("TWD=X", "usdtwd.csv", lambda d: pd.DataFrame({"Date": d.index.strftime("%Y-%m-%d"), "Country": "Taiwan",
                                                                      "Exchange rate": d.Close.round(4)}))):
        d = yf.download(tk, start="2014-01-01", progress=False, auto_adjust=False)
        d.columns = [c[0] if isinstance(c, tuple) else c for c in d.columns]
        f(d.dropna(subset=["Close"])).to_csv(os.path.join(EXT, fn), index=False)
    need = ["ohlc.jsonl", "turnover.jsonl", "ext/vix.csv", "ext/usdtwd.csv", "ext/inst_finmind.csv", "ext/margin_finmind.csv",
            "ext/sox.csv", "ext/fut_tx_daily.csv", "ext/fut_tx_inst.csv", "ext/opt_txo_inst.csv"]
    miss = [n for n in need if not os.path.exists(os.path.join(D, n))]
    print("完成。" if not miss else f"以下檔案沒抓到，請重跑或檢查網路：{miss}")

if __name__ == "__main__":
    if "--init" in sys.argv:
        init_all(os.environ.get("FINMIND_TOKEN"))
    elif "--daily" in sys.argv:'''


def transform(fn, s):
    if fn == "risk_model.py":
        s = add_header(s, fn)
        s = sub(s, r"# 2026-11-22 使用者決定：[^\n]*", "# v3.2 預設：大跌警訊減碼到 1/2（v3.1 為 1/4）；--grid 改用網格選出值", fn, regex=True)
        s = s.replace("# 通知判斷（每日排程用）", "# 通知判斷")
    elif fn == "model.py":
        s = add_header(s, fn)
        s = sub(s, r"資料（雲端版）：[^\n]*\n[^\n]*雲端被擋）",
                "資料：證交所 OHLC/成交值、FinMind 三大法人/融資、yfinance VIX、USD/TWD、費半 ^SOX、美債 10 年 ^TNX\n"
                "（第一次使用請先跑 python fetch_local.py --init）", fn, regex=True)
    elif fn == "fetch_twse.py":
        s = add_header(s, fn)
        s = sub(s, r"抓證交所歷史資料（[^）\n]*）", "抓證交所歷史資料（請遵守證交所網站使用規範，勿調高抓取頻率）", fn, regex=True)
    elif fn == "fetch_local.py":
        if not s.startswith('"""'):
            raise BuildError(f"{fn}：檔頭不是 docstring")
        s = FETCH_DOC + s[s.index('"""', 3) + 3:]
        s = sub(s, '    """與證交所原始 jsonl 比對外資買賣超、融資金額。"""\n',
                '    """與證交所原始 jsonl 比對外資買賣超、融資金額（沒有證交所逐日檔就略過）。"""\n'
                '    if not all(os.path.exists(os.path.join(D, f)) for f in ("inst.jsonl", "margin.jsonl")): return "略過（無證交所逐日檔）"\n', fn)
        s = sub(s, 'if __name__ == "__main__":\n    if "--daily" in sys.argv:', INIT_FN, fn)
    elif fn == "futures_eval.py":
        s = add_header(s, fn)
    elif fn == "weekly_chart.py":
        s = add_header(s, fn)
        s = re.sub(r"每天 \d{1,2}:\d{2} 的預警。", "每個交易日收盤後的模型讀數。", s)
        s = sub(s, '投資有風險，請自行判斷並承擔損益。")', '投資有風險，請自行判斷並承擔損益。模型：Alvin-J 原創（CC BY-NC 4.0）。")', fn)
    return s


def check_private(root):
    bad = []
    for dp, _, fs in os.walk(root):
        for f in fs:
            p = os.path.join(dp, f)
            try:
                t = open(p, encoding="utf-8").read()
            except (UnicodeDecodeError, OSError):
                continue
            bad += [f"{os.path.relpath(p, HERE)}：{w}" for w in FORBIDDEN if w in t]
    if bad:
        raise BuildError("公開包含私人字串：" + "；".join(bad))


def build_model(src):
    out = os.path.join(PKG_RISK, "model"); os.makedirs(out, exist_ok=True)
    for fn in FILES:
        s = open(os.path.join(src, fn), encoding="utf-8").read()
        p = os.path.join(out, fn)
        open(p, "w", encoding="utf-8", newline="\n").write(transform(fn, s))
        py_compile.compile(p, doraise=True)
    shutil.copyfile(os.path.join(src, "requirements.txt"), os.path.join(PKG_RISK, "requirements.txt"))
    shutil.rmtree(os.path.join(out, "__pycache__"), ignore_errors=True)


def build_zips():
    dist = os.path.join(HERE, "dist"); os.makedirs(dist, exist_ok=True)
    for zname, pkg in PKGS.items():
        root = os.path.join(HERE, pkg); entries = []
        for dp, dns, fs in os.walk(root):
            dns[:] = sorted(d for d in dns if d not in ("__pycache__", "data", "reports"))
            entries += [os.path.join(dp, f) for f in fs if not f.endswith(".pyc")]
        with zipfile.ZipFile(os.path.join(dist, zname), "w", zipfile.ZIP_DEFLATED) as z:
            for p in sorted(entries):
                zi = zipfile.ZipInfo(os.path.join(pkg, os.path.relpath(p, root)).replace(os.sep, "/"), (2026, 1, 1, 0, 0, 0))
                zi.compress_type = zipfile.ZIP_DEFLATED; zi.external_attr = 0o644 << 16
                z.writestr(zi, open(p, "rb").read())
        print("寫入", os.path.relpath(os.path.join(dist, zname), HERE), len(entries), "個檔案")


if __name__ == "__main__":
    src = sys.argv[sys.argv.index("--src") + 1] if "--src" in sys.argv else os.path.join(HERE, "..", "tw-model")
    try:
        build_model(src)
        check_private(os.path.join(HERE, "tw-risk-weekly")); check_private(os.path.join(HERE, "stock-health-check"))
        build_zips()
    except BuildError as e:
        print("打包失敗：", e); sys.exit(1)
