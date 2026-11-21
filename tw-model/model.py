"""
台股加權指數 多因子機率模型 v1
- 目標：次日漲跌（y1）、未來 20 交易日漲跌（y20）
- 模型：邏輯迴歸（標準化）、XGBoost；walk-forward 擴張視窗，每 63 交易日重訓，20 日標的加 20 日 embargo
- 部位：依 20 日上漲機率決定持股比例，再乘上波動度目標（年化 18%）縮放
- 輸出：reports/backtest_summary.csv、reports/today.json、reports/equity.png

資料（雲端版）：證交所 OHLC/成交值/三大法人/融資 + GitHub datasets 的 VIX、USD/TWD（FRED H.10）
本機有 yfinance 時加 --yf 會再加入費半 ^SOX、美債 10 年 ^TNX（雲端被擋）
"""
import json, os, sys, glob
import numpy as np, pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
D, R = os.path.join(HERE, "data"), os.path.join(HERE, "reports")
os.makedirs(R, exist_ok=True)

def num(x):
    try: return float(str(x).replace(",", ""))
    except ValueError: return np.nan

def roc(s):
    a, b, c = s.split("/"); return pd.Timestamp(int(a) + 1911, int(b), int(c))

def jl(fn):
    p = os.path.join(D, fn)
    return [json.loads(l) for l in open(p) if l.strip()] if os.path.exists(p) else []

# ---------- 載入 ----------
def merge_finmind(df):
    """FinMind（來源同為證交所）補 2016 起法人與融資；與證交所原始檔重疊日用中位比對齊單位，缺日以證交所補。"""
    ext = os.path.join(D, "ext"); pi, pm = os.path.join(ext, "inst_finmind.csv"), os.path.join(ext, "margin_finmind.csv")
    if os.path.exists(pi):
        a = pd.read_csv(pi, parse_dates=["date"]); a["net"] = (a.buy - a.sell) / 1e8  # 億元
        t = a.pivot_table(index="date", columns="name", values="net", aggfunc="sum")
        fm = pd.DataFrame({"foreign": t.get("Foreign_Investor"), "trust": t.get("Investment_Trust"),
                           "dealer": t[[c for c in t if c.startswith("Dealer")]].sum(axis=1, min_count=1)})
        for k in fm:
            df[k] = fm[k].reindex(df.index).combine_first(df[k]) if k in df else fm[k].reindex(df.index)
    if os.path.exists(pm):
        b = pd.read_csv(pm, parse_dates=["date"])
        b = b[b.name == "MarginPurchaseMoney"].set_index("date").TodayBalance.astype(float)
        if "margin" in df:
            ov = pd.concat([df.margin, b], axis=1, keys=["tw", "fm"]).dropna()
            k = (ov.tw / ov.fm).median() if len(ov) >= 20 else 1e-5
            df["margin"] = (b * k).reindex(df.index).combine_first(df.margin)
        else:
            df["margin"] = (b * 1e-5).reindex(df.index)  # 假設仟元
    return df

def load():
    o = {}
    for r in jl("ohlc.jsonl"):
        for x in r["j"].get("data", []): o[roc(x[0])] = [num(v) for v in x[1:5]]
    df = pd.DataFrame.from_dict(o, orient="index", columns=["open", "high", "low", "close"]).sort_index()
    t = {}
    for r in jl("turnover.jsonl"):
        for x in r["j"].get("data", []): t[roc(x[0])] = num(x[2]) / 1e8  # 億元
    df["turn"] = pd.Series(t)
    inst = {}
    for r in jl("inst.jsonl"):
        rows = {x[0]: num(x[3]) / 1e8 for x in r["j"].get("data", [])}
        if rows:
            k = pd.Timestamp(r["key"])
            inst[k] = {"foreign": rows.get("外資及陸資", rows.get("外資及陸資(不含外資自營商)", np.nan)),
                       "trust": rows.get("投信", np.nan),
                       "dealer": sum(v for n, v in rows.items() if n.startswith("自營商") and "合計" not in n)}
    if inst: df = df.join(pd.DataFrame.from_dict(inst, orient="index"))
    mg = {}
    for r in jl("margin.jsonl"):
        tb = r["j"].get("tables") or []
        for x in (tb[0]["data"] if tb else []):
            if x[0].startswith("融資金額"): mg[pd.Timestamp(r["key"])] = num(x[5]) / 1e5  # 億元
    if mg: df["margin"] = pd.Series(mg)
    df = merge_finmind(df)
    # 美方資料：以「台股日期 t 之前」最後一筆（美股 t-1 收盤，台股 t 收盤時已知）
    ext = os.path.join(D, "ext")
    vix = pd.read_csv(os.path.join(ext, "vix.csv"), parse_dates=["DATE"]).set_index("DATE")["CLOSE"]
    fx = pd.read_csv(os.path.join(ext, "usdtwd.csv"), parse_dates=["Date"]).set_index("Date")["Exchange rate"]
    us = {"vix": vix, "usdtwd": fx}
    for k in ("sox", "tnx"):  # fetch_local.py 存下的 yfinance 資料
        p = os.path.join(ext, f"{k}.csv")
        if os.path.exists(p): us[k] = pd.read_csv(p, parse_dates=["date"]).set_index("date")["close"]
    if "--yf" in sys.argv:
        import yfinance as yf
        for k, tk in {"sox": "^SOX", "tnx": "^TNX"}.items():
            s = yf.download(tk, start="2014-01-01", progress=False)["Close"]
            us[k] = s.squeeze()
    for k, s in us.items():
        s = s.dropna().sort_index(); s.index = s.index + pd.Timedelta(days=1)  # d 日資料最早於 d+1 台股可用
        df[k] = s.reindex(df.index, method="ffill")
    return df.dropna(subset=["close"])

# ---------- 因子 ----------
def features(df):
    c = df.close; f = pd.DataFrame(index=df.index); lr = np.log(c).diff()
    for n in (1, 5, 20, 60): f[f"ret{n}"] = c.pct_change(n)
    for n in (20, 60, 240): f[f"bias{n}"] = c / c.rolling(n).mean() - 1
    f["vol20"] = lr.rolling(20).std() * np.sqrt(252)
    f["vol_ratio"] = f.vol20 / (lr.rolling(120).std() * np.sqrt(252))
    f["dd60"] = c / c.rolling(60).max() - 1
    f["range"] = (df.high - df.low) / c
    g = lr.clip(lower=0).rolling(14).mean(); l = (-lr.clip(upper=0)).rolling(14).mean()
    f["rsi14"] = g / (g + l)
    f["turn_ratio"] = np.log(df.turn / df.turn.rolling(20).mean())
    if "foreign" in df:
        f["fgn_1"] = df.foreign / df.turn
        f["fgn_5"] = df.foreign.rolling(5).sum() / df.turn.rolling(5).sum()
        f["fgn_20"] = df.foreign.rolling(20).sum() / df.turn.rolling(20).sum()
        f["trust_5"] = df.trust.rolling(5).sum() / df.turn.rolling(5).sum()
        f["dealer_5"] = df.dealer.rolling(5).sum() / df.turn.rolling(5).sum()
    if "margin" in df:
        f["margin_5"] = df.margin.pct_change(5)
        f["margin_20"] = df.margin.pct_change(20)
        f["margin_vs_px"] = df.margin.pct_change(20) - c.pct_change(20)
    f["vix"] = df.vix; f["vix_5"] = df.vix.pct_change(5)
    f["fx_5"] = df.usdtwd.pct_change(5); f["fx_20"] = df.usdtwd.pct_change(20)
    if "sox" in df:
        f["sox_1"] = df.sox.pct_change(); f["sox_5"] = df.sox.pct_change(5); f["sox_20"] = df.sox.pct_change(20)
        f["sox_rel20"] = df.sox.pct_change(20) - c.pct_change(20)  # 費半相對台股
    if "tnx" in df:  # 殖利率用變動（bp），不用百分比變動
        f["tnx_1"] = df.tnx.diff(); f["tnx_20"] = df.tnx.diff(20); f["tnx_lvl"] = df.tnx
    # 籌碼延後 1 日：三大法人 15:00、融資約 21:30 才公布，收盤時尚未得知
    for k in [c for c in f if c.split("_")[0] in ("fgn", "trust", "dealer", "margin")]:
        f[k] = f[k].shift(CHIP_LAG)
    return f.replace([np.inf, -np.inf], np.nan)

CHIP_LAG = 1
GROUPS = {
    "量價": ["ret1", "ret5", "ret20", "ret60", "bias20", "bias60", "bias240", "vol20", "vol_ratio", "dd60", "range", "rsi14", "turn_ratio"],
    "VIX匯率": ["vix", "vix_5", "fx_5", "fx_20"],
    "法人": ["fgn_1", "fgn_5", "fgn_20", "trust_5", "dealer_5"],
    "融資": ["margin_5", "margin_20", "margin_vs_px"],
    "費半美債": ["sox_1", "sox_5", "sox_20", "sox_rel20", "tnx_1", "tnx_20", "tnx_lvl"],
}
SETS = {"v1(量價+VIX匯率)": ["量價", "VIX匯率"], "只用籌碼": ["法人", "融資"], "只用費半美債": ["費半美債"],
        "v1+籌碼": ["量價", "VIX匯率", "法人", "融資"], "全部": ["量價", "VIX匯率", "法人", "融資", "費半美債"]}

# ---------- 模型 ----------
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.metrics import roc_auc_score, brier_score_loss
from xgboost import XGBClassifier

def mk(name):
    if name == "logit":
        return make_pipeline(StandardScaler(), LogisticRegression(C=0.1, max_iter=2000))
    return XGBClassifier(n_estimators=300, max_depth=3, learning_rate=0.03, subsample=0.8,
                         colsample_bytree=0.7, min_child_weight=20, reg_lambda=5.0, verbosity=0)

def walk_forward(X, y, h, min_train=500, step=63, models=("logit", "xgb")):
    """擴張視窗；訓練集只用標籤在測試開始前已實現的樣本（embargo = h）。"""
    out = {m: pd.Series(np.nan, index=X.index) for m in models}
    idx = np.arange(len(X)); start = min_train
    while start < len(X):
        end = min(start + step, len(X)); tr = idx[: max(0, start - h)]
        tr = tr[~y.iloc[tr].isna().values]
        for m in out:
            mdl = mk(m).fit(X.iloc[tr], y.iloc[tr].astype(int))
            out[m].iloc[start:end] = mdl.predict_proba(X.iloc[start:end])[:, 1]
        start = end
    return out

def stats(ret):
    ret = ret.dropna(); eq = (1 + ret).cumprod(); yrs = len(ret) / 252
    return {"CAGR": eq.iloc[-1] ** (1 / yrs) - 1, "年化波動": ret.std() * np.sqrt(252),
            "Sharpe": ret.mean() / ret.std() * np.sqrt(252) if ret.std() > 0 else np.nan,
            "最大回撤": (eq / eq.cummax() - 1).min(), "日勝率": (ret[ret != 0] > 0).mean()}

def exposure(p20, vol20, target=0.18):
    base = np.select([p20 >= 0.6, p20 >= 0.5, p20 >= 0.4], [1.0, 0.7, 0.4], 0.1)
    return np.clip(base * np.minimum(1.0, target / vol20), 0, 1)

def block_auc_ci(y, p, block, n=500, seed=0):
    """循環區塊 bootstrap 的 AUC 95% 區間；20 日標籤高度重疊，有效樣本遠少於天數。"""
    y, p = np.asarray(y), np.asarray(p); T = len(y); rng = np.random.default_rng(seed); out = []
    for _ in range(n):
        st = rng.integers(0, T, int(np.ceil(T / block)))
        ix = (st[:, None] + np.arange(block)).ravel()[:T] % T
        if 0 < y[ix].mean() < 1: out.append(roc_auc_score(y[ix], p[ix]))
    return np.percentile(out, [2.5, 97.5])

def score(tgt, name, p, y, block):
    ok = p.notna() & y.notna(); yy, pp = y[ok], p[ok]
    lo, hi = block_auc_ci(yy.values, pp.values, block)
    return {"標的": tgt, "模型": name, "測試天數": int(ok.sum()), "AUC": roc_auc_score(yy, pp),
            "AUC_95%下界": lo, "AUC_95%上界": hi, "命中率": ((pp > .5) == yy).mean(),
            "基準(恆猜漲)": yy.mean(), "Brier": brier_score_loss(yy, pp)}

def backtest(p20, vol20, r):
    w = pd.Series(exposure(p20.values, vol20.values), index=p20.index)
    return w, w * r - w.diff().abs().fillna(w.iloc[0]) * 0.001

def main():
    df = load(); F = features(df)
    fwd1 = df.close.shift(-1) / df.close - 1; fwd20 = df.close.shift(-20) / df.close - 1
    cols = [c for c in F if F[c].notna().mean() > 0.5]
    data = F[cols].dropna()
    print(f"樣本 {data.index[0].date()} ~ {data.index[-1].date()}，{len(data)} 天，因子 {len(cols)} 個")
    y1 = (fwd1 > 0).astype(float).where(fwd1.notna()).reindex(data.index)
    y20 = (fwd20 > 0).astype(float).where(fwd20.notna()).reindex(data.index)
    T = (("次日", y1, 1, 5), ("20日", y20, 20, 20))  # (名稱, 標籤, embargo, bootstrap 區塊)

    # 1) 單因子：每個因子單獨跑 walk-forward 邏輯迴歸
    fa = []
    for c in cols:
        g = next((k for k, v in GROUPS.items() if c in v), "其他")
        row = {"因子": c, "類別": g}
        for tgt, y, h, blk in T:
            p = walk_forward(data[[c]], y, h, models=("logit",))["logit"]; sc = score(tgt, "logit", p, y, blk)
            row[f"{tgt}AUC"], row[f"{tgt}下界"], row[f"{tgt}上界"] = sc["AUC"], sc["AUC_95%下界"], sc["AUC_95%上界"]
        fa.append(row)
    fa = pd.DataFrame(fa).sort_values("20日AUC", ascending=False)
    print(fa.round(3).to_string(index=False)); fa.to_csv(os.path.join(R, "factor_auc.csv"), index=False, encoding="utf-8-sig")

    # 2) 因子組合：同一段樣本比較
    rows, P, seen = [], {}, {}
    for sn, gs in SETS.items():
        sc_ = [c for g in gs for c in GROUPS[g] if c in cols]
        if not sc_ or any(sorted(sc_) == sorted(v) for v in seen.values()): continue
        seen[sn] = sc_
        P[sn] = {}
        for tgt, y, h, blk in T:
            pr = walk_forward(data[sc_], y, h); pr["ens"] = (pr["logit"] + pr["xgb"]) / 2; P[sn][tgt] = pr
            for m, p in pr.items(): rows.append({"因子組": sn, "因子數": len(sc_), **score(tgt, m, p, y, blk)})
    perf = pd.DataFrame(rows); print(perf.round(3).to_string(index=False))
    perf.to_csv(os.path.join(R, "model_accuracy.csv"), index=False, encoding="utf-8-sig")

    # 3) 回測：t 日收盤決定部位，賺 t+1 報酬；成本單邊 0.1%（期貨/ETF 級別）
    main_set = "全部" if "全部" in P else list(P)[-1]
    ens20 = P[main_set]["20日"]["ens"]
    test = ens20.dropna().index; r = fwd1.reindex(test); vol = F.vol20[test]
    bt, ws = {}, {}
    for sn in P:
        ws[sn], st = backtest(P[sn]["20日"]["ens"][test], vol, r); bt[f"模型策略：{sn}"] = {**stats(st), "平均持股": ws[sn].mean()}
    bt["買進持有"] = {**stats(r), "平均持股": 1.0}
    bt["波動目標(無模型)"] = stats(pd.Series(exposure(np.full(len(test), .6), vol.values), index=test) * r)
    bt = pd.DataFrame(bt).T; print(bt.round(3).to_string())
    bt.to_csv(os.path.join(R, "backtest_summary.csv"), encoding="utf-8-sig")
    w, strat = backtest(ens20[test], vol, r)
    data = data[[c for g in SETS[main_set] for c in GROUPS[g] if c in cols]]; cols = list(data.columns)

    # 分組校準：機率高的那組真的比較常漲嗎
    cal = pd.DataFrame({"p": ens20[test], "y": y20.reindex(test), "r20": fwd20.reindex(test)}).dropna()
    cal["組"] = pd.qcut(cal.p, 5, labels=["最低", "低", "中", "高", "最高"])
    calt = cal.groupby("組", observed=True).agg(平均機率=("p", "mean"), 實際上漲率=("y", "mean"), 平均20日報酬=("r20", "mean"), 天數=("y", "size"))
    print(calt.round(3).to_string()); calt.to_csv(os.path.join(R, "calibration_20d.csv"), encoding="utf-8-sig")

    # 今日：用全部已實現標籤重訓
    last = data.index[-1]; Xl = data.loc[[last]]; today = {}
    for tgt, y, h in (("p_up_1d", y1, 1), ("p_up_20d", y20, 20)):
        ok = y.notna(); ps = {m: float(mk(m).fit(data[ok], y[ok].astype(int)).predict_proba(Xl)[0, 1]) for m in ("logit", "xgb")}
        today[tgt] = ps; today[tgt]["ens"] = (ps["logit"] + ps["xgb"]) / 2
    lr = np.log(df.close).diff().dropna()
    ew = np.sqrt((lr ** 2).ewm(alpha=0.06).mean().iloc[-1])  # RiskMetrics λ=0.94
    c = float(df.close.iloc[-1])
    rng = lambda k, n: [round(c * np.exp(-k * ew * np.sqrt(n))), round(c * np.exp(k * ew * np.sqrt(n)))]
    expo = float(exposure(np.array([today["p_up_20d"]["ens"]]), np.array([F.vol20.iloc[-1]]))[0])
    imp = None
    xg = mk("xgb").fit(data[y20.notna()], y20[y20.notna()].astype(int))
    imp = pd.Series(xg.feature_importances_, index=cols).sort_values(ascending=False).head(8).round(3).to_dict()
    today.update({"date": str(last.date()), "close": c, "ewma_daily_vol": float(ew),
                  "range_1d_68": rng(1, 1), "range_1d_95": rng(1.96, 1),
                  "range_20d_68": rng(1, 20), "range_20d_95": rng(1.96, 20),
                  "VaR95_1d_pct": float(-1.645 * ew), "suggest_equity": expo, "suggest_cash": 1 - expo,
                  "factors_today": {k: (None if pd.isna(v) else round(float(v), 4)) for k, v in data.loc[last].items()},
                  "xgb_top_features_20d": imp})
    json.dump(today, open(os.path.join(R, "today.json"), "w"), ensure_ascii=False, indent=2)
    print(json.dumps({k: today[k] for k in ("date", "close", "p_up_1d", "p_up_20d", "range_1d_95", "range_20d_68", "range_20d_95", "suggest_cash")}, ensure_ascii=False))

    try:
        import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
        fig, ax = plt.subplots(2, 1, figsize=(11, 7), sharex=True, gridspec_kw={"height_ratios": [3, 1]})
        ax[0].plot((1 + r).cumprod(), label="Buy & Hold"); ax[0].plot((1 + strat).cumprod(), label="Model")
        ax[0].set_yscale("log"); ax[0].legend(); ax[0].set_title("TAIEX walk-forward backtest")
        ax[1].fill_between(w.index, w.values, step="post", alpha=.5); ax[1].set_ylabel("Equity weight")
        fig.tight_layout(); fig.savefig(os.path.join(R, "equity.png"), dpi=110)
    except Exception as e:
        print("圖檔略過：", e)

if __name__ == "__main__":
    main()
