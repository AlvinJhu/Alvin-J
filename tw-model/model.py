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
    # 美方資料：以「台股日期 t 之前」最後一筆（美股 t-1 收盤，台股 t 收盤時已知）
    ext = os.path.join(D, "ext")
    vix = pd.read_csv(os.path.join(ext, "vix.csv"), parse_dates=["DATE"]).set_index("DATE")["CLOSE"]
    fx = pd.read_csv(os.path.join(ext, "usdtwd.csv"), parse_dates=["Date"]).set_index("Date")["Exchange rate"]
    us = {"vix": vix, "usdtwd": fx}
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
    for k in ("sox", "tnx"):
        if k in df: f[f"{k}_1"] = df[k].pct_change(); f[f"{k}_20"] = df[k].pct_change(20)
    return f.replace([np.inf, -np.inf], np.nan)

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

def walk_forward(X, y, h, min_train=500, step=63):
    """擴張視窗；訓練集只用標籤在測試開始前已實現的樣本（embargo = h）。"""
    out = {m: pd.Series(np.nan, index=X.index) for m in ("logit", "xgb")}
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

def main():
    df = load(); F = features(df)
    fwd1 = df.close.shift(-1) / df.close - 1; fwd20 = df.close.shift(-20) / df.close - 1
    cols = [c for c in F if F[c].notna().mean() > 0.5]
    data = F[cols].dropna()
    print(f"樣本 {data.index[0].date()} ~ {data.index[-1].date()}，{len(data)} 天，因子 {len(cols)} 個")
    y1 = (fwd1 > 0).astype(float).where(fwd1.notna()).reindex(data.index)
    y20 = (fwd20 > 0).astype(float).where(fwd20.notna()).reindex(data.index)
    p1, p20 = walk_forward(data, y1, 1), walk_forward(data, y20, 20)

    rows = []
    for tgt, P, y in (("次日", p1, y1), ("20日", p20, y20)):
        for m, p in P.items():
            ok = p.notna() & y.notna()
            rows.append({"標的": tgt, "模型": m, "測試天數": int(ok.sum()),
                         "AUC": roc_auc_score(y[ok], p[ok]), "命中率": ((p[ok] > .5) == y[ok]).mean(),
                         "基準(恆猜漲)": y[ok].mean(), "Brier": brier_score_loss(y[ok], p[ok])})
    perf = pd.DataFrame(rows); print(perf.round(3).to_string(index=False))

    # 回測：t 日收盤決定部位，賺 t+1 報酬；成本單邊 0.1%（期貨/ETF 級別）
    ens20 = (p20["logit"] + p20["xgb"]) / 2; ens1 = (p1["logit"] + p1["xgb"]) / 2
    test = ens20.dropna().index; r = fwd1.reindex(test)
    w = pd.Series(exposure(ens20[test].values, F.vol20[test].values), index=test)
    strat = w * r - w.diff().abs().fillna(w.iloc[0]) * 0.001
    bt = pd.DataFrame({"模型策略": stats(strat), "買進持有": stats(r),
                       "波動目標(無模型)": stats(pd.Series(exposure(np.full(len(test), .6), F.vol20[test].values), index=test) * r)}).T
    bt["平均持股"] = [w.mean(), 1.0, np.nan]
    print(bt.round(3).to_string())
    perf.to_csv(os.path.join(R, "model_accuracy.csv"), index=False, encoding="utf-8-sig")
    bt.to_csv(os.path.join(R, "backtest_summary.csv"), encoding="utf-8-sig")

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
