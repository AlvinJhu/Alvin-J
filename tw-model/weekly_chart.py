"""
週報圖：reports/risk_weekly.png
讀 reports/risk_history.csv（近 120 日樣本外 walk-forward 值）＋ risk_today.json（最新一天，全樣本重訓值）
四格同一 x 軸：加權指數（警訊期間加底色）、大跌機率（25% 警戒線、基準率）、預測波動、建議持股；標題列出下週 5 日區間。
用法：python weekly_chart.py [--days 90]
"""
import os, sys, json
import numpy as np, pandas as pd
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import font_manager as fm

HERE = os.path.dirname(os.path.abspath(__file__)); R = os.path.join(HERE, "reports")
DAYS = int(sys.argv[sys.argv.index("--days") + 1]) if "--days" in sys.argv else 90
for f in ("Noto Sans CJK TC", "Noto Sans CJK JP", "Noto Sans TC", "Microsoft JhengHei", "WenQuanYi Zen Hei", "PingFang TC"):
    if any(f == x.name for x in fm.fontManager.ttflist):
        plt.rcParams["font.family"] = f; break
plt.rcParams["axes.unicode_minus"] = False

INK, INK2, GRID, SURF = "#0b0b0b", "#52514e", "#e4e3df", "#fcfcfb"
BLUE, ORANGE, RED = "#2a78d6", "#eb6834", "#e34948"

def main():
    t = json.load(open(os.path.join(R, "risk_today.json"), encoding="utf-8"))
    h = pd.read_csv(os.path.join(R, "risk_history.csv"), parse_dates=["date"]).set_index("date")
    d = pd.Timestamp(t["date"])
    h.loc[d, ["close", "p_drawdown", "pred_vol", "warning", "suggest_equity"]] = [
        t["close"], t["p_drawdown_7pct_within_20d"], t["pred_vol_20d_ann"], t["warning_active"], t["suggest_equity"]]
    h = h.sort_index().tail(DAYS); h["warning"] = h.warning.astype(str).str.lower().isin(["true", "1", "1.0"])
    c, v = t["close"], t["pred_vol_20d_ann"]; s5 = v * np.sqrt(5 / 252)
    r68 = [round(c * np.exp(-s5)), round(c * np.exp(s5))]; r95 = [round(c * np.exp(-1.96 * s5)), round(c * np.exp(1.96 * s5))]
    thr = t["params"]["thr"]

    fig, ax = plt.subplots(4, 1, figsize=(10, 10.5), sharex=True, gridspec_kw={"height_ratios": [2.2, 1.4, 1.1, 1.1]})
    fig.patch.set_facecolor(SURF)
    for a in ax:
        a.set_facecolor(SURF); a.grid(axis="y", color=GRID, lw=.8); a.tick_params(colors=INK2, labelsize=9)
        for sp in ("top", "right"): a.spines[sp].set_visible(False)
        for sp in ("left", "bottom"): a.spines[sp].set_color(GRID)
        # 警訊期間底色
        w = h.warning.values; idx = h.index
        i = 0
        while i < len(w):
            if w[i]:
                j = i
                while j + 1 < len(w) and w[j + 1]: j += 1
                a.axvspan(idx[i], idx[min(j + 1, len(idx) - 1)], color=RED, alpha=.08, lw=0)
                i = j + 1
            else: i += 1

    ax[0].plot(h.index, h.close, color=BLUE, lw=2)
    ax[0].scatter([d], [c], color=BLUE, s=40, zorder=3, edgecolor=SURF, linewidth=2)
    ax[0].annotate(f"{c:,.0f}", (d, c), xytext=(6, -4), textcoords="offset points", color=INK, fontsize=9)
    nxt = pd.bdate_range(d + pd.Timedelta(days=1), periods=5)[-1]
    ax[0].fill_between([d, nxt], [c, r95[0]], [c, r95[1]], color=BLUE, alpha=.08, lw=0)
    ax[0].fill_between([d, nxt], [c, r68[0]], [c, r68[1]], color=BLUE, alpha=.18, lw=0)
    for y, lab in ((r68[0], "68%"), (r68[1], "68%"), (r95[0], "95%"), (r95[1], "95%")):
        ax[0].annotate(f"{y:,} ({lab})", (nxt, y), xytext=(4, 0), textcoords="offset points", fontsize=8, color=INK2, va="center")
    ax[0].set_title("加權指數收盤（紅底＝大跌警訊期間；右側扇形＝下週 5 日區間）", loc="left", fontsize=10, color=INK2)
    ax[0].yaxis.set_major_formatter(matplotlib.ticker.FuncFormatter(lambda x, _: f"{x:,.0f}"))

    ax[1].plot(h.index, h.p_drawdown * 100, color=ORANGE, lw=2)
    ax[1].axhline(thr * 100, color=RED, ls="--", lw=1.2); ax[1].text(1.0, thr * 100, f" 警戒線 {thr:.0%}", color=INK2, fontsize=8, va="center", transform=ax[1].get_yaxis_transform())
    ax[1].axhline(t["base_rate"] * 100, color=INK2, ls=":", lw=1); ax[1].text(1.0, t["base_rate"] * 100, f" 歷史平均 {t['base_rate']:.0%}", color=INK2, fontsize=8, va="center", transform=ax[1].get_yaxis_transform())
    ax[1].annotate(f"{t['p_drawdown_7pct_within_20d']:.1%}", (d, t["p_drawdown_7pct_within_20d"] * 100), xytext=(6, 0), textcoords="offset points", fontsize=9, color=INK)
    ax[1].set_ylim(0, max(50, h.p_drawdown.max() * 110)); ax[1].set_title("未來 20 日跌超過 7% 的機率（%）", loc="left", fontsize=10, color=INK2)

    ax[2].plot(h.index, h.pred_vol * 100, color=BLUE, lw=2)
    ax[2].annotate(f"{v:.0%}", (d, v * 100), xytext=(6, 0), textcoords="offset points", fontsize=9, color=INK)
    ax[2].set_title(f"預測年化波動（%）；目前在 2016 年以來第 {t['vol_percentile_since_2016']:.0%} 百分位", loc="left", fontsize=10, color=INK2)

    ax[3].step(h.index, h.suggest_equity * 100, where="post", color=BLUE, lw=2)
    ax[3].annotate(f"{t['suggest_equity']:.0%}", (d, t["suggest_equity"] * 100), xytext=(6, 0), textcoords="offset points", fontsize=9, color=INK)
    ax[3].set_ylim(0, 105); ax[3].set_title("建議持股比例（%）", loc="left", fontsize=10, color=INK2)
    ax[3].set_xlim(h.index[0], nxt + pd.Timedelta(days=6))
    ax[3].xaxis.set_major_formatter(matplotlib.dates.DateFormatter("%m/%d"))

    warn = "大跌警訊中" if t["warning_active"] else "無大跌警訊"
    left = f"，還要維持 {t['warning_trading_days_left']} 個交易日" if t.get("warning_trading_days_left") else ""
    fig.suptitle(f"台股風險週報｜資料至 {d:%Y/%m/%d}", x=.01, ha="left", fontsize=15, color=INK, y=.993)
    fig.text(.01, .965, f"建議持股 {t['suggest_equity']:.0%}｜{warn}{left}｜大跌機率 {t['p_drawdown_7pct_within_20d']:.1%}｜"
             f"預測波動 {v:.0%}｜外資選擇權偏向 {t['opt_fgn_bias']:+.3f}\n"
             f"下週區間：68% {r68[0]:,}–{r68[1]:,}；95% {r95[0]:,}–{r95[1]:,}。模型只預測風險與波動大小，不預測漲跌方向。",
             fontsize=9.5, color=INK2, va="top")
    fig.tight_layout(rect=(0, 0, .96, .935))
    out = os.path.join(R, "risk_weekly.png"); fig.savefig(out, dpi=150, facecolor=SURF); print(out)

if __name__ == "__main__":
    main()
