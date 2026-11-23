"""
週報圖：reports/risk_weekly.png（指數圖另畫 200 日線，只當趨勢參考）
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

def vis_len(s):
    return sum(0.55 if ord(ch) < 0x2E80 else 1.0 for ch in s)

def wrap(text, usable_px, size, dpi):
    """依字寬換行（中文 1、英數 0.55 個字寬）；數字與英文整段不拆開，標點不放行首。"""
    import re
    per = usable_px / (size * dpi / 72); out, cur = [], ""
    for tok in re.findall(r"[0-9A-Za-z.,%+\-/:~]+|.", text):
        if vis_len(cur + tok) > per and cur.strip() and tok not in "，。、；：）」":  # 標點允許凸出行尾
            out.append(cur); cur = tok.lstrip()
        else:
            cur += tok
    if cur: out.append(cur)
    return out

def report_blocks(t, hall, r68, r95, thr):
    """週報文字：依當週數據自動產生。回傳 [(文字, 字級, 顏色, 粗細, 上方間距 inch)]。"""
    p, v, w, base = t["p_drawdown_7pct_within_20d"], t["pred_vol_20d_ann"], t["suggest_equity"], t["base_rate"]
    pct, cut, tgt = t["vol_percentile_since_2016"], t["params"]["cut"], t["params"]["target"]
    d = pd.Timestamp(t["date"]); lw = t.get("last_warning_day")
    hi = "極高" if pct >= .9 else "偏高" if pct >= .7 else "正常" if pct >= .3 else "偏低"
    wk = hall.warning.astype(str).str.lower().isin(["true", "1", "1.0"]).iloc[-6:-1]  # 本週前幾個交易日
    if t.get("rebuy_alert") or (not t["warning_active"] and wk.any()):  # 本週剛結束減碼期
        core = f"風險解除，持股可加回到 {w:.0%}。這只代表減碼期結束，不是看漲訊號。"
    elif t["warning_active"]:
        k = t.get("warning_trading_days_left", 0)
        core = f"大跌警訊中，維持減碼，建議持股 {w:.0%}；若之後不再超過 {thr:.0%}，最快 {k} 個交易日後解除。"
    else:
        recent = lw and (d - pd.Timestamp(lw)).days <= 45
        core = ("大跌警訊已經解除" if recent else "目前沒有大跌警訊") + (f"，但波動{hi}" if pct >= .7 else "") + \
               f"，建議持股 {w:.0%}" + ("，不要回到滿倉。" if w < .8 else "。")
    top = 1 - pct
    volrank = f"是 2016 年以來最高的 {max(top, .01):.0%}" if pct >= .5 else f"在 2016 年以來第 {pct:.0%} 百分位"
    ob = t.get("opt_fgn_bias")
    def lab(x): return "偏多" if x > .02 else "偏空" if x < -.02 else "大致中性"
    if ob is not None and "opt_fgn_bias" in hall:
        prev = hall.opt_fgn_bias.iloc[-40:-15].mean()
        opt = f"外資選擇權：{d:%m/%d} 是 {ob:+.3f}，{lab(ob)}"
        if lab(prev) != lab(ob): opt += f"；一個月前平均 {prev:+.3f}，{lab(prev)}"
        opt += "。"
    else:
        opt = "外資選擇權：無資料。"
    risk = []
    dd = t["close"] / hall.close.max() - 1
    if dd < -.1: risk.append(f"指數已從近半年高點跌了 {-dd:.0%}")
    r1 = t["close"] / hall.close.iloc[-2] - 1 if len(hall) > 1 else 0
    if abs(r1) >= .03: risk.append(f"{d:%m/%d} 單日{'跌' if r1 < 0 else '漲'} {abs(r1):.1%}")
    try:
        vix = float(pd.read_csv(os.path.join(HERE, "data", "ext", "vix.csv")).CLOSE.iloc[-1])
        if vix >= 25: risk.append(f"VIX {vix:.0f} 處高檔，模型此時傾向預期反彈，這點宜保守看待")
    except Exception:
        pass
    tr = t.get("trend_ma200")
    trend = (f"・趨勢參考：指數{'站上' if tr['above'] else '跌破'} 200 日線（{tr['ma200']:,} 點），"
             f"在線{'上' if tr['above'] else '下'} {abs(tr['dist_pct']):.1%}。僅供參考，不作買賣依據。") if tr else None
    riskt = ("，".join(risk) + "。" if risk else "") + "模型只預測風險，不預測漲跌方向。"
    watch = (f"每天 20:43 的預警。大跌機率如果回到 {thr:.0%} 以上，就要減碼到{'一半' if cut == .5 else f'{cut:.0%}'}。"
             if not t["warning_active"] else
             f"每天 20:43 的預警。警訊期間維持減碼；連續 {int(t['params']['hold'])} 個交易日未再超過 {thr:.0%} 才回到正常部位。")
    B, S, G = 18, 16, .16
    return [
        (f"台股風險週報｜資料至 {d:%Y/%m/%d}", 17, INK2, "normal", 0),
        (f"建議持股 {w:.0%}", 40, INK, "bold", .05),
        ("核心結論：" + core, 20, INK, "bold", .12),
        (f"・大跌機率：未來 20 日跌 7% 以上的機率是 {p:.1%}，歷史平均 {base:.1%}，警戒線 {thr:.0%}。"
         + (f"最後一次超過警戒是 {pd.Timestamp(lw):%m/%d}。" if lw else ""), B, INK, "normal", G),
        (f"・預測波動：年化 {v:.1%}，{volrank}。持股比例是 {tgt:.0%} 除以 {v:.1%}"
         + (f"，再乘上減碼 {cut:g}" if t["warning_active"] else "") + f"，也就是 {w:.0%}。", B, INK, "normal", G),
        (f"・下週區間：68% 機率落在 {r68[0]:,} 到 {r68[1]:,} 點，95% 機率落在 {r95[0]:,} 到 {r95[1]:,} 點。", B, INK, "normal", G),
        ("・" + opt, B, INK, "normal", G),
    ] + ([(trend, B, INK2, "normal", G)] if trend else []) + [
        ("・風險提醒：" + riskt, B, INK, "normal", G),
        ("觀察重點：" + watch, B, INK, "bold", .2),
    ]

def main():
    t = json.load(open(os.path.join(R, "risk_today.json"), encoding="utf-8"))
    h = pd.read_csv(os.path.join(R, "risk_history.csv"), parse_dates=["date"]).set_index("date")
    d = pd.Timestamp(t["date"])
    h.loc[d, ["close", "p_drawdown", "pred_vol", "warning", "suggest_equity"]] = [
        t["close"], t["p_drawdown_7pct_within_20d"], t["pred_vol_20d_ann"], t["warning_active"], t["suggest_equity"]]
    if t.get("trend_ma200") and "ma200" in h: h.loc[d, "ma200"] = t["trend_ma200"]["ma200"]
    h = h.sort_index(); hall = h.copy(); h = h.tail(DAYS); h["warning"] = h.warning.astype(str).str.lower().isin(["true", "1", "1.0"])
    c, v = t["close"], t["pred_vol_20d_ann"]; s5 = v * np.sqrt(5 / 252)
    r68 = [round(c * np.exp(-s5)), round(c * np.exp(s5))]; r95 = [round(c * np.exp(-1.96 * s5)), round(c * np.exp(1.96 * s5))]
    thr = t["params"]["thr"]

    plt.rcParams.update({"font.size": 14})
    blocks = report_blocks(t, hall, r68, r95, thr)
    W, DPI = 8.0, 150; usable = (W - 0.5) * DPI
    lines = []  # (text, size, color, weight, gap_before_in)
    for txt, size, color, weight, gap in blocks:
        for k, ln in enumerate(wrap(txt, usable, size, DPI)):
            lines.append((ln, size, color, weight, gap if k == 0 else 0.0))
    text_h = sum(sz * 1.45 / 72 + g for _, sz, _, _, g in lines) + 0.35
    CH = 12.5; DISC_H = 1.25
    H = text_h + CH + DISC_H
    fig = plt.figure(figsize=(W, H)); fig.patch.set_facecolor(SURF)
    y = H - 0.25
    for ln, sz, color, weight, g in lines:
        y -= g + sz * 1.45 / 72
        fig.text(0.25 / W, y / H, ln, fontsize=sz, color=color, fontweight=weight, va="bottom")
    gs = fig.add_gridspec(4, 1, height_ratios=[2.2, 1.4, 1.1, 1.1], hspace=.42,
                          top=(H - text_h - 0.45) / H, bottom=(DISC_H + 0.45) / H, left=.13, right=.80)
    ax = [fig.add_subplot(gs[0])]; ax += [fig.add_subplot(gs[i], sharex=ax[0]) for i in (1, 2, 3)]
    disc = ("免責說明：本報告為量化模型的研究輸出，僅供參考，不構成任何投資建議或買賣邀約。"
            "模型以歷史資料估計，過去表現不代表未來結果；資料可能有延遲或錯誤。投資有風險，請自行判斷並承擔損益。")
    yy = DISC_H - 0.15
    for ln in wrap(disc, usable, 12, DPI):
        fig.text(0.25 / W, yy / H, ln, fontsize=12, color=INK2, va="top"); yy -= 12 * 1.45 / 72
    for a in ax:
        a.set_facecolor(SURF); a.grid(axis="y", color=GRID, lw=.8); a.tick_params(colors=INK2, labelsize=13)
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
    if "ma200" in h and h.ma200.notna().any():
        ax[0].plot(h.index, h.ma200, color=INK2, lw=1.4, ls="--")
        ax[0].annotate("200 日線（參考）", (h.index[0], h.ma200.dropna().iloc[0]), xytext=(2, 10), textcoords="offset points", fontsize=12, color=INK2)
    ax[0].scatter([d], [c], color=BLUE, s=40, zorder=3, edgecolor=SURF, linewidth=2)
    ax[0].annotate(f"{c:,.0f}", (d, c), xytext=(6, -4), textcoords="offset points", color=INK, fontsize=15, fontweight="bold")
    nxt = pd.bdate_range(d + pd.Timedelta(days=1), periods=5)[-1]
    ax[0].fill_between([d, nxt], [c, r95[0]], [c, r95[1]], color=BLUE, alpha=.08, lw=0)
    ax[0].fill_between([d, nxt], [c, r68[0]], [c, r68[1]], color=BLUE, alpha=.18, lw=0)
    for y, lab in ((r68[0], "68%"), (r68[1], "68%"), (r95[0], "95%"), (r95[1], "95%")):
        ax[0].annotate(f"{y:,} ({lab})", (nxt, y), xytext=(4, 0), textcoords="offset points", fontsize=12, color=INK2, va="center")
    ax[0].set_title("加權指數（紅底＝警訊期間，扇形＝下週區間）", loc="left", fontsize=15, color=INK)
    ax[0].yaxis.set_major_formatter(matplotlib.ticker.FuncFormatter(lambda x, _: f"{x:,.0f}"))

    ax[1].plot(h.index, h.p_drawdown * 100, color=ORANGE, lw=2)
    ax[1].axhline(thr * 100, color=RED, ls="--", lw=1.2); ax[1].text(1.0, thr * 100, f" 警戒線 {thr:.0%}", color=INK2, fontsize=12, va="center", transform=ax[1].get_yaxis_transform())
    ax[1].axhline(t["base_rate"] * 100, color=INK2, ls=":", lw=1); ax[1].text(1.0, t["base_rate"] * 100, f" 歷史平均 {t['base_rate']:.0%}", color=INK2, fontsize=12, va="center", transform=ax[1].get_yaxis_transform())
    ax[1].annotate(f"{t['p_drawdown_7pct_within_20d']:.1%}", (d, t["p_drawdown_7pct_within_20d"] * 100), xytext=(-10, 22), textcoords="offset points", fontsize=15, color=INK, fontweight="bold")
    ax[1].set_ylim(0, max(50, h.p_drawdown.max() * 110)); ax[1].set_title("大跌機率：20 日內跌超過 7%（%）", loc="left", fontsize=15, color=INK)

    ax[2].plot(h.index, h.pred_vol * 100, color=BLUE, lw=2)
    ax[2].annotate(f"{v:.0%}", (d, v * 100), xytext=(6, 0), textcoords="offset points", fontsize=15, color=INK, fontweight="bold")
    ax[2].set_title(f"預測年化波動（%），2016 年來第 {t['vol_percentile_since_2016']:.0%} 百分位", loc="left", fontsize=15, color=INK)

    ax[3].step(h.index, h.suggest_equity * 100, where="post", color=BLUE, lw=2)
    ax[3].annotate(f"{t['suggest_equity']:.0%}", (d, t["suggest_equity"] * 100), xytext=(6, 0), textcoords="offset points", fontsize=15, color=INK, fontweight="bold")
    ax[3].set_ylim(0, 105); ax[3].set_title("建議持股比例（%）", loc="left", fontsize=15, color=INK)
    ax[3].set_xlim(h.index[0], nxt + pd.Timedelta(days=6))
    ax[3].xaxis.set_major_locator(matplotlib.dates.MonthLocator()); ax[3].xaxis.set_major_formatter(matplotlib.dates.DateFormatter("%-m月"))

    for a in ax[:-1]: plt.setp(a.get_xticklabels(), visible=False)
    out = os.path.join(R, "risk_weekly.png"); fig.savefig(out, dpi=DPI, facecolor=SURF); print(out)

if __name__ == "__main__":
    main()
