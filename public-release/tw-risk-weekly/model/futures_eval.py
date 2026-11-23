"""
Alvin-J 原創｜CC BY-NC 4.0（非商業使用，須標示出處）
期貨、選擇權籌碼因子驗證（資料：FinMind 轉載期交所；法人資料自 2018-06 起）
問題：加入台指期未平倉、三大法人期貨淨部位、期現價差、法人選擇權部位，能否提升
  (a) 次日/20 日方向 AUC（model.py 的設定）
  (b) 20 日內跌 7% 的機率 AUC（risk_model.py 的設定）
所有期貨因子延後 1 日（期交所約 15:00 後公布）。同一段樣本比較有無期貨因子。
輸出：reports/futures_factor_auc.csv、futures_set_auc.csv
"""
import os, sys
import numpy as np, pandas as pd
import model as M
import risk_model as RM
from sklearn.metrics import roc_auc_score

EXT = os.path.join(M.D, "ext"); LAG = 1

def load_futures(index):
    fd = pd.read_csv(os.path.join(EXT, "fut_tx_daily.csv"), parse_dates=["date"])
    fd = fd[(fd.trading_session == "position") & ~fd.contract_date.astype(str).str.contains("/") & (fd.volume > 0)]
    fd["cd"] = fd.contract_date.astype(str)
    oi = fd.groupby("date").open_interest.sum()
    near = fd.sort_values(["date", "cd"]).groupby("date").first()
    fi = pd.read_csv(os.path.join(EXT, "fut_tx_inst.csv"), parse_dates=["date"])
    fi["net"] = fi.long_open_interest_balance_volume - fi.short_open_interest_balance_volume
    net = fi.pivot_table(index="date", columns="institutional_investors", values="net", aggfunc="sum")
    op = pd.read_csv(os.path.join(EXT, "opt_txo_inst.csv"), parse_dates=["date"])
    op["net_amt"] = op.long_open_interest_balance_amount - op.short_open_interest_balance_amount
    op["gross_amt"] = op.long_open_interest_balance_amount + op.short_open_interest_balance_amount
    fo = op[op.institutional_investors == "外資"].pivot_table(index="date", columns="call_put", values=["net_amt", "gross_amt"], aggfunc="sum")
    pl = op.pivot_table(index="date", columns="call_put", values="long_open_interest_balance_volume", aggfunc="sum")
    out = pd.DataFrame(index=index)
    out["oi"] = oi.reindex(index); out["near_settle"] = near.settlement_price.reindex(index)
    for k, n in (("外資", "fgn"), ("投信", "trust"), ("自營商", "dealer")):
        if k in net: out[f"fut_{n}_net"] = net[k].reindex(index)
    if len(fo):
        out["opt_fgn_bias"] = ((fo[("net_amt", "買權")] - fo[("net_amt", "賣權")]) /
                               (fo[("gross_amt", "買權")] + fo[("gross_amt", "賣權")])).reindex(index)
    if {"買權", "賣權"} <= set(pl.columns): out["opt_pc_long"] = (pl["賣權"] / pl["買權"]).reindex(index)
    return out

def fut_features(df, fu):
    f = pd.DataFrame(index=df.index); c = df.close
    f["oi_chg5"] = np.log(fu.oi).diff(5); f["oi_chg20"] = np.log(fu.oi).diff(20)
    f["basis"] = fu.near_settle / c - 1; f["basis_chg5"] = f.basis.diff(5)
    for n in ("fgn", "trust", "dealer"):
        k = f"fut_{n}_net"
        if k in fu: f[f"{k}_oi"] = fu[k] / fu.oi
    if "fut_fgn_net" in fu:
        f["fut_fgn_chg5"] = fu.fut_fgn_net.diff(5) / fu.oi; f["fut_fgn_chg20"] = fu.fut_fgn_net.diff(20) / fu.oi
        f["fut_fgn_z60"] = (fu.fut_fgn_net - fu.fut_fgn_net.rolling(60).mean()) / fu.fut_fgn_net.rolling(60).std()
    if "opt_fgn_bias" in fu: f["opt_fgn_bias"] = fu.opt_fgn_bias; f["opt_fgn_bias_chg5"] = fu.opt_fgn_bias.diff(5)
    if "opt_pc_long" in fu: f["opt_pc_long"] = np.log(fu.opt_pc_long)
    return f.shift(LAG).replace([np.inf, -np.inf], np.nan)

def main():
    df = M.load(); F = M.features(df); fu = load_futures(df.index); FF = fut_features(df, fu)
    fcols = list(FF.columns)
    print("期貨因子：", fcols)
    base = [c for g in ("量價", "VIX匯率", "法人", "融資", "費半美債") for c in M.GROUPS[g] if c in F]
    v1 = M.GROUPS["量價"] + M.GROUPS["VIX匯率"]
    X, yv, ydd, ew, lr = RM.build(df)
    risk_cols = [c for c in X.columns]
    allX = pd.concat([F[base], FF, X.add_prefix("r_")], axis=1).dropna()
    print(f"共同樣本 {allX.index[0].date()} ~ {allX.index[-1].date()}，{len(allX)} 天")
    fwd1 = df.close.shift(-1) / df.close - 1; fwd20 = df.close.shift(-20) / df.close - 1
    y1 = (fwd1 > 0).astype(float).where(fwd1.notna()).reindex(allX.index)
    y20 = (fwd20 > 0).astype(float).where(fwd20.notna()).reindex(allX.index)
    ydd = ydd.reindex(allX.index)
    T = (("次日", y1, 1, 5), ("20日", y20, 20, 20), ("20日內跌7%", ydd, 20, 20))

    # 單因子
    rows = []
    for c in fcols:
        row = {"因子": c}
        for tgt, y, h, blk in T:
            p = M.walk_forward(allX[[c]], y, h, models=("logit",))["logit"]; s = M.score(tgt, "logit", p, y, blk)
            row[f"{tgt}AUC"], row[f"{tgt}下界"], row[f"{tgt}上界"] = s["AUC"], s["AUC_95%下界"], s["AUC_95%上界"]
        rows.append(row)
    fa = pd.DataFrame(rows); print(fa.round(3).to_string(index=False))
    fa.to_csv(os.path.join(M.R, "futures_factor_auc.csv"), index=False, encoding="utf-8-sig")

    # 組合：同一段樣本
    rc = ["r_" + c for c in risk_cols]
    sets = {"方向": {"v1": v1, "只用期貨選擇權": fcols, "v1+期貨選擇權": v1 + fcols, "全部v2": base, "全部v2+期貨選擇權": base + fcols},
            "大跌": {"v3風險因子": rc, "只用期貨選擇權": fcols, "v3風險因子+期貨選擇權": rc + fcols}}
    rows = []
    for tgt, y, h, blk in T:
        for sn, cols in sets["大跌" if "跌" in tgt else "方向"].items():
            pr = M.walk_forward(allX[cols], y, h); pr["ens"] = (pr["logit"] + pr["xgb"]) / 2
            s = M.score(tgt, "ens", pr["ens"], y, blk)
            rows.append({"標的": tgt, "因子組": sn, "因子數": len(cols), **{k: s[k] for k in ("測試天數", "AUC", "AUC_95%下界", "AUC_95%上界", "基準(恆猜漲)")}})
    st = pd.DataFrame(rows).rename(columns={"基準(恆猜漲)": "基準率"}); print(st.round(3).to_string(index=False))
    st.to_csv(os.path.join(M.R, "futures_set_auc.csv"), index=False, encoding="utf-8-sig")
    print("今日期貨因子：", FF.iloc[-1].round(4).to_dict())

if __name__ == "__main__":
    main()
