"""
機器學習方向預測：梯度提升樹與神經網路，能否讓加權指數方向 AUC 明顯高於 0.55？
文獻：Gu, Kelly & Xiu (2020) 機器學習資產定價（月頻個股樣本外 R² 約 0.3–0.4%）；
      López de Prado (2018) 標籤重疊的 embargo；Holm (1979) 多重檢定；Harvey, Liu & Zhu (2016)。
設計（事先固定，不調參；所有組合全部列出）：
  特徵：現有全部 58 個（量價、VIX 匯率、法人、融資、費半美債、期貨選擇權籌碼、v3 波動特徵），樣本 2018-08 起
        另以不含期貨選擇權的 45 個特徵跑 2016 起長樣本
  模型（固定超參數）：
    LOGIT  邏輯迴歸 L2（C=0.1）— 線性基準
    XGB    XGBoost 深 2、300 棵、學習率 0.03、強正則化（同 v2）
    HGB    sklearn HistGradientBoosting 深 3、200 輪、學習率 0.05、葉最少 50 筆
    MLP    兩層神經網路 (32, 16)、L2 1e-3、提早停止，三個種子平均
  標的：次日、5 日、20 日漲跌；另以迴歸預測 20 日報酬，算 Gu et al. 的樣本外 R²（基準為 0 預測）
  驗證：擴張視窗、每 63 日重訓、embargo = 標的天數；AUC 95% 區間用區塊 bootstrap（區塊 = max(5, 標的天數)）；
        單邊 p 值（AUC ≤ 0.5 的 bootstrap 比例）做 Holm 校正
  部位：持股 = min(1, 22% / v3 預測波動) ×（模型機率 ≥ 0.5 → 1；否則 0.25），對照「只做波動目標」；成本 0.1%
輸出：reports/ml_auc.csv、ml_r2oos.csv、ml_strategy.csv、ml_importance.csv
"""
import os, sys, warnings
import numpy as np, pandas as pd
sys.argv = sys.argv[:1]; warnings.filterwarnings("ignore")
import model as M, risk_model as RM, futures_eval as FE, cta_model as CT
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.ensemble import HistGradientBoostingClassifier, HistGradientBoostingRegressor
from sklearn.neural_network import MLPClassifier, MLPRegressor
from sklearn.metrics import roc_auc_score
from xgboost import XGBClassifier, XGBRegressor

R = M.R; SPLIT = RM.SPLIT; ANN = np.sqrt(252)
HS = (1, 5, 20)

class Avg:
    def __init__(self, ms): self.ms = ms
    def fit(self, X, y): [m.fit(X, y) for m in self.ms]; return self
    def predict_proba(self, X): return np.mean([m.predict_proba(X) for m in self.ms], axis=0)
    def predict(self, X): return np.mean([m.predict(X) for m in self.ms], axis=0)

def clf(name):
    if name == "LOGIT": return make_pipeline(StandardScaler(), LogisticRegression(C=0.1, max_iter=3000))
    if name == "XGB": return XGBClassifier(n_estimators=300, max_depth=2, learning_rate=0.03, subsample=0.8, colsample_bytree=0.7,
                                           min_child_weight=30, reg_lambda=5.0, verbosity=0)
    if name == "HGB": return HistGradientBoostingClassifier(max_depth=3, max_iter=200, learning_rate=0.05, min_samples_leaf=50,
                                                            l2_regularization=1.0, early_stopping=False, random_state=0)
    return Avg([make_pipeline(StandardScaler(), MLPClassifier(hidden_layer_sizes=(32, 16), alpha=1e-3, early_stopping=True, max_iter=500, random_state=s))
                for s in range(3)])

def reg(name):
    if name == "LOGIT": return make_pipeline(StandardScaler(), Ridge(alpha=100.0))
    if name == "XGB": return XGBRegressor(n_estimators=300, max_depth=2, learning_rate=0.03, subsample=0.8, colsample_bytree=0.7,
                                          min_child_weight=30, reg_lambda=5.0, verbosity=0)
    if name == "HGB": return HistGradientBoostingRegressor(max_depth=3, max_iter=200, learning_rate=0.05, min_samples_leaf=50,
                                                           l2_regularization=1.0, early_stopping=False, random_state=0)
    return Avg([make_pipeline(StandardScaler(), MLPRegressor(hidden_layer_sizes=(32, 16), alpha=1e-3, early_stopping=True, max_iter=500, random_state=s))
                for s in range(3)])

MODELS = ("LOGIT", "XGB", "HGB", "MLP")

def wf(X, y, maker, h, kind="clf", min_train=500, step=63):
    out = pd.Series(np.nan, index=X.index); start = min_train; Xv = X.values
    while start < len(X):
        end = min(start + step, len(X)); tr = np.arange(max(0, start - h)); tr = tr[y.iloc[tr].notna().values]
        m = maker().fit(Xv[tr], y.iloc[tr].values.astype(int if kind == "clf" else float))
        out.iloc[start:end] = m.predict_proba(Xv[start:end])[:, 1] if kind == "clf" else m.predict(Xv[start:end])
        start = end
    return out

def boot_auc(y, p, block, n=500, seed=0):
    y, p = np.asarray(y), np.asarray(p); T = len(y); rng = np.random.default_rng(seed); out = []
    for _ in range(n):
        st = rng.integers(0, T, int(np.ceil(T / block))); ix = (st[:, None] + np.arange(block)).ravel()[:T] % T
        if 0 < y[ix].mean() < 1: out.append(roc_auc_score(y[ix], p[ix]))
    out = np.array(out); return np.percentile(out, 2.5), np.percentile(out, 97.5), float((out <= 0.5).mean())

def holm(p):
    p = np.asarray(p); o = np.argsort(p); m = len(p); adj = np.empty(m); run = 0
    for k, i in enumerate(o): run = max(run, min(1, (m - k) * p[i])); adj[i] = run
    return adj

def main():
    df = M.load(); c = df.close
    F = M.features(df)
    Xr, y_vol, _, _, _ = RM.build(df)
    for k in ("rv1", "rv5", "rv22", "rv66", "park5", "neg5"): F[k] = Xr[k]
    G = FE.fut_features(df, FE.load_futures(df.index))
    sets = {"長樣本 45 特徵（2016 起）": F, "全部 58 特徵（2018-08 起）": F.join(G)}
    rows, r2rows, preds = [], [], {}
    for sname, X in sets.items():
        X = X.dropna()
        for h in HS:
            fwd = c.shift(-h) / c - 1
            y = (fwd > 0).astype(float).where(fwd.notna()).reindex(X.index)
            yr = fwd.reindex(X.index)
            for mn in MODELS:
                p = wf(X, y, lambda: clf(mn), h); preds[(sname, h, mn)] = p
                for per, sel in (("2023 以前", X.index < SPLIT), ("2023 起（保留期）", X.index >= SPLIT), ("全期", X.index == X.index)):
                    ok = sel & p.notna() & y.notna(); yy, pp = y[ok], p[ok]
                    lo, hi, pv = boot_auc(yy.values, pp.values, max(5, h))
                    rows.append({"特徵": sname, "標的": f"{h}日", "模型": mn, "期間": per, "起": str(yy.index[0].date()), "天數": int(ok.sum()),
                                 "AUC": roc_auc_score(yy, pp), "95%下界": lo, "95%上界": hi, "單邊p": pv,
                                 "命中率": float(((pp >= .5) == yy).mean()), "恆猜漲命中率": float(yy.mean())})
                if h == 20:
                    pr = wf(X, yr, lambda: reg(mn), h, kind="reg")
                    for per, sel in (("2023 以前", X.index < SPLIT), ("2023 起（保留期）", X.index >= SPLIT), ("全期", X.index == X.index)):
                        ok = sel & pr.notna() & yr.notna(); a, b = yr[ok], pr[ok]
                        r2rows.append({"特徵": sname, "模型": mn, "期間": per, "天數": int(ok.sum()),
                                       "R2oos(基準=0)": 1 - ((a - b) ** 2).sum() / (a ** 2).sum(),
                                       "R2oos(基準=歷史平均)": 1 - ((a - b) ** 2).sum() / ((a - a.expanding().mean().shift(h).fillna(a.mean())) ** 2).sum(),
                                       "預測與實際相關": float(np.corrcoef(a, b)[0, 1])})
                print(sname, h, mn, round(rows[-1]["AUC"], 3), flush=True)
    au = pd.DataFrame(rows)
    full = au.期間 == "全期"; au["Holm校正p(全期 24 檢定)"] = np.nan
    au.loc[full, "Holm校正p(全期 24 檢定)"] = holm(au.loc[full, "單邊p"].values)
    au.to_csv(os.path.join(R, "ml_auc.csv"), index=False, encoding="utf-8-sig")
    r2 = pd.DataFrame(r2rows); r2.to_csv(os.path.join(R, "ml_r2oos.csv"), index=False, encoding="utf-8-sig")
    with pd.option_context("display.width", 250):
        print(au[au.期間 != "2023 以前"].round(3).to_string(index=False)); print(r2.round(4).to_string(index=False))

    # 部位：v3 預測波動 × 方向傾斜（全部 58 特徵 × 4 模型 × 3 標的 全列）
    Xa = sets["全部 58 特徵（2018-08 起）"].dropna()
    vol_fc = np.exp(RM.wf(Xr.dropna()[CT.HARX], y_vol.reindex(Xr.dropna().index), RM.ols))
    rr = (c.shift(-1) / c - 1)
    idx = Xa.index.intersection(vol_fc.dropna().index)
    idx = idx[preds[("全部 58 特徵（2018-08 起）", 20, "XGB")].reindex(idx).notna() & rr.reindex(idx).notna()]
    base_w = RM.weights(vol_fc[idx], np.zeros(len(idx)), 0.22, 1, 1, 0.2, 1)
    cands = {"只做波動目標（22%）": base_w}
    for h in HS:
        for mn in MODELS:
            p = preds[("全部 58 特徵（2018-08 起）", h, mn)][idx].values
            raw = np.minimum(1.0, 0.22 / vol_fc[idx].values) * np.where(p >= 0.5, 1.0, 0.25)
            w = np.empty(len(raw)); cur = raw[0]
            for i, x in enumerate(raw):
                if abs(x - cur) > 0.2: cur = x
                w[i] = cur
            cands[f"{mn} {h}日方向傾斜"] = pd.Series(w, index=idx)
    cands["買進持有"] = pd.Series(1.0, index=idx)
    rows, rets = [], {}
    for n, w in cands.items():
        s = rr[idx] if n == "買進持有" else RM.strat(w, rr[idx]); rets[n] = s
        for per, m in (("2023 以前", idx < SPLIT), ("2023 起（保留期）", idx >= SPLIT)):
            (lo, hi), pw = CT.boot_diff(s[m].values, rets["只做波動目標（22%）"][m].values) if n != "只做波動目標（22%）" else ((np.nan, np.nan), np.nan)
            rows.append({"策略": n, "期間": per, **M.stats(s[m]), "平均持股": w[m].mean(), "年換手": w[m].diff().abs().sum() / (m.sum() / 252),
                         "Sharpe差 vs 波動目標 95%下界": lo, "95%上界": hi})
    st = pd.DataFrame(rows); st.to_csv(os.path.join(R, "ml_strategy.csv"), index=False, encoding="utf-8-sig")
    with pd.option_context("display.width", 250): print(st.round(3).to_string(index=False))

    # 特徵重要性（全期重訓 XGB 20 日，僅供參考）
    y20 = ((c.shift(-20) / c - 1) > 0).astype(float).reindex(Xa.index); ok = y20.notna() & (Xa.index < Xa.index[-20])
    m = clf("XGB").fit(Xa[ok].values, y20[ok].values.astype(int))
    imp = pd.Series(m.feature_importances_, index=Xa.columns).sort_values(ascending=False)
    imp.rename("gain_share").to_csv(os.path.join(R, "ml_importance.csv"), encoding="utf-8-sig")
    print(imp.head(10).round(3).to_string())

if __name__ == "__main__":
    main()
