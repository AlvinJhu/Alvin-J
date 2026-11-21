# 台股加權指數 多因子機率模型 v2（2026-11-21）

## 檔案
- `fetch_twse.py`：抓證交所資料，可中斷續抓。`python fetch_twse.py monthly 2015-01`（OHLC、成交值）、`python fetch_twse.py daily 20160101`（三大法人、融資，由近到遠，每 4 秒 1 次）
- `fetch_local.py`：本機或 GitHub Actions 執行。FinMind 開放 API 抓 2016 起三大法人、融資（來源為證交所，與證交所原始檔重疊 375 天逐日 100% 一致），yfinance 抓費半 ^SOX、美債 10 年 ^TNX，存到 `data/ext/`
- `model.py`：因子、walk-forward 回測、今日預測。`python model.py`（`data/ext/` 有 sox/tnx 檔就自動納入；`--yf` 改為即時下載）
- `data/`：原始資料（jsonl）、`data/ext/` VIX、USD/TWD
- `risk_model.py`：v3 風險模型（波動預測、大跌機率、部位），`python risk_model.py`，輸出 `reports/risk_*`
- `reports/`：`model_accuracy.csv`、`backtest_summary.csv`、`calibration_20d.csv`、`today.json`、`equity.png`

## 方法
- 標的：次日、20 交易日後收盤是否高於今日
- 模型：邏輯迴歸（L2、標準化）＋ XGBoost（淺樹、強正則化），兩者平均
- 驗證：擴張視窗 walk-forward，最少 500 天訓練，每 63 天重訓；20 日標的 embargo 20 天，避免標籤重疊造成前視偏誤（López de Prado, 2018）
- 美國資料（VIX、匯率）一律延後一天對齊，確保台股收盤時已知
- 預期區間：RiskMetrics EWMA（λ=0.94）日波動，±1σ／±1.96σ 換算
- 部位：波動度目標 18% 年化（Moreira & Muir, 2017）

## 結果 v2（樣本 2016-02～2026-11，樣本外約 2,130 天）
新增：法人（外資、投信、自營 1/5/20 日買賣超占成交值）、融資（5/20 日變化、融資相對指數）、費半（1/5/20 日報酬、相對台股）、美債 10 年（1/20 日變動 bp、水準）。籌碼因子延後 1 日（法人 15:00、融資約 21:30 公布）。AUC 附 95% 區塊 bootstrap 區間（20 日標籤重疊，區塊 20 天）。

因子組合（logit＋XGBoost 平均）樣本外 AUC：
| 因子組 | 次日 AUC | 20日 AUC [95% 區間] |
|---|---|---|
| v1（量價＋VIX＋匯率） | 0.500 | 0.448 [0.381, 0.507] |
| 只用籌碼（法人＋融資） | 0.498 | 0.494 [0.432, 0.554] |
| 只用費半美債 | 0.506 | 0.544 [0.467, 0.614] |
| v1＋籌碼 | 0.494 | 0.472 [0.394, 0.538] |
| 全部 32 因子 | 0.499 | 0.478 [0.410, 0.545] |

單因子（`reports/factor_auc.csv`）：32 個因子中，20 日 AUC 最高為美債殖利率水準 0.532 [0.447, 0.610]；其餘全部低於 0.5，bias240、vol_ratio、margin_20 等的區間上界也低於 0.5，代表訓練期學到的方向在樣本外反轉（關係不穩定，不是可反向操作的訊號）。

| 策略（成本單邊 0.1%） | CAGR | 波動 | Sharpe | 最大回撤 |
|---|---|---|---|---|
| 模型：全部因子 | 4.9% | 14.0% | 0.41 | -29.0% |
| 模型：只用費半美債 | 7.5% | 14.1% | 0.58 | -25.1% |
| 模型：只用籌碼 | 7.0% | 14.8% | 0.53 | -32.5% |
| 買進持有（價格指數，未含息） | 15.5% | 20.2% | 0.82 | -31.6% |
| 只做波動度目標 | 11.9% | 16.4% | 0.77 | -31.0% |

結論：
1. 補齊 2016 起法人、融資後，籌碼因子沒有樣本外預測力，加進 v1 也沒有改善。
2. 費半美債組 20 日 AUC 0.544 最高，但未達 0.55 門檻，95% 區間含 0.5；且是 5 組 × 3 模型中挑出的最佳值，有多重比較偏誤（Harvey et al., 2016）。不採用。
3. 所有模型策略的 Sharpe 都低於「只做波動度目標」0.77。維持原則：模型機率不當買賣訊號，部位只用波動度目標控制（Moreira & Muir, 2017）。
4. 與文獻一致：多數預測變數在樣本外不勝過歷史平均（Welch & Goyal, 2008）。

## v3 風險模型（`risk_model.py`，2026-11-21）
方向猜不準，改預測兩個「可預測」的量：未來 20 日波動、未來 20 日內跌超過 7% 的機率。部位參數只用 2018–2022 選，2023–2026 完全保留驗證。

波動度預測（未來 20 日實現波動，`reports/risk_vol_forecast.csv`）：
| 模型 | 全期 R²（對數） | MAE（年化波動點） | QLIKE |
|---|---|---|---|
| 過去 20 日波動 | -0.08 | 6.5 | 0.51 |
| EWMA 0.94 | 0.05 | 6.1 | **0.42** |
| HAR＋幅度＋VIX | **0.24** | **4.9** | 0.48 |
HAR 類模型在 R²、MAE 勝過 EWMA，但 QLIKE（對低估較敏感）EWMA 仍較佳，改善屬溫和。

20 日內跌 7% 以上（基準率 9–11%，`reports/risk_crash_auc.csv`）：
| 期間 | AUC [95% 區間] | 機率最高 20% 的日子實際大跌率 | 其餘日子 |
|---|---|---|---|
| 2018–2022 | 0.57 [0.36, 0.77] | 20.9% | 8.9% |
| 2023–2026 保留 | 0.78 [0.66, 0.89] | 26.1% | 5.9% |
注意：大跌事件少且成群（2020/3、2022、2024/8、2025/4 等），獨立樣本有限，2018–2022 區間含 0.5。單用預測波動排序就有大部分效果。

部位（`reports/risk_strategy.csv`，持股 = min(1, 目標波動/預測波動)，大跌機率 > 30% 時再乘 0.25）：
| 策略 | 2023–2026 保留 Sharpe | 全期 Sharpe | 全期最大回撤 | 全期 CAGR |
|---|---|---|---|---|
| v3 風險模型 | 1.21 | 0.83 | -30.6% | 14.3% |
| v3 只用預測波動 | 1.22 | 0.84 | -31.6% | 15.1% |
| v2 波動目標 | 1.27 | 0.75 | -31.2% | 11.6% |
| 買進持有 | 1.19 | 0.82 | -31.6% | 15.5% |
結論：v3 讓部位與 Sharpe 接近買進持有、波動較低，但沒有穩定勝過買進持有；價值在「風險量化與部位上限」，不是超額報酬。大跌機率減碼會使年換手升至約 7.5 倍，成本須注意。

## 限制
- 雲端工作階段連不到證交所、FinMind、Yahoo；資料用 GitHub Actions 或本機 `fetch_local.py` 更新
- 尚未納入期交所（外資期貨未平倉、Put/Call、台指 VIX）
- 回測用價格指數，未含股息，買進持有報酬被低估約每年 3%
- 極端事件（如 2026-11-20 單日 -6.23%）屬於厚尾，常態假設的區間會低估

## 參考文獻
- Corsi, F. (2009). A simple approximate long-memory model of realized volatility. *Journal of Financial Econometrics, 7*(2), 174–196. https://doi.org/10.1093/jjfinec/nbp001
- Engle, R. F. (1982). Autoregressive conditional heteroscedasticity with estimates of the variance of United Kingdom inflation. *Econometrica, 50*(4), 987–1007. https://doi.org/10.2307/1912773
- Harvey, C. R., Liu, Y., & Zhu, H. (2016). …and the cross-section of expected returns. *The Review of Financial Studies, 29*(1), 5–68. https://doi.org/10.1093/rfs/hhv059
- J.P. Morgan/Reuters. (1996). *RiskMetrics—Technical document* (4th ed.).
- Künsch, H. R. (1989). The jackknife and the bootstrap for general stationary observations. *The Annals of Statistics, 17*(3), 1217–1241. https://doi.org/10.1214/aos/1176347265
- Parkinson, M. (1980). The extreme value method for estimating the variance of the rate of return. *The Journal of Business, 53*(1), 61–65. https://doi.org/10.1086/296071
- Patton, A. J. (2011). Volatility forecast comparison using imperfect volatility proxies. *Journal of Econometrics, 160*(1), 246–256. https://doi.org/10.1016/j.jeconom.2010.03.034
- López de Prado, M. (2018). *Advances in financial machine learning*. Wiley.
- Moreira, A., & Muir, T. (2017). Volatility-managed portfolios. *The Journal of Finance, 72*(4), 1611–1644. https://doi.org/10.1111/jofi.12513
- Welch, I., & Goyal, A. (2008). A comprehensive look at the empirical performance of equity premium prediction. *The Review of Financial Studies, 21*(4), 1455–1508. https://doi.org/10.1093/rfs/hhm014
