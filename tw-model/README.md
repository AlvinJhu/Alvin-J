# 台股加權指數 多因子機率模型 v1（2026-11-20）

## 檔案
- `fetch_twse.py`：抓證交所資料，可中斷續抓。`python fetch_twse.py monthly 2015-01`（OHLC、成交值）、`python fetch_twse.py daily 20160101`（三大法人、融資，由近到遠，每 4 秒 1 次）
- `model.py`：因子、walk-forward 回測、今日預測。`python model.py`；本機有 yfinance 加 `--yf` 會加入費半 ^SOX、美債 ^TNX
- `data/`：原始資料（jsonl）、`data/ext/` VIX、USD/TWD
- `reports/`：`model_accuracy.csv`、`backtest_summary.csv`、`calibration_20d.csv`、`today.json`、`equity.png`

## 方法
- 標的：次日、20 交易日後收盤是否高於今日
- 模型：邏輯迴歸（L2、標準化）＋ XGBoost（淺樹、強正則化），兩者平均
- 驗證：擴張視窗 walk-forward，最少 500 天訓練，每 63 天重訓；20 日標的 embargo 20 天，避免標籤重疊造成前視偏誤（López de Prado, 2018）
- 美國資料（VIX、匯率）一律延後一天對齊，確保台股收盤時已知
- 預期區間：RiskMetrics EWMA（λ=0.94）日波動，±1σ／±1.96σ 換算
- 部位：波動度目標 18% 年化（Moreira & Muir, 2017）

## 結果（2015-12～2026-11，樣本外 2,155 天）
| | AUC | 命中率 | 恆猜漲 |
|---|---|---|---|
| 次日 logit / xgb | 0.487 / 0.514 | 53.5% / 53.0% | 55.0% |
| 20日 logit / xgb | 0.437 / 0.465 | 56.3% / 56.2% | 62.8% |

| 策略 | CAGR | 波動 | Sharpe | 最大回撤 |
|---|---|---|---|---|
| 模型機率調部位 | 4.8% | 14.8% | 0.39 | -31.4% |
| 買進持有（價格指數，未含息） | 15.0% | 20.2% | 0.79 | -31.6% |
| 只做波動度目標 | 11.4% | 16.5% | 0.74 | -31.0% |

結論：量價＋VIX＋匯率因子沒有樣本外預測力；20 日機率甚至反向校準。只有波動度控部位有效。
籌碼因子（外資、投信、融資）雲端只抓到 2025-06 起約 375 天就被證交所擋，樣本不足，尚未納入。

## 限制
- 雲端抓不到費半、美債、期交所（外資期貨未平倉、Put/Call、台指 VIX）
- 回測用價格指數，未含股息，買進持有報酬被低估約每年 3%
- 極端事件（如 2026-11-20 單日 -6.23%）屬於厚尾，常態假設的區間會低估

## 參考文獻
- J.P. Morgan/Reuters. (1996). *RiskMetrics—Technical document* (4th ed.).
- López de Prado, M. (2018). *Advances in financial machine learning*. Wiley.
- Moreira, A., & Muir, T. (2017). Volatility-managed portfolios. *The Journal of Finance, 72*(4), 1611–1644. https://doi.org/10.1111/jofi.12513
- Welch, I., & Goyal, A. (2008). A comprehensive look at the empirical performance of equity premium prediction. *The Review of Financial Studies, 21*(4), 1455–1508. https://doi.org/10.1093/rfs/hhm014
