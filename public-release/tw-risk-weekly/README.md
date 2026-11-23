# 台股大跌預警模型 tw-risk-weekly（v3.2）

> **Alvin-J 原創**｜授權 CC BY-NC 4.0（可分享、可修改，但須標示「Alvin-J 原創」且不得商業使用）
> 本套件僅供研究與學習，**不構成投資建議**，使用前請先讀 [DISCLAIMER.md](DISCLAIMER.md)。

## 這是什麼
一個預測台股加權指數「風險」的量化模型，不猜漲跌，只回答三件事：

1. **未來 20 個交易日內跌 7% 以上的機率**（大跌機率）
2. **未來 20 日的年化波動**（預測波動）
3. **建議持股比例**：平時 = 目標波動 22% ÷ 預測波動（上限 100%）；大跌機率 >25% 時減碼到 1/2，至少維持 10 個交易日

定位是「空頭保險」：開發者的驗證顯示，大跌前的預警在各年代都有效（主要來自波動群聚），但減碼規則平時會少賺，長期報酬不一定勝過買進持有。

## 內容
| 檔案 | 說明 |
|---|---|
| `SKILL.md` | Claude SKILL：讓 Claude 自動跑模型、寫週報 |
| `model/fetch_local.py` | 資料抓取：`--init` 第一次全抓、`--daily` 每日補資料 |
| `model/fetch_twse.py` | 證交所加權指數、成交金額（由 fetch_local.py 呼叫） |
| `model/risk_model.py` | 主模型，輸出 `model/reports/risk_today.json` 等 |
| `model/model.py`、`model/futures_eval.py` | 資料載入、因子與共用函式 |
| `model/weekly_chart.py` | 週報圖片 `model/reports/risk_weekly.png` |
| `requirements.txt` | Python 套件 |

本套件**不附任何原始資料**，請用抓取腳本自行下載。

## 安裝
需要 Python 3.10 以上。

```bash
pip install -r requirements.txt
```

（選用）到 FinMind 免費註冊取得 token，設成環境變數可提高抓取額度：

```bash
# macOS / Linux
export FINMIND_TOKEN=你的token
# Windows PowerShell
$env:FINMIND_TOKEN="你的token"
```

## 使用步驟
```bash
# 1. 第一次：抓全部歷史資料（約 20–40 分鐘，證交所有限速，請勿調快）
python model/fetch_local.py --init

# 2. 跑模型（約 3–10 分鐘，含 walk-forward 回測）
python model/risk_model.py

# 3. 產生週報圖片（需中文字型）
python model/weekly_chart.py

# 之後每個交易日收盤後（建議晚上 7 點後，法人與融資資料已公布）
python model/fetch_local.py --daily
python model/risk_model.py
```

### 讀結果：`model/reports/risk_today.json`
| 欄位 | 意思 |
|---|---|
| `date`、`close` | 資料基準日、加權指數收盤 |
| `p_drawdown_7pct_within_20d` | 未來 20 日內跌 7% 以上的機率 |
| `base_rate` | 歷史平均大跌機率（約 10%） |
| `pred_vol_20d_ann` | 預測年化波動 |
| `warning_active` | 是否在警訊減碼期間 |
| `suggest_equity` | 建議持股比例 |
| `range_20d_68` / `range_20d_95` | 20 日 68% / 95% 區間 |
| `opt_fgn_bias` | 外資台指選擇權偏向（正偏多、負偏空） |
| `alerts` | 新警訊、減碼、警訊期滿、資料過期提醒 |
| `data_last_date` | 各資料來源最後日期，請確認沒有過期 |

### 參數
- `--cut 0.25`：警訊時減碼到 1/4（v3.1 設定，回撤更小但少賺更多）
- `--grid`：改用 2018–2022 網格搜尋選出的參數
- `--no-opt`：不用外資選擇權因子

### 搭配 Claude 使用
把整個資料夾壓成 zip，在 Claude 的「設定 → 功能 → Skills」上傳，之後說「跑台股大跌預警週報」即可。Claude 所在環境若連不到證交所、FinMind 或 Yahoo，請在自己電腦抓資料後再交給 Claude。

## 方法摘要
- **波動預測**：HAR-RV（日／週／月實現波動，Corsi, 2009）＋ Parkinson 高低幅度（Parkinson, 1980）＋ VIX，對數線性迴歸；以 QLIKE 評估（Patton, 2011）。
- **大跌機率**：邏輯迴歸＋XGBoost 平均；因子含波動、回撤、乖離、費半、法人、融資、外資選擇權偏向；籌碼與美股資料一律延後一天，避免前視偏誤。
- **驗證**：擴張視窗 walk-forward，每 63 天重訓，20 日標籤加 embargo（López de Prado, 2018）；參數只用 2018–2022 選，2023 起保留驗證。
- **部位**：波動度目標（Moreira & Muir, 2017）＋大跌警訊減碼＋調整帶降低換手。

## 開發者驗證結果（僅供參考，不代表未來）
- 2023–2026 保留期（v3.1，減碼 1/4）：Sharpe 1.66、最大回撤 -18.7%。
- 1999 起長樣本：大跌預警 AUC 各年代約 0.66–0.80；但 2001–2017 減碼規則 Sharpe 0.29，輸給買進持有 0.41。近年成績有部分是運氣。
- 整體效果：回撤約減半，年報酬約少 3–4%。v3.2 把減碼改為 1/2 以降低「保費」。
- 只能預測波動大小，**無法預測大漲**，也無法預測方向。

## 參考文獻
Corsi, F. (2009). A simple approximate long-memory model of realized volatility. *Journal of Financial Econometrics, 7*(2), 174–196. https://doi.org/10.1093/jjfinec/nbp001

López de Prado, M. (2018). *Advances in financial machine learning*. Wiley.

Moreira, A., & Muir, T. (2017). Volatility-managed portfolios. *The Journal of Finance, 72*(4), 1611–1644. https://doi.org/10.1111/jofi.12513

Parkinson, M. (1980). The extreme value method for estimating the variance of the rate of return. *The Journal of Business, 53*(1), 61–65. https://doi.org/10.1086/296071

Patton, A. J. (2011). Volatility forecast comparison using imperfect volatility proxies. *Journal of Econometrics, 160*(1), 246–256. https://doi.org/10.1016/j.jeconom.2010.03.034

## 授權
Alvin-J 原創，採 [CC BY-NC 4.0](https://creativecommons.org/licenses/by-nc/4.0/deed.zh-hant)。轉載或修改請標示「Alvin-J 原創」並附授權連結；禁止商業使用（販售、付費課程、付費訂閱、代操、投顧服務等）。詳見 [LICENSE](LICENSE)。
