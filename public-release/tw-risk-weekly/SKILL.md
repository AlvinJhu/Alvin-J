---
name: tw-risk-weekly
description: 台股大跌預警模型（Alvin-J 原創）：跑 risk_model.py 產出台股加權指數大跌機率、預測波動、建議持股與下週區間，並生成大字體週報圖片（含免責說明）。
---

# 台股大跌預警模型（v3.2）｜Alvin-J 原創

授權：CC BY-NC 4.0（非商業使用，轉載須標示「Alvin-J 原創」）。本 SKILL 產出僅供參考，不構成投資建議，詳見 DISCLAIMER.md。

## 定位
空頭保險：只預測風險（20 日內大跌機率、未來波動大小），不預測漲跌方向，不做大漲預測。

## 步驟
1. 準備環境：把本 SKILL 的 `model/` 資料夾複製到可寫入的工作資料夾，`pip install -r requirements.txt`。
2. 準備資料（本 SKILL 不附原始資料）：
   - 第一次：`python model/fetch_local.py --init`（約 20–40 分鐘）
   - 之後每個交易日收盤後：`python model/fetch_local.py --daily`
   - 若環境無法連到證交所、FinMind 或 Yahoo，請使用者改在自己電腦執行，不要用舊資料硬跑。
3. 執行 `python model/risk_model.py`（預設 v3.2：警訊時減碼到 1/2；`--cut 0.25` 改為減到 1/4），讀 `model/reports/risk_today.json`。
4. 檢查資料日期：`data_last_date` 各來源若落後超過 3 個交易日（`alerts` 會列出），報告開頭要明講哪個資料過期。
5. 規則：
   - 未來 20 日內跌 7% 以上的機率 >25% → 持股減至 1/2，至少維持 10 個交易日。
   - 連續 10 個交易日無警訊 → 回到正常部位：目標波動 22% ÷ 預測年化波動（上限 100%）。
6. 下週區間：收盤 ×(1 ± 預測波動×√(5/252))，68% 用 1σ、95% 用 2σ。

## 輸出格式（繁體中文、白話、簡潔）
- 資料基準日
- 核心結論：警訊狀態＋建議持股%
- 大跌機率（對照歷史基準率與 25% 警戒線、最後一次超標日期）
- 預測波動與持股計算式
- 下週 68% / 95% 區間
- 外資選擇權偏向（`opt_fgn_bias`，正值偏多、負值偏空）
- 風險提醒
- 結尾固定附：「本報告由 Alvin-J 原創量化模型自動產生，僅供參考，不構成投資建議；模型只預測風險不預測方向，投資盈虧自負。」

## 週報圖片
- 執行 `python model/weekly_chart.py`，輸出 `model/reports/risk_weekly.png`（四格：指數與警訊、大跌機率、預測波動、建議持股，圖內附文字結論與免責說明）。
- 需要中文字型（Noto Sans CJK TC、Microsoft JhengHei 或 PingFang TC 任一）。
