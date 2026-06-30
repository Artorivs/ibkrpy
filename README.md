# 盈透證券非同步量化交易系統

**免責聲明**: 本系統僅供學術研究與個人投資策略開發使用，切勿視為任何形式之投資建議。資本市場瞬息萬變，使用者須自行承擔參與市場所衍生之交易風險，開發者概不對任何資產減損負責。

**入市有風險，投資須謹慎。**

## 1. 系統準備要求 (Prerequisites)

IBKR TWS 或 IB Gateway: 必須在本機運行並登入。

至 `設定` -> `API` -> `設定` 中，勾選「啟用 ActiveX 和 Socket 客戶端」。

取消勾選「只讀 API (Read-Only API)」，否則無法下單。

Python 環境: 推薦使用 Python 3.10+；必須提前安裝 `Poetry` 以管理虛擬環境與依賴；推薦使用獨立虛擬環境。

安裝依賴:

```bash
poetry install
```

## 2. 系統啟動模式 (Operating Modes)

系統全部透過 `ibkrpy/core/main.py` 來啟動，請使用 --mode 參數來指定執行階段。

階段一：獲取歷史數據 (`--mode download`)

系統將透過 IBKR API 增量下載並縫合歷史數據，存入 SQLite 資料庫與 Feather 快取中。

```bash
python ibkrpy/core/main.py --mode download
```

階段二：模型訓練與參數尋優 (`--mode train`)

讀取資料庫數據，執行深度學習訓練，並利用 Optuna 進行策略參數的百次迭代尋優。
(⚠️ 警告：必須先執行過階段一，否則會找不到數據。)

```bash
python ibkrpy/core/main.py --mode train
```

成功後，`weights/` 資料夾下會生成對應的 `.json` (參數) 與 `.keras` / `.pkl` (權重)。

捷徑：一鍵全自動管線 (`--mode autopilot`)

如果你想一次性執行階段一與階段二，請使用此模式。這是週末大保養最常用的指令。

```bash
python ibkrpy/core/main.py --mode autopilot
```

階段三：啟動交易引擎 (`--mode live`)

載入訓練好的 AI 模型與策略參數，開始即時掃描市場並下單。

```bash
# 啟動實盤，但套用安全沙盒 (僅虛擬下單，不會真的發送電文給券商)
python ibkrpy/core/main.py --mode live --dry-run 

# 啟動並同時開啟視覺化儀表板
python ibkrpy/core/main.py --mode live --dry-run --ui
```

終極階段：24/7 全天候無人值守 (`--mode daemon`)

啟動系統守護進程。盤中自動執行交易，盤後/週末自動進行資料回補與模型重新訓練。

```bash
python ibkrpy/core/main.py --mode daemon --ui
```

## 3. 監控儀表板 (Dashboard) 使用指南

如果在啟動時加上了 `--ui` 參數，或單獨執行 `--mode ui`，系統將彈出一個獨立的監控視窗。

左側面板: 顯示實時的 AI 交易決策日誌與進出場方向。

右上方面板: 顯示從 IBKR 即時同步的淨清算值 (NLV) 與現有庫存。

右下方面板: 透過下拉選單，可以即時切換查看帳戶總收益曲線 (Equity Curve)，或是個別監控標的之即時走勢圖。

## 4. 核心配置檔 (`config.yaml`) 詳解

系統之靜態參數均由根目錄（`./`）之 `config.yaml` 統一控管。你可於該檔案內配置：

- 通訊埠號與客戶端編號。
- 外部總體經濟數據介面（如 FRED API）之金鑰。
- 交易標的池（長線、中線、短線）。
- 預設之策略閾值（如預測漲幅門檻、停損與停利倍數）。

以下為範例：

```yaml
ib_settings:
  host: "127.0.0.1"
  port: 7497       # 7497為模擬盤(Paper)，實盤(Live)請改為7496
  client_id: 1     # 系統主進程使用的 Client ID

api_keys_settings:
  fred_api_key: "YOUR_FRED_API_KEY_HERE" # 用於抓取 VIX 宏觀指標

strategy_settings:
  # 預設參數 (若 weights/ 目錄下有 Optuna 產生的 JSON 檔，將會覆蓋此處設定)
  min_prediction_threshold_pct: 0.005
  volatility_stop_loss_multiplier: 1.5
  volatility_take_profit_multiplier: 2.0

assets:
   symbol: "NVDA"
    term: "mid_term"
    tags: "Semiconductor, GPU, AI, Data_Center, Tech"

  - symbol: "TSM"
    term: "mid_term"
    tags: "Semiconductor, Foundry, Tech"

  - symbol: "AVGO"
    term: "mid_term"
    tags: "Semiconductor, Networking, Tech"
```
