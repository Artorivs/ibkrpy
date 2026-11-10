# 盈透證券非同步量化交易系統

[English](README.md) | **繁體中文**

**免責聲明**：本系統僅供學術研究與個人投資策略開發使用，切勿視為任何形式之投資建議。資本市場瞬息萬變，使用者須自行承擔參與市場所衍生之交易風險，開發者概不對任何資產減損負責。

**入市有風險，投資須謹慎。**

## 1. 系統準備要求

IBKR TWS 或 IB Gateway：必須在本機運行並登入。

至 `設定` -> `API` -> `設定` 中，勾選「啟用 ActiveX 和 Socket 客戶端」。

取消勾選「只讀 API (Read-Only API)」，否則無法下單。

Python 環境：推薦使用 Python 3.10+；必須提前安裝 `Poetry` 以管理虛擬環境與依賴；推薦使用獨立虛擬環境。

安裝依賴：

```bash
poetry install
```

由範本建立本機設定檔：

```bash
cp config.yaml.sample config.yaml
cp env.sample .env
```

## 2. 系統啟動模式

系統全部透過 `ibkrpy/core/main.py` 啟動，請以 `--mode` 參數指定執行階段。

階段一：獲取歷史數據 (`--mode download`)

透過 IBKR API 增量下載並縫合歷史數據，存入 SQLite 資料庫。每次執行也會補回先前斷線留下的缺口，並在股票分割後重新調整舊 K 線。

```bash
python ibkrpy/core/main.py --mode download
```

階段二：模型訓練與參數尋優 (`--mode train`)

讀取資料庫數據，訓練模型，並以 Optuna 尋優策略參數。各週期的模型會預測未來多根 K 線（`forecast_settings.horizon_bars`：日線 1 週、小時線約 1 個交易日、5 分鐘線 1 小時）；若某週期完全沒有交易，訓練日誌會列出是哪道關卡擋下。更改預測週期後須重新訓練。
（⚠️ 必須先執行過階段一，否則會找不到數據。）

```bash
python ibkrpy/core/main.py --mode train
```

成功後，`weights/` 資料夾下會生成對應的 `.json`（參數）與 `.keras` / `.pkl`（權重）。

捷徑：一鍵全自動管線 (`--mode autopilot`)

一次執行階段一與階段二，是週末保養最常用的指令。

```bash
python ibkrpy/core/main.py --mode autopilot
```

階段三：啟動交易引擎 (`--mode live`)

載入訓練好的模型與策略參數，開始即時掃描市場並下單。

```bash
# 啟動交易迴圈，但套用安全沙盒（僅虛擬下單，不會真的送單給券商）
python ibkrpy/core/main.py --mode live --dry-run

# 同時開啟視覺化儀表板
python ibkrpy/core/main.py --mode live --dry-run --ui
```

24/7 全天候無人值守 (`--mode daemon`)

啟動系統守護進程：盤中自動交易，盤後與週末自動回補資料並重新訓練模型。

```bash
python ibkrpy/core/main.py --mode daemon --ui
```

預測帳本 (`--mode ledger`)

即時與守護進程模式會把每次掃描的預測寫入 `prediction_ledger` 資料表。此模式會在預測週期過後補上實際漲跌，並依模型世代與預測週期，彙總及逐檔列出命中率、IC（資訊係數）與校準斜率。不需連線 IBKR。建議模擬交易約兩週後執行，以判斷模型是否具備優勢。

```bash
python ibkrpy/core/main.py --mode ledger
python ibkrpy/core/main.py --mode ledger SPY   # 單一標的
```

## 3. 監控儀表板使用指南

啟動時加上 `--ui` 參數，或單獨執行 `--mode ui`，系統會彈出獨立的監控視窗。

左側面板：即時的交易決策日誌與進出場方向。

右上方面板：從 IBKR 即時同步的淨清算值 (NLV) 與現有持倉。

右下方面板：透過下拉選單切換帳戶淨值曲線，或個別標的之走勢圖。

## 4. 核心配置檔 (`config.yaml`)

系統的靜態參數由根目錄（`./`）的 `config.yaml` 統一控管。API 金鑰、郵件帳密等機密可改放 `.env`，其優先順序較高。可設定的項目包括：

- 通訊埠號與客戶端編號。
- 外部總體經濟數據介面（如 FRED API）的金鑰。
- 交易標的池與各標的的週期。
- 各週期的預測步數（`forecast_settings.horizon_bars`）。
- 策略、風控與成本設定（部位大小、曝險上限、停損開關、佣金方案）。

完整選項請見 `config.yaml.sample`。簡短範例如下：

```yaml
ib_settings:
  host: "127.0.0.1"
  port: 7497       # 7497 = paper; use 7496 for live
  client_id: 1     # Client ID of the main process

api_keys_settings:
  fred_api_key: "YOUR_FRED_API_KEY_HERE" # Used for the VIX macro indicator

forecast_settings:
  horizon_bars:                  # Bars ahead each term predicts; retrain after changing
    short_term: 12
    mid_term: 7
    long_term: 5

strategy_settings:
  base_position_pct: 0.08        # Base size per position (share of net liquidation)
  max_gross_exposure: 1.0
  allow_shorting: true

risk_settings:
  max_daily_loss_pct: 0.02       # Block new entries after a 2% daily loss
  max_drawdown_pct: 0.10         # Block new entries during a 10% drawdown from peak

assets:
  - symbol: "NVDA"
    term: "mid_term"
    tags: "Semiconductor, GPU, AI, Data_Center, Tech"

  - symbol: "TSM"
    term: "mid_term"
    tags: "Semiconductor, Foundry, Tech"

  - symbol: "AVGO"
    term: "mid_term"
    tags: "Semiconductor, Networking, Tech"
```

（範例中的設定檔註解維持英文，與 `config.yaml.sample` 一致。）
