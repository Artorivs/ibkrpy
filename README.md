# IBKR Async Quantitative Trading System

**English** | [繁體中文](README.zh-TW.md)

**Disclaimer**: This system is for academic research and personal strategy development only. It is not investment advice of any kind. Markets change quickly; you bear all trading risk from using it, and the developers accept no responsibility for any loss.

**Trading involves risk. Invest with care.**

## 1. Prerequisites

IBKR TWS or IB Gateway must be running and logged in on this machine.

In `Configure` -> `API` -> `Settings`, tick "Enable ActiveX and Socket Clients".

Untick "Read-Only API", or orders cannot be placed.

Python: 3.10+ is recommended. Install `Poetry` first to manage the virtual environment and dependencies; a dedicated virtual environment is recommended.

Install dependencies:

```bash
poetry install
```

Create your local settings from the samples:

```bash
cp config.yaml.sample config.yaml
cp env.sample .env
```

## 2. Operating modes

Everything starts from `ibkrpy/core/main.py`; choose the stage with `--mode`.

Stage 1: download history (`--mode download`)

Downloads and stitches historical data incrementally through the IBKR API into the SQLite database. Each run also refills holes left by earlier outages and re-adjusts older bars after a stock split.

```bash
python ibkrpy/core/main.py --mode download
```

Stage 2: train models and tune parameters (`--mode train`)

Reads the database, trains the models and tunes strategy parameters with Optuna. Each term's models predict several bars ahead (`forecast_settings.horizon_bars`: 1 week of daily bars, about 1 day of hourly bars, 1 hour of 5-minute bars), and the training log shows which gate held a term that made no trades. Retrain after changing a horizon.
(⚠️ Run stage 1 first, or there is no data to train on.)

```bash
python ibkrpy/core/main.py --mode train
```

On success, `weights/` holds the `.json` parameters and the `.keras` / `.pkl` weights.

Shortcut: full pipeline (`--mode autopilot`)

Runs stage 1 and stage 2 in one go. This is the usual weekend maintenance command.

```bash
python ibkrpy/core/main.py --mode autopilot
```

Stage 3: start the trading engine (`--mode live`)

Loads the trained models and strategy parameters, then scans the market and places orders.

```bash
# Live loop in the safety sandbox (simulated orders only; nothing is sent to the broker)
python ibkrpy/core/main.py --mode live --dry-run

# Same, with the dashboard
python ibkrpy/core/main.py --mode live --dry-run --ui
```

Unattended 24/7 (`--mode daemon`)

Starts the daemon: trades during market hours, and backfills data and retrains models after hours and at weekends.

```bash
python ibkrpy/core/main.py --mode daemon --ui
```

Prediction ledger (`--mode ledger`)

Live and daemon modes record every scan's prediction in the `prediction_ledger` table. This mode fills in the realized move once the forecast horizon has passed, then reports hit rate, IC (information coefficient) and calibration slope per model generation and horizon, pooled and per symbol. It needs no IBKR connection. Run it after about two weeks of paper trading to judge whether the models have an edge.

```bash
python ibkrpy/core/main.py --mode ledger
python ibkrpy/core/main.py --mode ledger SPY   # one symbol
```

## 3. Dashboard

Add `--ui` at startup, or run `--mode ui` on its own, to open a separate monitoring window.

Left panel: the live trade decision log with entry and exit direction.

Top right: net liquidation value (NLV) and positions synced from IBKR.

Bottom right: a drop-down switches between the account equity curve and the price chart of each symbol.

## 4. Configuration (`config.yaml`)

Static settings live in `config.yaml` at the project root (`./`). Secrets such as API keys and email credentials can go in `.env` instead, which takes precedence. There you set:

- The port and client ID.
- API keys for external macro data (e.g. FRED).
- The asset pool and each symbol's term.
- The forecast horizon per term (`forecast_settings.horizon_bars`).
- Strategy, risk and cost settings (position size, exposure caps, kill switch, commission plan).

See `config.yaml.sample` for every option. A short example:

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
