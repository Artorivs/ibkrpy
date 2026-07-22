# evaluation/backtest_engine.py
# 整合模擬交易與高階機構級績效評估

import pandas as pd
import numpy as np
from typing import Dict, Any, List, Optional


class BacktestEngine:
    """提供高保真度歷史回測，並自動結算機構級績效指標"""

    def __init__(
        self,
        initial_capital: float = 100000.0,
        commission_per_share: float = 0.005,
        slippage_pct: float = 0.0005,
        use_bracket_exits: bool = True,
    ):
        self.initial_capital = initial_capital
        self.commission = commission_per_share
        self.slippage_pct = slippage_pct
        self.use_bracket_exits = use_bracket_exits

    # ------------------------------------------------------------------
    # 工具
    # ------------------------------------------------------------------

    @staticmethod
    def _infer_periods_per_year(index: pd.Index) -> float:
        """
        由索引的時間間隔推導年化係數。
        日線 -> 252；1 小時 (美股 RTH 6.5h) -> 1638；5 分鐘 -> 19656。
        推導失敗時保守回退為 252。
        """
        try:
            if not isinstance(index, pd.DatetimeIndex) or len(index) < 3:
                return 252.0

            deltas = pd.Series(index[1:]) - pd.Series(index[:-1])
            median_sec = deltas.dt.total_seconds().median()

            if not np.isfinite(median_sec) or median_sec <= 0:
                return 252.0

            # 間隔 >= 20 小時視為日線 (跨週末的 gap 會被中位數濾掉)
            if median_sec >= 20 * 3600:
                return 252.0

            # 盤中資料：一個交易日 6.5 小時 = 23400 秒
            bars_per_day = 23400.0 / median_sec
            return 252.0 * max(bars_per_day, 1.0)
        except Exception:
            return 252.0

    def _fill_price(self, price: float, is_buy: bool) -> float:
        """買入滑價使成本變高，賣出滑價使收入變少"""
        return price * (1 + self.slippage_pct) if is_buy else price * (1 - self.slippage_pct)

    # ------------------------------------------------------------------
    # 主迴圈
    # ------------------------------------------------------------------

    def run(
        self,
        df: pd.DataFrame,
        signals: List[Dict[str, Any]],
        benchmark_df: pd.DataFrame = None,
    ) -> Dict[str, Any]:
        """
        執行回測 (支援做多/做空，逐根 K 線計算 MTM 淨值，並模擬 bracket 出場)

        :param df: 歷史 K 線數據。需含 Close；若含 High/Low 則啟用 bracket 模擬。
        :param signals: 策略生成的信號列表 (含 stop_loss_price / take_profit_price)
        :param benchmark_df: (可選) 大盤數據，用於 Alpha、資訊比率等指標
        """
        if df.empty or 'Close' not in df.columns:
            return self._empty_metrics()

        has_hl = 'High' in df.columns and 'Low' in df.columns
        simulate_bracket = self.use_bracket_exits and has_hl

        capital = self.initial_capital
        position = 0
        active_sl: Optional[float] = None
        active_tp: Optional[float] = None

        equity_curve = []
        trades = []          # 每次部位變動
        round_trips = 0      # 完整的進出場來回次數

        signal_dict = {sig["timestamp"]: sig for sig in signals}

        for ts, row in df.iterrows():
            price = float(row['Close'])
            if not np.isfinite(price) or price <= 0:
                equity_curve.append({"timestamp": ts, "equity": capital + position * price})
                continue

            # ========== 1. 先檢查既有部位是否觸發 bracket 出場 ==========
            # 必須在處理新訊號之前，因為停損停利在盤中隨時可能成交。
            if simulate_bracket and position != 0 and (active_sl is not None or active_tp is not None):
                high = float(row['High'])
                low = float(row['Low'])
                exit_price = None
                exit_kind = None

                if position > 0:
                    hit_sl = active_sl is not None and low <= active_sl
                    hit_tp = active_tp is not None and high >= active_tp
                else:
                    hit_sl = active_sl is not None and high >= active_sl
                    hit_tp = active_tp is not None and low <= active_tp

                # 同一根 K 線同時觸及兩者時，保守假設先觸發停損。
                # 沒有 tick 資料就無法判定先後，樂觀假設會系統性高估績效。
                if hit_sl:
                    exit_price, exit_kind = active_sl, "STOP_LOSS"
                elif hit_tp:
                    exit_price, exit_kind = active_tp, "TAKE_PROFIT"

                if exit_price is not None:
                    closing_qty = -position
                    fill = self._fill_price(exit_price, is_buy=closing_qty > 0)
                    capital -= closing_qty * fill
                    capital -= abs(closing_qty) * self.commission

                    trades.append({
                        "type": exit_kind,
                        "price": exit_price,
                        "size": closing_qty,
                        "timestamp": ts,
                    })
                    round_trips += 1

                    position = 0
                    active_sl = None
                    active_tp = None

            # ========== 2. 處理新訊號 ==========
            sig = signal_dict.get(ts)
            if sig:
                action = sig.get("action")
                target_position = 0

                # 留 5% 資金作為保證金與滑價緩衝
                if action == "BUY":
                    target_position = int((capital * 0.95) / price)
                elif action == "SELL":
                    target_position = -int((capital * 0.95) / price)

                trade_size = target_position - position

                if trade_size != 0:
                    fill = self._fill_price(price, is_buy=trade_size > 0)
                    capital -= trade_size * fill
                    capital -= abs(trade_size) * self.commission

                    # 反手 (多翻空或空翻多) 同時包含一次出場與一次進場
                    if position != 0 and np.sign(target_position) != np.sign(position):
                        round_trips += 1

                    position = target_position

                    trades.append({
                        "type": action,
                        "price": price,
                        "size": trade_size,
                        "timestamp": ts,
                    })

                    if position != 0:
                        active_sl = sig.get("stop_loss_price")
                        active_tp = sig.get("take_profit_price")
                    else:
                        active_sl = None
                        active_tp = None

            equity_curve.append({"timestamp": ts, "equity": capital + position * price})

        # ========== 3. 回測結束強制平倉結算 ==========
        if position != 0:
            last_price = float(df['Close'].iloc[-1])
            closing_qty = -position
            fill = self._fill_price(last_price, is_buy=closing_qty > 0)
            capital -= closing_qty * fill
            capital -= abs(closing_qty) * self.commission
            round_trips += 1
            position = 0
            if equity_curve:
                equity_curve[-1]["equity"] = capital

        return self._evaluate_performance(equity_curve, round_trips, benchmark_df)

    # ------------------------------------------------------------------
    # 績效結算
    # ------------------------------------------------------------------

    def _evaluate_performance(
        self,
        equity_curve: List[Dict],
        round_trips: int,
        benchmark_df: pd.DataFrame = None,
    ) -> Dict[str, Any]:
        if not equity_curve:
            return self._empty_metrics()

        df_eq = pd.DataFrame(equity_curve).set_index("timestamp")
        df_eq['return'] = df_eq['equity'].pct_change().fillna(0)

        periods_per_year = self._infer_periods_per_year(df_eq.index)
        ann_factor = np.sqrt(periods_per_year)

        total_return_pct = (df_eq['equity'].iloc[-1] / self.initial_capital - 1) * 100

        bar_returns = df_eq['return']
        std = bar_returns.std()
        sharpe_ratio = (bar_returns.mean() / std * ann_factor) if std > 0 else 0.0

        # 索提諾比率 —— 僅懲罰下行風險
        downside = bar_returns[bar_returns < 0]
        downside_std = downside.std() * ann_factor if len(downside) > 1 else 0.0
        sortino_ratio = (
            (bar_returns.mean() * periods_per_year / downside_std) if downside_std > 0 else 0.0
        )

        # 獲利因子 (以每根 K 線的損益衡量，非逐筆交易的損益)
        gains = bar_returns[bar_returns > 0].sum()
        losses = abs(bar_returns[bar_returns < 0].sum())
        profit_factor = (gains / losses) if losses > 0 else float('inf')

        # 最大回撤
        roll_max = df_eq['equity'].cummax()
        drawdown = df_eq['equity'] / roll_max - 1
        max_drawdown_pct = abs(drawdown.min()) * 100

        # 資訊比率與追蹤誤差
        tracking_error = 0.0
        information_ratio = 0.0

        if benchmark_df is not None and not benchmark_df.empty and 'Close' in benchmark_df.columns:
            bench_returns = benchmark_df['Close'].pct_change().fillna(0)
            aligned = pd.concat([bar_returns, bench_returns], axis=1, join='inner').dropna()
            if len(aligned) > 1:
                active_returns = aligned.iloc[:, 0] - aligned.iloc[:, 1]
                tracking_error = active_returns.std() * ann_factor
                if tracking_error > 0:
                    information_ratio = (active_returns.mean() * periods_per_year) / tracking_error

        return {
            "total_return_pct": round(total_return_pct, 2),
            "sharpe_ratio": round(sharpe_ratio, 2),
            "sortino_ratio": round(sortino_ratio, 2),
            "profit_factor": round(profit_factor, 2),
            "information_ratio": round(information_ratio, 2),
            "tracking_error_pct": round(tracking_error * 100, 2),
            "max_drawdown_pct": round(max_drawdown_pct, 2),
            "total_trades": round_trips,
            "periods_per_year": round(periods_per_year, 1),
            "final_equity": round(df_eq['equity'].iloc[-1], 2),
        }

    def _empty_metrics(self) -> Dict[str, Any]:
        return {
            "total_return_pct": 0.0, "sharpe_ratio": 0.0,
            "sortino_ratio": 0.0, "profit_factor": 0.0,
            "information_ratio": 0.0, "tracking_error_pct": 0.0,
            "max_drawdown_pct": 0.0, "total_trades": 0,
            "periods_per_year": 252.0,
            "final_equity": self.initial_capital,
        }