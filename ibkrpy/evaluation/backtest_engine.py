# evaluation/backtest_engine.py
# Bar-by-bar backtest with performance metrics

import pandas as pd
import numpy as np
from typing import Any, Callable, Dict, List, Optional

from ibkrpy.strategy.order_intent import OrderSnapshot, decide_order

Decide = Callable[[Any, Any, int, float], Optional[Dict[str, Any]]]


class BacktestEngine:
    """Historical backtest that follows the live order rules and computes metrics.

    Every signal goes through order_intent.decide_order, the function live uses, so
    sizing, close-only exits, no-add, shorting, the cost check and min_trade_usd match.
    A signal may carry target_weight (default position_pct) and entries_allowed.
    Fill costs come from the live ExecutionCostModel when one is given.
    """

    def __init__(
        self,
        initial_capital: float = 100000.0,
        commission_per_share: float = 0.005,
        slippage_pct: float = 0.0005,
        use_bracket_exits: bool = True,
        position_pct: float = 0.08,
        allow_shorting: bool = True,
        cost_model=None,
        min_trade_usd: float = 0.0,
        max_gross_exposure: float = 1.0,
        safety_multiple: float = 1.5,
    ):
        self.initial_capital = initial_capital
        self.commission = commission_per_share
        self.slippage_pct = slippage_pct
        self.use_bracket_exits = use_bracket_exits
        self.position_pct = float(position_pct)
        self.allow_shorting = bool(allow_shorting)
        self.cost_model = cost_model
        self.min_trade_usd = float(min_trade_usd)
        self.max_gross_exposure = float(max_gross_exposure)
        self.safety_multiple = float(safety_multiple)

    def _snapshot(
        self,
        sig: Dict[str, Any],
        position: int,
        price: float,
        cash: float,
        equity: float,
    ) -> OrderSnapshot:
        return OrderSnapshot(
            action=sig["action"],
            current_pos=position,
            price=price,
            net_liquidation=max(equity, 0.0),
            available_funds=max(cash, 0.0),
            gross_exposure=abs(position) * price,
            take_profit_price=sig.get("take_profit_price"),
            entries_allowed=bool(sig.get("entries_allowed", True)),
            allow_shorting=self.allow_shorting,
            target_weight=float(sig.get("target_weight", self.position_pct)),
            max_gross_exposure=self.max_gross_exposure,
            min_trade_usd=self.min_trade_usd,
            safety_multiple=self.safety_multiple,
        )

    @staticmethod
    def _infer_periods_per_year(index: pd.Index) -> float:
        """
        Infer the annualization factor from the index spacing.
        Daily -> 252; 1 hour (6.5h RTH) -> 1638; 5 minutes -> 19656.
        Falls back to 252 if it can't be inferred.
        """
        try:
            if not isinstance(index, pd.DatetimeIndex) or len(index) < 3:
                return 252.0

            deltas = pd.Series(index[1:]) - pd.Series(index[:-1])
            median_sec = deltas.dt.total_seconds().median()

            if not np.isfinite(median_sec) or median_sec <= 0:
                return 252.0

            if median_sec >= 20 * 3600:
                return 252.0

            bars_per_day = 23400.0 / median_sec
            return 252.0 * max(bars_per_day, 1.0)
        except Exception:
            return 252.0

    def _fill_cost(self, price: float, shares: int) -> float:
        """Cash cost of one fill (one side of a round trip)."""
        shares = abs(int(shares))
        if shares == 0:
            return 0.0
        if self.cost_model is not None:
            return self.cost_model.estimate(price, shares).total / 2.0
        return shares * (price * self.slippage_pct + self.commission)

    def run(
        self,
        df: pd.DataFrame,
        signals: List[Dict[str, Any]],
        benchmark_df: pd.DataFrame = None,
    ) -> Dict[str, Any]:
        """Backtest a precomputed signal list (signals keyed by "timestamp")."""
        signal_dict = {sig["timestamp"]: sig for sig in signals}
        return self.simulate(
            df, lambda ts, row, position, equity: signal_dict.get(ts), benchmark_df
        )

    def simulate(
        self,
        df: pd.DataFrame,
        decide: Decide,
        benchmark_df: pd.DataFrame = None,
    ) -> Dict[str, Any]:
        """
        Run bar by bar, asking decide(ts, row, position, equity) for a signal at each close.

        The decision sees the current position, as the live strategy does. Brackets from
        a fill on bar t are checked from bar t+1 on, against that bar's High/Low.

        :param df: OHLCV bars. Needs Close; High/Low enable bracket simulation.
        :param decide: returns a signal dict (action, stop_loss_price, take_profit_price) or None
        :param benchmark_df: optional benchmark bars for alpha / information ratio
        """
        if df.empty or "Close" not in df.columns:
            return self._empty_metrics()

        has_hl = "High" in df.columns and "Low" in df.columns
        simulate_bracket = self.use_bracket_exits and has_hl

        cash = self.initial_capital
        position = 0
        active_sl: Optional[float] = None
        active_tp: Optional[float] = None

        equity_curve = []
        round_trips = 0

        def close_at(fill_price: float) -> None:
            nonlocal cash, position, active_sl, active_tp, round_trips
            cash += position * fill_price
            cash -= self._fill_cost(fill_price, position)
            position = 0
            active_sl = active_tp = None
            round_trips += 1

        for ts, row in df.iterrows():
            price = float(row["Close"])
            if not np.isfinite(price) or price <= 0:
                last = equity_curve[-1]["equity"] if equity_curve else cash
                equity_curve.append({"timestamp": ts, "equity": last})
                continue

            if simulate_bracket and position != 0:
                high = float(row["High"])
                low = float(row["Low"])
                if position > 0:
                    hit_sl = active_sl is not None and low <= active_sl
                    hit_tp = active_tp is not None and high >= active_tp
                else:
                    hit_sl = active_sl is not None and high >= active_sl
                    hit_tp = active_tp is not None and low <= active_tp

                # A bar that opens beyond a level fills at the open, not at the level.
                opened = float(row["Open"]) if "Open" in row.index else None
                if opened is not None and not np.isfinite(opened):
                    opened = None
                if hit_sl:
                    fill = active_sl
                    if opened is not None:
                        fill = min(fill, opened) if position > 0 else max(fill, opened)
                    close_at(fill)
                elif hit_tp:
                    fill = active_tp
                    if opened is not None:
                        fill = max(fill, opened) if position > 0 else min(fill, opened)
                    close_at(fill)

            equity = cash + position * price
            sig = decide(ts, row, position, equity)
            action = sig.get("action") if sig else None

            if action in ("BUY", "SELL"):
                intent = decide_order(
                    self._snapshot(sig, position, price, cash, equity), self.cost_model
                )
                if intent.is_order and intent.closing:
                    close_at(price)
                elif intent.is_order:
                    position = intent.quantity if action == "BUY" else -intent.quantity
                    cash -= position * price
                    cash -= self._fill_cost(price, intent.quantity)
                    active_sl = sig.get("stop_loss_price")
                    active_tp = sig.get("take_profit_price")

            equity_curve.append({"timestamp": ts, "equity": cash + position * price})

        if position != 0:
            close_at(float(df["Close"].iloc[-1]))
            if equity_curve:
                equity_curve[-1]["equity"] = cash

        return self._evaluate_performance(equity_curve, round_trips, benchmark_df)

    def _evaluate_performance(
        self,
        equity_curve: List[Dict],
        round_trips: int,
        benchmark_df: pd.DataFrame = None,
    ) -> Dict[str, Any]:
        if not equity_curve:
            return self._empty_metrics()

        df_eq = pd.DataFrame(equity_curve).set_index("timestamp")
        df_eq["return"] = df_eq["equity"].pct_change().fillna(0)

        periods_per_year = self._infer_periods_per_year(df_eq.index)
        ann_factor = np.sqrt(periods_per_year)

        total_return_pct = (df_eq["equity"].iloc[-1] / self.initial_capital - 1) * 100

        bar_returns = df_eq["return"]
        std = bar_returns.std()
        sharpe_ratio = (bar_returns.mean() / std * ann_factor) if std > 0 else 0.0

        downside = bar_returns[bar_returns < 0]
        downside_std = downside.std() * ann_factor if len(downside) > 1 else 0.0
        sortino_ratio = (
            (bar_returns.mean() * periods_per_year / downside_std)
            if downside_std > 0
            else 0.0
        )

        gains = bar_returns[bar_returns > 0].sum()
        losses = abs(bar_returns[bar_returns < 0].sum())
        profit_factor = (gains / losses) if losses > 0 else float("inf")

        roll_max = df_eq["equity"].cummax()
        drawdown = df_eq["equity"] / roll_max - 1
        max_drawdown_pct = abs(drawdown.min()) * 100

        tracking_error = 0.0
        information_ratio = 0.0

        if (
            benchmark_df is not None
            and not benchmark_df.empty
            and "Close" in benchmark_df.columns
        ):
            bench_returns = benchmark_df["Close"].pct_change().fillna(0)
            aligned = pd.concat(
                [bar_returns, bench_returns], axis=1, join="inner"
            ).dropna()
            if len(aligned) > 1:
                active_returns = aligned.iloc[:, 0] - aligned.iloc[:, 1]
                tracking_error = active_returns.std() * ann_factor
                if tracking_error > 0:
                    information_ratio = (
                        active_returns.mean() * periods_per_year
                    ) / tracking_error

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
            "final_equity": round(df_eq["equity"].iloc[-1], 2),
        }

    def _empty_metrics(self) -> Dict[str, Any]:
        return {
            "total_return_pct": 0.0,
            "sharpe_ratio": 0.0,
            "sortino_ratio": 0.0,
            "profit_factor": 0.0,
            "information_ratio": 0.0,
            "tracking_error_pct": 0.0,
            "max_drawdown_pct": 0.0,
            "total_trades": 0,
            "periods_per_year": 252.0,
            "final_equity": self.initial_capital,
        }
