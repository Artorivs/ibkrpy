# ibkrpy/strategy/market_analyzer.py
# Cross-asset correlation, beta, macro trend and risk-parity weights

import logging
import pandas as pd
import numpy as np
from typing import Dict, Any
from ibkrpy.shared.db_manager import DatabaseManager
from ibkrpy.shared.config_manager import ConfigManager

logger = logging.getLogger("ibkrpy")


class MarketAnalyzer:
    """
    Global market context:
    correlation matrix, betas, macro trend and risk-parity position weights.
    """

    def __init__(self, db_manager: DatabaseManager, config_manager: ConfigManager):
        self.db = db_manager
        self.config = config_manager
        self.lookback_periods = config_manager.get(
            "general_settings.analyzer_lookback", 60
        )
        self.benchmark_symbol = config_manager.get(
            "general_settings.benchmark_symbol", "QQQ"
        )

        self.base_position_pct = float(
            config_manager.get("strategy_settings.base_position_pct", 0.08)
        )
        self.min_tilt = float(
            config_manager.get("strategy_settings.min_risk_parity_tilt", 0.5)
        )
        self.max_tilt = float(
            config_manager.get("strategy_settings.max_risk_parity_tilt", 2.0)
        )

    def get_global_context(self) -> Dict[str, Any]:
        """
        Compute the global market context.
        (Call once at the start of each trading loop and pass it to every symbol.)
        """
        context = {
            "correlation_matrix": {},
            "beta_values": {},
            "macro_trend": "NEUTRAL",
            "symbols": [],
            "optimal_weights": {},
            "risk_parity_tilt": {},
            "is_valid": False,
        }

        symbols = [p.symbol for p in self.config.asset_profiles]
        if not symbols:
            return context

        price_dict = {}
        for profile in self.config.asset_profiles:
            sym = profile.symbol
            context["symbols"].append(sym)

            df = self.db.get_market_data_sync(sym)
            if not df.empty and len(df) >= self.lookback_periods:
                price_dict[sym] = df["Close"].tail(self.lookback_periods)

        if not price_dict:
            return context

        prices_df = pd.DataFrame(price_dict).ffill().dropna()
        if len(prices_df) < 10:
            logger.warning(
                f"Not enough data for global analysis ({len(prices_df)} aligned rows, need >= 10); "
                f"correlation / beta / risk parity fall back to defaults this loop."
            )
            return context

        returns_df = np.log(prices_df / prices_df.shift(1)).dropna()
        context["is_valid"] = True

        corr_matrix = returns_df.corr()
        context["correlation_matrix"] = corr_matrix.to_dict()

        if self.benchmark_symbol in prices_df.columns:
            benchmark_cum_ret = (
                prices_df[self.benchmark_symbol].iloc[-1]
                / prices_df[self.benchmark_symbol].iloc[0]
            ) - 1
            if benchmark_cum_ret > 0.003:
                context["macro_trend"] = "BULLISH"
            elif benchmark_cum_ret < -0.003:
                context["macro_trend"] = "BEARISH"

            bench_var = returns_df[self.benchmark_symbol].var()
            if bench_var > 0:
                for sym in returns_df.columns:
                    if sym != self.benchmark_symbol:
                        cov = returns_df[sym].cov(returns_df[self.benchmark_symbol])
                        context["beta_values"][sym] = round(cov / bench_var, 3)
                    else:
                        context["beta_values"][sym] = 1.0

        variances = returns_df.var()
        if not variances.empty:
            variances = variances.replace(0, 1e-6)
            inv_variances = 1.0 / variances
            weights = inv_variances / inv_variances.sum()
            context["optimal_weights"] = weights.to_dict()

            mean_weight = float(weights.mean())
            if mean_weight > 0:
                tilts = (weights / mean_weight).clip(
                    lower=self.min_tilt, upper=self.max_tilt
                )
                context["risk_parity_tilt"] = tilts.to_dict()

        return context

    def analyze_stock_risk(
        self,
        symbol: str,
        context: Dict[str, Any],
        action: str = "BUY",
        current_positions: Dict[str, float] = None,
    ) -> Dict[str, Any]:
        """
        Cross-sectional analysis of one symbol against the global context.
        """
        analysis = {
            "macro_alignment": "NEUTRAL",
            "benchmark_correlation": 0.0,
            "beta": 1.0,
            "conviction_multiplier": 1.0,
            "target_weight": self.base_position_pct,
            "warnings": [],
        }

        if not context.get("is_valid") or symbol not in context.get("symbols", []):
            return analysis

        tilt = context.get("risk_parity_tilt", {}).get(symbol, 1.0)
        analysis["risk_parity_tilt"] = tilt
        analysis["target_weight"] = self.base_position_pct * tilt

        macro_trend = context.get("macro_trend", "NEUTRAL")
        beta = context.get("beta_values", {}).get(symbol, 1.0)
        analysis["macro_alignment"] = macro_trend
        analysis["beta"] = beta

        if macro_trend == "BULLISH" and action == "BUY":
            analysis["conviction_multiplier"] += 0.15
        elif macro_trend == "BEARISH" and action == "SELL":
            analysis["conviction_multiplier"] += 0.15
        elif macro_trend == "BULLISH" and action == "SELL":
            analysis["conviction_multiplier"] -= 0.15
            analysis["warnings"].append(
                "Counter-trend: market is rising; shorting is riskier."
            )
        elif macro_trend == "BEARISH" and action == "BUY":
            analysis["conviction_multiplier"] -= 0.15
            analysis["warnings"].append(
                "Counter-trend: market is falling; longs win less often."
            )

            if beta > 1.3:
                analysis["conviction_multiplier"] -= 0.1
                analysis["warnings"].append(
                    f"High beta: bearish market and this symbol's beta is {beta:.2f}; it may fall more than the market."
                )
            elif beta < 0.8:
                analysis["conviction_multiplier"] += 0.1
                analysis["warnings"].append(
                    f"Defensive: low beta ({beta:.2f}), likely to fall less than the market."
                )

        corr_matrix = context.get("correlation_matrix", {})

        if current_positions and symbol in corr_matrix:
            max_corr_with_holdings = 0.0
            highly_correlated_peers = []

            for pos_sym, pos_qty in current_positions.items():
                if (
                    pos_qty != 0
                    and pos_sym != symbol
                    and pos_sym in corr_matrix[symbol]
                ):
                    corr = corr_matrix[symbol][pos_sym]

                    if (action == "BUY" and pos_qty > 0) or (
                        action == "SELL" and pos_qty < 0
                    ):
                        if corr > max_corr_with_holdings:
                            max_corr_with_holdings = corr
                        if corr > 0.75:
                            highly_correlated_peers.append(pos_sym)

            if highly_correlated_peers:
                analysis["conviction_multiplier"] -= 0.20
                analysis["warnings"].append(
                    f"Concentration: highly correlated with {highly_correlated_peers} (max r={max_corr_with_holdings:.2f}); reducing size."
                )

        analysis["conviction_multiplier"] = max(
            0.4, min(1.6, analysis["conviction_multiplier"])
        )

        return analysis
