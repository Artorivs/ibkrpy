# ibkrpy/strategy/market_analyzer.py
# 負責跨資產的相關性分析、產業板塊動能、大盤宏觀狀態評估與投資組合權重最佳化

import logging
import pandas as pd
import numpy as np
from typing import Dict, Any, List
from ibkrpy.shared.db_manager import DatabaseManager
from ibkrpy.shared.config_manager import ConfigManager

logger = logging.getLogger("ibkrpy")


class MarketAnalyzer:
    """
    機構級市場全局分析器 (Global Market Context)
    提取股票間的相關性矩陣、Beta 值、產業板塊資金流向、宏觀趨勢，並計算最佳化投資組合權重。
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
        計算並回傳全局市場狀態
        (建議在 trading_engine 的每一輪迴圈開頭呼叫一次，然後傳給各個標的)
        """
        context = {
            "correlation_matrix": {},  # 標的間的相關係數矩陣
            "beta_values": {},  # 各標的相對於大盤的 Beta 值 (系統性風險)
            "macro_trend": "NEUTRAL",  # 大盤短期趨勢
            "symbols": [],  # 納入本次計算的標的
            "optimal_weights": {},  # 投資組合最佳化目標權重 (總和為 1)
            "risk_parity_tilt": {},  # 相對傾斜倍數 (平均為 1.0)
            "is_valid": False,  # 數據是否足夠計算
        }

        symbols = [p.symbol for p in self.config.asset_profiles]
        if not symbols:
            return context

        # 1. 獲取所有標的近期的收盤價並對齊
        price_dict = {}
        for profile in self.config.asset_profiles:
            sym = profile.symbol
            context["symbols"].append(sym)

            df = self.db.get_market_data_sync(sym)
            if not df.empty and len(df) >= self.lookback_periods:
                price_dict[sym] = df["Close"].tail(self.lookback_periods)

        if not price_dict:
            return context

        # 2. 組裝成 DataFrame 並計算對數報酬率
        prices_df = pd.DataFrame(price_dict).ffill().dropna()
        if len(prices_df) < 10:
            logger.warning(
                f"全局分析資料不足 (對齊後僅 {len(prices_df)} 列，需 >= 10)，"
                f"本輪相關性 / Beta / 風險平價全部退回預設值。"
            )
            return context

        returns_df = np.log(prices_df / prices_df.shift(1)).dropna()
        context["is_valid"] = True

        # 3. 計算相關係數矩陣
        corr_matrix = returns_df.corr()
        context["correlation_matrix"] = corr_matrix.to_dict()

        # 4. 評估大盤宏觀趨勢與 Beta 值
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

        # 6. 投資組合最佳化: 基於風險平價 (Risk Parity / Inverse Variance)
        # 讓波動大的股票權重小，波動小的股票權重大，實現整體 Portfolio 波動率最小化與夏普最大化
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
        針對單一股票，結合全局上下文進行橫截面分析 (Cross-Sectional Analysis)
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

        # 目標部位 = 基準大小 × 風險平價傾斜。低波動標的拿到較大部位，
        # 高波動標的較小，但兩者都圍繞在一個明確、可設定的基準值附近。
        tilt = context.get("risk_parity_tilt", {}).get(symbol, 1.0)
        analysis["risk_parity_tilt"] = tilt
        analysis["target_weight"] = self.base_position_pct * tilt

        # --- A. 大盤宏觀對齊與 Beta 風險分析 ---
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
            analysis["warnings"].append("逆勢警告: 大盤處於上升趨勢，做空風險較高。")
        elif macro_trend == "BEARISH" and action == "BUY":
            analysis["conviction_multiplier"] -= 0.15
            analysis["warnings"].append("逆勢警告: 大盤處於下降趨勢，做多勝率較低。")

            if beta > 1.3:
                analysis["conviction_multiplier"] -= 0.1
                analysis["warnings"].append(
                    f"高 Beta 警告: 大盤偏空且該標的 Beta 極高 ({beta:.2f})，跌幅可能超越大盤。"
                )
            elif beta < 0.8:
                analysis["conviction_multiplier"] += 0.1
                analysis["warnings"].append(
                    f"防禦屬性: 該標的 Beta 較低 ({beta:.2f})，具備一定的抗跌能力。"
                )

        # --- B. 投資組合過度集中風險 (Portfolio Concentration Risk) ---
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
                    f"集中度風險: 與當前持倉 {highly_correlated_peers} 高度正相關 (Max R={max_corr_with_holdings:.2f})，將縮減倉位分散風險。"
                )

        analysis["conviction_multiplier"] = max(
            0.4, min(1.6, analysis["conviction_multiplier"])
        )

        return analysis
