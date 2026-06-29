# ibkrpy/strategy/market_analyzer.py
# 負責跨資產的相關性分析、產業板塊動能、大盤宏觀狀態評估與投資組合權重最佳化

import pandas as pd
import numpy as np
from typing import Dict, Any, List
from ibkrpy.shared.db_manager import DatabaseManager
from ibkrpy.shared.config_manager import ConfigManager

class MarketAnalyzer:
    """
    機構級市場全局分析器 (Global Market Context)
    提取股票間的相關性矩陣、Beta 值、產業板塊資金流向、宏觀趨勢，並計算最佳化投資組合權重。
    """
    def __init__(self, db_manager: DatabaseManager, config_manager: ConfigManager):
        self.db = db_manager
        self.config = config_manager
        self.lookback_periods = 60  # 計算相關性與 Beta 使用的 K 線數量
        self.benchmark_symbol = "SPY"

    def get_global_context(self) -> Dict[str, Any]:
        """
        計算並回傳全局市場狀態 
        (建議在 trading_engine 的每一輪迴圈開頭呼叫一次，然後傳給各個標的)
        """
        context = {
            "correlation_matrix": {},   # 標的間的相關係數矩陣
            "beta_values": {},          # 各標的相對於大盤的 Beta 值 (系統性風險)
            "sector_momentum": {},      # 各產業板塊的短期平均報酬率
            "macro_trend": "NEUTRAL",   # 大盤短期趨勢
            "symbol_tags": {},          # 標的對應的產業標籤快取
            "optimal_weights": {},      # 投資組合最佳化目標權重
            "is_valid": False           # 數據是否足夠計算
        }
        
        symbols = [p.symbol for p in self.config.asset_profiles]
        if not symbols:
            return context

        # 1. 獲取所有標的近期的收盤價並對齊
        price_dict = {}
        for profile in self.config.asset_profiles:
            sym = profile.symbol
            context["symbol_tags"][sym] = profile.tags or ["General"]
            
            df = self.db.get_market_data_sync(sym)
            if not df.empty and len(df) >= self.lookback_periods:
                price_dict[sym] = df['Close'].tail(self.lookback_periods)

        if not price_dict:
            return context

        # 2. 組裝成 DataFrame 並計算對數報酬率
        prices_df = pd.DataFrame(price_dict).ffill().dropna()
        if len(prices_df) < 10: 
            return context
            
        returns_df = np.log(prices_df / prices_df.shift(1)).dropna()
        context["is_valid"] = True

        # 3. 計算相關係數矩陣
        corr_matrix = returns_df.corr()
        context["correlation_matrix"] = corr_matrix.to_dict()

        # 4. 評估大盤宏觀趨勢與 Beta 值
        if self.benchmark_symbol in prices_df.columns:
            spy_cum_ret = (prices_df[self.benchmark_symbol].iloc[-1] / prices_df[self.benchmark_symbol].iloc[0]) - 1
            if spy_cum_ret > 0.003:   
                context["macro_trend"] = "BULLISH"
            elif spy_cum_ret < -0.003: 
                context["macro_trend"] = "BEARISH"

            bench_var = returns_df[self.benchmark_symbol].var()
            if bench_var > 0:
                for sym in returns_df.columns:
                    if sym != self.benchmark_symbol:
                        cov = returns_df[sym].cov(returns_df[self.benchmark_symbol])
                        context["beta_values"][sym] = round(cov / bench_var, 3)
                    else:
                        context["beta_values"][sym] = 1.0

        # 5. 評估產業板塊資金動能
        sector_returns = {}
        for sym in returns_df.columns:
            sym_cum_ret = (prices_df[sym].iloc[-1] / prices_df[sym].iloc[0]) - 1
            tags = context["symbol_tags"].get(sym, ["General"])
            for tag in tags:
                if tag not in sector_returns:
                    sector_returns[tag] = []
                sector_returns[tag].append(sym_cum_ret)

        for tag, rets in sector_returns.items():
            context["sector_momentum"][tag] = np.mean(rets)

        # 6. 投資組合最佳化: 基於風險平價 (Risk Parity / Inverse Variance)
        # 讓波動大的股票權重小，波動小的股票權重大，實現整體 Portfolio 波動率最小化與夏普最大化
        variances = returns_df.var()
        if not variances.empty:
            # 避免變異數為 0 導致除以零錯誤
            variances = variances.replace(0, 1e-6)
            inv_variances = 1.0 / variances
            optimal_weights = (inv_variances / inv_variances.sum()).to_dict()
            context["optimal_weights"] = optimal_weights

        return context

    def analyze_stock_risk(self, symbol: str, context: Dict[str, Any], action: str = "BUY", current_positions: Dict[str, float] = None) -> Dict[str, Any]:
        """
        針對單一股票，結合全局上下文進行橫截面分析 (Cross-Sectional Analysis)
        """
        analysis = {
            "sector_health": "NEUTRAL",
            "macro_alignment": "NEUTRAL",
            "benchmark_correlation": 0.0,
            "beta": 1.0,
            "conviction_multiplier": 1.0,  
            "target_weight": 0.10,
            "warnings": []
        }
        
        if not context.get("is_valid") or symbol not in context["symbol_tags"]:
            return analysis

        # 提取投資組合最佳化目標權重
        analysis["target_weight"] = context.get("optimal_weights", {}).get(symbol, 0.10)

        # --- A. 產業板塊健康度分析 ---
        tags = context["symbol_tags"][symbol]
        sector_scores = [context["sector_momentum"].get(t, 0.0) for t in tags]
        avg_sector_score = np.mean(sector_scores) if sector_scores else 0.0
        
        if avg_sector_score > 0.005:
            analysis["sector_health"] = "STRONG"
            analysis["conviction_multiplier"] += 0.2 if action == "BUY" else -0.2
        elif avg_sector_score < -0.005:
            analysis["sector_health"] = "WEAK"
            analysis["conviction_multiplier"] += 0.2 if action == "SELL" else -0.2
            if action == "BUY":
                analysis["warnings"].append(f"逆風警告: {tags} 板塊資金正在流出。")

        # --- B. 大盤宏觀對齊與 Beta 風險分析 ---
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
                analysis["warnings"].append(f"高 Beta 警告: 大盤偏空且該標的 Beta 極高 ({beta:.2f})，跌幅可能超越大盤。")
            elif beta < 0.8:
                analysis["conviction_multiplier"] += 0.1 
                analysis["warnings"].append(f"防禦屬性: 該標的 Beta 較低 ({beta:.2f})，具備一定的抗跌能力。")

        # --- C. 投資組合過度集中風險 (Portfolio Concentration Risk) ---
        corr_matrix = context.get("correlation_matrix", {})
        
        if current_positions and symbol in corr_matrix:
            max_corr_with_holdings = 0.0
            highly_correlated_peers = []

            for pos_sym, pos_qty in current_positions.items():
                if pos_qty != 0 and pos_sym != symbol and pos_sym in corr_matrix[symbol]:
                    corr = corr_matrix[symbol][pos_sym]
                    
                    if (action == "BUY" and pos_qty > 0) or (action == "SELL" and pos_qty < 0):
                        if corr > max_corr_with_holdings:
                            max_corr_with_holdings = corr
                        if corr > 0.75:
                            highly_correlated_peers.append(pos_sym)

            if highly_correlated_peers:
                analysis["conviction_multiplier"] -= 0.20
                analysis["warnings"].append(
                    f"集中度風險: 與當前持倉 {highly_correlated_peers} 高度正相關 (Max R={max_corr_with_holdings:.2f})，將縮減倉位分散風險。"
                )

        analysis["conviction_multiplier"] = max(0.4, min(1.6, analysis["conviction_multiplier"]))
        
        return analysis