# evaluation/backtest_engine.py
# 整合模擬交易與高階機構級績效評估

import pandas as pd
import numpy as np
from typing import Dict, Any, List

class BacktestEngine:
    """提供高保真度歷史回測，並自動結算機構級績效指標"""

    def __init__(self, initial_capital: float = 100000.0, commission_per_share: float = 0.005, slippage_pct: float = 0.0005):
        self.initial_capital = initial_capital
        self.commission = commission_per_share
        self.slippage_pct = slippage_pct

    def run(self, df: pd.DataFrame, signals: List[Dict[str, Any]], benchmark_df: pd.DataFrame = None) -> Dict[str, Any]:
        """
        執行回測 (支援做多/做空，並逐根 K 線計算 MTM 淨值)
        :param df: 歷史 K 線數據 (需包含 Close)
        :param signals: 策略生成的信號列表
        :param benchmark_df: (可選) 大盤數據 (如 SPY)，用於計算 Alpha、資訊比率等進階指標
        """
        if df.empty:
            return self._empty_metrics()

        capital = self.initial_capital
        position = 0
        equity_curve = []
        trades = []
        
        # 將信號轉換為字典，以 timestamp 為鍵，提升查詢速度
        signal_dict = {sig["timestamp"]: sig for sig in signals}

        # 模擬逐筆交易與逐根 K 線的 MTM (Mark-to-Market) 淨值結算
        for ts, row in df.iterrows():
            price = row['Close']
            sig = signal_dict.get(ts)

            if sig:
                action = sig["action"]
                target_position = 0
                
                # 決定目標倉位 (留 5% 資金作為保證金與滑價緩衝)
                if action == "BUY":
                    target_position = int((capital * 0.95) / price)
                elif action == "SELL":
                    target_position = -int((capital * 0.95) / price) # 允許做空
                    
                trade_size = target_position - position
                
                if trade_size != 0:
                    # 買入時滑價讓成本變高，賣出時滑價讓收入變少
                    if trade_size > 0: 
                        cost = trade_size * price * (1 + self.slippage_pct)
                    else:
                        cost = trade_size * price * (1 - self.slippage_pct)
                        
                    # 更新現金
                    capital -= cost
                    capital -= abs(trade_size) * self.commission
                    position = target_position
                    
                    trades.append({"type": action, "price": price, "size": trade_size, "timestamp": ts})

            current_equity = capital + (position * price)
            equity_curve.append({"timestamp": ts, "equity": current_equity})

        # 回測結束強制平倉結算
        if position != 0:
            last_price = df['Close'].iloc[-1]
            capital += position * last_price - abs(position) * self.commission
            equity_curve[-1]["equity"] = capital

        return self._evaluate_performance(equity_curve, trades, benchmark_df)

    def _evaluate_performance(self, equity_curve: List[Dict], trades: List[Dict], benchmark_df: pd.DataFrame = None) -> Dict[str, Any]:
        """純 Python/Pandas 向量化計算機構級績效"""
        if not equity_curve:
            return self._empty_metrics()

        df_eq = pd.DataFrame(equity_curve).set_index("timestamp")
        df_eq['return'] = df_eq['equity'].pct_change().fillna(0)

        total_return_pct = (df_eq['equity'].iloc[-1] / self.initial_capital - 1) * 100
        
        # 基礎指標 (夏普比率)
        daily_returns = df_eq['return']
        sharpe_ratio = (daily_returns.mean() / daily_returns.std() * np.sqrt(252)) if daily_returns.std() > 0 else 0
        
        # 索提諾比率 (Sortino Ratio) - 僅懲罰下行風險
        downside_std = daily_returns[daily_returns < 0].std() * np.sqrt(252)
        sortino_ratio = (daily_returns.mean() * 252 / downside_std) if downside_std > 0 else 0
        
        # 獲利因子 (Profit Factor) - 總獲利 / 總虧損
        winning_days = daily_returns[daily_returns > 0].sum()
        losing_days = abs(daily_returns[daily_returns < 0].sum())
        profit_factor = (winning_days / losing_days) if losing_days > 0 else float('inf')

        # 最大回撤
        roll_max = df_eq['equity'].cummax()
        drawdown = df_eq['equity'] / roll_max - 1
        max_drawdown_pct = abs(drawdown.min()) * 100
        
        # 資訊比率 (Information Ratio) 與追蹤誤差 (Tracking Error)
        tracking_error = 0.0
        information_ratio = 0.0
        
        if benchmark_df is not None and not benchmark_df.empty:
            bench_returns = benchmark_df['Close'].pct_change().fillna(0)
            # 對齊日期索引
            aligned = pd.concat([daily_returns, bench_returns], axis=1, join='inner').dropna()
            if not aligned.empty:
                strat_ret = aligned.iloc[:, 0]
                bench_ret = aligned.iloc[:, 1]
                active_returns = strat_ret - bench_ret
                
                # 追蹤誤差 (主動風險)
                tracking_error = active_returns.std() * np.sqrt(252)
                # 資訊比率 (超額報酬 / 主動風險)
                if tracking_error > 0:
                    information_ratio = (active_returns.mean() * 252) / tracking_error

        return {
            "total_return_pct": round(total_return_pct, 2),
            "sharpe_ratio": round(sharpe_ratio, 2),
            "sortino_ratio": round(sortino_ratio, 2),
            "profit_factor": round(profit_factor, 2),
            "information_ratio": round(information_ratio, 2),
            "tracking_error_pct": round(tracking_error * 100, 2),
            "max_drawdown_pct": round(max_drawdown_pct, 2),
            "total_trades": len(trades) // 2,
            "final_equity": round(df_eq['equity'].iloc[-1], 2)
        }

    def _empty_metrics(self) -> Dict[str, Any]:
        return {
            "total_return_pct": 0.0, "sharpe_ratio": 0.0, 
            "sortino_ratio": 0.0, "profit_factor": 0.0,
            "information_ratio": 0.0, "tracking_error_pct": 0.0,
            "max_drawdown_pct": 0.0, "total_trades": 0, "final_equity": self.initial_capital
        }