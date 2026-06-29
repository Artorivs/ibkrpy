# ibkrpy/strategy/core_strategy.py
# 核心策略 (支援多模型 Ensemble 與動態進場閾值)，專注於信號生成與情境應對

from typing import Dict, Any, Optional
import numpy as np
from .strategy_components import MarketRegime

class CoreStrategy:
    """因市場情境而變，動靜皆合其宜的決策者。支援單一預測或多模型混合 (Ensemble)。"""

    def __init__(self, symbol: str, config: Dict[str, Any] = None):
        self.symbol = symbol
        self.config = config or {}
        
        # 簡化參數讀取
        self.term = self.config.get("term", "long_term") # 預設為長線
        self.min_pred_threshold = self.config.get("min_prediction_threshold_pct", 0.005)
        self.sl_multiplier = self.config.get("volatility_stop_loss_multiplier", 2.0)
        self.tp_multiplier = self.config.get("volatility_take_profit_multiplier", 3.0)

    def _calculate_ensemble_prediction(self, predictions: Dict[str, float], regime: MarketRegime) -> float:
        """動態權重 (Dynamic Weighting) 加上 MAD 極端值防護機制"""
        if not predictions:
            return 0.0

        # --- 1. 極端值防護 (Outlier Rejection using MAD) ---
        pred_values = list(predictions.values())
        filtered_predictions = {}
        
        if len(pred_values) >= 3:
            median_val = np.median(pred_values)
            abs_deviations = [abs(p - median_val) for p in pred_values]
            mad = np.median(abs_deviations)
            threshold = 3.0 * mad if mad > 0 else 1e-5
            
            for model_name, pred_val in predictions.items():
                if abs(pred_val - median_val) <= threshold or mad == 0.0:
                    filtered_predictions[model_name] = pred_val
                else:
                    print(f"[{self.symbol}] 🛡️ 剔除 {model_name} 的極端預測值 ({pred_val:.2f})")
        else:
            filtered_predictions = predictions

        if not filtered_predictions:
            return 0.0

        # --- 2. 依據市場情境分配動態權重 ---
        weights = {}
        if regime in [MarketRegime.BULL_TREND, MarketRegime.BEAR_TREND]:
            weights = {'LSTM': 0.45, 'Transformer': 0.45, 'ARIMA': 0.10}
        elif regime == MarketRegime.SIDEWAYS_VOLATILE:
            weights = {'LSTM': 0.33, 'Transformer': 0.33, 'ARIMA': 0.34}
        else:
            weights = {'LSTM': 0.15, 'Transformer': 0.15, 'ARIMA': 0.70}
            
        # --- 3. 計算過濾後的加權平均 ---
        total_weight = 0.0
        weighted_sum = 0.0
        
        for model_name, pred_val in filtered_predictions.items():
            w = weights.get(model_name, 1.0 / len(filtered_predictions))
            weighted_sum += pred_val * w
            total_weight += w
            
        return weighted_sum / total_weight if total_weight > 0 else 0.0

    def generate_signal(
        self,
        current_price: float,
        prediction: float = None,
        volatility: float = 0.02,
        regime: MarketRegime = MarketRegime.SIDEWAYS_QUIET,
        ensemble_predictions: Dict[str, float] = None
    ) -> Optional[Dict[str, Any]]:
        """純函數式信號生成，支援長短期邏輯鑑別。"""
        if current_price <= 0 or not np.isfinite(volatility):
            return None

        final_prediction = prediction
        if ensemble_predictions:
            final_prediction = self._calculate_ensemble_prediction(ensemble_predictions, regime)
            print(f"[{self.symbol}] 觸發混合模型 (Ensemble)，綜合預測價格為: {final_prediction:.2f}")
            
        if final_prediction is None or not np.isfinite(final_prediction):
            return None

        # 盤整靜默期過濾：長線策略容忍度高，可忽視短期靜默；短線則嚴格拒絕
        if regime == MarketRegime.SIDEWAYS_QUIET and self.term == "short_term":
            return None

        # 動態進場閾值：短線易受雜訊干擾，需放大波動率懲罰 (x5)；長線看大趨勢，波動率懲罰降低 (x2)
        vol_penalty = 5 if self.term == "short_term" else 2
        dynamic_threshold = self.min_pred_threshold * (1 + volatility * vol_penalty)
        
        price_diff_pct = (final_prediction / current_price) - 1
        action = "HOLD"
        
        print(f"[{self.symbol}] 策略診斷 ({self.term}): 預期漲跌幅 {price_diff_pct*100:.3f}% | 動態進場閾值 {dynamic_threshold*100:.3f}%")
        
        if price_diff_pct > dynamic_threshold:
            action = "BUY"
        elif price_diff_pct < -dynamic_threshold:
            action = "SELL"

        # 逆勢過濾
        if (regime == MarketRegime.BEAR_TREND and action == "BUY") or \
           (regime == MarketRegime.BULL_TREND and action == "SELL"):
            return None

        if action == "HOLD":
            return None

        sl_mult, tp_mult = self.sl_multiplier, self.tp_multiplier
        if regime == MarketRegime.SIDEWAYS_VOLATILE:
            sl_mult *= 1.2
            tp_mult *= 0.8

        if action == "BUY":
            stop_loss = current_price * (1 - volatility * sl_mult)
            take_profit = current_price * (1 + volatility * tp_mult)
        else:
            stop_loss = current_price * (1 + volatility * sl_mult)
            take_profit = current_price * (1 - volatility * tp_mult)

        return {
            "symbol": self.symbol,
            "action": action,
            "price": current_price,
            "stop_loss_price": round(stop_loss, 2),
            "take_profit_price": round(take_profit, 2),
            "prediction_diff_pct": round(price_diff_pct, 4),
            "regime": regime.name,
            "term": self.term
        }