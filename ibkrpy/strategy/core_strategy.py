# ibkrpy/strategy/core_strategy.py
# 核心策略 (支援多模型 Ensemble 與動態進場閾值)，專注於信號生成與情境應對

from typing import Dict, Any, Optional
import numpy as np
from .strategy_components import MarketRegime

class CoreStrategy:
    """因市場情境而變，動靜皆合其宜的決策者。支援單一預測或多模型混合 (Ensemble)。"""

    def __init__(self, symbol: str, config: Dict[str, Any] = None):
        self.symbol = symbol
        self.config = dict(config) if config else {}

        self.term = self.config.get("term", "long_term")
        self.min_pred_threshold = self.config.get("min_prediction_threshold_pct", 0.005)
        self.sl_multiplier = self.config.get("volatility_stop_loss_multiplier", 2.0)
        self.tp_multiplier = self.config.get("volatility_take_profit_multiplier", 3.0)

        # 進場門檻至少需達到 σ 的幾倍
        self.entry_sigma_mult = self.config.get("entry_sigma_multiplier", 1.0)

        # 停利要擷取預測漲跌幅的幾成
        self.tp_capture_ratio = self.config.get("tp_capture_ratio", 0.8)

        # 單邊最小獲利空間，需覆蓋手續費 + 滑價
        self.min_edge_pct = self.config.get("min_edge_pct", 0.0025)

        # 停損不得緊於 σ 的幾倍，避免被單根 K 線的正常雜訊掃出
        self.sl_noise_floor_mult = self.config.get("sl_noise_floor_multiplier", 0.5)

        # 可接受的最低風險報酬比
        self.min_reward_risk = self.config.get("min_reward_risk_ratio", 1.0)

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
            filtered_predictions = predictions

        if not filtered_predictions:
            return 0.0

        # --- 2. 依據市場情境分配動態權重 ---
        if regime in (MarketRegime.BULL_TREND, MarketRegime.BEAR_TREND):
            weights = {'LSTM': 0.45, 'Transformer': 0.45, 'ARIMA': 0.10}
        elif regime == MarketRegime.SIDEWAYS_VOLATILE:
            weights = {'LSTM': 0.33, 'Transformer': 0.33, 'ARIMA': 0.34}
        else:
            weights = {'LSTM': 0.15, 'Transformer': 0.15, 'ARIMA': 0.70}
            
        # --- 3. 計算過濾後的加權平均 ---
        total_weight = 0.0
        weighted_sum = 0.0
        fallback_w = 1.0 / len(filtered_predictions)

        for model_name, pred_val in filtered_predictions.items():
            w = weights.get(model_name, fallback_w)
            weighted_sum += pred_val * w
            total_weight += w
            
        return weighted_sum / total_weight if total_weight > 0 else 0.0

    # ------------------------------------------------------------------
    # 出場幾何
    # ------------------------------------------------------------------

    def _build_exit_geometry(
        self,
        expected_move: float,
        volatility: float,
        regime: MarketRegime,
    ) -> Optional[Dict[str, float]]:
        """
        由「預測幅度」與「波動率」共同決定停利／停損距離 (皆為正的百分比)。

        設計原則：
          1. 停利必須落在預測價之內 —— tp_dist = expected_move × tp_capture_ratio
          2. 停利必須大於交易成本 —— tp_dist >= min_edge_pct
          3. 停損由波動率決定，但不得緊於 σ × sl_noise_floor_multiplier
          4. 風險報酬比不足時回傳 None，代表「此訊號不值得執行」

        回傳 None 表示該筆訊號應被放棄。
        """
        sl_mult = self.sl_multiplier
        tp_mult = self.tp_multiplier

        # 高波動盤整：放寬停損、收緊停利
        if regime == MarketRegime.SIDEWAYS_VOLATILE:
            sl_mult *= 1.2
            tp_mult *= 0.8

        # --- 停利：錨定預測，並以波動率設一個上限 ---
        # tp_mult 在此的角色從「決定距離」降級為「相對於 σ 的硬上限」，
        # 避免模型出現異常大的預測時，停利被拉到不合理的位置。
        tp_dist = expected_move * self.tp_capture_ratio
        tp_ceiling = volatility * tp_mult
        if tp_ceiling > 0:
            tp_dist = min(tp_dist, tp_ceiling)
        tp_dist = max(tp_dist, self.min_edge_pct)

        # --- 停損：由波動率決定，設雜訊下限 ---
        sl_dist = max(
            volatility * sl_mult,
            volatility * self.sl_noise_floor_mult,
            self.min_edge_pct,
        )

        if sl_dist <= 0 or tp_dist <= 0:
            return None

        reward_risk = tp_dist / sl_dist
        if reward_risk < self.min_reward_risk:
            # 預測優勢小於必須承擔的波動 —— 放棄，而非硬掛一個永遠打不到的停利。
            return None

        return {
            "tp_dist": tp_dist,
            "sl_dist": sl_dist,
            "reward_risk": reward_risk,
        }

    # ------------------------------------------------------------------
    # 主入口
    # ------------------------------------------------------------------

    def generate_signal(
        self,
        current_price: float,
        prediction: float = None,
        volatility: float = 0.02,
        regime: MarketRegime = MarketRegime.SIDEWAYS_QUIET,
        ensemble_predictions: Dict[str, float] = None,
    ) -> Optional[Dict[str, Any]]:
        """
        純函數式信號生成，支援長短期邏輯鑑別。

        注意：`volatility` 預期是「每根 K 線」的波動率 (已由 TradingEngine 去年化)，
        而非年化波動率。傳入年化值會讓停損停利放大約 √252 倍。
        """
        if current_price <= 0 or not np.isfinite(volatility) or volatility <= 0:
            return None

        final_prediction = prediction
        if ensemble_predictions:
            final_prediction = self._calculate_ensemble_prediction(ensemble_predictions, regime)

        if final_prediction is None or not np.isfinite(final_prediction) or final_prediction <= 0:
            return None

        # 盤整靜默期過濾：長線策略容忍度高，短線則嚴格拒絕
        if regime == MarketRegime.SIDEWAYS_QUIET and self.term == "short_term":
            return None

        # 進場門檻
        dynamic_threshold = max(
            self.min_pred_threshold,
            volatility * self.entry_sigma_mult,
        )

        price_diff_pct = (final_prediction / current_price) - 1

        if price_diff_pct > dynamic_threshold:
            action = "BUY"
        elif price_diff_pct < -dynamic_threshold:
            action = "SELL"
        else:
            return None

        # 逆勢過濾
        if (regime == MarketRegime.BEAR_TREND and action == "BUY") or \
           (regime == MarketRegime.BULL_TREND and action == "SELL"):
            return None

        # --- 出場幾何 ---
        geometry = self._build_exit_geometry(
            expected_move=abs(price_diff_pct),
            volatility=volatility,
            regime=regime,
        )
        if geometry is None:
            return None

        tp_dist = geometry["tp_dist"]
        sl_dist = geometry["sl_dist"]

        if action == "BUY":
            stop_loss = current_price * (1 - sl_dist)
            take_profit = current_price * (1 + tp_dist)
        else:
            stop_loss = current_price * (1 + sl_dist)
            take_profit = current_price * (1 - tp_dist)

        stop_loss = round(stop_loss, 2)
        take_profit = round(take_profit, 2)

        # 四捨五入後若與現價塌陷成同一檔位 (低價股 / 極小距離)，該訊號無法執行
        if abs(take_profit - current_price) < 0.01 or abs(stop_loss - current_price) < 0.01:
            return None
        if stop_loss <= 0 or take_profit <= 0:
            return None

        return {
            "symbol": self.symbol,
            "action": action,
            "price": current_price,
            "predicted_price": round(float(final_prediction), 2),
            "stop_loss_price": stop_loss,
            "take_profit_price": take_profit,
            "prediction_diff_pct": round(price_diff_pct, 4),
            "entry_threshold_pct": round(dynamic_threshold, 4),
            "reward_risk_ratio": round(geometry["reward_risk"], 2),
            "regime": regime.name,
            "term": self.term,
        }