# ibkrpy/strategy/core_strategy.py
# 核心策略 (支援多模型 Ensemble 與動態進場閾值)。
#
# 本版唯一的結構性改動：generate_signal 的每一條 return None 都會先寫入
# self.last_decision。舊版有 9 條靜默的 return None —— 策略天天在做決定，
# 但外界完全看不到它決定了什麼、為什麼，這正是「交易好像沒在動」的成因。

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
        self.entry_sigma_mult = self.config.get("entry_sigma_multiplier", 1.0)
        self.tp_capture_ratio = self.config.get("tp_capture_ratio", 0.8)
        self.min_edge_pct = self.config.get("min_edge_pct", 0.0025)
        self.sl_noise_floor_mult = self.config.get("sl_noise_floor_multiplier", 0.5)
        self.min_reward_risk = self.config.get("min_reward_risk_ratio", 1.0)

        # 最近一次決策的完整說明。TradingEngine 每個 tick 都會讀它並寫進日誌。
        self.last_decision: Dict[str, Any] = {}

    # ------------------------------------------------------------------
    # 決策紀錄
    # ------------------------------------------------------------------

    def _reject(self, code: str, detail: str, **fields) -> None:
        self.last_decision = {
            "symbol": self.symbol,
            "outcome": "HOLD",
            "reason_code": code,
            "detail": detail,
            **fields,
        }
        return None

    def _accept(self, signal: Dict[str, Any], **fields) -> Dict[str, Any]:
        self.last_decision = {
            "symbol": self.symbol,
            "outcome": signal["action"],
            "reason_code": "SIGNAL",
            "detail": "通過全部進場條件",
            **fields,
        }
        return signal

    def describe_last_decision(self) -> str:
        """把 last_decision 壓成一行可讀的日誌字串。"""
        d = self.last_decision
        if not d:
            return "尚未做出任何決策"

        parts = [f"{d.get('outcome', '?')}", f"[{d.get('reason_code', '?')}]", d.get("detail", "")]
        nums = []
        if d.get("predicted_pct") is not None:
            nums.append(f"預測 {d['predicted_pct'] * 100:+.2f}%")
        if d.get("threshold_pct") is not None:
            nums.append(f"門檻 ±{d['threshold_pct'] * 100:.2f}%")
        if d.get("sigma") is not None:
            nums.append(f"σ {d['sigma'] * 100:.2f}%")
        if d.get("reward_risk") is not None:
            nums.append(f"R:R {d['reward_risk']:.2f}")
        if d.get("regime"):
            nums.append(f"情境 {d['regime']}")
        if d.get("models"):
            nums.append(f"模型 {d['models']}")
        if nums:
            parts.append("| " + "·".join(nums))
        return " ".join(p for p in parts if p)

    # ------------------------------------------------------------------
    # Ensemble
    # ------------------------------------------------------------------

    def _calculate_ensemble_prediction(self, predictions: Dict[str, float],
                                       regime: MarketRegime) -> float:
        """動態權重 (Dynamic Weighting) 加上 MAD 極端值防護機制"""
        if not predictions:
            return 0.0

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

        self._last_filtered_models = sorted(filtered_predictions.keys())

        if regime in (MarketRegime.BULL_TREND, MarketRegime.BEAR_TREND):
            weights = {'LSTM': 0.45, 'Transformer': 0.45, 'ARIMA': 0.10}
        elif regime == MarketRegime.SIDEWAYS_VOLATILE:
            weights = {'LSTM': 0.33, 'Transformer': 0.33, 'ARIMA': 0.34}
        else:
            weights = {'LSTM': 0.15, 'Transformer': 0.15, 'ARIMA': 0.70}

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

    def _build_exit_geometry(self, expected_move: float, volatility: float,
                             regime: MarketRegime) -> Optional[Dict[str, float]]:
        """
        由「預測幅度」與「波動率」共同決定停利／停損距離 (皆為正的百分比)。
        回傳 None 表示該筆訊號應被放棄，並附上量化後的原因。
        """
        sl_mult = self.sl_multiplier
        tp_mult = self.tp_multiplier

        if regime == MarketRegime.SIDEWAYS_VOLATILE:
            sl_mult *= 1.2
            tp_mult *= 0.8

        tp_dist = expected_move * self.tp_capture_ratio
        tp_ceiling = volatility * tp_mult
        capped_by_ceiling = tp_ceiling > 0 and tp_ceiling < tp_dist
        if tp_ceiling > 0:
            tp_dist = min(tp_dist, tp_ceiling)
        floored_by_edge = tp_dist < self.min_edge_pct
        tp_dist = max(tp_dist, self.min_edge_pct)

        sl_dist = max(
            volatility * sl_mult,
            volatility * self.sl_noise_floor_mult,
            self.min_edge_pct,
        )

        if sl_dist <= 0 or tp_dist <= 0:
            return None

        reward_risk = tp_dist / sl_dist
        return {
            "tp_dist": tp_dist,
            "sl_dist": sl_dist,
            "reward_risk": reward_risk,
            "capped_by_ceiling": capped_by_ceiling,
            "floored_by_edge": floored_by_edge,
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
        純函數式信號生成。每一條退出路徑都會寫入 self.last_decision。

        注意：`volatility` 預期是「每根 K 線」的波動率 (已由 TradingEngine 去年化)。
        """
        regime_name = regime.name if hasattr(regime, "name") else str(regime)
        self._last_filtered_models = sorted((ensemble_predictions or {}).keys())

        if current_price <= 0:
            return self._reject("BAD_PRICE", f"現價無效 ({current_price})", regime=regime_name)

        if not np.isfinite(volatility) or volatility <= 0:
            return self._reject("BAD_VOLATILITY",
                                f"波動率無效 ({volatility})，GARCH 可能未訓練",
                                regime=regime_name)

        final_prediction = prediction
        if ensemble_predictions:
            final_prediction = self._calculate_ensemble_prediction(ensemble_predictions, regime)

        if final_prediction is None or not np.isfinite(final_prediction) or final_prediction <= 0:
            return self._reject("NO_PREDICTION", "無有效的 Ensemble 預測值",
                                regime=regime_name, sigma=volatility,
                                models=self._last_filtered_models)

        price_diff_pct = (final_prediction / current_price) - 1
        dynamic_threshold = max(self.min_pred_threshold, volatility * self.entry_sigma_mult)

        common = dict(
            regime=regime_name,
            sigma=volatility,
            predicted_pct=price_diff_pct,
            threshold_pct=dynamic_threshold,
            models=self._last_filtered_models,
        )

        if regime == MarketRegime.SIDEWAYS_QUIET and self.term == "short_term":
            return self._reject("REGIME_QUIET",
                                "短線策略在低波動盤整期不進場", **common)

        if abs(price_diff_pct) <= dynamic_threshold:
            return self._reject(
                "BELOW_THRESHOLD",
                f"預測幅度未達進場門檻 (差 {(dynamic_threshold - abs(price_diff_pct)) * 100:.2f} %)",
                **common,
            )

        action = "BUY" if price_diff_pct > 0 else "SELL"

        if (regime == MarketRegime.BEAR_TREND and action == "BUY") or \
           (regime == MarketRegime.BULL_TREND and action == "SELL"):
            return self._reject("COUNTER_TREND",
                                f"{action} 訊號與 {regime_name} 趨勢相反，已過濾", **common)

        geometry = self._build_exit_geometry(
            expected_move=abs(price_diff_pct), volatility=volatility, regime=regime,
        )
        if geometry is None:
            return self._reject("BAD_GEOMETRY", "停損／停利距離計算失敗", **common)

        if geometry["reward_risk"] < self.min_reward_risk:
            note = ""
            if geometry["capped_by_ceiling"]:
                note = f"（停利被 σ×{self.tp_multiplier} 的上限壓住）"
            return self._reject(
                "POOR_REWARD_RISK",
                f"風報比 {geometry['reward_risk']:.2f} 低於門檻 {self.min_reward_risk:.2f}{note}",
                reward_risk=geometry["reward_risk"], **common,
            )

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

        if abs(take_profit - current_price) < 0.01 or abs(stop_loss - current_price) < 0.01:
            return self._reject("PRICE_COLLAPSE",
                                "停損／停利四捨五入後與現價同檔，無法執行",
                                reward_risk=geometry["reward_risk"], **common)
        if stop_loss <= 0 or take_profit <= 0:
            return self._reject("NON_POSITIVE_EXIT", "停損／停利價位非正數",
                                reward_risk=geometry["reward_risk"], **common)

        signal = {
            "symbol": self.symbol,
            "action": action,
            "price": current_price,
            "predicted_price": round(float(final_prediction), 2),
            "stop_loss_price": stop_loss,
            "take_profit_price": take_profit,
            "prediction_diff_pct": round(price_diff_pct, 4),
            "entry_threshold_pct": round(dynamic_threshold, 4),
            "reward_risk_ratio": round(geometry["reward_risk"], 2),
            "regime": regime_name,
            "term": self.term,
        }
        return self._accept(signal, reward_risk=geometry["reward_risk"], **common)