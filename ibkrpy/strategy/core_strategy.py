# ibkrpy/strategy/core_strategy.py
# 核心策略 (支援多模型 Ensemble 與動態進場閾值)

from typing import Dict, Any, Optional
import numpy as np
from .strategy_components import MarketRegime


class CoreStrategy:
    """因市場情境而變，動靜皆合其宜的決策者。支援單一預測或多模型混合 (Ensemble)。"""

    def __init__(
        self,
        symbol: str,
        config: Dict[str, Any] = None,
        threshold_policy=None,
        prediction_history=None,
        collapse_detector=None,
    ):
        self.symbol = symbol
        self.config = dict(config) if config else {}

        self.term = self.config.get("term", "long_term")
        self.min_pred_threshold = self.config.get("min_prediction_threshold_pct", 0.005)
        self.sl_multiplier = self.config.get("volatility_stop_loss_multiplier", 2.0)
        self.tp_multiplier = self.config.get("volatility_take_profit_multiplier", 3.0)
        self.entry_sigma_mult = self.config.get("entry_sigma_multiplier", 1.0)
        self.tp_capture_ratio = self.config.get("tp_capture_ratio", 0.8)
        self.min_edge_pct = self.config.get("min_edge_pct", 0.0005)
        self.sl_noise_floor_mult = self.config.get("sl_noise_floor_multiplier", 0.5)
        self.min_reward_risk = self.config.get("min_reward_risk_ratio", 1.0)

        # --- 進出場的非對稱設定 ---
        self.exit_threshold_ratio = self.config.get("exit_threshold_ratio", 0.5)
        self.reversal_caution_level = self.config.get("reversal_caution_level", 0.5)
        self.reversal_threshold_boost = self.config.get("reversal_threshold_boost", 1.0)
        self.reversal_block_level = self.config.get("reversal_block_level", 0.75)

        if threshold_policy is None:
            from .prediction_calibrator import SigmaThresholdPolicy

            threshold_policy = SigmaThresholdPolicy(
                multiplier=self.entry_sigma_mult, static_floor=self.min_pred_threshold
            )
        self.threshold_policy = threshold_policy
        self.prediction_history = prediction_history
        self.collapse_detector = collapse_detector

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

        parts = [
            f"{d.get('outcome', '?')}",
            f"[{d.get('reason_code', '?')}]",
            d.get("detail", ""),
        ]
        nums = []
        if d.get("predicted_pct") is not None:
            nums.append(f"預測 {d['predicted_pct'] * 100:+.2f}%")
        if d.get("threshold_pct") is not None:
            t = d["threshold_pct"]
            src = d.get("threshold_source")
            label = f"門檻 ±{t * 100:.3f}%" if np.isfinite(t) else "門檻 校準中"
            nums.append(f"{label}{f'({src})' if src else ''}")
        if d.get("sigma") is not None:
            nums.append(f"σ {d['sigma'] * 100:.2f}%")
        if d.get("reward_risk") is not None:
            nums.append(f"R:R {d['reward_risk']:.2f}")
        if d.get("regime"):
            nums.append(f"情境 {d['regime']}")
        if d.get("models"):
            nums.append(f"模型 {d['models']}")
        if nums:
            parts.append("| " + " · ".join(nums))
        return " ".join(p for p in parts if p)

    # ------------------------------------------------------------------
    # Ensemble
    # ------------------------------------------------------------------

    MAD_MIN_MODELS = 5

    def _calculate_ensemble_prediction(
        self, predictions: Dict[str, float], regime: MarketRegime
    ) -> float:
        """動態權重 (Dynamic Weighting)，並依模型實際離散度調整話語權"""
        if not predictions:
            return 0.0

        filtered_predictions = dict(predictions)
        self._last_dropped = []

        if len(predictions) >= self.MAD_MIN_MODELS:
            pred_values = list(predictions.values())
            median_val = float(np.median(pred_values))
            mad = float(np.median([abs(p - median_val) for p in pred_values]))
            spread = float(np.max(pred_values) - np.min(pred_values))
            threshold = max(3.0 * mad, 0.25 * spread, 1e-9)

            kept = {
                k: v for k, v in predictions.items() if abs(v - median_val) <= threshold
            }
            if kept:
                self._last_dropped = sorted(set(predictions) - set(kept))
                filtered_predictions = kept

        self._last_filtered_models = sorted(filtered_predictions.keys())

        if regime in (MarketRegime.BULL_TREND, MarketRegime.BEAR_TREND):
            base_weights = {"LSTM": 0.45, "Transformer": 0.45, "ARIMA": 0.10}
        elif regime == MarketRegime.SIDEWAYS_VOLATILE:
            base_weights = {"LSTM": 0.33, "Transformer": 0.33, "ARIMA": 0.34}
        else:
            base_weights = {"LSTM": 0.15, "Transformer": 0.15, "ARIMA": 0.70}

        fallback_w = 1.0 / len(filtered_predictions)
        weights = {k: base_weights.get(k, fallback_w) for k in filtered_predictions}

        live = getattr(self, "_model_liveness", None)
        if live:
            adjusted = {
                k: w * max(float(live.get(k, 1.0)), 0.0) for k, w in weights.items()
            }
            if sum(adjusted.values()) > 1e-12:
                weights = adjusted

        total_weight = sum(weights.values())
        if total_weight <= 1e-12:
            return float(np.mean(list(filtered_predictions.values())))

        return (
            sum(filtered_predictions[k] * w for k, w in weights.items()) / total_weight
        )

    def set_model_liveness(self, liveness: Dict[str, float]):
        """
        由 TradingEngine 注入：每個模型的「存活度」(0~1)。
        1.0 = 輸出具備正常變異；接近 0 = 已塌陷成常數。
        """
        self._model_liveness = dict(liveness or {})

    # ------------------------------------------------------------------
    # 出場幾何
    # ------------------------------------------------------------------

    def _build_exit_geometry(
        self, expected_move: float, volatility: float, regime: MarketRegime
    ) -> Optional[Dict[str, float]]:
        """
        由「預測幅度」與「波動率」共同決定停利／停損距離 (皆為正的百分比)

        先由預測幅度決定停利，再由目標風報比反推停損，
        最後才用波動率設「雜訊下限」與「風險上限」兩道護欄。

            tp_dist = clamp(預測幅度 × tp_capture, 成本地板, σ × tp_mult)
            sl_dist = clamp(tp_dist / 目標風報比,  σ × noise_floor, σ × sl_mult)

        於是風報比在正常情況下「依構造成立」；只有當雜訊下限把停損頂開時才會不足，
        而那正是應該拒絕的情形 —— 預測出來的優勢小於必須忍受的雜訊。
        這種拒絕會帶著明確的原因碼 NOISE_EXCEEDS_EDGE 回報，而不是含糊的風報比不足。
        """
        sl_mult = self.sl_multiplier
        tp_mult = self.tp_multiplier

        if regime == MarketRegime.SIDEWAYS_VOLATILE:
            sl_mult *= 1.2
            tp_mult *= 0.8

        # --- 停利：錨定預測，上下都有護欄 ---
        tp_dist = expected_move * self.tp_capture_ratio
        tp_ceiling = volatility * tp_mult
        capped_by_ceiling = tp_ceiling > 0 and tp_ceiling < tp_dist
        if tp_ceiling > 0:
            tp_dist = min(tp_dist, tp_ceiling)
        tp_dist = max(tp_dist, self.min_edge_pct)

        # --- 停損：由目標風報比反推，再套雜訊下限與風險上限 ---
        target_rr = max(self.min_reward_risk, 1e-6)
        noise_floor = volatility * self.sl_noise_floor_mult
        risk_cap = max(volatility * sl_mult, noise_floor)

        sl_ideal = tp_dist / target_rr
        sl_dist = min(max(sl_ideal, noise_floor, self.min_edge_pct), risk_cap)

        if sl_dist <= 0 or tp_dist <= 0:
            return None

        reward_risk = tp_dist / sl_dist
        return {
            "tp_dist": tp_dist,
            "sl_dist": sl_dist,
            "reward_risk": reward_risk,
            "capped_by_ceiling": capped_by_ceiling,
            "noise_floor": noise_floor,
            "noise_bound": sl_dist > sl_ideal + 1e-12,
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
        current_position: float = 0.0,
        reversal_risk: float = 0.0,
    ) -> Optional[Dict[str, Any]]:
        """純函數式信號生成

        current_position 讓策略能區分「平倉」與「開倉」——兩者的風險性質
        相反，套用同一組過濾規則會在轉折點上鎖死出場 (見逆勢過濾處的說明)。

        reversal_risk 是 0~1 的頂部反轉風險。它不會阻擋出場，只會提高
        新增曝險的門檻——「現在減碼」永遠比「現在加碼」安全。

        注意：`volatility` 預期是「每根 K 線」的波動率 (已由 TradingEngine 去年化)
        """
        regime_name = regime.name if hasattr(regime, "name") else str(regime)
        self._last_filtered_models = sorted((ensemble_predictions or {}).keys())

        if current_price <= 0:
            return self._reject(
                "BAD_PRICE", f"現價無效 ({current_price})", regime=regime_name
            )

        if not np.isfinite(volatility) or volatility <= 0:
            return self._reject(
                "BAD_VOLATILITY",
                f"波動率無效 ({volatility})，GARCH 可能未訓練",
                regime=regime_name,
            )

        final_prediction = prediction
        if ensemble_predictions:
            final_prediction = self._calculate_ensemble_prediction(
                ensemble_predictions, regime
            )

        if (
            final_prediction is None
            or not np.isfinite(final_prediction)
            or final_prediction <= 0
        ):
            return self._reject(
                "NO_PREDICTION",
                "無有效的 Ensemble 預測值",
                regime=regime_name,
                sigma=volatility,
                models=self._last_filtered_models,
            )

        price_diff_pct = (final_prediction / current_price) - 1

        if self.prediction_history is not None:
            self.prediction_history.append(self.symbol, price_diff_pct)

        history = (
            self.prediction_history.get(self.symbol)
            if self.prediction_history is not None
            else []
        )

        from .prediction_calibrator import ThresholdContext

        decision = self.threshold_policy.compute(
            ThresholdContext(
                symbol=self.symbol,
                sigma=volatility,
                history=history,
                min_edge_pct=self.min_edge_pct,
                term=self.term,
            )
        )
        dynamic_threshold = decision.value

        common = dict(
            regime=regime_name,
            sigma=volatility,
            predicted_pct=price_diff_pct,
            threshold_pct=dynamic_threshold,
            threshold_source=decision.source,
            reversal_risk=reversal_risk,
            models=self._last_filtered_models,
        )

        holds_position = abs(current_position) > 0
        wants_to_reduce = (current_position > 0 and price_diff_pct < 0) or (
            current_position < 0 and price_diff_pct > 0
        )

        if holds_position and wants_to_reduce:
            dynamic_threshold *= self.exit_threshold_ratio
            common["threshold_pct"] = dynamic_threshold
            common["threshold_source"] = (
                f"{decision.source}×{self.exit_threshold_ratio:g}(出場)"
            )
        elif reversal_risk >= self.reversal_caution_level:
            # 盤頭跡象明顯時，新增曝險必須拿出更強的證據。
            inflation = 1.0 + reversal_risk * self.reversal_threshold_boost
            dynamic_threshold *= inflation
            common["threshold_pct"] = dynamic_threshold
            common["threshold_source"] = f"{decision.source}×{inflation:.2f}(反轉風險)"

        if self.collapse_detector is not None and history:
            report = self.collapse_detector.inspect(history)
            if report.collapsed:
                return self._reject(
                    "MODEL_COLLAPSED",
                    f"模型輸出已退化成常數 ({report.detail})，拒絕以此下單。請重新訓練。",
                    **common,
                )

        if regime == MarketRegime.SIDEWAYS_QUIET and self.term == "short_term":
            return self._reject(
                "REGIME_QUIET", "短線策略在低波動盤整期不進場", **common
            )

        if not np.isfinite(dynamic_threshold):
            return self._reject(
                "CALIBRATING",
                f"門檻尚在校準中 ({decision.detail})，樣本累積足夠後才會開始交易",
                **common,
            )

        if abs(price_diff_pct) <= dynamic_threshold:
            gap = (dynamic_threshold - abs(price_diff_pct)) * 100
            if decision.source == "cost_floor":
                return self._reject(
                    "EDGE_BELOW_COST",
                    f"預測優勢不足以覆蓋交易成本 ({decision.detail})",
                    **common,
                )
            return self._reject(
                "BELOW_THRESHOLD",
                f"預測幅度未達進場門檻 (差 {gap:.3f} 個百分點, 門檻來源: {decision.source})",
                **common,
            )

        action = "BUY" if price_diff_pct > 0 else "SELL"

        is_reducing = (current_position > 0 and action == "SELL") or (
            current_position < 0 and action == "BUY"
        )

        if not is_reducing:
            if reversal_risk >= self.reversal_block_level:
                return self._reject(
                    "REVERSAL_IMMINENT",
                    f"頂部反轉風險 {reversal_risk:.0%} 已達封鎖水位 "
                    f"{self.reversal_block_level:.0%}，暫停新增曝險 (出場不受此限)",
                    **common,
                )
            if regime == MarketRegime.BEAR_TREND and action == "BUY":
                return self._reject(
                    "COUNTER_TREND",
                    f"{action} 為新增曝險且與 {regime_name} 相反，已過濾",
                    **common,
                )
            if regime == MarketRegime.BULL_TREND and action == "SELL":
                return self._reject(
                    "COUNTER_TREND",
                    f"{action} 為新增曝險且與 {regime_name} 相反，已過濾",
                    **common,
                )

        geometry = self._build_exit_geometry(
            expected_move=abs(price_diff_pct),
            volatility=volatility,
            regime=regime,
        )
        if geometry is None:
            return self._reject("BAD_GEOMETRY", "停損／停利距離計算失敗", **common)

        if geometry["reward_risk"] < self.min_reward_risk:
            if geometry.get("noise_bound"):
                return self._reject(
                    "NOISE_EXCEEDS_EDGE",
                    f"停損被雜訊下限頂到 {geometry['noise_floor'] * 100:.2f}% "
                    f"(σ×{self.sl_noise_floor_mult:g})，而停利只有 {geometry['tp_dist'] * 100:.2f}% —— "
                    f"預測優勢小於必須忍受的雜訊",
                    reward_risk=geometry["reward_risk"],
                    **common,
                )
            note = "（停利被 σ 上限壓住）" if geometry["capped_by_ceiling"] else ""
            return self._reject(
                "POOR_REWARD_RISK",
                f"風報比 {geometry['reward_risk']:.2f} 低於門檻 {self.min_reward_risk:.2f}{note}",
                reward_risk=geometry["reward_risk"],
                **common,
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

        if (
            abs(take_profit - current_price) < 0.01
            or abs(stop_loss - current_price) < 0.01
        ):
            return self._reject(
                "PRICE_COLLAPSE",
                "停損／停利四捨五入後與現價同檔，無法執行",
                reward_risk=geometry["reward_risk"],
                **common,
            )
        if stop_loss <= 0 or take_profit <= 0:
            return self._reject(
                "NON_POSITIVE_EXIT",
                "停損／停利價位非正數",
                reward_risk=geometry["reward_risk"],
                **common,
            )

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
