# ibkrpy/strategy/core_strategy.py
# Core strategy: ensemble prediction with a dynamic entry threshold

from typing import Dict, Any, Optional
import numpy as np
from .strategy_components import MarketRegime


class CoreStrategy:
    """Turns predictions into signals, adapting to the market regime. Accepts one prediction or an ensemble."""

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
        self.horizon_bars = int(self.config.get("horizon_bars", 1) or 1)
        self.min_pred_threshold = self.config.get("min_prediction_threshold_pct", 0.005)
        self.sl_multiplier = self.config.get("volatility_stop_loss_multiplier", 2.0)
        self.tp_multiplier = self.config.get("volatility_take_profit_multiplier", 3.0)
        self.entry_sigma_mult = self.config.get("entry_sigma_multiplier", 1.0)
        self.tp_capture_ratio = self.config.get("tp_capture_ratio", 0.8)
        self.min_edge_pct = self.config.get("min_edge_pct", 0.0005)
        self.sl_noise_floor_mult = self.config.get("sl_noise_floor_multiplier", 0.5)
        self.min_reward_risk = self.config.get("min_reward_risk_ratio", 1.0)

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

        self.last_decision: Dict[str, Any] = {}

    @property
    def history_key(self) -> str:
        """Prediction-history key. Moves over different horizons have different scales,
        so each horizon keeps its own history for the quantile threshold."""
        h = int(self.horizon_bars or 1)
        return self.symbol if h <= 1 else f"{self.symbol}@{h}"

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
            "detail": "passed all entry checks",
            **fields,
        }
        return signal

    def describe_last_decision(self) -> str:
        """Format last_decision as one readable log line."""
        d = self.last_decision
        if not d:
            return "no decision yet"

        parts = [
            f"{d.get('outcome', '?')}",
            f"[{d.get('reason_code', '?')}]",
            d.get("detail", ""),
        ]
        nums = []
        if d.get("predicted_pct") is not None:
            nums.append(f"pred {d['predicted_pct'] * 100:+.2f}%")
        if d.get("threshold_pct") is not None:
            t = d["threshold_pct"]
            src = d.get("threshold_source")
            label = (
                f"threshold ±{t * 100:.3f}%"
                if np.isfinite(t)
                else "threshold calibrating"
            )
            nums.append(f"{label}{f'({src})' if src else ''}")
        if d.get("sigma") is not None:
            nums.append(f"σ {d['sigma'] * 100:.2f}%")
        if d.get("reward_risk") is not None:
            nums.append(f"R:R {d['reward_risk']:.2f}")
        if d.get("regime"):
            nums.append(f"regime {d['regime']}")
        if d.get("models"):
            nums.append(f"models {d['models']}")
        if nums:
            parts.append("| " + " · ".join(nums))
        return " ".join(p for p in parts if p)

    MAD_MIN_MODELS = 5

    def _calculate_ensemble_prediction(
        self, predictions: Dict[str, float], regime: MarketRegime
    ) -> float:
        """Equal-weight mean of the models that pass the outlier filter and are not collapsed.

        Equal weights are the robust default for a few noisy forecasters (the forecast
        combination puzzle): estimated or hand-set weights rarely beat 1/N out of sample.
        """
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

        live = getattr(self, "_model_liveness", None) or {}
        alive = {k: v for k, v in filtered_predictions.items() if live.get(k, 1.0) > 0}
        if alive:
            filtered_predictions = alive

        self._last_filtered_models = sorted(filtered_predictions.keys())
        return float(np.mean(list(filtered_predictions.values())))

    def set_model_liveness(self, liveness: Dict[str, float]):
        """
        Injected by TradingEngine: 1.0 for a live model, 0.0 for one whose output has
        collapsed to a constant. Collapsed models are left out of the mean.
        """
        self._model_liveness = dict(liveness or {})

    def _build_exit_geometry(
        self, expected_move: float, volatility: float, regime: MarketRegime
    ) -> Optional[Dict[str, float]]:
        """
        Take-profit and stop-loss distances (positive fractions) from the predicted move and volatility.

        Take-profit comes from the predicted move, the stop from the target reward:risk,
        and volatility only sets a noise floor and a risk cap:

                tp_dist = clamp(predicted move × tp_capture, cost floor, σ × tp_mult)
                sl_dist = clamp(tp_dist / target R:R,  σ × noise_floor, σ × sl_mult)

        So reward:risk holds by construction, and only falls short when the noise floor pushes
        the stop out, i.e. the predicted edge is smaller than the noise to sit through. That case
        is rejected with reason code NOISE_EXCEEDS_EDGE.
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
        tp_dist = max(tp_dist, self.min_edge_pct)

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
        """Generate a signal (no side effects beyond last_decision).

        current_position separates closing from opening, which carry opposite risk; one set of
        filters for both would lock exits at turning points (see the counter-trend filter).

        reversal_risk (0-1) never blocks exits; it only raises the bar for new exposure.

        `volatility` is σ over the forecast horizon (per-bar σ × √horizon_bars), the same
        span as the predicted move.
        """
        regime_name = regime.name if hasattr(regime, "name") else str(regime)
        self._last_filtered_models = sorted((ensemble_predictions or {}).keys())

        if current_price <= 0:
            return self._reject(
                "BAD_PRICE", f"invalid price ({current_price})", regime=regime_name
            )

        if not np.isfinite(volatility) or volatility <= 0:
            return self._reject(
                "BAD_VOLATILITY",
                f"invalid volatility ({volatility}); GARCH may be untrained",
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
                "no valid ensemble prediction",
                regime=regime_name,
                sigma=volatility,
                models=self._last_filtered_models,
            )

        price_diff_pct = (final_prediction / current_price) - 1

        history_key = self.history_key
        if self.prediction_history is not None:
            self.prediction_history.append(history_key, price_diff_pct)

        history = (
            self.prediction_history.get(history_key)
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
                f"{decision.source}×{self.exit_threshold_ratio:g}(exit)"
            )
        elif reversal_risk >= self.reversal_caution_level:
            inflation = 1.0 + reversal_risk * self.reversal_threshold_boost
            dynamic_threshold *= inflation
            common["threshold_pct"] = dynamic_threshold
            common["threshold_source"] = (
                f"{decision.source}×{inflation:.2f}(reversal risk)"
            )

        if self.collapse_detector is not None and history:
            report = self.collapse_detector.inspect(history)
            if report.collapsed:
                return self._reject(
                    "MODEL_COLLAPSED",
                    f"model output collapsed to a constant ({report.detail}); not trading on it. Retrain.",
                    **common,
                )

        if regime == MarketRegime.SIDEWAYS_QUIET and self.term == "short_term":
            return self._reject(
                "REGIME_QUIET",
                "short-term strategy skips quiet sideways markets",
                **common,
            )

        if not np.isfinite(dynamic_threshold):
            return self._reject(
                "CALIBRATING",
                f"threshold still calibrating ({decision.detail}); trading starts once enough samples accumulate",
                **common,
            )

        if abs(price_diff_pct) <= dynamic_threshold:
            gap = (dynamic_threshold - abs(price_diff_pct)) * 100
            if decision.source == "cost_floor":
                return self._reject(
                    "EDGE_BELOW_COST",
                    f"predicted edge doesn't cover trading costs ({decision.detail})",
                    **common,
                )
            return self._reject(
                "BELOW_THRESHOLD",
                f"prediction below entry threshold (short by {gap:.3f} pp, source: {decision.source})",
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
                    f"top-reversal risk {reversal_risk:.0%} reached the block level "
                    f"{self.reversal_block_level:.0%}; no new exposure (exits unaffected)",
                    **common,
                )
            if regime == MarketRegime.BEAR_TREND and action == "BUY":
                return self._reject(
                    "COUNTER_TREND",
                    f"{action} adds exposure against {regime_name}; filtered",
                    **common,
                )
            if regime == MarketRegime.BULL_TREND and action == "SELL":
                return self._reject(
                    "COUNTER_TREND",
                    f"{action} adds exposure against {regime_name}; filtered",
                    **common,
                )

        geometry = self._build_exit_geometry(
            expected_move=abs(price_diff_pct),
            volatility=volatility,
            regime=regime,
        )
        if geometry is None:
            return self._reject(
                "BAD_GEOMETRY", "stop/target distance calculation failed", **common
            )

        if geometry["reward_risk"] < self.min_reward_risk:
            if geometry.get("noise_bound"):
                return self._reject(
                    "NOISE_EXCEEDS_EDGE",
                    f"stop pushed out to the noise floor {geometry['noise_floor'] * 100:.2f}% "
                    f"(σ×{self.sl_noise_floor_mult:g}) while the target is only {geometry['tp_dist'] * 100:.2f}%: "
                    f"predicted edge is smaller than the noise",
                    reward_risk=geometry["reward_risk"],
                    **common,
                )
            note = (
                " (target capped by σ ceiling)" if geometry["capped_by_ceiling"] else ""
            )
            return self._reject(
                "POOR_REWARD_RISK",
                f"reward:risk {geometry['reward_risk']:.2f} below {self.min_reward_risk:.2f}{note}",
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
                "stop/target round to the current price tick; cannot execute",
                reward_risk=geometry["reward_risk"],
                **common,
            )
        if stop_loss <= 0 or take_profit <= 0:
            return self._reject(
                "NON_POSITIVE_EXIT",
                "stop/target price not positive",
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
