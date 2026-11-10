# ibkrpy/evaluation/model_tuner.py
# Optuna strategy tuning and term selection

import logging
from typing import Any, Dict, Optional, Tuple

import numpy as np
import optuna
import pandas as pd

from .backtest_engine import BacktestEngine

logger = logging.getLogger(__name__)

optuna.logging.set_verbosity(optuna.logging.WARNING)

# Parameters live reads per symbol from weights/global_best_params.json.
TUNED_PARAMS = {
    "volatility_stop_loss_multiplier": (0.5, 2.0),
    "volatility_take_profit_multiplier": (1.0, 3.0),
}

MIN_SEGMENT_ROWS = 30


class _DictConfig:
    """Minimal stand-in for ConfigManager.get (dot paths) when no config is given."""

    def __init__(self, data: Optional[dict] = None):
        self._data = data or {}

    def get(self, dot_path: str, default: Any = None) -> Any:
        node: Any = self._data
        for part in dot_path.split("."):
            if not isinstance(node, dict) or part not in node:
                return default
            node = node[part]
        return node


class ModelTuner:
    """Tunes strategy parameters through the same strategy, threshold policy, sizing and
    costs as the live engine, and scores them on data the search never saw."""

    def __init__(self, model_orchestrator=None, data_manager=None, config=None):
        self.models = model_orchestrator
        self.data = data_manager
        self.config = config if config is not None else _DictConfig()

    def _calculate_composite_score(self, perf: Dict[str, Any]) -> float:
        """Composite score built on Sortino, profit factor, drawdown and trade count."""
        sortino = perf.get("sortino_ratio", 0.0)
        n_trades = perf.get("total_trades", 0)

        if sortino <= 0:
            mdd = perf.get("max_drawdown_pct", 100.0)
            return sortino - 1.0 - (mdd / 100.0)

        sample_penalty = 1.0
        if n_trades < 5:
            sample_penalty = max(0.05, n_trades / 5.0)

        pf = perf.get("profit_factor", 0.0)
        pf_multiplier = (
            np.log(pf) if (1.0 < pf < 100) else (pf - 1.0 if pf <= 1.0 else 4.6)
        )

        mdd = perf.get("max_drawdown_pct", 100.0)
        mdd_penalty = max(0.1, 1.0 - (mdd / 100.0))

        trade_penalty = 1.0
        if n_trades > 40:
            trade_penalty = 40.0 / n_trades

        return (
            sortino
            * (1.0 + pf_multiplier)
            * mdd_penalty
            * trade_penalty
            * sample_penalty
        )

    def _engine(self) -> BacktestEngine:
        from ibkrpy.strategy.execution_costs import build_execution_cost_model

        get = self.config.get
        return BacktestEngine(
            position_pct=float(get("strategy_settings.base_position_pct", 0.08)),
            allow_shorting=bool(get("strategy_settings.allow_shorting", False)),
            cost_model=build_execution_cost_model(self.config),
            min_trade_usd=float(get("strategy_settings.min_trade_usd", 500.0)),
            max_gross_exposure=float(get("strategy_settings.max_gross_exposure", 1.0)),
            safety_multiple=float(get("execution_cost_settings.safety_multiple", 1.5)),
        )

    def evaluate(
        self,
        symbol: str,
        rows: pd.DataFrame,
        params: Dict[str, Any],
        term: str,
        warmup: Optional[pd.DataFrame] = None,
    ) -> Dict[str, Any]:
        """
        Backtest one parameter set the way live would trade it.

        rows needs Close (High/Low/Open for brackets), prediction, volatility and regime,
        and may carry reversal_risk; each prediction is made at that bar's close for the
        next bar. Orders go through decide_order with the live regime size scale. warmup
        rows only seed the prediction history used by the quantile threshold, as live
        history would.
        """
        from ibkrpy.strategy.core_strategy import CoreStrategy
        from ibkrpy.strategy.prediction_calibrator import (
            InMemoryPredictionHistory,
            build_threshold_policy,
        )
        from ibkrpy.strategy.order_intent import (
            cost_floor_pct,
            is_reducing,
            regime_size_scale,
        )
        from ibkrpy.strategy.strategy_components import MarketRegime

        cfg = dict(self.config.get("strategy_settings") or {})
        cfg.update(params)
        cfg["term"] = term

        history = InMemoryPredictionHistory(
            maxlen=int(self.config.get("threshold_settings.history_length", 500))
        )
        if warmup is not None and not warmup.empty:
            for p, c in zip(warmup["prediction"], warmup["Close"]):
                if c and c > 0:
                    history.append(symbol, p / c - 1)

        strategy = CoreStrategy(
            symbol,
            cfg,
            threshold_policy=build_threshold_policy(self.config),
            prediction_history=history,
        )
        engine = self._engine()
        configured_floor = float(cfg.get("min_edge_pct", 0.0010))

        def decide(ts, row, position, equity):
            price = float(row["Close"])
            strategy.min_edge_pct = max(
                cost_floor_pct(
                    engine.cost_model,
                    price,
                    equity,
                    engine.position_pct,
                    engine.safety_multiple,
                ),
                configured_floor,
            )

            regime_name = row.get("regime", "SIDEWAYS_QUIET")
            regime = MarketRegime.__members__.get(
                str(regime_name), MarketRegime.SIDEWAYS_QUIET
            )
            reversal_risk = float(row.get("reversal_risk", 0.0) or 0.0)
            signal = strategy.generate_signal(
                current_price=price,
                prediction=float(row["prediction"]),
                volatility=float(row.get("volatility", 0.02)),
                regime=regime,
                current_position=position,
                reversal_risk=reversal_risk,
            )
            if signal:
                scale = regime_size_scale(
                    regime,
                    reversal_risk,
                    is_reducing(signal["action"], position),
                    self.config,
                )
                signal = {**signal, "target_weight": engine.position_pct * scale}
            return signal

        return engine.simulate(rows, decide)

    def optimize_strategy_params(
        self,
        symbol: str,
        df: pd.DataFrame,
        precomputed_data: pd.DataFrame,
        n_trials: int = 50,
        term: str = "long_term",
    ) -> Tuple[Dict[str, Any], float]:
        """
        Tune on the first half of the out-of-sample rows, then score the best parameters
        on the second half. Returns (best params, validation score).

        The validation score is what selects the term, so the choice between terms is not
        biased by the search that picked each term's parameters.
        """
        rows = precomputed_data.copy()
        for col in ("Open", "High", "Low"):
            if col not in rows.columns and col in df.columns:
                rows[col] = df[col].reindex(rows.index)

        cut = len(rows) // 2
        tune, validate = rows.iloc[:cut], rows.iloc[cut:]
        if len(tune) < MIN_SEGMENT_ROWS or len(validate) < MIN_SEGMENT_ROWS:
            logger.warning(
                f"[{symbol}] {term}: {len(rows)} OOS rows is too few to split into "
                f"tune and validate (need {2 * MIN_SEGMENT_ROWS})."
            )
            return {}, -999.0

        def objective(trial):
            params = {
                name: trial.suggest_float(name, lo, hi)
                for name, (lo, hi) in TUNED_PARAMS.items()
            }
            perf = self.evaluate(symbol, tune, params, term)
            return self._calculate_composite_score(perf)

        # Fresh in-memory study per run, trials run serially: the objective is pure
        # Python (no gain from threads), and a shared SQLite study locked under
        # concurrent writers and mixed trials scored on older data.
        try:
            study = optuna.create_study(
                study_name=f"strategy_{symbol}_{term}",
                direction="maximize",
            )
            study.optimize(objective, n_trials=n_trials, n_jobs=1)
        except Exception as e:
            logger.warning(f"⚠️ Optuna optimization failed: {e}")
            return {}, -999.0

        best = study.best_params
        perf = self.evaluate(symbol, validate, best, term, warmup=tune)
        score = self._calculate_composite_score(perf)
        logger.info(
            f"[{symbol}] {term}: tune score {study.best_value:.2f}, "
            f"validation score {score:.2f} "
            f"({perf.get('total_trades', 0)} trades, Sortino {perf.get('sortino_ratio', 0.0):.2f})"
        )
        return best, score
