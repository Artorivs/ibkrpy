# ibkrpy/evaluation/model_tuner.py
# Optuna strategy tuning and term selection

import optuna
import pandas as pd
import numpy as np
from typing import Dict, Any, Tuple
from .backtest_engine import BacktestEngine
import logging

logger = logging.getLogger(__name__)

optuna.logging.set_verbosity(optuna.logging.WARNING)


class ModelTuner:
    """Tunes strategy parameters and scores candidates."""

    def __init__(self, model_orchestrator, data_manager):
        self.models = model_orchestrator
        self.data = data_manager
        self.engine = BacktestEngine()

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

    def optimize_strategy_params(
        self,
        symbol: str,
        df: pd.DataFrame,
        precomputed_data: pd.DataFrame,
        n_trials: int = 50,
        term: str = "long_term",
    ) -> Tuple[Dict[str, Any], float]:
        """Tune risk parameters with Optuna; returns (best params, composite score)."""
        from ibkrpy.strategy.core_strategy import CoreStrategy
        from ibkrpy.strategy.strategy_components import MarketRegime

        def objective(trial):
            min_pred_pct = trial.suggest_float(
                "min_prediction_threshold_pct", 0.001, 0.020
            )

            sl_mult = trial.suggest_float("volatility_stop_loss_multiplier", 0.5, 2.0)
            tp_mult = trial.suggest_float("volatility_take_profit_multiplier", 1.0, 3.0)

            config = {
                "min_prediction_threshold_pct": min_pred_pct,
                "volatility_stop_loss_multiplier": sl_mult,
                "volatility_take_profit_multiplier": tp_mult,
                "term": term,
            }
            strategy = CoreStrategy(symbol, config)
            signals = []

            for row in precomputed_data.itertuples(index=True):
                regime_name = getattr(row, "regime", "SIDEWAYS_QUIET")
                regime = (
                    MarketRegime[regime_name]
                    if hasattr(MarketRegime, regime_name)
                    else MarketRegime.SIDEWAYS_QUIET
                )

                close = getattr(row, "Close")
                sig = strategy.generate_signal(
                    current_price=close,
                    prediction=getattr(row, "prediction", close),
                    volatility=getattr(row, "volatility", 0.02),
                    regime=regime,
                )
                if sig:
                    sig["timestamp"] = row.Index
                    signals.append(sig)

            perf = self.engine.run(df, signals)
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

            return study.best_params, study.best_value
        except Exception as e:
            logger.warning(f"⚠️ Optuna optimization failed: {e}")
            return {}, -999.0
