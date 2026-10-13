# ibkrpy/evaluation/model_tuner.py
# 結合 Optuna 優化與模型選拔 (支援跨週期錦標賽)

import optuna
import pandas as pd
import numpy as np
from typing import Dict, Any, Tuple
from .backtest_engine import BacktestEngine
import logging

logger = logging.getLogger(__name__)

optuna.logging.set_verbosity(optuna.logging.WARNING)


class ModelTuner:
    """負責模型的超參數尋優與最終模型選拔"""

    def __init__(self, model_orchestrator, data_manager):
        self.models = model_orchestrator
        self.data = data_manager
        self.engine = BacktestEngine()

    def _calculate_composite_score(self, perf: Dict[str, Any]) -> float:
        """超越單一夏普指數的機構級複合評分 (Composite Score)"""
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
        """使用 Optuna 尋找最佳風控參數並回傳 (最佳參數, 複合評分)"""
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

        # [修正] 舊版用 n_jobs=-1 讓每個 CPU 核心各開一條執行緒，同時寫入同一個
        # SQLite 檔 (data/optuna_study.db)。SQLite 一次只容許一個寫入者，
        # 寫入排隊超過 5 秒就拋 "database is locked"，整個週期被記為 -999 而淘汰。
        # objective 是純 Python 迴圈，受 GIL 限制，多執行緒本來也快不了。
        #
        # 另外 load_if_exists=True 會把歷次重訓的 trial 累積在同一個 study，
        # best_params 可能來自用舊資料評分的舊 trial。改為每次重訓一個全新的
        # 記憶體內 study，只依本次資料選出參數。
        try:
            study = optuna.create_study(
                study_name=f"strategy_{symbol}_{term}",
                direction="maximize",
            )
            study.optimize(objective, n_trials=n_trials, n_jobs=1)

            return study.best_params, study.best_value
        except Exception as e:
            logger.warning(f"⚠️ Optuna 優化失敗: {e}")
            return {}, -999.0
