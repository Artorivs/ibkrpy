# ibkrpy/evaluation/model_tuner.py
# 結合 Optuna 優化與模型選拔 (支援跨週期錦標賽)

import optuna
import pandas as pd
import numpy as np
import os
from typing import Dict, Any, List, Tuple
from .backtest_engine import BacktestEngine
import logging

logger = logging.getLogger(__name__)

optuna.logging.set_verbosity(optuna.logging.WARNING)

PROJECT_ROOT = os.path.dirname(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
)
DATA_DIR = os.path.join(PROJECT_ROOT, "data")


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

        # 虧損策略：仍依虧損程度排序，讓 TPE 知道往哪個方向走
        if sortino <= 0:
            mdd = perf.get("max_drawdown_pct", 100.0)
            return sortino - 1.0 - (mdd / 100.0)

        # 樣本數不足：線性遞減而非硬切，避免與虧損策略混為一談
        sample_penalty = 1.0
        if n_trades < 5:
            sample_penalty = max(0.05, n_trades / 5.0)

        # 2. 獲利因子 (Profit Factor)
        pf = perf.get("profit_factor", 0.0)
        pf_multiplier = (
            np.log(pf) if (1.0 < pf < 100) else (pf - 1.0 if pf <= 1.0 else 4.6)
        )

        # 3. 最大回撤懲罰
        mdd = perf.get("max_drawdown_pct", 100.0)
        mdd_penalty = max(0.1, 1.0 - (mdd / 100.0))

        # 4. 交易頻率懲罰
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

            # 止損與停利的探索區間，產生買賣跨度
            sl_mult = trial.suggest_float("volatility_stop_loss_multiplier", 0.5, 2.0)
            tp_mult = trial.suggest_float("volatility_take_profit_multiplier", 1.0, 3.0)

            config = {
                "min_prediction_threshold_pct": min_pred_pct,
                "volatility_stop_loss_multiplier": sl_mult,
                "volatility_take_profit_multiplier": tp_mult,
                "term": term,  # 動態使用外部傳入的競技週期
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

        try:
            os.makedirs(DATA_DIR, exist_ok=True)
            db_path = os.path.join(DATA_DIR, "optuna_study.db")
            storage_url = f"sqlite:///{db_path}"

            study_name = f"strategy_{symbol}_{term}"

            study = optuna.create_study(
                study_name=study_name,
                storage=storage_url,
                load_if_exists=True,
                direction="maximize",
            )

            study.optimize(objective, n_trials=n_trials, n_jobs=-1)

            return study.best_params, study.best_value
        except Exception as e:
            logger.warning(f"⚠️ Optuna 優化失敗: {e}")
            return {}, -999.0


# 註：原 optimize_hyperparameters 與 select_best_model 已移除。
#   - optimize_hyperparameters 的目標函式是 `-abs(look_back-60) + dropout*2`，
#     沒有訓練任何模型，回傳的「最佳超參數」恆為 look_back=60 加最大 dropout。
#   - select_best_model 對每個候選模型都套用同一組硬編碼的 simulated_perf，
#     分數完全相同，永遠回傳 candidate_models[0]。
#   兩者都未被 pipeline_manager 呼叫。留著只會讓人誤以為系統具備這些能力。
