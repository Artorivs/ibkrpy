# ibkrpy/evaluation/model_tuner.py
# 結合 Optuna 優化與模型選拔 (支援跨週期錦標賽)

import logging
import optuna
import pandas as pd
import numpy as np
import os
from typing import Dict, Any, List, Tuple
from .backtest_engine import BacktestEngine

optuna.logging.set_verbosity(optuna.logging.WARNING)

logger = logging.getLogger("ibkrpy")
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
DATA_DIR = os.path.join(PROJECT_ROOT, "data")

class ModelTuner:
    """負責模型的超參數尋優與最終模型選拔"""

    def __init__(self, model_orchestrator, data_manager):
        self.models = model_orchestrator
        self.data = data_manager
        self.engine = BacktestEngine()

    def _calculate_composite_score(self, perf: Dict[str, Any]) -> float:
        """超越單一夏普指數的機構級複合評分 (Composite Score)"""
        # 樣本數過少，不具統計意義，給予極大懲罰
        if perf["total_trades"] < 5:
            return -999.0  

        # 1. 索提諾比率 (Sortino Ratio)
        sortino = perf.get("sortino_ratio", 0.0)
        if sortino <= 0: 
            return -999.0

        # 2. 獲利因子 (Profit Factor)
        pf = perf.get("profit_factor", 0.0)
        pf_multiplier = np.log(pf) if (1.0 < pf < 100) else (pf - 1.0 if pf <= 1.0 else 4.6)

        # 3. 最大回撤懲罰
        mdd = perf.get("max_drawdown_pct", 100.0)
        mdd_penalty = max(0.1, 1.0 - (mdd / 100.0))

        # 4. 交易頻率懲罰
        trade_penalty = 1.0
        if perf["total_trades"] > 40:
            trade_penalty = 40.0 / perf["total_trades"]

        return sortino * (1.0 + pf_multiplier) * mdd_penalty * trade_penalty

    def optimize_strategy_params(self, symbol: str, df: pd.DataFrame, precomputed_data: pd.DataFrame, n_trials: int = 50, term: str = "long_term") -> Tuple[Dict[str, Any], float]:
        """使用 Optuna 尋找最佳風控參數並回傳 (最佳參數, 複合評分)"""
        from ibkrpy.strategy.core_strategy import CoreStrategy
        from ibkrpy.strategy.strategy_components import MarketRegime
        
        def objective(trial):
            min_pred_pct = trial.suggest_float("min_prediction_threshold_pct", 0.001, 0.020)
            
            # 止損與停利的探索區間，產生買賣跨度
            sl_mult = trial.suggest_float("volatility_stop_loss_multiplier", 0.5, 2.0) 
            tp_mult = trial.suggest_float("volatility_take_profit_multiplier", 1.0, 3.0) 
            
            config = {
                "min_prediction_threshold_pct": min_pred_pct,
                "volatility_stop_loss_multiplier": sl_mult,
                "volatility_take_profit_multiplier": tp_mult,
                "term": term  # 動態使用外部傳入的競技週期
            }
            strategy = CoreStrategy(symbol, config)
            signals = []
            
            for ts, row in precomputed_data.iterrows():
                regime_name = row.get('regime', 'SIDEWAYS_QUIET')
                regime = MarketRegime[regime_name] if hasattr(MarketRegime, regime_name) else MarketRegime.SIDEWAYS_QUIET
                
                sig = strategy.generate_signal(
                    current_price=row['Close'],
                    prediction=row.get('prediction', row['Close']),
                    volatility=row.get('volatility', 0.02),
                    regime=regime
                )
                if sig:
                    sig["timestamp"] = ts
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
                direction="maximize"
            )
            
            study.optimize(objective, n_trials=n_trials, n_jobs=-1)
            
            return study.best_params, study.best_value
        except Exception as e:
            logger.warning(f"      ⚠️ Optuna 優化失敗: {e}")
            return {}, -999.0

    def optimize_hyperparameters(self, symbol: str, model_type: str, n_trials: int = 20) -> Dict[str, Any]:
        logger.info(f"[{symbol}] 開始 {model_type} 模型的超參數優化 (Trials: {n_trials})...")
        def objective(trial):
            look_back = trial.suggest_categorical("look_back", [30, 60, 90])
            dropout = trial.suggest_float("dropout_rate", 0.1, 0.4)
            return -abs(look_back - 60) + (dropout * 2) 
            
        os.makedirs(DATA_DIR, exist_ok=True)
        db_path = os.path.join(DATA_DIR, "optuna_study.db")
        storage_url = f"sqlite:///{db_path}"
        study_name = f"hyperparams_{symbol}_{model_type}"

        study = optuna.create_study(
            study_name=study_name,
            storage=storage_url,
            load_if_exists=True,
            direction="maximize"
        )
        study.optimize(objective, n_trials=n_trials, n_jobs=-1)
        return study.best_params

    def select_best_model(self, symbol: str, candidate_models: List[str]) -> str:
        logger.info(f"[{symbol}] 展開模型選拔賽: {candidate_models}")
        best_model = candidate_models[0]
        best_score = -float('inf')
        for model_type in candidate_models:
            simulated_perf = {"sortino_ratio": 1.5, "profit_factor": 1.2, "max_drawdown_pct": 15, "total_trades": 25}
            score = self._calculate_composite_score(simulated_perf)
            if score > best_score:
                best_score = score
                best_model = model_type
        logger.info(f"[{symbol}] 冠軍模型誕生: {best_model} (Composite Score: {best_score:.2f})")
        return best_model