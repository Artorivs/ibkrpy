# ibkrpy/manager/model_orchestrator.py
# Manages model lifetime, caching and prediction dispatch, so weights load once.

import os
import pandas as pd
from typing import Tuple, Dict, Any, Optional
import logging

logger = logging.getLogger(__name__)


class ModelOrchestrator:
    """Manages LSTM, Transformer, ARIMA and GARCH instances and their lifetime."""

    _WEIGHT_FILE_PATTERNS = {
        "LSTM": "{symbol}_LSTM.keras",
        "Transformer": "{symbol}_Transformer.keras",
        "ARIMA": "{symbol}_classical.pkl",
        "GARCH": "{symbol}_classical.pkl",
    }

    _REQUIRES_TRAINED_WEIGHTS = {"LSTM", "Transformer"}

    def __init__(self, model_factory, weights_dir: str = None, data_pipeline=None):
        """
        :param model_factory: factory with create_model(model_type)
        :param weights_dir:   weights directory; defaults to factory.weights_dir
        :param data_pipeline: optional DataPipeline. When given, reloading weights also
                                                    clears its cached training artifacts, so a new model is never
                                                    paired with an old scaler (which decodes to the wrong price range).
        """
        self.factory = model_factory
        self.pipeline = data_pipeline

        self.weights_dir = weights_dir or getattr(
            model_factory, "weights_dir", "weights"
        )

        self._loaded_models: Dict[str, Any] = {}
        self._loaded_mtimes: Dict[str, float] = {}

    def _weight_path(self, symbol: str, model_type: str) -> Optional[str]:
        pattern = self._WEIGHT_FILE_PATTERNS.get(model_type)
        if not pattern:
            return None
        return os.path.join(self.weights_dir, pattern.format(symbol=symbol))

    @staticmethod
    def _mtime_of(path: Optional[str]) -> float:
        """File mtime, or -1.0 when the file doesn't exist (no weights)."""
        if not path:
            return -1.0
        try:
            return os.path.getmtime(path)
        except OSError:
            return -1.0

    def invalidate(self, symbol: str = None, model_type: str = None):
        """
        Clear the cache.
        - no arguments: everything
        - symbol only: all models for that symbol
        - both: one model
        """
        if symbol is None:
            self._loaded_models.clear()
            self._loaded_mtimes.clear()
            return

        if model_type is not None:
            keys = [f"{symbol}_{model_type}"]
        else:
            keys = [k for k in self._loaded_models if k.startswith(f"{symbol}_")]

        for k in keys:
            self._loaded_models.pop(k, None)
            self._loaded_mtimes.pop(k, None)

        self._invalidate_artifacts(symbol)

    def _invalidate_artifacts(self, symbol: str):
        """
        After weights change, cached training artifacts must reload too, or they won't match the model.
        """
        if self.pipeline is None:
            return
        try:
            self.pipeline.invalidate(symbol)
        except Exception as e:
            logger.warning(f"[{symbol}] Failed to clear training-artifact cache: {e}")

    _invalidate_scaler = _invalidate_artifacts

    def _is_model_usable(self, model, symbol: str, model_type: str) -> bool:
        """
        Whether the model has usable trained weights.

        Two checks, for two failure modes:
            1. The model reports is_trained = False (the file exists but failed to load,
                  e.g. an old .keras after an architecture change).
            2. The weights file doesn't exist (never trained for this symbol).

        lstm.py has no is_trained attribute yet, so check 2 covers it; check 1 applies
        automatically if one is added.
        """
        if model_type not in self._REQUIRES_TRAINED_WEIGHTS:
            return True

        if getattr(model, "is_trained", None) is False:
            return False

        return self._mtime_of(self._weight_path(symbol, model_type)) >= 0

    def is_ready(self, symbol: str, model_type: str) -> bool:
        """
        Lets TradingEngine skip models known to be unusable before building the ensemble.
        Usage:
            usable = [m for m in ("LSTM", "Transformer", "ARIMA")
                      if orchestrator.is_ready(symbol, m)]
        """
        if model_type not in self._REQUIRES_TRAINED_WEIGHTS:
            return True
        try:
            model = self._get_or_load_model(symbol, model_type)
        except Exception:
            return False
        return self._is_model_usable(model, symbol, model_type)

    def _get_or_load_model(self, symbol: str, model_type: str = "LSTM"):
        """
        Lazy loading with mtime invalidation: when the weights file changes on disk,
        the old instance is dropped and reloaded.
        """
        cache_key = f"{symbol}_{model_type}"
        path = self._weight_path(symbol, model_type)
        current_mtime = self._mtime_of(path)

        cached = self._loaded_models.get(cache_key)
        if cached is not None:
            if self._loaded_mtimes.get(cache_key) == current_mtime:
                return cached

            logger.debug(f"[{symbol}] {model_type} weights changed; reloading...")
            self._loaded_models.pop(cache_key, None)
            self._loaded_mtimes.pop(cache_key, None)
            self._invalidate_artifacts(symbol)
        else:
            logger.debug(f"[{symbol}] Loading {model_type} model...")

        model = self.factory.create_model(model_type, symbol=symbol)
        model.load_weights(symbol)
        self._verify_input_shape(model, symbol, model_type)

        self._loaded_models[cache_key] = model
        self._loaded_mtimes[cache_key] = current_mtime
        return model

    @staticmethod
    def _verify_input_shape(model, symbol: str, model_type: str):
        keras_model = getattr(model, "model", None)
        expected = getattr(model, "feature_cols", None)
        if keras_model is None or not expected:
            return
        try:
            shape = keras_model.input_shape
            trained_features = shape[-1]
        except Exception:
            return

        if trained_features != len(expected):
            model.is_trained = False
            logger.error(
                f"[{symbol}] {model_type} weights were trained on {trained_features} features "
                f"but the current manifest has {len(expected)}; they must match. "
                f"Model excluded. Check that weights/training_artifacts.json has a manifest for "
                f"{symbol} and that ModelOrchestrator passes the symbol to the factory; "
                f"otherwise run --mode train again."
            )

    def predict(
        self, symbol: str, df: pd.DataFrame, model_type: str = "LSTM"
    ) -> Tuple[float, float]:
        """
        Run a numeric prediction and return (predicted price, predicted volatility).
        For LSTM, Transformer, ARIMA and GARCH.
        """
        if df.empty:
            return 0.0, 0.0

        try:
            model = self._get_or_load_model(symbol, model_type)

            if not self._is_model_usable(model, symbol, model_type):
                logger.warning(
                    f"[{symbol}] ⛔ {model_type} is not trained; excluded from the ensemble."
                )
                return 0.0, 0.0

            prediction = model.predict_next_price(df)
            volatility = model.predict_volatility(df)
            return prediction, volatility
        except Exception as e:
            logger.warning(f"[{symbol}] {model_type} prediction failed: {e}")
            current_price = df["Close"].iloc[-1]
            return current_price, 0.02
