# ibkrpy/models/lstm.py
# Sequence model that forecasts the next bar.

import os
from typing import List, Optional
import numpy as np
import pandas as pd
import tensorflow as tf
import keras
from keras.models import Sequential, load_model
from keras.layers import LSTM, Dense, Dropout, LayerNormalization
import warnings

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"
warnings.filterwarnings("ignore")


class LSTMModel:
    """LSTM model behind the ModelOrchestrator interface."""

    def __init__(
        self,
        look_back: int = 60,
        feature_cols: Optional[List[str]] = None,
        weights_dir: str = "weights",
    ):
        self.look_back = look_back
        self.feature_cols = feature_cols or ["Open", "High", "Low", "Close", "Volume"]
        self.features = len(self.feature_cols)
        self.input_shape = (look_back, self.features)
        self.weights_dir = weights_dir
        self.model = None

    def _build_model(self) -> Sequential:
        """Build the two-layer LSTM network."""
        model = Sequential(
            [
                LSTM(64, return_sequences=True, input_shape=self.input_shape),
                LayerNormalization(),
                Dropout(0.2),
                LSTM(32, return_sequences=False),
                LayerNormalization(),
                Dropout(0.2),
                Dense(24, activation="swish"),
                Dense(1),
            ]
        )

        optimizer = keras.optimizers.Adam(learning_rate=0.001, clipnorm=1.0)
        model.compile(optimizer=optimizer, loss=tf.keras.losses.Huber(delta=1.0))

        return model

    def load_weights(self, symbol: str):
        """Load .keras or .h5 weights."""
        file_path = os.path.join(self.weights_dir, f"{symbol}_LSTM.keras")

        if os.path.exists(file_path):
            self.model = load_model(file_path)
            print(f"[{symbol}] LSTM weights loaded.")
        else:
            print(
                f"[{symbol}] LSTM weights not found ({file_path}); using an untrained model."
            )
            self.model = self._build_model()

    def _prepare_features(self, df: pd.DataFrame) -> np.ndarray:
        """Take the last look_back rows as a (1, look_back, features) tensor."""
        if len(df) < self.look_back:
            return np.array([])

        missing_cols = [col for col in self.feature_cols if col not in df.columns]
        if missing_cols:
            raise ValueError(f"Missing required feature columns: {missing_cols}")

        data = df[self.feature_cols].iloc[-self.look_back :].values
        return np.expand_dims(data, axis=0)

    def predict_next_price(self, df: pd.DataFrame) -> float:
        """Forecast the next bar (scaled output; decode it outside)."""
        if self.model is None:
            return df["Close"].iloc[-1]

        X = self._prepare_features(df)
        if len(X) == 0:
            return df["Close"].iloc[-1]

        X_tensor = tf.convert_to_tensor(X, dtype=tf.float32)
        prediction = self.model(X_tensor, training=False)
        return float(prediction[0, 0])

    def predict_volatility(self, df: pd.DataFrame) -> float:
        """Fallback volatility from recent history (the model does not forecast it)."""
        returns = np.log(df["Close"] / df["Close"].shift(1)).dropna()
        vol = returns.tail(20).std() * np.sqrt(252) if len(returns) >= 20 else 0.02
        return float(vol)
