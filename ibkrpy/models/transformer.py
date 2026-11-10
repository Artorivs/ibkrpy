# ibkrpy/models/transformer.py
# Captures long-range dependencies across the input window.

import os
import numpy as np
import pandas as pd
import tensorflow as tf
import keras
from keras.layers import (
    Input,
    Dense,
    Dropout,
    LayerNormalization,
    MultiHeadAttention,
    Cropping1D,
    Flatten,
)
from keras.models import Model, load_model
from typing import List, Optional


@keras.saving.register_keras_serializable(package="ibkrpy")
class PositionalEmbedding(keras.layers.Layer):
    """
    Learned positional embedding.

    Learns a d_model vector for each step t in [0, look_back) and adds it to that step,
    so self-attention can tell bars apart by position.

    Implementation note:
        The weight is created with add_weight() in build(), shape (look_back, d_model),
        and broadcast onto (batch, look_back, d_model).

        A keras.layers.Embedding sub-layer is deliberately avoided. Keras 3 doesn't build
        sub-layers created in __init__ unless the parent's build() does, so saved weights
        fail to load with "Layer 'pos_emb' was never built".
        add_weight() is the recommended path and avoids that ordering issue.
    """

    def __init__(self, look_back: int, d_model: int, **kwargs):
        super().__init__(**kwargs)
        self.look_back = look_back
        self.d_model = d_model

    def build(self, input_shape):
        self.pos_emb = self.add_weight(
            name="pos_emb",
            shape=(self.look_back, self.d_model),
            initializer=keras.initializers.RandomNormal(stddev=0.02),
            trainable=True,
        )
        super().build(input_shape)

    def call(self, x):
        return x + self.pos_emb

    def compute_output_shape(self, input_shape):
        return input_shape

    def get_config(self):
        config = super().get_config()
        config.update({"look_back": self.look_back, "d_model": self.d_model})
        return config


class TransformerModel:
    """Transformer model behind the ModelOrchestrator interface."""

    def __init__(
        self,
        look_back: int = 60,
        feature_cols: Optional[List[str]] = None,
        d_model: int = 64,
        head_size: int = 32,
        num_heads: int = 4,
        ff_dim: int = 128,
        num_blocks: int = 2,
        dropout_rate: float = 0.1,
        weights_dir: str = "weights",
    ):
        self.look_back = look_back
        self.feature_cols = feature_cols or ["Open", "High", "Low", "Close", "Volume"]
        self.features = len(self.feature_cols)

        self.d_model = d_model
        self.head_size = head_size
        self.num_heads = num_heads
        self.ff_dim = ff_dim
        self.num_blocks = num_blocks
        self.dropout_rate = dropout_rate

        self.weights_dir = weights_dir
        self.model = None
        self.is_trained = False

    def _build_model(self) -> Model:
        """Build the Transformer encoder with positional embedding."""
        inputs = Input(shape=(self.look_back, self.features), name="ohlcv_window")

        x = Dense(self.d_model, name="input_projection")(inputs)

        x = PositionalEmbedding(self.look_back, self.d_model, name="positional")(x)
        x = Dropout(self.dropout_rate)(x)

        for i in range(self.num_blocks):
            attn = MultiHeadAttention(
                key_dim=self.head_size,
                num_heads=self.num_heads,
                dropout=self.dropout_rate,
                name=f"mha_{i}",
            )(x, x)
            x = LayerNormalization(epsilon=1e-6, name=f"ln_attn_{i}")(x + attn)

            ff = Dense(self.ff_dim, activation="swish", name=f"ff_in_{i}")(x)
            ff = Dropout(self.dropout_rate)(ff)
            ff = Dense(self.d_model, name=f"ff_out_{i}")(ff)
            x = LayerNormalization(epsilon=1e-6, name=f"ln_ff_{i}")(x + ff)

        x = Cropping1D(cropping=(self.look_back - 1, 0), name="take_last_step")(x)
        x = Flatten(name="flatten_last")(x)

        x = Dense(20, activation="swish", name="head_dense")(x)
        x = Dropout(self.dropout_rate)(x)
        outputs = Dense(1, name="prediction")(x)

        model = Model(inputs, outputs, name="ibkrpy_transformer")

        optimizer = keras.optimizers.Adam(learning_rate=0.001, clipnorm=1.0)
        model.compile(optimizer=optimizer, loss=tf.keras.losses.Huber(delta=1.0))

        return model

    def load_weights(self, symbol: str):
        """Load .keras weights."""
        file_path = os.path.join(self.weights_dir, f"{symbol}_Transformer.keras")

        if os.path.exists(file_path):
            try:
                self.model = load_model(file_path)
                self.is_trained = True
                print(f"[{symbol}] Transformer weights loaded.")
                return
            except Exception as e:
                print(f"❌ [{symbol}] Failed to load Transformer weights: {e}")
                print(
                    f"❌ [{symbol}] If this file was saved by the old architecture (no positional embedding), run --mode train again."
                )

        print(
            f"❌ [{symbol}] No usable Transformer weights; using an untrained model (do not trade on it)."
        )
        self.model = self._build_model()
        self.is_trained = False

    def _prepare_features(self, df: pd.DataFrame) -> np.ndarray:
        """Take the input window as a 3D tensor, checking the feature columns."""
        if len(df) < self.look_back:
            return np.array([])

        missing_cols = [col for col in self.feature_cols if col not in df.columns]
        if missing_cols:
            raise ValueError(f"Missing required feature columns: {missing_cols}")

        data = df[self.feature_cols].iloc[-self.look_back :].values
        return np.expand_dims(data, axis=0)

    def predict_next_price(self, df: pd.DataFrame) -> float:
        """Forecast the next bar (scaled output; the caller decodes it)."""
        if self.model is None:
            return float(df["Close"].iloc[-1])

        try:
            X = self._prepare_features(df)
            if len(X) == 0:
                return float(df["Close"].iloc[-1])

            X_tensor = tf.convert_to_tensor(X, dtype=tf.float32)
            prediction = self.model(X_tensor, training=False)
            return float(prediction[0, 0])
        except Exception as e:
            print(f"Transformer prediction failed: {e}")
            return float(df["Close"].iloc[-1])

    def predict_volatility(self, df: pd.DataFrame) -> float:
        """Fallback: historical volatility."""
        returns = np.log(df["Close"] / df["Close"].shift(1)).dropna()
        vol = returns.tail(20).std() * np.sqrt(252) if len(returns) >= 20 else 0.02
        return float(vol)
