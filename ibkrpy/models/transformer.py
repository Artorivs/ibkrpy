# ibkrpy/models/transformer.py
# 善於捕捉長時序之關聯與全局特徵。

import os
import numpy as np
import pandas as pd
import tensorflow as tf
import keras
from keras.layers import (
    Input, Dense, Dropout, LayerNormalization,
    MultiHeadAttention, Cropping1D, Flatten,
)
from keras.models import Model, load_model
from typing import List, Optional


@keras.saving.register_keras_serializable(package="ibkrpy")
class PositionalEmbedding(keras.layers.Layer):
    """
    可學習的位置嵌入 (Learned Positional Embedding)。

    對每一個時間步 t in [0, look_back) 學一個 d_model 維向量並加到該步的表徵上，
    讓後續的 self-attention 能夠分辨「第幾根 K 線」。

    實作註記：
      權重以 add_weight() 在 build() 內直接建立，形狀為 (look_back, d_model)，
      靠 broadcasting 加到 (batch, look_back, d_model) 上。

      這裡刻意「不」使用 keras.layers.Embedding 子層。Keras 3 不會自動 build
      在 __init__ 中建立的子層，除非父層的 build() 明確地去建立它的狀態；
      否則存檔時子層變數會被寫入權重檔，載入時該層卻尚未 build，導致
      「Layer 'pos_emb' was never built」的載入失敗。
      add_weight() 是官方建議的標準路徑，沒有這個順序問題。
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
        # (look_back, d_model) 廣播加到 (batch, look_back, d_model)
        return x + self.pos_emb

    def compute_output_shape(self, input_shape):
        return input_shape

    def get_config(self):
        config = super().get_config()
        config.update({"look_back": self.look_back, "d_model": self.d_model})
        return config


class TransformerModel:
    """標準化 Transformer 模型，對接 ModelOrchestrator 介面"""

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
        # 預設對齊 OHLCV 5 特徵。若訓練時使用了更多特徵，
        # 務必在此傳入同一份清單，否則會 shape mismatch。
        self.feature_cols = feature_cols or ['Open', 'High', 'Low', 'Close', 'Volume']
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
        """構建具備位置編碼的 Transformer Encoder 架構"""
        inputs = Input(shape=(self.look_back, self.features), name="ohlcv_window")

        # --- 輸入投影：把 5 維特徵升到 d_model，讓殘差流有足夠寬度 ---
        x = Dense(self.d_model, name="input_projection")(inputs)

        # --- 位置編碼 ---
        x = PositionalEmbedding(self.look_back, self.d_model, name="positional")(x)
        x = Dropout(self.dropout_rate)(x)

        # --- Transformer Blocks (Post-LN) ---
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

        # --- 取最後一個時間步 ---
        # 用 Cropping1D + Flatten 而非 Lambda，確保 .keras 序列化不需要 custom_objects。
        x = Cropping1D(cropping=(self.look_back - 1, 0), name="take_last_step")(x)
        x = Flatten(name="flatten_last")(x)

        x = Dense(20, activation="swish", name="head_dense")(x)
        x = Dropout(self.dropout_rate)(x)
        outputs = Dense(1, name="prediction")(x)

        model = Model(inputs, outputs, name="ibkrpy_transformer")

        # 使用 Huber Loss 對抗極端值 (黑天鵝防護)
        optimizer = keras.optimizers.Adam(learning_rate=0.001, clipnorm=1.0)
        model.compile(optimizer=optimizer, loss=tf.keras.losses.Huber(delta=1.0))
        

        return model

    def load_weights(self, symbol: str):
        """載入 .keras 權重檔"""
        file_path = os.path.join(self.weights_dir, f"{symbol}_Transformer.keras")
        
        if os.path.exists(file_path):
            try:
                # PositionalEmbedding 已透過 register_keras_serializable 註冊，
                # 只要本模組被 import 過就不需要傳 custom_objects。
                self.model = load_model(file_path)
                self.is_trained = True
                print(f"[{symbol}] Transformer 模型權重載入成功。")
                return
            except Exception as e:
                # 架構變更後，舊的 .keras 會在此失敗 —— 這是預期行為，需重新訓練。
                print(f"[{symbol}] ⚠️ Transformer 權重載入失敗: {e}")
                print(f"[{symbol}]    若此檔為舊架構 (無位置編碼) 所存，請重新執行 --mode train。")

        print(f"[{symbol}] ⚠️ 無可用的 Transformer 權重，初始化未訓練模型 (不應用於實盤決策)。")
        self.model = self._build_model()
        self.is_trained = False

    def _prepare_features(self, df: pd.DataFrame) -> np.ndarray:
        """特徵工程：擷取視窗並升維 (包含維度過濾防護)"""
        if len(df) < self.look_back:
            return np.array([])

        missing_cols = [col for col in self.feature_cols if col not in df.columns]
        if missing_cols:
            raise ValueError(f"數據缺失必要特徵欄位: {missing_cols}")
            
        # 嚴格過濾欄位，避免 df 的指標無限膨脹導致 keras shape 報錯
        data = df[self.feature_cols].iloc[-self.look_back:].values
        return np.expand_dims(data, axis=0)

    def predict_next_price(self, df: pd.DataFrame) -> float:
        """預測下一個時間步的價格 (輸出為縮放值，需由呼叫端還原)"""
        if self.model is None:
            return float(df['Close'].iloc[-1])
            
        try:
            X = self._prepare_features(df)
            if len(X) == 0:
                return float(df['Close'].iloc[-1])
            
            X_tensor = tf.convert_to_tensor(X, dtype=tf.float32)
            prediction = self.model(X_tensor, training=False)
            return float(prediction[0, 0])
        except Exception as e:
            print(f"Transformer 預測發生異常: {e}")
            return float(df['Close'].iloc[-1])

    def predict_volatility(self, df: pd.DataFrame) -> float:
        """Fallback：計算歷史波動率"""
        returns = np.log(df['Close'] / df['Close'].shift(1)).dropna()
        vol = returns.tail(20).std() * np.sqrt(252) if len(returns) >= 20 else 0.02
        return float(vol)
