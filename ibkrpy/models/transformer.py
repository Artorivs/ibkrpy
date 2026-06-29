# ibkrpy/models/transformer.py
# transformer.py: Transformer 模型
# 善於捕捉長時序之關聯與全局特徵。

import os
import numpy as np
import pandas as pd
import tensorflow as tf
import keras
from keras.layers import Input, Dense, Dropout, LayerNormalization, MultiHeadAttention
from keras.models import Model, load_model
from typing import List, Optional

class TransformerModel:
    """標準化 Transformer 模型，對接 ModelOrchestrator 介面"""
    
    def __init__(
        self, 
        look_back: int = 60, 
        feature_cols: Optional[List[str]] = None, 
        head_size: int = 256, 
        num_heads: int = 4, 
        num_blocks: int = 2,
        weights_dir: str = "weights"
    ):
        self.look_back = look_back
        # 預設對齊 OHLCV 5特徵，避免 Pipeline 灌入過多未經訓練的指標導致 Shape Mismatch
        self.feature_cols = feature_cols or ['Open', 'High', 'Low', 'Close', 'Volume']
        self.features = len(self.feature_cols)
        
        self.head_size = head_size
        self.num_heads = num_heads
        self.num_blocks = num_blocks
        self.weights_dir = weights_dir
        self.model = None

    def _build_model(self) -> Model:
        """構建 Transformer Encoder 架構"""
        inputs = Input(shape=(self.look_back, self.features))
        x = inputs
        
        # 建立多個 Transformer Block
        for _ in range(self.num_blocks):
            # 自注意力機制 (Self-Attention)
            x_res = MultiHeadAttention(key_dim=self.head_size, num_heads=self.num_heads, dropout=0.1)(x, x)
            x = LayerNormalization(epsilon=1e-6)(x + x_res)
            
            # 前饋網路 (Feed Forward)
            x_res = Dense(self.head_size, activation="swish")(x)
            x_res = Dropout(0.1)(x_res)
            x_res = Dense(self.features)(x_res)
            x = LayerNormalization(epsilon=1e-6)(x + x_res)
            
        # 展平與輸出
        x = tf.keras.layers.GlobalAveragePooling1D()(x)
        x = Dense(20, activation="swish")(x)
        x = Dropout(0.1)(x)
        outputs = Dense(1)(x)

        model = Model(inputs, outputs)
        
        # 使用 Huber Loss 對抗極端值 (黑天鵝防護)
        optimizer = keras.optimizers.Adam(learning_rate=0.001, clipnorm=1.0)
        model.compile(optimizer=optimizer, loss=tf.keras.losses.Huber(delta=1.0))
        
        return model

    def load_weights(self, symbol: str):
        """載入 .keras 或 .h5 權重檔"""
        file_path = os.path.join(self.weights_dir, f"{symbol}_Transformer.keras")
        
        if os.path.exists(file_path):
            self.model = load_model(file_path)
            print(f"[{symbol}] Transformer 模型權重載入成功。")
        else:
            print(f"[{symbol}] 找不到 Transformer 權重 ({file_path})，初始化全新模型。")
            self.model = self._build_model()

    def _prepare_features(self, df: pd.DataFrame) -> np.ndarray:
        """特徵工程：擷取視窗並升維 (包含維度過濾防護)"""
        if len(df) < self.look_back:
            return np.array([])
            
        # 確保 DataFrame 包含所有必需的欄位，若缺少則拋出明確錯誤
        missing_cols = [col for col in self.feature_cols if col not in df.columns]
        if missing_cols:
            raise ValueError(f"數據缺失必要特徵欄位: {missing_cols}")
            
        # 嚴格過濾欄位，避免 df 的指標無限膨脹導致 keras shape 報錯
        data = df[self.feature_cols].iloc[-self.look_back:].values
        return np.expand_dims(data, axis=0)

    def predict_next_price(self, df: pd.DataFrame) -> float:
        """預測下一個時間步的價格"""
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