# ibkrpy/models/lstm.py
# lstm.py: LSTM 模型
# 專責捕捉時間序列的序列記憶效應，輸出下一個時間步的預測值。

import os
from typing import List, Optional
import numpy as np
import pandas as pd
import tensorflow as tf
import keras
from keras.models import Sequential, load_model
from keras.layers import LSTM, Dense, Dropout, LayerNormalization
import warnings

# 抑制 TF 的繁雜日誌
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
warnings.filterwarnings("ignore")

class LSTMModel:
    """標準化 LSTM 模型，對接 ModelOrchestrator 介面"""
    
    def __init__(self, 
                 look_back: int = 60, 
                 feature_cols: Optional[List[str]] = None, 
                 weights_dir: str = "weights"
                 ):
        self.look_back = look_back
        self.feature_cols = feature_cols or ['Open', 'High', 'Low', 'Close', 'Volume']
        self.features = len(self.feature_cols)
        self.input_shape = (look_back, self.features)
        self.weights_dir = weights_dir
        self.model = None

    def _build_model(self) -> Sequential:
        """構建標準的 LSTM 雙層網路架構 (具備防神經元壞死機制)"""
        model = Sequential([
            LSTM(64, return_sequences=True, input_shape=self.input_shape),
            LayerNormalization(),
            Dropout(0.2),
            LSTM(32, return_sequences=False),
            LayerNormalization(),
            Dropout(0.2),
            Dense(24, activation='swish'),
            Dense(1)
        ])
        
        # [升級] 加入 clipnorm=1.0 梯度裁剪，防範黑天鵝數據引發的梯度爆炸
        optimizer = keras.optimizers.Adam(learning_rate=0.001, clipnorm=1.0)
        model.compile(optimizer=optimizer, loss=tf.keras.losses.Huber(delta=1.0))
        
        return model

    def load_weights(self, symbol: str):
        """載入 .keras 或 .h5 權重檔"""
        file_path = os.path.join(self.weights_dir, f"{symbol}_LSTM.keras")
        
        if os.path.exists(file_path):
            self.model = load_model(file_path)
            print(f"[{symbol}] LSTM 模型權重載入成功。")
        else:
            print(f"[{symbol}] 找不到 LSTM 權重 ({file_path})，初始化全新未經訓練之模型。")
            self.model = self._build_model()

    def _prepare_features(self, df: pd.DataFrame) -> np.ndarray:
        """提取最後 look_back 筆資料，轉換為 (1, look_back, features) 的 3D 張量"""
        if len(df) < self.look_back:
            return np.array([])
            
        missing_cols = [col for col in self.feature_cols if col not in df.columns]
        if missing_cols:
            raise ValueError(f"數據缺失必要特徵欄位: {missing_cols}")
            
        data = df[self.feature_cols].iloc[-self.look_back:].values 
        return np.expand_dims(data, axis=0) 

    def predict_next_price(self, df: pd.DataFrame) -> float:
        """預測下一個時間步的價格 (已縮放數值需在外部還原)"""
        if self.model is None:
            return df['Close'].iloc[-1]
            
        X = self._prepare_features(df)
        if len(X) == 0:
            return df['Close'].iloc[-1]
            
        X_tensor = tf.convert_to_tensor(X, dtype=tf.float32)
        prediction = self.model(X_tensor, training=False)
        return float(prediction[0, 0])

    def predict_volatility(self, df: pd.DataFrame) -> float:
        """若模型本身不預測波動率，提供基於近期歷史的波動率作為 fallback"""
        returns = np.log(df['Close'] / df['Close'].shift(1)).dropna()
        vol = returns.tail(20).std() * np.sqrt(252) if len(returns) >= 20 else 0.02
        return float(vol)