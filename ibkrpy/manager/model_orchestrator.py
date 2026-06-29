# model_orchestrator.py: 模型調度器
# 專注於管理 ML 模型的記憶體生命週期、快取與預測分發，避免重複載入。

import pandas as pd
from typing import Tuple, Dict, Any, Union

class ModelOrchestrator:
    """管理各類預測模型 (LSTM, Transformer, ARIMA, HMM) 的單例與生命週期"""
    
    def __init__(self, model_factory):
        self.factory = model_factory
        self._loaded_models = {}  # Cache: { "AAPL_LSTM": model_instance, "AAPL_HMM": hmm_instance }

    def _get_or_load_model(self, symbol: str, model_type: str = "LSTM"):
        """惰性載入 (Lazy Loading)：只有在需要時才將權重/模型載入記憶體"""
        cache_key = f"{symbol}_{model_type}"
        
        if cache_key not in self._loaded_models:
            print(f"[{symbol}] 載入 {model_type} 模型至記憶體...")
            model = self.factory.create_model(model_type)
            
            # 注意：深度學習模型可能使用 .h5 或 .keras，而 HMM (sklearn/hmmlearn) 通常使用 .pkl 或 joblib
            # 這裡假設您的 model 封裝內部已經處理好了副檔名與載入邏輯
            model.load_weights(symbol) 
            self._loaded_models[cache_key] = model
            
        return self._loaded_models[cache_key]

    def predict(self, symbol: str, df: pd.DataFrame, model_type: str = "LSTM") -> Tuple[float, float]:
        """
        執行連續數值預測並返回 (預測價格, 預測波動率)
        主要適用於: LSTM, Transformer, ARIMA, GARCH
        """
        if df.empty:
            return 0.0, 0.0
            
        model = self._get_or_load_model(symbol, model_type)
        
        try:
            prediction = model.predict_next_price(df)
            volatility = model.predict_volatility(df)
            
            return prediction, volatility
            
        except Exception as e:
            print(f"[{symbol}] {model_type} 數值預測失敗: {e}")
            # 模型失效時的安全回退機制 (Fallback)
            current_price = df['Close'].iloc[-1]
            return current_price, 0.02

    def detect_regime(self, symbol: str, df: pd.DataFrame, model_type: str = "HMM") -> int:
        """
        執行市場隱藏狀態偵測 (Regime Detection)
        主要適用於: HMM (Hidden Markov Model)
        返回: 狀態 ID (例如 0 代表低波熊市，1 代表高波牛市)
        """
        if df.empty:
            return -1 # 未知狀態
            
        model = self._get_or_load_model(symbol, model_type)
        
        try:
            # 假設 HMM 模型類別實作了 predict_state，接收 df 並返回最新的隱藏狀態
            state = model.predict_state(df)
            return state
            
        except Exception as e:
            print(f"[{symbol}] {model_type} 狀態偵測失敗: {e}")
            # 安全回退機制：返回預設狀態 0
            return 0