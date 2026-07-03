# ibkrpy/models/arima.py
# 專司時間序列之線性預測，善於捕捉趨勢與均值回歸特性。

import os
import numpy as np
import pandas as pd
import joblib
import warnings
from statsmodels.tsa.arima.model import ARIMA, ARIMAResultsWrapper

# 抑制 statsmodels 的收斂與日期索引警告
warnings.filterwarnings("ignore")

class ARIMAModel:
    """標準化 ARIMA 模型，對接 ModelOrchestrator 介面"""
    
    def __init__(self, p: int = 5, d: int = 1, q: int = 0, weights_dir: str = "weights"):
        self.order = (p, d, q)
        self.weights_dir = weights_dir
        self.model_results: ARIMAResultsWrapper = None

    def load_weights(self, symbol: str):
        """從整合包載入 ARIMA 模型狀態 (.pkl)"""
        file_path = os.path.join(self.weights_dir, f"{symbol}_classical.pkl")
        
        if os.path.exists(file_path):
            try:
                bundle = joblib.load(file_path)
                self.model_results = bundle.get('arima')
                if self.model_results:
                    print(f"[{symbol}] ARIMA 模型權重載入成功。")
                else:
                    print(f"[{symbol}] 整合包內無 ARIMA 權重，將回退實時擬合。")
            except Exception as e:
                print(f"[{symbol}] ARIMA 權重載入失敗: {e}，將回退至實時擬合。")
        else:
            print(f"[{symbol}] 找不到傳統模型整合包 ({file_path})，將在預測時進行實時擬合。")

    def predict_next_price(self, df: pd.DataFrame) -> float:
        """預測下一個時間步的價格"""
        if df.empty or 'Close' not in df.columns:
            return 0.0

        series = df['Close'].dropna()
        last_price = float(series.iloc[-1])
        
        # 確保數據量大於 ARIMA 的最大滯後階數
        if len(series) < max(self.order) + 1:
            return last_price

        try:
            if self.model_results is not None:
                # 模式 1: 使用已保存的模型，灌入最新數據更新狀態並預測
                updated_model = self.model_results.apply(series.values)
                forecast = updated_model.forecast(steps=1)
                return float(forecast[0])
            else:
                # 模式 2: 實時擬合 (Fallback)
                model = ARIMA(series.values, order=self.order)
                res = model.fit()
                forecast = res.forecast(steps=1)
                return float(forecast[0])
                
        except Exception as e:
            print(f"ARIMA 預測失敗: {e}")
            return last_price # 發生任何錯誤時，回退到最後已知價格

    def predict_volatility(self, df: pd.DataFrame) -> float:
        """Fallback：計算歷史波動率 (ARIMA 本身不負責波動率)"""
        returns = np.log(df['Close'] / df['Close'].shift(1)).dropna()
        vol = returns.tail(20).std() * np.sqrt(252) if len(returns) >= 20 else 0.02
        return float(vol)