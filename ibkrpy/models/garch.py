# ibkrpy/models/garch.py
# garch.py: GARCH 模型
# 專司預測市場之波動率，是風險控制模組的強力護盾。

import os
import numpy as np
import pandas as pd
import joblib
import warnings
from arch import arch_model

warnings.filterwarnings("ignore")

class GARCHModel:
    """標準化 GARCH 模型，對接 ModelOrchestrator 介面"""
    
    def __init__(self, p: int = 1, q: int = 1, dist: str = "normal", weights_dir: str = "weights"):
        self.p = p
        self.q = q
        self.dist = dist
        self.weights_dir = weights_dir
        self.saved_params = None

    def load_weights(self, symbol: str):
        """從整合包載入 GARCH 模型參數 (.pkl)"""
        file_path = os.path.join(self.weights_dir, f"{symbol}_classical.pkl")
        
        if os.path.exists(file_path):
            try:
                bundle = joblib.load(file_path)
                self.saved_params = bundle.get('garch')
                if self.saved_params is not None:
                    print(f"[{symbol}] GARCH 模型權重載入成功。")
                else:
                    print(f"[{symbol}] 整合包內無 GARCH 權重，將回退實時擬合。")
            except Exception as e:
                print(f"[{symbol}] GARCH 權重載入失敗: {e}，將回退至實時擬合。")
        else:
            print(f"[{symbol}] 找不到傳統模型整合包 ({file_path})，將在預測時進行實時擬合。")

    def predict_volatility(self, df: pd.DataFrame) -> float:
        """預測下一期的年化波動率"""
        if df.empty or 'Close' not in df.columns:
            return 0.02

        # arch_model 預期數值較大的百分比收益率 (避免優化器收斂失敗)
        returns = np.log(df['Close'] / df['Close'].shift(1)).dropna() * 100.0

        if len(returns) < 20:
            return 0.02

        try:
            model = arch_model(returns, vol='Garch', p=self.p, q=self.q, dist=self.dist)

            if self.saved_params is not None:
                # 模式 1: 使用已儲存的參數 (固定權重) 直接預測
                res = model.fix(self.saved_params)
            else:
                # 模式 2: 實時擬合最新數據
                res = model.fit(disp='off')

            # 預測下一期方差 (此為百分比單位的方差)
            forecasts = res.forecast(horizon=1, method='analytic')
            predicted_var = forecasts.variance.iloc[-1].item()

            # 還原為小數形式的日波動率，再轉化為年化波動率
            predicted_vol_daily = np.sqrt(predicted_var) / 100.0
            annual_vol = predicted_vol_daily * np.sqrt(252)
            
            return float(annual_vol)

        except Exception as e:
            print(f"GARCH 預測失敗: {e}")
            # Fallback: 若預測失敗，返回簡單的歷史年化標準差
            return float((returns.tail(20).std() / 100.0) * np.sqrt(252))

    def predict_next_price(self, df: pd.DataFrame) -> float:
        """Fallback：GARCH 本身不預測價格"""
        return float(df['Close'].iloc[-1]) if not df.empty else 0.0