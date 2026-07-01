# ibkrpy/data/data_pipeline.py
# 數據管線與緩存

import json
import os
from pathlib import Path
import pandas as pd
import numpy as np
from typing import Tuple, List, Dict

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
WEIGHTS_DIR = os.path.join(PROJECT_ROOT, "weights")

class DataPipeline:
    """負責數據的本地快取存取，以及機器學習所需的預處理"""
    
    def __init__(self):
        self.scalers = {}

    def add_technical_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        """使用 pandas_ta 快速計算技術指標"""
        import pandas_ta as ta
        df_copy = df.copy()
        df_copy.ta.sma(length=10, append=True)
        df_copy.ta.ema(length=20, append=True)
        df_copy.ta.rsi(length=14, append=True)
        df_copy.ta.atr(length=14, append=True)
        
        # 增加 MACD 與 布林通道，增強對 5m 雜訊的過濾能力
        df_copy.ta.macd(fast=12, slow=26, signal=9, append=True)
        df_copy.ta.bbands(length=20, std=2, append=True)
        return df_copy.dropna()

    def engineer_advanced_features(self, df: pd.DataFrame, benchmark_df: pd.DataFrame = None, macro_dict: Dict[str, pd.Series] = None) -> pd.DataFrame:
        """進階特徵工程：融合技術指標、大盤相關性與宏觀經濟數據"""
        df_adv = self.add_technical_indicators(df)

        # 1. 跨資產相關性 (例如 QQQ 大盤表現)
        if benchmark_df is not None and not benchmark_df.empty:
            bench_ret = np.log(benchmark_df['Close'] / benchmark_df['Close'].shift(1))
            # 將大盤的收益率對齊到當前個股的時間軸
            df_adv['bench_return'] = bench_ret.reindex(df_adv.index).fillna(0)
            # 計算個股與大盤的 20 根 K 線滾動相關性
            stock_ret = np.log(df_adv['Close'] / df_adv['Close'].shift(1))
            df_adv['bench_correlation'] = stock_ret.rolling(20).corr(df_adv['bench_return']).fillna(0)

        # 2. 宏觀數據 (例如 VIX 恐慌指數)
        if macro_dict:
            for name, series in macro_dict.items():
                if series is not None and not series.empty:
                    df_adv[name] = series.reindex(df_adv.index).ffill().bfill()
        
        return df_adv.dropna()

    def fit_scale(self, df: pd.DataFrame, columns: List[str], symbol: str):
        """
        [訓練階段專用]
        僅在 Training Set 上呼叫。計算各特徵的 Min-Max 並保存為 JSON。
        """
        scaler_dict = {}
        for col in columns:
            if col in df.columns:
                scaler_dict[col] = {
                    'min': float(df[col].min()), 
                    'max': float(df[col].max())
                }
        
        self.scalers[symbol] = scaler_dict
        
        # 實體化為絕對路徑存檔
        os.makedirs(WEIGHTS_DIR, exist_ok=True)
        scaler_path = os.path.join(WEIGHTS_DIR, f"{symbol}_scaler.json")
        with open(scaler_path, "w", encoding="utf-8") as f:
            json.dump(scaler_dict, f, indent=4)
            
        print(f"[{symbol}] 特徵縮放參數 (Scaler) 已成功適配並儲存。")

    def transform_scale(self, df: pd.DataFrame, columns: List[str], symbol: str) -> pd.DataFrame:
        """
        [回測與實盤專用]
        載入訓練階段算好的 Min-Max 參數進行轉換。
        """
        df_scaled = df.copy()
        
        # 1. 嘗試從記憶體或硬碟載入 Scaler
        if symbol not in self.scalers:
            scaler_path = os.path.join(WEIGHTS_DIR, f"{symbol}_scaler.json")
            if os.path.exists(scaler_path):
                with open(scaler_path, "r", encoding="utf-8") as f:
                    self.scalers[symbol] = json.load(f)
            else:
                print(f"[{symbol}] ⚠️ 找不到預訓練的 Scaler，退回動態計算！")
                self.fit_scale(df, columns, symbol)
                
        scaler_dict = self.scalers[symbol]
        
        # 2. 嚴格使用已知的 Min-Max 進行轉換
        for col in columns:
            if col not in df.columns or col not in scaler_dict: 
                continue
                
            min_val = scaler_dict[col]['min']
            max_val = scaler_dict[col]['max']
            
            if max_val > min_val:
                df_scaled[col] = (df_scaled[col] - min_val) / (max_val - min_val)
            else:
                df_scaled[col] = 0.5
                
        return df_scaled

    def inverse_transform_scale(self, value: float, column: str, symbol: str) -> float:
        """反向還原縮放值 (供預測價格還原時使用)"""
        if symbol not in self.scalers or column not in self.scalers[symbol]:
            return value
            
        min_val = self.scalers[symbol][column]['min']
        max_val = self.scalers[symbol][column]['max']
        
        if max_val > min_val:
            return value * (max_val - min_val) + min_val
        return min_val

    def create_sequences(self, df: pd.DataFrame, features: List[str], target_col: str, look_back: int) -> Tuple[np.ndarray, np.ndarray]:
        """建立 LSTM/Transformer 所需的滑動視窗時間序列數據"""
        if len(df) <= look_back:
            return np.array([]), np.array([])
            
        data = df[features].values
        target = df[target_col].values
        
        X, y = [], []
        for i in range(len(data) - look_back):
            X.append(data[i:(i + look_back)])
            y.append(target[i + look_back])
            
        return np.array(X), np.array(y)