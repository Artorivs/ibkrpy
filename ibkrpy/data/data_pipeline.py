# ibkrpy/data/data_pipeline.py
# 數據管線與緩存

import json
import os
import logging
import pandas as pd
import numpy as np
from typing import Tuple, List, Dict, Optional

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
WEIGHTS_DIR = os.path.join(PROJECT_ROOT, "weights")

logger = logging.getLogger("ibkrpy")

# 永遠不會進入模型的欄位 (資料庫識別欄位)
_EXCLUDED_FEATURES = {"symbol", "timeframe"}

# 基礎價量欄位，永遠排在特徵清單最前面以保證順序穩定
_BASE_FEATURES = ["Open", "High", "Low", "Close", "Volume"]


class DataPipeline:
    """負責數據的本地快取存取，以及機器學習所需的預處理"""
    
    def __init__(self):
        self.scalers: Dict[str, Dict] = {}
        self._manifests: Dict[str, List[str]] = {}

    # ------------------------------------------------------------------
    # 特徵工程
    # ------------------------------------------------------------------

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

    def engineer_advanced_features(
        self,
        df: pd.DataFrame,
        benchmark_df: pd.DataFrame = None,
        macro_dict: Dict[str, pd.Series] = None,
    ) -> pd.DataFrame:
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

    # ------------------------------------------------------------------
    # 特徵清單 (Manifest)
    # ------------------------------------------------------------------

    def select_model_features(self, df: pd.DataFrame) -> List[str]:
        """
        從 engineer_advanced_features 的輸出挑出要送進模型的欄位。

        規則：
          - OHLCV 永遠納入且固定排在最前面 (保證欄位順序在訓練/推論間一致)
          - 其餘取所有數值型、無 NaN/Inf、非全常數的欄位，按名稱排序
        """
        features = [c for c in _BASE_FEATURES if c in df.columns]

        extras = []
        for col in df.columns:
            if col in features or col in _EXCLUDED_FEATURES:
                continue
            if not pd.api.types.is_numeric_dtype(df[col]):
                continue
            series = df[col].replace([np.inf, -np.inf], np.nan)
            if series.isna().any():
                continue
            # 全常數欄位沒有資訊量，且會讓 Min-Max 退化成 0.5
            if float(series.max()) == float(series.min()):
                continue
            extras.append(col)

        return features + sorted(extras)

    def _manifest_path(self, symbol: str) -> str:
        return os.path.join(WEIGHTS_DIR, f"{symbol}_features.json")

    def save_feature_manifest(self, symbol: str, features: List[str]):
        """[訓練階段專用] 保存本次訓練實際使用的特徵欄位與順序"""
        os.makedirs(WEIGHTS_DIR, exist_ok=True)
        with open(self._manifest_path(symbol), "w", encoding="utf-8") as f:
            json.dump({"features": list(features)}, f, indent=4, ensure_ascii=False)
        self._manifests[symbol] = list(features)
        logger.info(f"[{symbol}] 特徵清單已保存 ({len(features)} 個欄位)。")

    def load_feature_manifest(self, symbol: str) -> Optional[List[str]]:
        """
        [推論階段] 讀回訓練時使用的特徵清單。
        回傳 None 代表沒有 manifest —— 呼叫端應視為「該標的尚未訓練」。
        """
        if symbol in self._manifests:
            return self._manifests[symbol]

        path = self._manifest_path(symbol)
        if not os.path.exists(path):
            return None

        try:
            with open(path, "r", encoding="utf-8") as f:
                features = json.load(f).get("features")
            if not features:
                return None
            self._manifests[symbol] = features
            return features
        except Exception as e:
            logger.error(f"[{symbol}] 特徵清單載入失敗: {e}")
            return None

    def align_to_manifest(self, df: pd.DataFrame, symbol: str) -> Tuple[pd.DataFrame, List[str]]:
        """
        把推論用的 DataFrame 對齊到訓練時的特徵清單。

        訓練時存在、推論時缺席的欄位 (例如 benchmark 抓取失敗導致
        bench_correlation 消失) 會補 0 並記錄警告 —— 靜默補值會讓模型
        在不知情的狀況下吃到錯誤的分布。
        """
        features = self.load_feature_manifest(symbol)
        if features is None:
            return df, [c for c in _BASE_FEATURES if c in df.columns]

        missing = [c for c in features if c not in df.columns]
        if missing:
            logger.warning(f"[{symbol}] 推論資料缺少 {len(missing)} 個訓練特徵，已補 0: {missing}")
            df = df.copy()
            for c in missing:
                df[c] = 0.0

        return df, features

    def invalidate(self, symbol: str = None):
        """清除記憶體快取。重訓後必須呼叫，否則會用舊 scaler 配新模型。"""
        if symbol is None:
            self.scalers.clear()
            self._manifests.clear()
        else:
            self.scalers.pop(symbol, None)
            self._manifests.pop(symbol, None)

    # ------------------------------------------------------------------
    # 特徵縮放
    # ------------------------------------------------------------------

    @staticmethod
    def _compute_scaler(df: pd.DataFrame, columns: List[str]) -> Dict[str, Dict[str, float]]:
        """純計算，不產生任何副作用 (不寫檔、不改 self.scalers)"""
        scaler_dict = {}
        for col in columns:
            if col in df.columns:
                scaler_dict[col] = {
                    'min': float(df[col].min()),
                    'max': float(df[col].max()),
                }
        return scaler_dict

    def fit_scale(self, df: pd.DataFrame, columns: List[str], symbol: str):
        """
        [訓練階段專用] 僅在 Training Set 上呼叫。

        此函式「會寫檔」，因此絕不可從 transform_scale 之類的讀取路徑呼叫。
        """
        scaler_dict = self._compute_scaler(df, columns)
        self.scalers[symbol] = scaler_dict
        
        # 實體化為絕對路徑存檔
        os.makedirs(WEIGHTS_DIR, exist_ok=True)
        scaler_path = os.path.join(WEIGHTS_DIR, f"{symbol}_scaler.json")
        with open(scaler_path, "w", encoding="utf-8") as f:
            json.dump(scaler_dict, f, indent=4)
            
        logger.info(f"[{symbol}] 特徵縮放參數 (Scaler) 已成功適配並儲存。")

    def transform_scale(self, df: pd.DataFrame, columns: List[str], symbol: str) -> pd.DataFrame:
        """
        [回測與實盤專用] 載入訓練階段算好的 Min-Max 參數進行轉換。

        找不到 scaler 時，只在記憶體中臨時計算，「不會」落盤。
        這種情況代表該標的尚未訓練，預測結果不應被信任。
        """
        df_scaled = df.copy()
        
        # 1. 嘗試從記憶體或硬碟載入 Scaler
        if symbol not in self.scalers:
            scaler_path = os.path.join(WEIGHTS_DIR, f"{symbol}_scaler.json")
            if os.path.exists(scaler_path):
                with open(scaler_path, "r", encoding="utf-8") as f:
                    self.scalers[symbol] = json.load(f)
            else:
                logger.error(
                    f"[{symbol}] 找不到預訓練的 Scaler，僅在記憶體中臨時計算 (不落盤)。"
                    f" 該標的尚未訓練，預測結果不可信 —— 請執行 --mode train。"
                )
                self.scalers[symbol] = self._compute_scaler(df, columns)

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

    def inverse_transform_scale(self, value: float, column: str, symbol: str) -> Optional[float]:
        """
        反向還原縮放值 (供預測價格還原時使用)。
        """
        if symbol not in self.scalers or column not in self.scalers[symbol]:
            logger.error(f"[{symbol}] 缺少 {column} 的 scaler，無法還原預測值。")
            return None

        min_val = self.scalers[symbol][column]['min']
        max_val = self.scalers[symbol][column]['max']
        
        if max_val > min_val:
            return value * (max_val - min_val) + min_val
        return min_val

    # ------------------------------------------------------------------
    # 序列建構
    # ------------------------------------------------------------------

    def create_sequences(
        self,
        df: pd.DataFrame,
        features: List[str],
        target_col: str,
        look_back: int,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """建立 LSTM/Transformer 所需的滑動視窗時間序列數據"""
        if len(df) <= look_back:
            return np.array([]), np.array([])
            
        missing = [c for c in features if c not in df.columns]
        if missing:
            raise ValueError(f"建立序列時缺少特徵欄位: {missing}")

        data = df[features].values
        target = df[target_col].values
        
        X, y = [], []
        for i in range(len(data) - look_back):
            X.append(data[i:(i + look_back)])
            y.append(target[i + look_back])
            
        return np.array(X), np.array(y)