# hmm.py: 隱馬爾可夫模型 (HMM)
# 專責市場隱藏狀態 (Regime) 之偵測，如牛熊市或波動率高低狀態。

import os
import numpy as np
import pandas as pd
import joblib
import warnings

# 抑制 hmmlearn 在計算過程中的收斂警告，保持日誌乾淨
warnings.filterwarnings("ignore")

try:
    from hmmlearn.hmm import GaussianHMM
except ImportError:
    GaussianHMM = None
    print("警告: 尚未安裝 hmmlearn，請執行 `pip install hmmlearn`")

class HMMModel:
    """
    隱馬爾可夫模型 (Hidden Markov Model)
    用於偵測市場的隱藏狀態 (如: 0 代表低波盤整，1 代表高波趨勢等)
    """
    
    def __init__(self, n_components: int = 2, weights_dir: str = "weights"):
        """
        初始化 HMM 模型
        :param n_components: 隱藏狀態的數量 (預設為 2，例如牛/熊)
        :param weights_dir: 模型權重存放的目錄
        """
        self.n_components = n_components
        self.weights_dir = weights_dir
        self.model = None
        self.is_trained = False

    def load_weights(self, symbol: str):
        """從整合包載入 HMM 模型權重 (.pkl)"""
        if GaussianHMM is None:
            raise ImportError("hmmlearn 未安裝，無法載入 HMM 模型。")

        file_path = os.path.join(self.weights_dir, f"{symbol}_classical.pkl")
        
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"找不到 {symbol} 的傳統模型整合包: {file_path}")

        try:
            bundle = joblib.load(file_path)
            self.model = bundle.get('hmm')
            if self.model:
                self.is_trained = True
                print(f"[{symbol}] HMM 模型權重載入成功。")
            else:
                raise ValueError("整合包內無 HMM 模型")
        except Exception as e:
            raise RuntimeError(f"[{symbol}] 載入 HMM 模型失敗: {e}")

    def _prepare_features(self, df: pd.DataFrame) -> np.ndarray:
        """
        內部特徵工程：將原始 OHLCV 轉換為 HMM 需要的觀測特徵。
        這裡實作了最經典的 HMM 雙特徵：對數收益率 (Log Return) 與 波動率 (Volatility)。
        """
        df_features = pd.DataFrame(index=df.index)
        
        # 1. 計算對數收益率 (Log Returns)
        df_features['log_return'] = np.log(df['Close'] / df['Close'].shift(1)) * 100.0
        
        # 2. 計算短期波動率 (例如 5 日標準差)
        df_features['volatility'] = df_features['log_return'].rolling(window=5).std()
        
        # 移除 NaN 值，因為 HMM 無法處理包含 NaN 的數據
        df_features = df_features.dropna()
        
        # hmmlearn 預期輸入形狀為 (n_samples, n_features)
        return df_features[['log_return', 'volatility']].values

    def predict_state(self, df: pd.DataFrame) -> int:
        """
        預測當前最新的市場狀態
        此方法供 ModelOrchestrator 呼叫
        :param df: 包含歷史 K 線數據的 DataFrame
        :return: 狀態 ID (int)
        """
        if not self.is_trained or self.model is None:
            print("警告: HMM 模型未載入，回退至預設狀態 -1")
            return -1

        # 提取特徵
        features = self._prepare_features(df)
        
        if len(features) == 0:
            print("警告: 傳入數據長度不足以計算特徵，無法進行 HMM 狀態偵測")
            return -1

        try:
            # 預測這段時間序列所有的隱藏狀態
            hidden_states = self.model.predict(features)
            
            # 我們只需要最新一根 K 線所處的狀態
            current_state = int(hidden_states[-1])
            return current_state
            
        except Exception as e:
            print(f"HMM 狀態預測失敗: {e}")
            return -1