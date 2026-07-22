# ibkrpy/manager/model_orchestrator.py
# 專注於管理 ML 模型的記憶體生命週期、快取與預測分發，避免重複載入。

import os
import pandas as pd
from typing import Tuple, Dict, Any, Optional


class ModelOrchestrator:
    """管理各類預測模型 (LSTM, Transformer, ARIMA, HMM) 的單例與生命週期"""

    # 各模型類型對應的權重檔名樣板。ARIMA / GARCH / HMM 共用同一個整合包。
    _WEIGHT_FILE_PATTERNS = {
        "LSTM": "{symbol}_LSTM.keras",
        "Transformer": "{symbol}_Transformer.keras",
        "ARIMA": "{symbol}_classical.pkl",
        "GARCH": "{symbol}_classical.pkl",
        "HMM": "{symbol}_classical.pkl",
    }

    # 沒有權重就完全不可用的模型。隨機初始化的神經網路輸出是純噪音，
    # 絕不能參與 Ensemble 投票。
    # ARIMA / GARCH 不在此列 —— 它們在缺少整合包時會退回實時擬合，那是合法的降級路徑。
    _REQUIRES_TRAINED_WEIGHTS = {"LSTM", "Transformer"}

    def __init__(self, model_factory, weights_dir: str = None, data_pipeline=None):
        """
        :param model_factory: 具備 create_model(model_type) 的工廠
        :param weights_dir:   權重目錄。未指定時自動沿用 factory.weights_dir
        :param data_pipeline: (可選) DataPipeline 實例。傳入後，權重更新時會一併
                              清掉對應的 scaler 快取 —— 否則會出現「新模型配舊 scaler」
                              的錯配，預測值會被還原到錯誤的價格區間。
        """
        self.factory = model_factory
        self.pipeline = data_pipeline

        self.weights_dir = weights_dir or getattr(model_factory, "weights_dir", "weights")

        # Cache: { "AAPL_LSTM": model_instance }
        self._loaded_models: Dict[str, Any] = {}
        # 對應的權重檔 mtime: { "AAPL_LSTM": 1712345678.9 }
        self._loaded_mtimes: Dict[str, float] = {}

    # ------------------------------------------------------------------
    # 快取管理
    # ------------------------------------------------------------------

    def _weight_path(self, symbol: str, model_type: str) -> Optional[str]:
        pattern = self._WEIGHT_FILE_PATTERNS.get(model_type)
        if not pattern:
            return None
        return os.path.join(self.weights_dir, pattern.format(symbol=symbol))

    @staticmethod
    def _mtime_of(path: Optional[str]) -> float:
        """取得檔案 mtime；檔案不存在時回傳 -1.0 (代表「無權重」狀態)"""
        if not path:
            return -1.0
        try:
            return os.path.getmtime(path)
        except OSError:
            return -1.0

    def invalidate(self, symbol: str = None, model_type: str = None):
        """
        手動清除快取。
        - 不帶參數：清空全部
        - 只給 symbol：清除該標的的所有模型
        - 兩者都給：只清除單一模型
        """
        if symbol is None:
            self._loaded_models.clear()
            self._loaded_mtimes.clear()
            return

        if model_type is not None:
            keys = [f"{symbol}_{model_type}"]
        else:
            keys = [k for k in self._loaded_models if k.startswith(f"{symbol}_")]

        for k in keys:
            self._loaded_models.pop(k, None)
            self._loaded_mtimes.pop(k, None)

        self._invalidate_scaler(symbol)

    def _invalidate_scaler(self, symbol: str):
        """權重更新後，記憶體中的 scaler 也必須跟著重讀，否則會與新模型錯配"""
        if self.pipeline is not None:
            try:
                self.pipeline.scalers.pop(symbol, None)
            except Exception:
                pass

    def _is_model_usable(self, model, symbol: str, model_type: str) -> bool:
        """
        判斷模型是否具備可用的訓練權重。

        兩道檢查，涵蓋兩種失敗模式：
          1. 模型自己回報 is_trained = False（權重檔存在但載入失敗，
             例如架構變更後的舊 .keras）
          2. 權重檔根本不存在（訓練從未跑過該標的）

        lstm.py 目前沒有 is_trained 屬性，會由第 2 道檢查兜住；
        若日後補上該屬性，第 1 道會自動生效，不需再改本檔。
        """
        if model_type not in self._REQUIRES_TRAINED_WEIGHTS:
            return True

        if getattr(model, "is_trained", None) is False:
            return False

        return self._mtime_of(self._weight_path(symbol, model_type)) >= 0

    def is_ready(self, symbol: str, model_type: str) -> bool:
        """
        供 TradingEngine 在組裝 Ensemble 前預先過濾，避免呼叫已知不可用的模型。
        用法：
            usable = [m for m in ("LSTM", "Transformer", "ARIMA")
                      if orchestrator.is_ready(symbol, m)]
        """
        if model_type not in self._REQUIRES_TRAINED_WEIGHTS:
            return True
        try:
            model = self._get_or_load_model(symbol, model_type)
        except Exception:
            return False
        return self._is_model_usable(model, symbol, model_type)

    def _get_or_load_model(self, symbol: str, model_type: str = "LSTM"):
        """
        惰性載入 (Lazy Loading) + mtime 失效檢查。
        權重檔在磁碟上被更新過時，自動丟棄舊實例並重新載入。
        """
        cache_key = f"{symbol}_{model_type}"
        path = self._weight_path(symbol, model_type)
        current_mtime = self._mtime_of(path)

        cached = self._loaded_models.get(cache_key)
        if cached is not None:
            if self._loaded_mtimes.get(cache_key) == current_mtime:
                return cached

            print(f"[{symbol}] 偵測到 {model_type} 權重已更新，重新載入至記憶體...")
            self._loaded_models.pop(cache_key, None)
            self._loaded_mtimes.pop(cache_key, None)
            self._invalidate_scaler(symbol)
        else:
            print(f"[{symbol}] 載入 {model_type} 模型至記憶體...")

        model = self.factory.create_model(model_type)
        model.load_weights(symbol)

        self._loaded_models[cache_key] = model
        self._loaded_mtimes[cache_key] = current_mtime
        return model

    # ------------------------------------------------------------------
    # 預測介面
    # ------------------------------------------------------------------

    def predict(self, symbol: str, df: pd.DataFrame, model_type: str = "LSTM") -> Tuple[float, float]:
        """
        執行連續數值預測並返回 (預測價格, 預測波動率)
        主要適用於: LSTM, Transformer, ARIMA, GARCH
        """
        if df.empty:
            return 0.0, 0.0

        try:
            # 載入也放進 try：HMMModel.load_weights 在找不到整合包時會主動拋錯，
            model = self._get_or_load_model(symbol, model_type)

            if not self._is_model_usable(model, symbol, model_type):
                print(f"[{symbol}] ⛔ {model_type} 尚未訓練，已排除於 Ensemble 之外。")
                return 0.0, 0.0

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
        返回: 狀態 ID (例如 0 代表低波熊市，1 代表高波牛市)；-1 代表未知
        """
        if df.empty:
            return -1

        try:
            model = self._get_or_load_model(symbol, model_type)
            return model.predict_state(df)
        except Exception as e:
            print(f"[{symbol}] {model_type} 狀態偵測失敗: {e}")
            # 安全回退機制：返回預設狀態 0
            return 0
