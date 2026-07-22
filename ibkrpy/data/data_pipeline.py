# ibkrpy/data/data_pipeline.py
# 數據管線與緩存，負責特徵工程、特徵清單、特徵縮放、序列建構。

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

# 與價格等比例縮放的欄位 (名稱前綴)。涵蓋 add_technical_indicators 產生的所有
# 價格量綱指標：移動平均、布林上中下軌、ATR、MACD 等。
_PRICE_RELATIVE_PREFIXES = (
    "Open", "High", "Low", "Close",
    "SMA_", "EMA_", "WMA_", "DEMA_", "TEMA_", "HMA_", "VWAP",
    "BBL_", "BBM_", "BBU_",          # 布林上中下軌 (價格量綱)
    "ATR_", "ATRr_", "TRUERANGE",
    "MACD_", "MACDh_", "MACDs_",     # MACD 三線皆為價格差量綱
    "HL2", "HLC3", "OHLC4", "STDEV",
)

# 明確不隨價格縮放的欄位 (振盪指標、比例、外部數據)
_NEVER_PRICE_RELATIVE = (
    "Volume", "RSI_", "BBB_", "BBP_", "ADX_", "DMP_", "DMN_",
    "STOCH", "CCI_", "MFI_", "WILLR_", "ROC_", "MOM_",
    "VIX", "bench_return", "bench_correlation",
)


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
            df_adv['bench_return'] = bench_ret.reindex(df_adv.index).fillna(0)
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

    @staticmethod
    def classify_price_relative(df: pd.DataFrame, features: List[str]) -> List[str]:
        """
        找出「隨價格等比例縮放」的欄位。

        這類欄位不能用 Min-Max —— 訓練窗的 min/max 一旦被突破 (趨勢股創新高
        幾乎必然發生)，縮放值就會超出 [0,1]，模型進入訓練分布之外的外推區間。
        改為在每個時間窗內除以該窗最後一根的收盤價，變成無量綱的相對值。

        判斷方式：
          1. 以名稱前綴為主。add_technical_indicators 產生哪些欄位是我們自己決定的，
             明確列舉遠比數值猜測可靠。
          2. 名稱不認得的欄位再走數值判準，且門檻刻意收得很嚴 ——
             要求 col/Close 的變異係數 < 0.05 且比值落在 [0.5, 2.0]。
             純用標準差會誤判：強趨勢下 RSI/Close 的絕對標準差同樣很小。
        """
        if "Close" not in df.columns:
            return []

        close = df["Close"].replace(0, np.nan)
        out = []

        for col in features:
            if col in _PRICE_RELATIVE_PREFIXES:
                out.append(col)
                continue
            if any(col.startswith(p) for p in _PRICE_RELATIVE_PREFIXES):
                out.append(col)
                continue
            if col in _NEVER_PRICE_RELATIVE or any(col.startswith(p) for p in _NEVER_PRICE_RELATIVE):
                continue

            # 未知欄位的保守數值判準
            if col not in df.columns or not pd.api.types.is_numeric_dtype(df[col]):
                continue
            ratio = (df[col] / close).replace([np.inf, -np.inf], np.nan).dropna()
            if len(ratio) < 20:
                continue
            mean = float(ratio.mean())
            if abs(mean) < 1e-9:
                continue
            cv = float(ratio.std()) / abs(mean)
            if cv < 0.05 and 0.5 <= mean <= 2.0:
                out.append(col)

        return out

    def _manifest_path(self, symbol: str) -> str:
        return os.path.join(WEIGHTS_DIR, f"{symbol}_features.json")

    def save_feature_manifest(
        self,
        symbol: str,
        features: List[str],
        price_relative: List[str] = None,
        target_mode: str = "log_return",
        target_scale: float = 100.0,
    ):
        """[訓練階段專用] 保存特徵欄位、其縮放方式與預測目標的定義"""
        manifest = {
            "features": list(features),
            "price_relative": list(price_relative or []),
            "target_mode": target_mode,     # "log_return" 或 "level"
            "target_scale": target_scale,   # log_return 乘上的倍率 (100 = 百分比)
        }
        os.makedirs(WEIGHTS_DIR, exist_ok=True)
        with open(self._manifest_path(symbol), "w", encoding="utf-8") as f:
            json.dump(manifest, f, indent=4, ensure_ascii=False)
        self._manifests[symbol] = manifest
        logger.debug(
            f"[{symbol}] 特徵清單已保存 ({len(features)} 欄，"
            f"其中 {len(manifest['price_relative'])} 欄為價格相對縮放；"
            f"目標: {target_mode})。"
        )

    def load_manifest(self, symbol: str) -> Optional[Dict]:
        """讀回完整 manifest。回傳 None 代表該標的尚未訓練。"""
        if symbol in self._manifests:
            return self._manifests[symbol]

        path = self._manifest_path(symbol)
        if not os.path.exists(path):
            return None

        try:
            with open(path, "r", encoding="utf-8") as f:
                manifest = json.load(f)
            if not manifest.get("features"):
                return None
            manifest.setdefault("price_relative", [])
            manifest.setdefault("target_mode", "level")
            manifest.setdefault("target_scale", 1.0)
            self._manifests[symbol] = manifest
            return manifest
        except Exception as e:
            logger.error(f"[{symbol}] 特徵清單載入失敗: {e}")
            return None

    def load_feature_manifest(self, symbol: str) -> Optional[List[str]]:
        """只取特徵欄位清單 (供 AutomatedModelFactory 決定 feature_cols)"""
        m = self.load_manifest(symbol)
        return m["features"] if m else None

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

        [修正] 找不到 scaler 時回傳 None 而非原值。
        舊版直接回傳原值，等於讓一個 0~1 的縮放值冒充真實價格，
        只能靠下游的 10% 偏差過濾網攔截 —— 那是保險絲，不是正確行為。
        """
        if symbol not in self.scalers or column not in self.scalers[symbol]:
            logger.error(f"[{symbol}] 缺少 {column} 的 scaler，無法還原預測值。")
            return None

        min_val = self.scalers[symbol][column]['min']
        max_val = self.scalers[symbol][column]['max']

        if max_val > min_val:
            return value * (max_val - min_val) + min_val
        return min_val

    def transform_for_inference(self, df: pd.DataFrame, symbol: str) -> Tuple[pd.DataFrame, List[str]]:
        """
        [實盤/回測專用] 依 manifest 產生模型輸入。

        - price_relative 欄位：除以「整段 df 最後一根的收盤價」。
          推論時只需要最後一個時間窗，其錨點就是最後一根收盤價，
          因此與訓練時 create_sequences 的逐窗錨定在語意上完全一致。
        - 其餘欄位：套用訓練時擬合的 Min-Max。
        """
        manifest = self.load_manifest(symbol)
        if manifest is None:
            features = [c for c in _BASE_FEATURES if c in df.columns]
            return self.transform_scale(df, features, symbol), features

        df, features = self.align_to_manifest(df, symbol)
        price_rel = set(manifest.get("price_relative", []))

        anchor = float(df["Close"].iloc[-1]) if "Close" in df.columns and len(df) else 0.0
        if anchor <= 0:
            logger.error(f"[{symbol}] 錨定價格無效，無法產生模型輸入。")
            return df, features

        minmax_cols = [c for c in features if c not in price_rel]
        out = self.transform_scale(df, minmax_cols, symbol)

        for col in features:
            if col in price_rel and col in out.columns:
                out[col] = df[col] / anchor

        return out, features

    def decode_prediction(self, raw: float, current_price: float, symbol: str) -> Optional[float]:
        """
        把模型原始輸出轉回「預測價格」。

        target_mode = "log_return" 時，模型輸出的是下一根 K 線的對數報酬 x target_scale，
        因此 price = current_price * exp(raw / target_scale)。

        [修正] 舊版模型預測的是 Min-Max 縮放後的「價格水位」。預測價格水位時，
        網路最省力的解就是「輸出約等於最後一根輸入」，收斂到近似隨機漫步，
        預測邊際自然趨近於 0 —— 這正是預測幅度長期停在 ±0.3% 的主因。
        """
        if not np.isfinite(raw) or current_price <= 0:
            return None

        manifest = self.load_manifest(symbol)
        mode = manifest.get("target_mode", "level") if manifest else "level"

        if mode == "log_return":
            scale = float(manifest.get("target_scale", 100.0)) or 100.0
            log_ret = float(raw) / scale
            # 單根 K 線的對數報酬超過 ±0.5 (約 ±65%) 必為模型失效
            if abs(log_ret) > 0.5:
                logger.warning(f"[{symbol}] 模型輸出的報酬率異常 ({log_ret:.3f})，已剔除。")
                return None
            return float(current_price * np.exp(log_ret))

        # 舊格式 (價格水位) 的相容路徑
        return self.inverse_transform_scale(raw, "Close", symbol)

    # ------------------------------------------------------------------
    # 序列建構
    # ------------------------------------------------------------------

    def create_sequences(
        self,
        df: pd.DataFrame,
        features: List[str],
        look_back: int,
        raw_close: pd.Series = None,
        price_relative: List[str] = None,
        target_mode: str = "log_return",
        target_scale: float = 100.0,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        建立 LSTM/Transformer 所需的滑動視窗時間序列數據。

        :param df:            已對 minmax 欄位縮放、price_relative 欄位仍為原值的資料
        :param raw_close:     未經任何縮放的原始收盤價 (逐窗錨定與計算目標報酬皆需要)
        :param price_relative: 需要在每個窗內除以該窗最後收盤價的欄位
        :param target_mode:   "log_return" (預設) 或 "level"

        逐窗錨定的因果性：第 i 個窗涵蓋 [i, i+look_back-1]，錨點取 i+look_back-1
        的收盤價，目標則是 i+look_back-1 -> i+look_back 的報酬。錨點只用到窗內
        已觀測到的資料，沒有前視。
        """
        if len(df) <= look_back:
            return np.array([]), np.array([])

        missing = [c for c in features if c not in df.columns]
        if missing:
            raise ValueError(f"建立序列時缺少特徵欄位: {missing}")

        data = df[features].values.astype("float64")

        if raw_close is None:
            raw_close = df["Close"]
        closes = np.asarray(raw_close, dtype="float64")

        price_idx = [features.index(c) for c in (price_relative or []) if c in features]

        X, y = [], []
        for i in range(len(data) - look_back):
            window = data[i:(i + look_back)].copy()
            anchor = closes[i + look_back - 1]

            if price_idx:
                if anchor <= 0 or not np.isfinite(anchor):
                    continue
                window[:, price_idx] = window[:, price_idx] / anchor

            if target_mode == "log_return":
                nxt = closes[i + look_back]
                if anchor <= 0 or nxt <= 0:
                    continue
                target = np.log(nxt / anchor) * target_scale
            else:
                target = df[features[features.index("Close")]].values[i + look_back]

            if not np.isfinite(window).all() or not np.isfinite(target):
                continue

            X.append(window)
            y.append(target)

        if not X:
            return np.array([]), np.array([])
        return np.array(X), np.array(y)