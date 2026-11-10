# ibkrpy/data/data_pipeline.py
# Data pipeline: feature engineering, feature manifest, scaling and sequence building

import os
import logging
import pandas as pd
import numpy as np
from typing import Tuple, List, Dict, Optional

PROJECT_ROOT = os.path.dirname(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
)
WEIGHTS_DIR = os.path.join(PROJECT_ROOT, "weights")

logger = logging.getLogger("ibkrpy")

_EXCLUDED_FEATURES = {"symbol", "timeframe"}

_BASE_FEATURES = ["Open", "High", "Low", "Close", "Volume"]

_PRICE_RELATIVE_PREFIXES = (
    "Open",
    "High",
    "Low",
    "Close",
    "SMA_",
    "EMA_",
    "WMA_",
    "DEMA_",
    "TEMA_",
    "HMA_",
    "VWAP",
    "BBL_",
    "BBM_",
    "BBU_",
    "ATR_",
    "ATRr_",
    "TRUERANGE",
    "MACD_",
    "MACDh_",
    "MACDs_",
    "HL2",
    "HLC3",
    "OHLC4",
    "STDEV",
)

_NEVER_PRICE_RELATIVE = (
    "Volume",
    "RSI_",
    "BBB_",
    "BBP_",
    "ADX_",
    "DMP_",
    "DMN_",
    "STOCH",
    "CCI_",
    "MFI_",
    "WILLR_",
    "ROC_",
    "MOM_",
    "VIX",
    "bench_return",
    "bench_correlation",
)


class DataPipeline:
    """Local data cache access and ML preprocessing."""

    def __init__(self, artifact_store=None, config=None):
        self.scalers: Dict[str, Dict] = {}

        self.config = config
        if artifact_store is None:
            from ibkrpy.data.artifact_store import build_artifact_store

            artifact_store = build_artifact_store(WEIGHTS_DIR, config)
        self.artifacts = artifact_store
        self._manifests: Dict[str, List[str]] = {}

    def add_technical_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        """Append the default technical indicators (pandas_ta 0.3.14b column names)."""
        from ibkrpy.shared.indicators import add_default_indicators

        return add_default_indicators(df).dropna()

    def engineer_advanced_features(
        self,
        df: pd.DataFrame,
        benchmark_df: pd.DataFrame = None,
        macro_dict: Dict[str, pd.Series] = None,
    ) -> pd.DataFrame:
        """Feature engineering: technical indicators, benchmark correlation and macro data."""
        df_adv = self.add_technical_indicators(df)

        if benchmark_df is not None and not benchmark_df.empty:
            bench_ret = np.log(benchmark_df["Close"] / benchmark_df["Close"].shift(1))
            df_adv["bench_return"] = bench_ret.reindex(df_adv.index).fillna(0)
            stock_ret = np.log(df_adv["Close"] / df_adv["Close"].shift(1))
            df_adv["bench_correlation"] = (
                stock_ret.rolling(20).corr(df_adv["bench_return"]).fillna(0)
            )

        if macro_dict:
            for name, series in macro_dict.items():
                if series is not None and not series.empty:
                    df_adv[name] = series.reindex(df_adv.index).ffill().bfill()

        return df_adv.dropna()

    def select_model_features(self, df: pd.DataFrame) -> List[str]:
        """
        Choose the columns from engineer_advanced_features that go into the model.

        Rules:
          - OHLCV is always included and always first (same column order in training and inference)
          - then every numeric column with no NaN/Inf that is not constant, sorted by name
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
            if float(series.max()) == float(series.min()):
                continue
            extras.append(col)

        return features + sorted(extras)

    @staticmethod
    def classify_price_relative(df: pd.DataFrame, features: List[str]) -> List[str]:
        """
        Find columns that scale proportionally with price.

        These cannot use Min-Max: once price breaks the training window's min/max (almost
        certain for a trending stock), scaled values leave [0, 1] and the model extrapolates.
        Instead each window is divided by its last close, giving a dimensionless value.

        Detection:
          1. Mainly by name prefix. We define which columns add_technical_indicators makes,
             so listing them is more reliable than numeric guessing.
          2. Unknown names fall back to a strict numeric test: col/Close coefficient of
             variation < 0.05 and ratio within [0.5, 2.0]. Plain std misfires, since in a
             strong trend RSI/Close also has a small absolute std.
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
            if col in _NEVER_PRICE_RELATIVE or any(
                col.startswith(p) for p in _NEVER_PRICE_RELATIVE
            ):
                continue

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

    def save_feature_manifest(
        self,
        symbol: str,
        features: List[str],
        price_relative: List[str] = None,
        target_mode: str = "log_return",
        target_scale: float = 100.0,
        benchmark: str = None,
    ):
        """[Training only] Save feature columns, their scaling and the target definition."""
        manifest = {
            "features": list(features),
            "price_relative": list(price_relative or []),
            "target_mode": target_mode,
            "target_scale": target_scale,
            "benchmark": (str(benchmark).upper() if benchmark else None),
        }
        self.artifacts.save_manifest(symbol, manifest)
        self._manifests[symbol] = manifest
        logger.debug(
            f"[{symbol}] Feature manifest saved ({len(features)} columns, "
            f"{len(manifest['price_relative'])} price-relative; "
            f"target: {target_mode}; benchmark: {manifest['benchmark'] or 'none'})."
        )

    def load_manifest(self, symbol: str) -> Optional[Dict]:
        """Read back the full manifest. None means the symbol is not trained yet."""
        if symbol in self._manifests:
            return self._manifests[symbol]

        try:
            manifest = self.artifacts.load_manifest(symbol)
            if not manifest or not manifest.get("features"):
                return None
            manifest.setdefault("price_relative", [])
            manifest.setdefault("target_mode", "level")
            manifest.setdefault("target_scale", 1.0)
            manifest.setdefault("benchmark", None)
            self._manifests[symbol] = manifest
            return manifest
        except Exception as e:
            logger.error(f"[{symbol}] Failed to load feature manifest: {e}")
            return None

    def load_feature_manifest(self, symbol: str) -> Optional[List[str]]:
        """Feature column list only (used by AutomatedModelFactory for feature_cols)."""
        m = self.load_manifest(symbol)
        return m["features"] if m else None

    def align_to_manifest(
        self, df: pd.DataFrame, symbol: str
    ) -> Tuple[pd.DataFrame, List[str]]:
        """
        Align an inference DataFrame to the training feature manifest.

        Columns present in training but missing at inference (e.g. bench_correlation when the
        benchmark fetch failed) are filled with 0 and logged; silent filling would feed the
        model the wrong distribution unnoticed.
        """
        features = self.load_feature_manifest(symbol)
        if features is None:
            return df, [c for c in _BASE_FEATURES if c in df.columns]

        missing = [c for c in features if c not in df.columns]
        if missing:
            logger.warning(
                f"[{symbol}] Inference data is missing {len(missing)} training features, filled with 0: {missing}"
            )
            df = df.copy()
            for c in missing:
                df[c] = 0.0

        return df, features

    def save_training_artifacts(
        self,
        symbol: str,
        scaler: Dict,
        features: List[str],
        price_relative: List[str] = None,
        target_mode: str = "log_return",
        target_scale: float = 100.0,
        benchmark: str = None,
    ):
        """
        [Training only] Write scaler and manifest in one atomic step.

        Calling save_feature_manifest() and fit_scale() separately can leave a new manifest
        with an old scaler on disk if the second write fails, with no error raised.
        """
        manifest = {
            "features": list(features),
            "price_relative": list(price_relative or []),
            "target_mode": target_mode,
            "target_scale": target_scale,
            "benchmark": (str(benchmark).upper() if benchmark else None),
        }
        self.artifacts.save_bundle(symbol, scaler=scaler, manifest=manifest)
        self.scalers[symbol] = scaler
        self._manifests[symbol] = manifest
        logger.debug(
            f"[{symbol}] Training artifacts written atomically ({len(features)} features, "
            f"{len(scaler)} scaled columns)."
        )

    def invalidate(self, symbol: str = None):
        """
        Clear the in-memory cache. Must be called after retraining, or an old scaler meets a new model.

        Scaler and manifest are two halves of one training run and must be cleared together.
        Other modules should call this method rather than touching the two dicts.
        """
        if symbol is None:
            self.scalers.clear()
            self._manifests.clear()
        else:
            self.scalers.pop(symbol, None)
            self._manifests.pop(symbol, None)

    @staticmethod
    def _compute_scaler(
        df: pd.DataFrame, columns: List[str]
    ) -> Dict[str, Dict[str, float]]:
        """Pure computation with no side effects (no file writes, self.scalers untouched)."""
        scaler_dict = {}
        for col in columns:
            if col in df.columns:
                scaler_dict[col] = {
                    "min": float(df[col].min()),
                    "max": float(df[col].max()),
                }
        return scaler_dict

    def fit_scale(self, df: pd.DataFrame, columns: List[str], symbol: str):
        """
        [Training only] Call on the training set only.

        This writes to disk, so it must never be called from a read path such as
        transform_scale. ScalerStore decides where it is written.
        """
        scaler_dict = self._compute_scaler(df, columns)
        self.scalers[symbol] = scaler_dict
        self.artifacts.save_scaler(symbol, scaler_dict)

        if "__wf" not in symbol:
            logger.info(f"[{symbol}] Feature scaler fitted and saved.")
        else:
            logger.debug(f"[{symbol}] Feature scaler fitted and saved.")

    def transform_scale(
        self, df: pd.DataFrame, columns: List[str], symbol: str
    ) -> pd.DataFrame:
        """
        [Backtest and live] Transform with the Min-Max parameters fitted in training.

        If no scaler exists, one is computed in memory only and not saved. That means the
        symbol is untrained and its predictions should not be trusted.
        """
        df_scaled = df.copy()

        if symbol not in self.scalers:
            stored = self.artifacts.load_scaler(symbol)
            if stored is not None:
                self.scalers[symbol] = stored
            else:
                logger.error(
                    f"[{symbol}] No pretrained scaler; computing in memory only (not saved). "
                    f"The symbol is untrained and its predictions are unreliable; run --mode train."
                )
                self.scalers[symbol] = self._compute_scaler(df, columns)

        scaler_dict = self.scalers[symbol]

        for col in columns:
            if col not in df.columns or col not in scaler_dict:
                continue

            min_val = scaler_dict[col]["min"]
            max_val = scaler_dict[col]["max"]

            if max_val > min_val:
                df_scaled[col] = (df_scaled[col] - min_val) / (max_val - min_val)
            else:
                df_scaled[col] = 0.5

        return df_scaled

    def inverse_transform_scale(
        self, value: float, column: str, symbol: str
    ) -> Optional[float]:
        """
        Invert scaling (used to recover predicted prices).

        Returns None when no scaler exists, rather than passing a 0-1 scaled value off as a price.
        """
        if symbol not in self.scalers or column not in self.scalers[symbol]:
            logger.error(
                f"[{symbol}] No scaler for {column}; cannot invert the prediction."
            )
            return None

        min_val = self.scalers[symbol][column]["min"]
        max_val = self.scalers[symbol][column]["max"]

        if max_val > min_val:
            return value * (max_val - min_val) + min_val
        return min_val

    def transform_for_inference(
        self, df: pd.DataFrame, symbol: str
    ) -> Tuple[pd.DataFrame, List[str]]:
        """
        [Live and backtest] Build model input from the manifest.

        - price_relative columns: divided by the last close of the whole df. Inference only
            needs the last window, whose anchor is that close, so this matches the per-window
            anchoring in create_sequences during training.
        - Other columns: Min-Max fitted in training.
        """
        manifest = self.load_manifest(symbol)
        if manifest is None:
            features = [c for c in _BASE_FEATURES if c in df.columns]
            return self.transform_scale(df, features, symbol), features

        df, features = self.align_to_manifest(df, symbol)
        price_rel = set(manifest.get("price_relative", []))

        anchor = (
            float(df["Close"].iloc[-1]) if "Close" in df.columns and len(df) else 0.0
        )
        if anchor <= 0:
            logger.error(f"[{symbol}] Invalid anchor price; cannot build model input.")
            return df, features

        minmax_cols = [c for c in features if c not in price_rel]
        out = self.transform_scale(df, minmax_cols, symbol)

        for col in features:
            if col in price_rel and col in out.columns:
                out[col] = df[col] / anchor

        return out, features

    def decode_prediction(
        self, raw: float, current_price: float, symbol: str
    ) -> Optional[float]:
        """
        Convert raw model output back to a predicted price.

        With target_mode = "log_return" the model outputs the next bar's log return x target_scale,
        so price = current_price * exp(raw / target_scale).
        """
        if not np.isfinite(raw) or current_price <= 0:
            return None

        manifest = self.load_manifest(symbol)
        mode = manifest.get("target_mode", "level") if manifest else "level"

        if mode == "log_return":
            scale = float(manifest.get("target_scale", 100.0)) or 100.0
            log_ret = float(raw) / scale
            if abs(log_ret) > 0.5:
                logger.warning(
                    f"[{symbol}] Abnormal model return output ({log_ret:.3f}); discarded."
                )
                return None
            return float(current_price * np.exp(log_ret))

        return self.inverse_transform_scale(raw, "Close", symbol)

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
        Build sliding-window sequences for LSTM/Transformer.

        :param df:            data with minmax columns scaled and price_relative columns raw
        :param raw_close:     unscaled close (needed for per-window anchors and target returns)
        :param price_relative: columns divided by each window's last close
        :param target_mode:   "log_return" (default) or "level"

        Causality: window i covers [i, i+look_back-1], anchored on the close at i+look_back-1;
        the target is the return from i+look_back-1 to i+look_back. The anchor only uses data
        observed inside the window, so there is no lookahead.
        """
        if len(df) <= look_back:
            return np.array([]), np.array([])

        missing = [c for c in features if c not in df.columns]
        if missing:
            raise ValueError(
                f"Missing feature columns when building sequences: {missing}"
            )

        data = df[features].values.astype("float64")

        if raw_close is None:
            raw_close = df["Close"]
        closes = np.asarray(raw_close, dtype="float64")

        price_idx = [features.index(c) for c in (price_relative or []) if c in features]

        X, y = [], []
        for i in range(len(data) - look_back):
            window = data[i : (i + look_back)].copy()
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
