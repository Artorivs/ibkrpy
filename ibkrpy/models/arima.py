# ibkrpy/models/arima.py
# Linear time-series forecaster (trend and mean reversion).

import os
import numpy as np
import pandas as pd
import joblib
import warnings
from statsmodels.tsa.arima.model import ARIMA, ARIMAResultsWrapper

warnings.filterwarnings("ignore")


class ARIMAModel:
    """ARIMA model behind the ModelOrchestrator interface."""

    def __init__(
        self, p: int = 5, d: int = 1, q: int = 0, weights_dir: str = "weights"
    ):
        self.order = (p, d, q)
        self.weights_dir = weights_dir
        self.model_results: ARIMAResultsWrapper = None

    def load_weights(self, symbol: str):
        """Load the fitted ARIMA from the classical bundle (.pkl)."""
        file_path = os.path.join(self.weights_dir, f"{symbol}_classical.pkl")

        if os.path.exists(file_path):
            try:
                bundle = joblib.load(file_path)
                self.model_results = bundle.get("arima")
                if self.model_results:
                    print(f"[{symbol}] ARIMA weights loaded.")
                else:
                    print(f"[{symbol}] No ARIMA in the bundle; will fit on the fly.")
            except Exception as e:
                print(
                    f"[{symbol}] Failed to load ARIMA weights: {e}; will fit on the fly."
                )
        else:
            print(
                f"[{symbol}] Classical bundle not found ({file_path}); will fit on the fly."
            )

    def predict_next_price(self, df: pd.DataFrame) -> float:
        """Forecast the next bar's price."""
        if df.empty or "Close" not in df.columns:
            return 0.0

        series = df["Close"].dropna()
        last_price = float(series.iloc[-1])

        if len(series) < max(self.order) + 1:
            return last_price

        try:
            if self.model_results is not None:
                updated_model = self.model_results.apply(series.values)
                forecast = updated_model.forecast(steps=1)
                return float(forecast[0])
            else:
                model = ARIMA(series.values, order=self.order)
                res = model.fit()
                forecast = res.forecast(steps=1)
                return float(forecast[0])

        except Exception as e:
            print(f"ARIMA forecast failed: {e}")
            return last_price

    def predict_volatility(self, df: pd.DataFrame) -> float:
        """Fallback: historical volatility (ARIMA does not model volatility)."""
        returns = np.log(df["Close"] / df["Close"].shift(1)).dropna()
        vol = returns.tail(20).std() * np.sqrt(252) if len(returns) >= 20 else 0.02
        return float(vol)
