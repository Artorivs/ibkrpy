# ibkrpy/models/garch.py
# Volatility forecaster used by risk control.

import os
import numpy as np
import pandas as pd
import joblib
import warnings
from arch import arch_model

warnings.filterwarnings("ignore")


class GARCHModel:
    """GARCH model behind the ModelOrchestrator interface."""

    def __init__(
        self, p: int = 1, q: int = 1, dist: str = "normal", weights_dir: str = "weights"
    ):
        self.p = p
        self.q = q
        self.dist = dist
        self.weights_dir = weights_dir
        self.saved_params = None

    def load_weights(self, symbol: str):
        """Load GARCH parameters from the classical bundle (.pkl)."""
        file_path = os.path.join(self.weights_dir, f"{symbol}_classical.pkl")

        if os.path.exists(file_path):
            try:
                bundle = joblib.load(file_path)
                self.saved_params = bundle.get("garch")
                if self.saved_params is not None:
                    print(f"[{symbol}] GARCH weights loaded.")
                else:
                    print(f"[{symbol}] No GARCH in the bundle; will fit on the fly.")
            except Exception as e:
                print(
                    f"[{symbol}] Failed to load GARCH weights: {e}; will fit on the fly."
                )
        else:
            print(
                f"[{symbol}] Classical bundle not found ({file_path}); will fit on the fly."
            )

    def predict_volatility(self, df: pd.DataFrame) -> float:
        """Next-bar volatility per bar (not annualized); NaN when there is too little data."""
        if df.empty or "Close" not in df.columns:
            return float("nan")

        returns = np.log(df["Close"] / df["Close"].shift(1)).dropna() * 100.0

        if len(returns) < 20:
            return float("nan")

        try:
            model = arch_model(returns, vol="Garch", p=self.p, q=self.q, dist=self.dist)

            if self.saved_params is not None:
                res = model.fix(self.saved_params)
            else:
                res = model.fit(disp="off")

            forecasts = res.forecast(horizon=1, method="analytic")
            predicted_var = forecasts.variance.iloc[-1].item()

            return float(np.sqrt(predicted_var) / 100.0)

        except Exception as e:
            print(f"GARCH forecast failed: {e}")
            return float(returns.tail(20).std() / 100.0)

    def predict_next_price(self, df: pd.DataFrame) -> float:
        """Fallback: GARCH does not forecast price."""
        return float(df["Close"].iloc[-1]) if not df.empty else 0.0
