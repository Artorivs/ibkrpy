# ibkrpy/models/hmm.py
# Hidden-state (regime) detection, e.g. bull/bear or high/low volatility.

import os
import numpy as np
import pandas as pd
import joblib
import warnings

warnings.filterwarnings("ignore")

try:
    from hmmlearn.hmm import GaussianHMM
except ImportError:
    GaussianHMM = None
    print("Warning: hmmlearn is not installed; run `pip install hmmlearn`")


class HMMModel:
    """
    Hidden Markov Model
    that detects hidden market states (e.g. 0 = quiet range, 1 = volatile trend).
    """

    def __init__(self, n_components: int = 2, weights_dir: str = "weights"):
        """
        Initialize the HMM.
        :param n_components: number of hidden states (default 2, e.g. bull/bear)
        :param weights_dir: directory holding model weights
        """
        self.n_components = n_components
        self.weights_dir = weights_dir
        self.model = None
        self.is_trained = False

    def load_weights(self, symbol: str):
        """Load the HMM from the classical bundle (.pkl)."""
        if GaussianHMM is None:
            raise ImportError("hmmlearn is not installed; cannot load the HMM.")

        file_path = os.path.join(self.weights_dir, f"{symbol}_classical.pkl")

        if not os.path.exists(file_path):
            raise FileNotFoundError(
                f"Classical bundle for {symbol} not found: {file_path}"
            )

        try:
            bundle = joblib.load(file_path)
            self.model = bundle.get("hmm")
            if self.model:
                self.is_trained = True
                print(f"[{symbol}] HMM weights loaded.")
            else:
                raise ValueError("No HMM in the bundle")
        except Exception as e:
            raise RuntimeError(f"[{symbol}] Failed to load the HMM: {e}")

    def _prepare_features(self, df: pd.DataFrame) -> np.ndarray:
        """
        Turn OHLCV into HMM observations:
        log return and volatility.
        """
        df_features = pd.DataFrame(index=df.index)

        df_features["log_return"] = np.log(df["Close"] / df["Close"].shift(1)) * 100.0

        df_features["volatility"] = df_features["log_return"].rolling(window=5).std()

        df_features = df_features.dropna()

        return df_features[["log_return", "volatility"]].values

    def predict_state(self, df: pd.DataFrame) -> int:
        """
        Predict the latest market state.
        Called by ModelOrchestrator.
        :param df: DataFrame of historical bars
        :return: state ID (int)
        """
        if not self.is_trained or self.model is None:
            print("Warning: HMM not loaded; returning default state -1")
            return -1

        features = self._prepare_features(df)

        if len(features) == 0:
            print("Warning: not enough data to compute HMM features")
            return -1

        try:
            hidden_states = self.model.predict(features)

            current_state = int(hidden_states[-1])
            return current_state

        except Exception as e:
            print(f"HMM state prediction failed: {e}")
            return -1
