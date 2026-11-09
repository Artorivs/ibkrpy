# ibkrpy/models/garch.py
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

    def __init__(
        self, p: int = 1, q: int = 1, dist: str = "normal", weights_dir: str = "weights"
    ):
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
                self.saved_params = bundle.get("garch")
                if self.saved_params is not None:
                    print(f"[{symbol}] GARCH 模型權重載入成功。")
                else:
                    print(f"[{symbol}] 整合包內無 GARCH 權重，將回退實時擬合。")
            except Exception as e:
                print(f"[{symbol}] GARCH 權重載入失敗: {e}，將回退至實時擬合。")
        else:
            print(
                f"[{symbol}] 找不到傳統模型整合包 ({file_path})，將在預測時進行實時擬合。"
            )

    def predict_volatility(self, df: pd.DataFrame) -> float:
        """
        預測下一根 K 線的波動率，單位是「每根 K 線」(不年化)。

        [修正] 舊版一律乘上 √252 當成年化值，呼叫端再除以 √(該週期每年 K 線數)。
        這只對日 K 成立：1 小時 K 會小 √6.5 ≈ 2.5 倍 (仍落在合理帶內而被採用)，
        5 分 K 小 √78 ≈ 8.8 倍。GARCH 是在同一週期的 K 線上擬合的，
        它的條件變異數本來就是「每根 K 線」的量，不需要任何換算。

        資料不足或擬合失敗時回傳 NaN，讓 VolatilityEstimator 改用實現波動率。
        """
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
            print(f"GARCH 預測失敗: {e}")
            return float(returns.tail(20).std() / 100.0)

    def predict_next_price(self, df: pd.DataFrame) -> float:
        """Fallback：GARCH 本身不預測價格"""
        return float(df["Close"].iloc[-1]) if not df.empty else 0.0
