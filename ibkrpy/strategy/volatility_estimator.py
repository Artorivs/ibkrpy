# ibkrpy/strategy/volatility_estimator.py
# 以「同一批 K 線」直接估計每根波動率，並用它驗證 / 取代模型輸出。

from __future__ import annotations

import logging
import math
from dataclasses import dataclass
from typing import Optional

import numpy as np
import pandas as pd

logger = logging.getLogger("ibkrpy.volatility")

_TRADING_DAYS = 252.0

# 每根 K 線的波動率下限 (小數)。低於此值代表估計失敗，不是市場真的不動。
# 最安靜的大型股日波動率也在 0.4% 以上; 取 0.05% 作為極寬鬆的下限。
_ABS_FLOOR_PER_BAR = 0.0005

# 模型值相對實現值的可接受倍數帶。超出即視為尺度錯誤。
_BAND_LO = 1.0 / 3.0
_BAND_HI = 3.0


@dataclass(frozen=True)
class VolatilityEstimate:
    """一次波動率估計的完整結果，含來源與診斷資訊。"""

    value: float  # 每根 K 線的波動率 (小數, 恆為正)
    source: str  # "model" | "realized" | "floor"
    realized: float  # 實現波動率 (每根 K 線, 小數)
    model: Optional[float]  # 模型給出的值 (每根 K 線, 小數); None 表示沒有
    ratio: Optional[float]  # model / realized
    detail: str = ""

    @property
    def annualized(self) -> float:
        """僅供顯示。回推年化需要 bars_per_year，故由呼叫端另行提供。"""
        return self.value * math.sqrt(_TRADING_DAYS)


# ----------------------------------------------------------------------
# 實現波動率
# ----------------------------------------------------------------------


def _close_to_close(
    close: pd.Series, lookback: int, halflife: Optional[float]
) -> float:
    """對數報酬的標準差。halflife 有值時改用 EWMA (較貼近近期狀態)。"""
    rets = np.log(close / close.shift(1)).dropna()
    if len(rets) < 5:
        return float("nan")
    rets = rets.iloc[-lookback:]
    if halflife:
        # EWMA 變異數: 近期樣本權重較高，但不像 GARCH 那樣需要擬合參數。
        weights = 0.5 ** (np.arange(len(rets))[::-1] / float(halflife))
        weights /= weights.sum()
        mean = float(np.sum(weights * rets.to_numpy()))
        var = float(np.sum(weights * (rets.to_numpy() - mean) ** 2))
        return math.sqrt(max(var, 0.0))
    return float(rets.std(ddof=1))


def _yang_zhang(df: pd.DataFrame, lookback: int) -> float:
    """
    Yang-Zhang 估計量。同時吸收隔夜跳空與盤中波動，對同樣的樣本數，
    效率約為單純收盤對收盤的 5~7 倍 —— 在只有 60 根 K 線可用時差別很大。

    需要 Open/High/Low/Close 四欄; 缺欄或資料不足時回傳 nan 由呼叫端退回。
    """
    need = ("Open", "High", "Low", "Close")
    if not all(c in df.columns for c in need):
        return float("nan")

    win = df.iloc[-(lookback + 1) :]
    o = pd.to_numeric(win["Open"], errors="coerce")
    h = pd.to_numeric(win["High"], errors="coerce")
    l = pd.to_numeric(win["Low"], errors="coerce")
    c = pd.to_numeric(win["Close"], errors="coerce")

    prev_c = c.shift(1)
    frame = pd.DataFrame({"o": o, "h": h, "l": l, "c": c, "pc": prev_c}).dropna()
    frame = frame[(frame > 0).all(axis=1)]
    n = len(frame)
    if n < 5:
        return float("nan")

    # 隔夜 (前收 -> 開盤)
    overnight = np.log(frame["o"] / frame["pc"])
    # 盤中 (開盤 -> 收盤)
    open_close = np.log(frame["c"] / frame["o"])
    # Rogers-Satchell: 對漂移免疫的盤中估計
    rs = np.log(frame["h"] / frame["c"]) * np.log(frame["h"] / frame["o"]) + np.log(
        frame["l"] / frame["c"]
    ) * np.log(frame["l"] / frame["o"])

    var_o = float(overnight.var(ddof=1))
    var_c = float(open_close.var(ddof=1))
    var_rs = float(rs.mean())

    k = 0.34 / (1.34 + (n + 1) / (n - 1))
    var = var_o + k * var_c + (1 - k) * var_rs
    if not np.isfinite(var) or var <= 0:
        return float("nan")
    return math.sqrt(var)


def realized_volatility(
    df: pd.DataFrame, lookback: int = 60, halflife: Optional[float] = 20.0
) -> float:
    """
    每根 K 線的實現波動率 (小數)。與傳入的 K 線同尺度，不做任何年化或去年化。

    優先 Yang-Zhang; 缺 OHLC 或退化時退回 EWMA 收盤對收盤。
    兩者皆失敗回傳 nan，由 VolatilityEstimator 決定如何處理。
    """
    if df is None or df.empty or "Close" not in df.columns:
        return float("nan")

    yz = _yang_zhang(df, lookback)
    if np.isfinite(yz) and yz > 0:
        return yz

    close = pd.to_numeric(df["Close"], errors="coerce").dropna()
    ctc = _close_to_close(close, lookback, halflife)
    return ctc if np.isfinite(ctc) and ctc > 0 else float("nan")


# ----------------------------------------------------------------------
# 估計器
# ----------------------------------------------------------------------


class VolatilityEstimator:
    """
    以實現波動率為基準，決定最終要交給策略的 σ。

    契約: estimate() 永遠回傳一個有限的正值，永不拋例外。
    這一點很重要 —— σ 有四個下游消費者 (門檻政策、出場幾何的 noise_floor 與
    risk_cap、情境判定、風險平價倉位縮放)，任何一個拿到 nan 都會靜默失效。
    """

    def __init__(
        self,
        lookback: int = 60,
        halflife: float = 20.0,
        band_lo: float = _BAND_LO,
        band_hi: float = _BAND_HI,
        abs_floor: float = _ABS_FLOOR_PER_BAR,
        trust_model: bool = True,
    ):
        self.lookback = int(lookback)
        self.halflife = float(halflife)
        self.band_lo = float(band_lo)
        self.band_hi = float(band_hi)
        self.abs_floor = float(abs_floor)
        self.trust_model = bool(trust_model)
        self._warned: set = set()

    def estimate(
        self,
        symbol: str,
        df: pd.DataFrame,
        model_vol_per_bar: Optional[float] = None,
    ) -> VolatilityEstimate:
        """
        :param df:                策略本輪實際使用的 K 線 (與 σ 同尺度)
        :param model_vol_per_bar: 模型給出的「每根 K 線」波動率。若手上只有年化值，
                                  請先自行換算後傳入 —— 本方法不猜測慣例。
        """
        realized = realized_volatility(df, self.lookback, self.halflife)

        if not np.isfinite(realized) or realized <= 0:
            # 連實現波動率都算不出來，代表 K 線本身有問題。
            fallback = max(self.abs_floor, 0.01)
            return VolatilityEstimate(
                fallback,
                "floor",
                float("nan"),
                model_vol_per_bar,
                None,
                "K 線不足以估計實現波動率，改用保守下限",
            )

        realized = max(realized, self.abs_floor)

        if not self.trust_model or model_vol_per_bar is None:
            return VolatilityEstimate(
                realized,
                "realized",
                realized,
                model_vol_per_bar,
                None,
                "直接採用實現波動率",
            )

        model = float(model_vol_per_bar)
        if not np.isfinite(model) or model <= 0:
            return VolatilityEstimate(
                realized, "realized", realized, model, None, "模型值非有限或非正"
            )

        ratio = model / realized
        if self.band_lo <= ratio <= self.band_hi:
            return VolatilityEstimate(
                model,
                "model",
                realized,
                model,
                ratio,
                f"模型值在合理帶內 (×{ratio:.2f})",
            )

        # 超出合理帶 —— 幾乎都是尺度錯誤，不是市場異常。
        if symbol not in self._warned:
            self._warned.add(symbol)
            logger.error(
                f"🌡️ [{symbol}] 模型波動率 {model * 100:.3f}% 與實現波動率 "
                f"{realized * 100:.3f}% 相差 {ratio:.2f} 倍，已改用實現值。"
                f"這通常代表該標的的 GARCH 權重尺度錯誤，需要重新訓練。"
            )
        return VolatilityEstimate(
            realized,
            "realized",
            realized,
            model,
            ratio,
            f"模型值偏離 {ratio:.2f} 倍，超出 [{self.band_lo:.2f}, {self.band_hi:.2f}]",
        )


def build_volatility_estimator(config) -> VolatilityEstimator:
    """Composition Root 使用。"""
    s = (config.get("volatility_settings") or {}) if config else {}
    return VolatilityEstimator(
        lookback=int(s.get("lookback", 60)),
        halflife=float(s.get("ewma_halflife", 20.0)),
        band_lo=float(s.get("model_band_lo", _BAND_LO)),
        band_hi=float(s.get("model_band_hi", _BAND_HI)),
        abs_floor=float(s.get("abs_floor_per_bar", _ABS_FLOOR_PER_BAR)),
        trust_model=bool(s.get("trust_model", True)),
    )
