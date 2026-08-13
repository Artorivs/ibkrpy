# ibkrpy/strategy/regime_detector.py
# 市場情境偵測：具備遲滯、非對稱確認，以及頂部反轉風險評分。

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Dict, List, Optional

import numpy as np
import pandas as pd
from .strategy_components import MarketRegime

logger = logging.getLogger("ibkrpy.regime")


@dataclass
class RegimeAssessment:
    """一次情境評估的完整結果。

    reversal_risk 是本類別存在的理由：它讓「多頭仍成立，但正在轉弱」
    成為一個可以被表達、被記錄、被據以行動的狀態，而不必等到
    均線交叉完成才發現趨勢已經結束。
    """

    regime: MarketRegime
    reversal_risk: float = 0.0  # 0~1，趨勢反轉的近期風險
    trend_strength: float = 0.0  # 0~1，由 ADX 正規化而來
    bars_in_regime: int = 0
    raw_regime: MarketRegime = MarketRegime.SIDEWAYS_QUIET  # 未經遲滯的即時判定
    reasons: List[str] = field(default_factory=list)

    @property
    def is_trending(self) -> bool:
        return self.regime in (MarketRegime.BULL_TREND, MarketRegime.BEAR_TREND)

    @property
    def is_turning(self) -> bool:
        """趨勢正在轉弱，但尚未翻轉成反向趨勢。"""
        return self.reversal_risk >= 0.5

    def describe(self) -> str:
        parts = [self.regime.name]
        if self.regime is not self.raw_regime:
            parts.append(f"(即時判定 {self.raw_regime.name}，遲滯中)")
        parts.append(f"反轉風險 {self.reversal_risk:.0%}")
        if self.reasons:
            parts.append("| " + " · ".join(self.reasons))
        return " ".join(parts)


def _safe_last(series: Optional[pd.Series], default: float = np.nan) -> float:
    """取序列最後一個有限值。取不到就回傳 default，並讓呼叫端決定如何處理。"""
    if series is None or len(series) == 0:
        return default
    value = pd.to_numeric(series, errors="coerce").dropna()
    if value.empty:
        return default
    return float(value.iloc[-1])


def _clamp01(value: float) -> float:
    if not np.isfinite(value):
        return 0.0
    return float(min(max(value, 0.0), 1.0))


def wilder_atr(df: pd.DataFrame, period: int) -> pd.Series:
    """Wilder 平均真實區間。

    自行實作而非依賴 pandas_ta：情境判定是決策路徑上的必經之處，
    不該因為一個選用套件缺席就整個退化成「盤整」。
    """
    high = pd.to_numeric(df["High"], errors="coerce")
    low = pd.to_numeric(df["Low"], errors="coerce")
    close = pd.to_numeric(df["Close"], errors="coerce")
    prev_close = close.shift(1)

    true_range = pd.concat(
        [high - low, (high - prev_close).abs(), (low - prev_close).abs()], axis=1
    ).max(axis=1)
    return true_range.ewm(alpha=1.0 / period, adjust=False, min_periods=period).mean()


def wilder_adx(df: pd.DataFrame, period: int) -> pd.Series:
    """Wilder 平均趨向指數 (ADX)。回傳單一序列，與舊版取用第一欄的行為一致。"""
    high = pd.to_numeric(df["High"], errors="coerce")
    low = pd.to_numeric(df["Low"], errors="coerce")

    up_move = high.diff()
    down_move = -low.diff()

    plus_dm = up_move.where((up_move > down_move) & (up_move > 0), 0.0)
    minus_dm = down_move.where((down_move > up_move) & (down_move > 0), 0.0)

    atr = wilder_atr(df, period)
    alpha = 1.0 / period
    plus_di = (
        100.0 * plus_dm.ewm(alpha=alpha, adjust=False, min_periods=period).mean() / atr
    )
    minus_di = (
        100.0 * minus_dm.ewm(alpha=alpha, adjust=False, min_periods=period).mean() / atr
    )

    denominator = (plus_di + minus_di).replace(0, np.nan)
    dx = 100.0 * (plus_di - minus_di).abs() / denominator
    return dx.ewm(alpha=alpha, adjust=False, min_periods=period).mean()


class ReversalRiskModel:
    """把數個獨立訊號合成為 0~1 的反轉風險分數。

    每個訊號都是一個純函式：吃 DataFrame，吐 0~1。彼此不知道對方存在，
    新增訊號只要多一個方法並登記權重即可 (Open/Closed)。
    """

    def __init__(self, weights: Dict[str, float] = None):
        self.weights = weights or {
            "momentum_fade": 0.22,
            "price_below_fast_ma": 0.20,
            "lower_highs": 0.20,
            "trend_exhaustion": 0.13,
            "volatility_expansion": 0.13,
            "distribution_volume": 0.12,
        }

    @staticmethod
    def momentum_fade(df: pd.DataFrame, fast: int, slow: int) -> float:
        """快線的斜率轉平或轉負 → 上升動能正在消失。

        衡量的是斜率而非乖離：盤頭期間快慢線的乖離可能還是正的，
        但快線已經走平——那才是最早出現的訊號。斜率以自身近期的
        正斜率常態做正規化，因此不受標的價位高低影響。
        """
        close = df["Close"]
        if len(close) < slow + fast:
            return 0.0

        ma_fast = close.rolling(fast).mean()
        slope = ma_fast.diff(5) / ma_fast.replace(0, np.nan)
        recent = slope.dropna()
        if len(recent) < 20:
            return 0.0

        now = float(recent.iloc[-1])
        positive = recent.tail(60)
        positive = positive[positive > 0]
        if positive.empty:
            return 1.0 if now <= 0 else 0.0

        norm = float(positive.median())
        if norm <= 0:
            return 0.0
        # 斜率從常態正值降到 0 → 0.7 分；轉為同幅度負值 → 滿分
        return _clamp01((norm - now) / (2.0 * norm))

    @staticmethod
    def price_below_fast_ma(df: pd.DataFrame, fast: int) -> float:
        """收盤跌破快線，且跌幅越深分數越高。多頭轉弱最直接的徵兆。"""
        close = df["Close"]
        if len(close) < fast + 2:
            return 0.0
        ma_fast = _safe_last(close.rolling(fast).mean())
        last = float(close.iloc[-1])
        if not np.isfinite(ma_fast) or ma_fast <= 0:
            return 0.0
        gap = (ma_fast - last) / ma_fast
        # 跌破 3% 視為滿分
        return _clamp01(gap / 0.03)

    @staticmethod
    def trend_exhaustion(df: pd.DataFrame, adx: Optional[pd.Series]) -> float:
        """ADX 從高點回落 → 趨勢正在耗盡。"""
        if adx is None or len(adx.dropna()) < 6:
            return 0.0
        recent = adx.dropna().tail(8)
        peak = float(recent.max())
        now = float(recent.iloc[-1])
        if peak < 20.0:
            return 0.0
        return _clamp01((peak - now) / peak)

    @staticmethod
    def lower_highs(df: pd.DataFrame, lookback: int) -> float:
        """距離區間高點的回落幅度 → 頭部特徵。"""
        high = df["High"]
        close = df["Close"]
        if len(high) < lookback + 2:
            return 0.0
        window_high = float(high.tail(lookback).max())
        last_close = float(close.iloc[-1])
        if window_high <= 0:
            return 0.0
        drawdown = (window_high - last_close) / window_high
        return _clamp01(drawdown / 0.06)

    @staticmethod
    def volatility_expansion(df: pd.DataFrame, atr: Optional[pd.Series]) -> float:
        """波動率相對自身基線放大 → 分歧加劇，多頭末端常見。"""
        if atr is None or len(atr.dropna()) < 25:
            return 0.0
        series = atr.dropna()
        baseline = float(series.tail(25).median())
        now = float(series.iloc[-1])
        if baseline <= 0:
            return 0.0
        return _clamp01((now / baseline - 1.0) / 0.6)

    @staticmethod
    def distribution_volume(df: pd.DataFrame, lookback: int) -> float:
        """下跌K棒的成交量大於上漲K棒 → 派發 (出貨) 特徵。"""
        if "Volume" not in df.columns or len(df) < lookback + 1:
            return 0.0
        recent = df.tail(lookback)
        change = recent["Close"].diff()
        volume = pd.to_numeric(recent["Volume"], errors="coerce")
        down = volume[change < 0].sum()
        up = volume[change > 0].sum()
        if up <= 0:
            return 1.0 if down > 0 else 0.0
        return _clamp01((down / up - 1.0) / 0.8)

    def score(
        self,
        df: pd.DataFrame,
        adx: Optional[pd.Series],
        atr: Optional[pd.Series],
        fast: int,
        slow: int,
        lookback: int,
    ) -> tuple[float, List[str]]:
        signals = {
            "momentum_fade": self.momentum_fade(df, fast, slow),
            "price_below_fast_ma": self.price_below_fast_ma(df, fast),
            "lower_highs": self.lower_highs(df, lookback),
            "trend_exhaustion": self.trend_exhaustion(df, adx),
            "volatility_expansion": self.volatility_expansion(df, atr),
            "distribution_volume": self.distribution_volume(df, lookback),
        }
        total_weight = sum(self.weights.get(k, 0.0) for k in signals)
        if total_weight <= 0:
            return 0.0, []

        score = (
            sum(v * self.weights.get(k, 0.0) for k, v in signals.items()) / total_weight
        )
        reasons = [
            f"{name} {value:.0%}"
            for name, value in sorted(signals.items(), key=lambda kv: -kv[1])
            if value >= 0.4
        ]
        return _clamp01(score), reasons


class MarketRegimeDetector:
    """以 ADX、均線與 ATR 判定情境，並附帶遲滯與反轉風險。

    與舊版的三個關鍵差異：

    1. 預設週期拉長 (5/10 → 20/50)。5/10 日均線在盤中會被未收盤的當根 K
       反覆穿越，日誌顯示這造成每日多達 8 次的情境翻轉。
    2. 遲滯 (hysteresis)：情境要連續數根 K 線維持同一判定才真正切換。
       且**非對稱**——轉為空頭所需的確認根數少於轉為多頭。市場下跌
       比上漲快，用同一組參數兩邊對稱處理，等於在出場時故意反應遲鈍。
    3. 反轉風險評分：在 BEAR_TREND 被確認之前就開始上升，填補
       「多頭已死但空頭未立」的那段空窗。
    """

    def __init__(self, config: Dict[str, object] = None):
        cfg = config or {}

        self.adx_period = int(cfg.get("regime_adx_period", 14))
        self.ma_short = int(cfg.get("regime_ma_short", 20))
        self.ma_long = int(cfg.get("regime_ma_long", 50))
        self.atr_period = int(cfg.get("regime_atr_period", 14))
        self.adx_threshold = float(cfg.get("regime_adx_trend_threshold", 20.0))
        self.vol_threshold = float(cfg.get("regime_volatility_threshold_pct", 0.02))

        # 非對稱確認：轉空比轉多快。這是刻意的防禦性偏誤。
        self.bull_confirm_bars = int(cfg.get("regime_bull_confirm_bars", 3))
        self.bear_confirm_bars = int(cfg.get("regime_bear_confirm_bars", 1))

        # 盤中重複評估時，未收盤的當根 K 會讓均線交叉反覆穿越。
        self.use_closed_bars_only = bool(cfg.get("regime_use_closed_bars_only", True))

        self.reversal_lookback = int(cfg.get("regime_reversal_lookback", 20))
        self.reversal_model = ReversalRiskModel(cfg.get("regime_reversal_weights"))

        self._state: Dict[str, Dict[str, object]] = {}

    def _indicators(self, df: pd.DataFrame) -> Dict[str, object]:
        """計算指標。失敗時記錄原因並回傳 None，由呼叫端明確處理。"""
        result: Dict[str, object] = {"adx": None, "atr": None}
        try:
            result["adx"] = wilder_adx(df, self.adx_period)
            result["atr"] = wilder_atr(df, self.atr_period)
        except Exception as exc:
            logger.warning(f"情境指標計算失敗，本次退回中性判定: {exc}")
        return result

    def _classify(
        self, df: pd.DataFrame, adx_value: float, atr_pct: float
    ) -> MarketRegime:
        close = df["Close"]
        ma_short = _safe_last(close.rolling(self.ma_short).mean())
        ma_long = _safe_last(close.rolling(self.ma_long).mean())

        if not np.isfinite(ma_short) or not np.isfinite(ma_long):
            return MarketRegime.SIDEWAYS_QUIET

        is_trending = np.isfinite(adx_value) and adx_value > self.adx_threshold
        if is_trending:
            return (
                MarketRegime.BULL_TREND
                if ma_short > ma_long
                else MarketRegime.BEAR_TREND
            )

        is_volatile = np.isfinite(atr_pct) and atr_pct > self.vol_threshold
        return (
            MarketRegime.SIDEWAYS_VOLATILE
            if is_volatile
            else MarketRegime.SIDEWAYS_QUIET
        )

    def _confirm_bars(self, candidate: MarketRegime) -> int:
        """轉為空頭只需要較少的確認根數。"""
        if candidate is MarketRegime.BEAR_TREND:
            return self.bear_confirm_bars
        if candidate is MarketRegime.BULL_TREND:
            return self.bull_confirm_bars
        return max(self.bear_confirm_bars, 1)

    def _apply_hysteresis(
        self, key: str, candidate: MarketRegime
    ) -> tuple[MarketRegime, int]:
        state = self._state.setdefault(
            key,
            {"regime": candidate, "pending": candidate, "streak": 0, "bars": 0},
        )
        state["bars"] = int(state["bars"]) + 1

        if candidate is state["regime"]:
            state["pending"] = candidate
            state["streak"] = 0
            return state["regime"], int(state["bars"])

        if candidate is state["pending"]:
            state["streak"] = int(state["streak"]) + 1
        else:
            state["pending"] = candidate
            state["streak"] = 1

        if int(state["streak"]) >= self._confirm_bars(candidate):
            state["regime"] = candidate
            state["streak"] = 0
            state["bars"] = 1

        return state["regime"], int(state["bars"])

    def assess(self, df: pd.DataFrame, key: str = None) -> RegimeAssessment:
        """完整評估。key 用來隔離不同標的的遲滯狀態。"""
        minimum = max(self.ma_long, self.adx_period, self.atr_period) + 2
        if df is None or len(df) < minimum:
            return RegimeAssessment(
                regime=MarketRegime.SIDEWAYS_QUIET,
                reasons=[f"資料不足 ({0 if df is None else len(df)}/{minimum} 根)"],
            )

        frame = df.iloc[:-1] if self.use_closed_bars_only and len(df) > minimum else df

        indicators = self._indicators(frame)
        adx_series = indicators["adx"]
        atr_series = indicators["atr"]

        adx_value = _safe_last(adx_series, default=np.nan)
        atr_value = _safe_last(atr_series, default=np.nan)
        last_close = float(frame["Close"].iloc[-1])
        atr_pct = (
            atr_value / last_close
            if last_close > 0 and np.isfinite(atr_value)
            else np.nan
        )

        raw = self._classify(frame, adx_value, atr_pct)
        regime, bars = self._apply_hysteresis(key or "__global__", raw)

        reversal, reasons = self.reversal_model.score(
            frame,
            adx_series,
            atr_series,
            fast=self.ma_short,
            slow=self.ma_long,
            lookback=self.reversal_lookback,
        )

        # 反轉風險只在多頭情境下有意義；空頭與盤整期不必再談「反轉向下」。
        if regime is not MarketRegime.BULL_TREND:
            reversal *= 0.5

        trend_strength = _clamp01(adx_value / 50.0) if np.isfinite(adx_value) else 0.0

        return RegimeAssessment(
            regime=regime,
            reversal_risk=reversal,
            trend_strength=trend_strength,
            bars_in_regime=bars,
            raw_regime=raw,
            reasons=reasons,
        )

    def detect(self, df: pd.DataFrame, key: str = None) -> MarketRegime:
        """相容既有呼叫端的簡化介面。"""
        return self.assess(df, key).regime
