# strategy_components.py: 策略基礎組件 (情境、風險、倉位)
# 將原本分散的組件集中，降低複雜度，提高內聚性

import math
import pandas as pd
import pandas_ta as ta
import numpy as np
from enum import Enum, auto
from abc import ABC, abstractmethod
from typing import Dict, Any, List, Tuple, Optional

# --- 1. 市場情境偵測 ---

class MarketRegime(Enum):
    BULL_TREND = auto()
    BEAR_TREND = auto()
    SIDEWAYS_VOLATILE = auto()
    SIDEWAYS_QUIET = auto()

class MarketRegimeDetector:
    """使用 ADX, SMA, ATR 來判斷市場當前所處情境"""
    def __init__(self, config: Dict[str, Any] = None):
        cfg = config or {}
        self.adx_period = cfg.get("regime_adx_period", 5)
        self.ma_short = cfg.get("regime_ma_short", 5)
        self.ma_long = cfg.get("regime_ma_long", 10)
        self.atr_period = cfg.get("regime_atr_period", 5)
        self.adx_threshold = cfg.get("regime_adx_trend_threshold", 20.0)
        self.vol_threshold = cfg.get("regime_volatility_threshold_pct", 0.02)

    def detect(self, df: pd.DataFrame) -> MarketRegime:
        if len(df) < max(self.ma_long, self.adx_period, self.atr_period) + 1:
            return MarketRegime.SIDEWAYS_QUIET
            
        # 計算技術指標
        adx = ta.adx(df['High'], df['Low'], df['Close'], length=self.adx_period)
        adx_val = adx.iloc[-1, 0] if adx is not None and not adx.empty else 0
        
        sma_short = ta.sma(df['Close'], length=self.ma_short).iloc[-1]
        sma_long = ta.sma(df['Close'], length=self.ma_long).iloc[-1]
        
        atr = ta.atr(df['High'], df['Low'], df['Close'], length=self.atr_period).iloc[-1]
        close_price = df['Close'].iloc[-1]
        atr_pct = (atr / close_price) if close_price > 0 else 0

        # 邏輯判定
        is_trending = adx_val > self.adx_threshold
        is_volatile = atr_pct > self.vol_threshold

        if is_trending:
            return MarketRegime.BULL_TREND if sma_short > sma_long else MarketRegime.BEAR_TREND
        return MarketRegime.SIDEWAYS_VOLATILE if is_volatile else MarketRegime.SIDEWAYS_QUIET

# --- 2. 風險控制器 ---

class IRiskRule(ABC):
    @abstractmethod
    def assess(self, context: Dict[str, Any]) -> Tuple[bool, str]:
        pass

class VIXHaltRule(IRiskRule):
    def __init__(self, threshold: float = 30.0):
        self.threshold = threshold

    def assess(self, context: Dict[str, Any]) -> Tuple[bool, str]:
        vix_series = context.get("vix_series")
        if vix_series is not None and not vix_series.empty:
            vix_val = vix_series.iloc[-1]
            if vix_val > self.threshold:
                return False, f"VIX 過高 ({vix_val:.2f})，暫停交易。"
        return True, "VIX 正常"

class RiskController:
    """統一管理所有風險規則"""
    def __init__(self, rules: List[IRiskRule]):
        self.rules = rules

    def check_trade_allowed(self, context: Dict[str, Any]) -> Tuple[bool, str]:
        for rule in self.rules:
            allowed, reason = rule.assess(context)
            if not allowed:
                return False, reason
        return True, "風險檢查通過"

# --- 3. 倉位計算器 ---

class PositionSizer:
    """動態計算倉位大小"""
    def __init__(self, max_equity_pct: float = 0.2, min_trade_usd: float = 10.0):
        self.max_equity_pct = max_equity_pct
        self.min_trade_usd = min_trade_usd

    def calculate_size(self, capital: float, price: float, risk_per_share: float = None) -> int:
        if price <= 0: return 0
        
        # 基本計算邏輯：若提供每股風險，則基於風險計算；否則基於資金比例
        if risk_per_share and risk_per_share > 0:
            risk_budget = capital * 0.01  # 預設單筆交易風險為總資金 1%
            quantity = math.floor(risk_budget / risk_per_share)
        else:
            quantity = math.floor((capital * self.max_equity_pct) / price)
            
        # 限制檢查
        if quantity * price < self.min_trade_usd:
            return 0
            
        return quantity