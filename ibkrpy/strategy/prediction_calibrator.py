# ibkrpy/strategy/prediction_calibrator.py
# 進場門檻的校準，以及模型塌陷偵測。

from __future__ import annotations

import json
import logging
import os
from abc import ABC, abstractmethod
from collections import deque
from dataclasses import dataclass, field
from typing import Deque, Dict, List, Optional, Sequence

import numpy as np

logger = logging.getLogger("ibkrpy.calibrator")


# 值物件


@dataclass(frozen=True)
class ThresholdContext:
    """計算門檻所需的全部輸入。新增欄位不會破壞既有 policy 的簽章。"""

    symbol: str
    sigma: float  # 每根 K 線的價格波動率 (小數)
    history: Sequence[float]  # 該標的過往的預測幅度 (小數, 有正負)
    min_edge_pct: float = 0.0005  # 交易成本地板
    term: str = "long_term"


@dataclass(frozen=True)
class ThresholdDecision:
    value: float
    source: str
    detail: str = ""


@dataclass
class CollapseReport:
    """模型塌陷的診斷結果。"""

    collapsed: bool
    samples: int
    dispersion: float  # 預測值的標準差
    unique_ratio: float  # 相異值 / 樣本數
    detail: str = ""


# 預測歷史 (Port + Adapters)


class PredictionHistoryStore(ABC):
    """記錄每檔標的的預測幅度，供門檻校準與塌陷偵測使用。"""

    @abstractmethod
    def append(self, symbol: str, value: float) -> None: ...

    @abstractmethod
    def get(self, symbol: str) -> List[float]: ...


class InMemoryPredictionHistory(PredictionHistoryStore):
    """滾動視窗。重啟後歸零，適合回測與測試。"""

    def __init__(self, maxlen: int = 500):
        self._maxlen = maxlen
        self._data: Dict[str, Deque[float]] = {}

    def append(self, symbol: str, value: float) -> None:
        if value is None or not np.isfinite(value):
            return
        key = str(symbol).upper()
        if key not in self._data:
            self._data[key] = deque(maxlen=self._maxlen)
        self._data[key].append(float(value))

    def get(self, symbol: str) -> List[float]:
        return list(self._data.get(str(symbol).upper(), ()))

    def snapshot(self) -> Dict[str, List[float]]:
        return {k: list(v) for k, v in self._data.items()}


class JsonPredictionHistory(InMemoryPredictionHistory):
    """
    落盤版本。實盤重啟後不必重新累積樣本 —— 否則每次重啟都要等
    min_samples 筆掃描才能開始交易。

    以 Decorator 的方式繼承記憶體版：寫入邏輯完全複用，只多一層節流落盤。
    """

    def __init__(self, path: str, maxlen: int = 500, flush_every: int = 25):
        super().__init__(maxlen=maxlen)
        self._path = path
        self._flush_every = max(1, int(flush_every))
        self._pending = 0
        self._load()

    def _load(self):
        if not os.path.exists(self._path):
            return
        try:
            with open(self._path, "r", encoding="utf-8") as f:
                raw = json.load(f) or {}
            for sym, values in raw.items():
                for v in values:
                    super().append(sym, v)
            logger.debug(f"預測歷史已載入 {len(raw)} 檔標的。")
        except Exception as e:
            logger.warning(f"預測歷史載入失敗 ({self._path}): {e}")

    def append(self, symbol: str, value: float) -> None:
        super().append(symbol, value)
        self._pending += 1
        if self._pending >= self._flush_every:
            self.flush()

    def flush(self):
        try:
            os.makedirs(os.path.dirname(self._path), exist_ok=True)
            with open(self._path, "w", encoding="utf-8") as f:
                json.dump(self.snapshot(), f)
            self._pending = 0
        except Exception as e:
            logger.warning(f"預測歷史寫入失敗: {e}")


# 門檻政策 (Strategy Pattern)


class ThresholdPolicy(ABC):
    """
    契約：回傳一個正的門檻值 (小數)，永不拋例外。
    所有實作可互換 —— 這是 LSP 的落腳處，也是能安全 A/B 切換的前提。
    """

    @abstractmethod
    def compute(self, ctx: ThresholdContext) -> ThresholdDecision: ...

    @property
    def name(self) -> str:
        return type(self).__name__


class SigmaThresholdPolicy(ThresholdPolicy):
    """
    舊版行為：門檻 = σ × multiplier。保留下來有兩個用途：
      1. 需要回退時可以一行切換
      2. 作為對照組，讓新政策的效果可被量化比較

    ⚠️ 不建議在生產使用。σ 是價格的尺度，不是預測值的尺度。
    """

    def __init__(self, multiplier: float = 1.0, static_floor: float = 0.0):
        self._mult = float(multiplier)
        self._floor = float(static_floor)

    def compute(self, ctx: ThresholdContext) -> ThresholdDecision:
        value = max(self._floor, ctx.sigma * self._mult)
        return ThresholdDecision(value, "sigma", f"σ×{self._mult:g}")


class QuantileThresholdPolicy(ThresholdPolicy):
    """
    門檻 = 該標的過往 |預測| 的第 q 分位數。

    語意變成「只在這個模型罕見地有強烈意見時進場」，而不是「等模型預測出
    一個大過市場波動的數字」。後者對任何真實的次根 K 線預測器都是不可能的：
    有效預測器的輸出標準差約為 σ × √R²，R² 只有幾個百分點時就是 0.1~0.2σ，
    要求 |預測| > 1σ 等於要求預測落在自身分布的 5~10 個標準差外。

    樣本不足時回傳 None-like 行為 (source="insufficient")，由 Composite
    決定要不要退回其他政策。
    """

    def __init__(self, quantile: float = 0.85, min_samples: int = 40):
        self._q = min(max(float(quantile), 0.5), 0.999)
        self._min_samples = int(min_samples)

    def compute(self, ctx: ThresholdContext) -> ThresholdDecision:
        hist = [abs(float(v)) for v in ctx.history if np.isfinite(v)]
        if len(hist) < self._min_samples:
            return ThresholdDecision(
                float("inf"),
                "insufficient",
                f"樣本 {len(hist)}/{self._min_samples}",
            )
        value = float(np.quantile(hist, self._q))
        if not np.isfinite(value) or value <= 0:
            return ThresholdDecision(float("inf"), "insufficient", "分位數退化為 0")
        return ThresholdDecision(
            value,
            "quantile",
            f"P{self._q * 100:.0f} of {len(hist)} 筆歷史",
        )


class CostFloorThresholdPolicy(ThresholdPolicy):
    """
    Decorator：在內層政策之上套一個交易成本地板。

    這一層是安全閥，不是優化。少了它，分位數政策會忠實地讓系統在
    「模型輸出 0.06% 而來回成本 0.05%」的情況下持續開單 —— 那不是交易，
    是把資金餵給價差。
    """

    def __init__(self, inner: ThresholdPolicy, absolute_floor: float = 0.0005):
        self._inner = inner
        self._floor = float(absolute_floor)

    def compute(self, ctx: ThresholdContext) -> ThresholdDecision:
        inner = self._inner.compute(ctx)
        floor = max(self._floor, float(ctx.min_edge_pct or 0.0))
        if inner.value >= floor:
            return inner
        return ThresholdDecision(
            floor,
            "cost_floor",
            f"{inner.source} 給出 {inner.value * 100:.3f}%，低於成本地板 {floor * 100:.3f}%",
        )

    @property
    def name(self) -> str:
        return f"CostFloor({self._inner.name})"


class FallbackThresholdPolicy(ThresholdPolicy):
    """
    Composite：依序嘗試，第一個給出有限值的獲勝。
    冷啟動時分位數政策樣本不足，自動退回 σ 政策 —— 保守但不會誤開單。
    """

    def __init__(self, policies: Sequence[ThresholdPolicy]):
        self._policies = list(policies)
        if not self._policies:
            raise ValueError("FallbackThresholdPolicy 至少需要一個政策")

    def compute(self, ctx: ThresholdContext) -> ThresholdDecision:
        last = None
        for policy in self._policies:
            try:
                decision = policy.compute(ctx)
            except Exception as e:
                logger.error(f"[{ctx.symbol}] {policy.name} 違反契約拋出例外: {e}")
                continue
            last = decision
            if np.isfinite(decision.value):
                return decision
        return last or ThresholdDecision(float("inf"), "none", "所有政策皆無法給出門檻")

    @property
    def name(self) -> str:
        return "Fallback(" + ", ".join(p.name for p in self._policies) + ")"


# 塌陷偵測


class CollapseDetector:
    """
    偵測「模型輸出已退化成常數」。

    這是 2026-08-07 日誌暴露出來的真正病灶：LSTM 與 Transformer 對同一檔標的，
    在一整個交易日的 33~34 次掃描中輸出「完全相同的單一數值」 ——
    連 5 分 K 標的也是如此，儘管輸入每根 K 線都在變。

    成因是以 MSE 迴歸次根 K 線對數報酬：報酬近似不可預測時，讓損失最小的常數解
    就是樣本平均 (≈0)，網路會直接收斂到那裡。這種失效不會拋例外、不會有 NaN，
    模型照樣輸出數字，只是那個數字沒有資訊 —— 必須主動偵測。
    """

    def __init__(
        self,
        min_samples: int = 20,
        dispersion_floor: float = 1e-5,  # 預測標準差低於此值視為常數
        unique_ratio_floor: float = 0.05,
    ):
        self._min_samples = int(min_samples)
        self._dispersion_floor = float(dispersion_floor)
        self._unique_floor = float(unique_ratio_floor)

    def inspect(self, history: Sequence[float]) -> CollapseReport:
        values = [float(v) for v in history if v is not None and np.isfinite(v)]
        n = len(values)
        if n < self._min_samples:
            return CollapseReport(
                False,
                n,
                float("nan"),
                float("nan"),
                f"樣本不足 ({n}/{self._min_samples})，暫不判定",
            )

        dispersion = float(np.std(values))
        unique_ratio = len(set(np.round(values, 10))) / n

        if dispersion <= self._dispersion_floor:
            return CollapseReport(
                True,
                n,
                dispersion,
                unique_ratio,
                f"預測標準差 {dispersion:.2e} ≈ 0，輸出已退化成常數",
            )
        if unique_ratio <= self._unique_floor:
            return CollapseReport(
                True,
                n,
                dispersion,
                unique_ratio,
                f"{n} 次預測僅 {int(unique_ratio * n)} 個相異值",
            )
        return CollapseReport(False, n, dispersion, unique_ratio, "輸出具備變異")


# 組裝


def build_threshold_policy(config) -> ThresholdPolicy:
    """
    Composition Root 使用。依 config.yaml 組出門檻政策。

    預設鏈：CostFloor( Fallback( Quantile, Sigma ) )
      - 有足夠歷史 -> 用預測分布的分位數
      - 冷啟動     -> 退回 σ 政策 (保守)
      - 兩者都不得低於交易成本地板
    """
    s = config.get("threshold_settings") or {}
    mode = str(s.get("mode", "quantile")).lower()

    sigma_policy = SigmaThresholdPolicy(
        multiplier=float(
            s.get(
                "sigma_multiplier",
                config.get("strategy_settings.entry_sigma_multiplier", 1.0),
            )
        ),
        static_floor=float(
            config.get("strategy_settings.min_prediction_threshold_pct", 0.0)
        ),
    )

    if mode == "sigma":
        inner: ThresholdPolicy = sigma_policy
    else:
        quantile_policy = QuantileThresholdPolicy(
            quantile=float(s.get("quantile", 0.85)),
            min_samples=int(s.get("min_samples", 40)),
        )
        cold_start = str(s.get("cold_start", "sigma")).lower()
        if cold_start == "block":
            inner = quantile_policy  # 樣本不足時門檻為 inf -> 不交易
        else:
            inner = FallbackThresholdPolicy([quantile_policy, sigma_policy])

    return CostFloorThresholdPolicy(
        inner, absolute_floor=float(s.get("cost_floor_pct", 0.0005))
    )


def build_prediction_history(config, weights_dir: str) -> PredictionHistoryStore:
    s = config.get("threshold_settings") or {}
    maxlen = int(s.get("history_length", 500))
    if s.get("persist_history", True):
        return JsonPredictionHistory(
            os.path.join(weights_dir, "_prediction_history.json"), maxlen=maxlen
        )
    return InMemoryPredictionHistory(maxlen=maxlen)
