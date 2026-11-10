# ibkrpy/data/benchmark_resolver.py
# Per-symbol benchmark ETF selection

import json
import logging
import os
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence

import numpy as np
import pandas as pd

logger = logging.getLogger("ibkrpy.benchmark")


@dataclass(frozen=True)
class AssetClassification:
    """Industry classification of a symbol. Either field may be None when the source lacks the symbol."""

    sector: Optional[str] = None
    industry: Optional[str] = None

    def is_empty(self) -> bool:
        return not self.sector and not self.industry


class ClassificationProvider(ABC):
    """Provides industry classification for a symbol."""

    @abstractmethod
    def classify(self, symbol: str) -> AssetClassification: ...


class ReturnsProvider(ABC):
    """Provides log-return series for correlation. Returns None when data is insufficient."""

    @abstractmethod
    def get_returns(
        self, symbol: str, timeframe: str, lookback: int
    ) -> Optional[pd.Series]: ...


class BenchmarkReader(ABC):
    """Read-only interface used by the live path."""

    @abstractmethod
    def get(self, symbol: str) -> Optional[str]: ...


class BenchmarkWriter(ABC):
    """Write-only interface used by the training path."""

    @abstractmethod
    def set(self, symbol: str, benchmark: str) -> None: ...


class BenchmarkResolver(ABC):

    @abstractmethod
    def resolve(self, symbol: str) -> Optional[str]: ...

    @property
    def name(self) -> str:
        return type(self).__name__


class PinnedBenchmarkResolver(BenchmarkResolver):
    """Read the choice locked in during training."""

    def __init__(self, reader: BenchmarkReader):
        self._reader = reader

    def resolve(self, symbol: str) -> Optional[str]:
        try:
            pinned = self._reader.get(symbol)
        except Exception as e:
            logger.warning(f"[{symbol}] Failed to read locked benchmark: {e}")
            return None
        if pinned and pinned != symbol:
            return pinned
        return None


class ExplicitMapResolver(BenchmarkResolver):
    """Explicit mapping from config.yaml."""

    def __init__(self, mapping: Dict[str, str] = None):
        self._map = {str(k).upper(): str(v).upper() for k, v in (mapping or {}).items()}

    def resolve(self, symbol: str) -> Optional[str]:
        chosen = self._map.get(str(symbol).upper())
        return chosen if chosen and chosen != symbol else None


class SectorEtfResolver(BenchmarkResolver):
    """Pick the sector ETF matching the industry classification."""

    DEFAULT_SECTOR_MAP = {
        "TECHNOLOGY": "XLK",
        "INFORMATION TECHNOLOGY": "XLK",
        "FINANCIAL SERVICES": "XLF",
        "FINANCIAL": "XLF",
        "FINANCIALS": "XLF",
        "HEALTHCARE": "XLV",
        "HEALTH CARE": "XLV",
        "CONSUMER CYCLICAL": "XLY",
        "CONSUMER DISCRETIONARY": "XLY",
        "CONSUMER DEFENSIVE": "XLP",
        "CONSUMER STAPLES": "XLP",
        "ENERGY": "XLE",
        "INDUSTRIALS": "XLI",
        "INDUSTRIAL": "XLI",
        "BASIC MATERIALS": "XLB",
        "MATERIALS": "XLB",
        "UTILITIES": "XLU",
        "REAL ESTATE": "XLRE",
        "COMMUNICATION SERVICES": "XLC",
    }

    DEFAULT_INDUSTRY_MAP = {
        "SEMICONDUCTORS": "SMH",
        "SEMICONDUCTOR EQUIPMENT & MATERIALS": "SMH",
        "SEMICONDUCTOR EQUIPMENT AND MATERIALS": "SMH",
        "SOFTWARE - INFRASTRUCTURE": "IGV",
        "SOFTWARE - APPLICATION": "IGV",
        "BIOTECHNOLOGY": "XBI",
        "AEROSPACE & DEFENSE": "ITA",
        "AEROSPACE AND DEFENSE": "ITA",
        "INTERNET RETAIL": "XRT",
        "BANKS - DIVERSIFIED": "KBE",
        "BANKS—DIVERSIFIED": "KBE",
    }

    def __init__(
        self,
        classifier: ClassificationProvider,
        sector_map: Dict[str, str] = None,
        industry_map: Dict[str, str] = None,
    ):
        self._classifier = classifier
        self._sector_map = {**self.DEFAULT_SECTOR_MAP, **self._upper(sector_map)}
        self._industry_map = {**self.DEFAULT_INDUSTRY_MAP, **self._upper(industry_map)}

    @staticmethod
    def _upper(m: Dict[str, str]) -> Dict[str, str]:
        return {str(k).upper(): str(v).upper() for k, v in (m or {}).items()}

    def resolve(self, symbol: str) -> Optional[str]:
        try:
            cls = self._classifier.classify(symbol)
        except Exception as e:
            logger.warning(f"[{symbol}] Classification lookup failed: {e}")
            return None

        if cls.is_empty():
            return None

        if cls.industry:
            hit = self._industry_map.get(cls.industry.strip().upper())
            if hit and hit != symbol:
                logger.debug(
                    f"[{symbol}] Benchmark {hit} chosen by industry='{cls.industry}'"
                )
                return hit

        if cls.sector:
            hit = self._sector_map.get(cls.sector.strip().upper())
            if hit and hit != symbol:
                logger.debug(
                    f"[{symbol}] Benchmark {hit} chosen by sector='{cls.sector}'"
                )
                return hit

        return None


class CorrelationBenchmarkResolver(BenchmarkResolver):
    """Pick the candidate whose historical returns correlate most with the symbol."""

    def __init__(
        self,
        returns: ReturnsProvider,
        candidates: Sequence[str],
        timeframe: str = "1 day",
        lookback: int = 250,
        min_observations: int = 60,
        min_correlation: float = 0.3,
    ):
        self._returns = returns
        self._candidates = [str(c).upper() for c in candidates]
        self._timeframe = timeframe
        self._lookback = lookback
        self._min_obs = min_observations
        self._min_corr = min_correlation

    def resolve(self, symbol: str) -> Optional[str]:
        try:
            target = self._returns.get_returns(symbol, self._timeframe, self._lookback)
        except Exception as e:
            logger.warning(f"[{symbol}] Failed to read return series: {e}")
            return None

        if target is None or len(target) < self._min_obs:
            return None

        best_symbol, best_corr = None, -np.inf
        for candidate in self._candidates:
            if candidate == str(symbol).upper():
                continue
            try:
                other = self._returns.get_returns(
                    candidate, self._timeframe, self._lookback
                )
            except Exception:
                continue
            if other is None or len(other) < self._min_obs:
                continue

            joined = pd.concat([target, other], axis=1, join="inner").dropna()
            if len(joined) < self._min_obs:
                continue

            corr = float(joined.iloc[:, 0].corr(joined.iloc[:, 1]))
            if not np.isfinite(corr):
                continue
            if corr > best_corr:
                best_symbol, best_corr = candidate, corr

        if best_symbol is None:
            return None

        if best_corr < self._min_corr:
            logger.debug(
                f"[{symbol}] Best candidate correlation is only {best_corr:.2f} "
                f"(below threshold {self._min_corr:.2f}); skipping correlation-based choice."
            )
            return None

        logger.debug(
            f"[{symbol}] Benchmark {best_symbol} chosen by correlation (ρ={best_corr:.2f})"
        )
        return best_symbol


class StaticBenchmarkResolver(BenchmarkResolver):
    """
    Terminal resolver that always returns an answer. If the default is the symbol itself
    (e.g. QQQ is in the asset pool), the fallback is used so bench_correlation is not
    identically 1.
    """

    def __init__(
        self, default_symbol: str = "SPY", self_reference_fallback: str = "SPY"
    ):
        self._default = str(default_symbol).upper()
        self._fallback = str(self_reference_fallback).upper()

    def resolve(self, symbol: str) -> Optional[str]:
        if str(symbol).upper() != self._default:
            return self._default
        if self._fallback != str(symbol).upper():
            return self._fallback
        return "SPY" if str(symbol).upper() != "SPY" else "VTI"


class ChainedBenchmarkResolver(BenchmarkResolver):
    """
    Chain of responsibility: resolvers are asked in order and the first answer wins.
    """

    def __init__(self, resolvers: Sequence[BenchmarkResolver]):
        self._resolvers = list(resolvers)
        if not self._resolvers:
            raise ValueError("ChainedBenchmarkResolver needs at least one resolver")

    def resolve(self, symbol: str) -> Optional[str]:
        for resolver in self._resolvers:
            try:
                result = resolver.resolve(symbol)
            except Exception as e:
                logger.error(
                    f"[{symbol}] {resolver.name} raised, breaking its contract: {e}"
                )
                continue
            if result:
                logger.debug(
                    f"[{symbol}] benchmark = {result} (decided by {resolver.name})"
                )
                return result
        return None


class CachingBenchmarkResolver(BenchmarkResolver):
    """
    Caching decorator. Correlation scans the database, so without a cache it would rerun
    every tick.
    """

    def __init__(self, inner: BenchmarkResolver):
        self._inner = inner
        self._cache: Dict[str, Optional[str]] = {}

    def resolve(self, symbol: str) -> Optional[str]:
        key = str(symbol).upper()
        if key not in self._cache:
            self._cache[key] = self._inner.resolve(symbol)
        return self._cache[key]

    def invalidate(self, symbol: str = None):
        if symbol is None:
            self._cache.clear()
        else:
            self._cache.pop(str(symbol).upper(), None)

    @property
    def name(self) -> str:
        return f"Caching({self._inner.name})"


class StaticClassificationProvider(ClassificationProvider):
    """Classification supplied directly by config.yaml; also the simplest test double."""

    def __init__(self, table: Dict[str, Dict[str, str]] = None):
        self._table = {
            str(k).upper(): AssetClassification(
                sector=(v or {}).get("sector"), industry=(v or {}).get("industry")
            )
            for k, v in (table or {}).items()
        }

    def classify(self, symbol: str) -> AssetClassification:
        return self._table.get(str(symbol).upper(), AssetClassification())


class FmpCacheClassificationProvider(ClassificationProvider):
    """
    Reads weights/fmp_cache.json, which pipeline_manager already fetches
    (best_params['fmp_sector']).
    """

    def __init__(self, cache_path: str):
        self._path = cache_path
        self._data: Optional[Dict] = None
        self._mtime: float = -1.0

    def _load(self) -> Dict:
        try:
            mtime = os.path.getmtime(self._path)
        except OSError:
            self._data = {}
            return self._data
        if self._data is None or mtime != self._mtime:
            try:
                with open(self._path, "r", encoding="utf-8") as f:
                    self._data = json.load(f) or {}
                self._mtime = mtime
            except Exception as e:
                logger.warning(f"Failed to read FMP cache ({self._path}): {e}")
                self._data = {}
        return self._data

    def classify(self, symbol: str) -> AssetClassification:
        entry = self._load().get(str(symbol).upper()) or self._load().get(str(symbol))
        if not isinstance(entry, dict):
            return AssetClassification()
        return AssetClassification(
            sector=entry.get("sector"), industry=entry.get("industry")
        )


class AssetProfileClassificationProvider(ClassificationProvider):
    """Derive classification from assets[].tags in config.yaml."""

    def __init__(
        self,
        asset_profiles,
        sector_vocabulary: Sequence[str] = None,
        industry_vocabulary: Sequence[str] = None,
    ):
        vocab_sector = {
            s.upper()
            for s in (sector_vocabulary or SectorEtfResolver.DEFAULT_SECTOR_MAP.keys())
        }
        vocab_industry = {
            s.upper()
            for s in (
                industry_vocabulary or SectorEtfResolver.DEFAULT_INDUSTRY_MAP.keys()
            )
        }

        self._table: Dict[str, AssetClassification] = {}
        for profile in asset_profiles or []:
            tags = [
                t.strip() for t in (getattr(profile, "tags", None) or []) if t.strip()
            ]
            if not tags:
                continue
            sector = next((t for t in tags if t.upper() in vocab_sector), None)
            industry = next((t for t in tags if t.upper() in vocab_industry), None)
            if sector or industry:
                self._table[profile.symbol.upper()] = AssetClassification(
                    sector, industry
                )

    def classify(self, symbol: str) -> AssetClassification:
        return self._table.get(str(symbol).upper(), AssetClassification())


class CompositeClassificationProvider(ClassificationProvider):
    """Ask several sources in order, filling each other's missing fields."""

    def __init__(self, providers: Sequence[ClassificationProvider]):
        self._providers = list(providers)

    def classify(self, symbol: str) -> AssetClassification:
        sector, industry = None, None
        for provider in self._providers:
            try:
                cls = provider.classify(symbol)
            except Exception:
                continue
            sector = sector or cls.sector
            industry = industry or cls.industry
            if sector and industry:
                break
        return AssetClassification(sector, industry)


class DatabaseReturnsProvider(ReturnsProvider):
    """
    Adapts DatabaseManager to the ReturnsProvider port, so
    CorrelationBenchmarkResolver does not depend on SQLite.
    """

    def __init__(self, db_manager, cache_enabled: bool = True):
        self._db = db_manager
        self._cache: Dict[str, pd.Series] = {}
        self._cache_enabled = cache_enabled

    def get_returns(
        self, symbol: str, timeframe: str, lookback: int
    ) -> Optional[pd.Series]:
        key = f"{symbol}|{timeframe}|{lookback}"
        if self._cache_enabled and key in self._cache:
            return self._cache[key]

        df = self._db.get_market_data_sync(symbol, timeframe=timeframe)
        if df is None or df.empty or "Close" not in df.columns:
            return None

        close = pd.to_numeric(df["Close"], errors="coerce").dropna()
        if len(close) < 2:
            return None

        close = close.tail(lookback)
        returns = np.log(close / close.shift(1)).dropna()
        returns.name = symbol
        returns.index = pd.to_datetime(returns.index, utc=True).normalize()
        returns = returns[~returns.index.duplicated(keep="last")]

        if self._cache_enabled:
            self._cache[key] = returns
        return returns

    def invalidate(self):
        self._cache.clear()


class ManifestBenchmarkReader(BenchmarkReader):
    """
    Reads the benchmark actually used in training from the DataPipeline feature
    manifest (weights/training_artifacts.json).
    """

    def __init__(self, data_pipeline):
        self._pipeline = data_pipeline

    def get(self, symbol: str) -> Optional[str]:
        if self._pipeline is None:
            return None
        try:
            manifest = self._pipeline.load_manifest(symbol)
        except Exception:
            return None
        if not manifest:
            return None
        value = manifest.get("benchmark")
        return str(value).upper() if value else None


class JsonBenchmarkStore(BenchmarkReader, BenchmarkWriter):
    """
    Persists training-time choices to weights/benchmark_map.json. Implements both the
    reader and writer interfaces; each consumer depends only on the one it needs.
    """

    def __init__(self, path: str):
        self._path = path
        self._data: Optional[Dict[str, str]] = None
        self._mtime: float = -1.0

    def _load(self) -> Dict[str, str]:
        try:
            mtime = os.path.getmtime(self._path)
        except OSError:
            if self._data is None:
                self._data = {}
            return self._data
        if self._data is None or mtime != self._mtime:
            try:
                with open(self._path, "r", encoding="utf-8") as f:
                    self._data = json.load(f) or {}
                self._mtime = mtime
            except Exception as e:
                logger.warning(f"Failed to read benchmark map: {e}")
                self._data = {}
        return self._data

    def get(self, symbol: str) -> Optional[str]:
        return self._load().get(str(symbol).upper())

    def set(self, symbol: str, benchmark: str) -> None:
        data = dict(self._load())
        data[str(symbol).upper()] = str(benchmark).upper()
        try:
            os.makedirs(os.path.dirname(self._path), exist_ok=True)
            with open(self._path, "w", encoding="utf-8") as f:
                json.dump(data, f, indent=4, ensure_ascii=False, sort_keys=True)
            self._data = data
            self._mtime = os.path.getmtime(self._path)
        except Exception as e:
            logger.error(f"Failed to write benchmark map: {e}")

    def as_dict(self) -> Dict[str, str]:
        return dict(self._load())


class NullBenchmarkStore(BenchmarkReader, BenchmarkWriter):
    """No-op store for backtests and tests, so weights/ is left untouched."""

    def get(self, symbol: str) -> Optional[str]:
        return None

    def set(self, symbol: str, benchmark: str) -> None:
        return None


DEFAULT_CANDIDATE_POOL = [
    "SPY",
    "QQQ",
    "IWM",
    "DIA",
    "XLK",
    "XLF",
    "XLV",
    "XLY",
    "XLP",
    "XLE",
    "XLI",
    "XLB",
    "XLU",
    "XLRE",
    "XLC",
    "SMH",
    "IGV",
    "XBI",
    "ITA",
    "XRT",
    "KBE",
]


def build_benchmark_resolver(
    config,
    db_manager=None,
    asset_profiles=None,
    store: BenchmarkReader = None,
    weights_dir: str = None,
    data_pipeline=None,
) -> BenchmarkResolver:
    """
    Build the full resolver chain from config.yaml.

    Priority:
      1. Choice recorded in the training feature manifest (the weights were trained with it)
      2. benchmark_map.json
      3. Explicit config mapping
      4. Industry / sector ETF mapping
      5. Historical return correlation
      6. Global default (terminal, always answers)
    """
    settings = config.get("benchmark_settings") or {}
    default_symbol = settings.get("default") or config.get(
        "general_settings.benchmark_symbol", "SPY"
    )

    resolvers: List[BenchmarkResolver] = []

    if settings.get("respect_trained_choice", True):
        if data_pipeline is not None:
            resolvers.append(
                PinnedBenchmarkResolver(ManifestBenchmarkReader(data_pipeline))
            )
        if store is not None:
            resolvers.append(PinnedBenchmarkResolver(store))

    overrides = settings.get("overrides") or {}
    if overrides:
        resolvers.append(ExplicitMapResolver(overrides))

    if settings.get("enable_sector_mapping", True):
        providers: List[ClassificationProvider] = []
        manual = settings.get("classifications") or {}
        if manual:
            providers.append(StaticClassificationProvider(manual))
        if asset_profiles:
            providers.append(AssetProfileClassificationProvider(asset_profiles))
        if weights_dir:
            providers.append(
                FmpCacheClassificationProvider(
                    os.path.join(weights_dir, "fmp_cache.json")
                )
            )
        if providers:
            resolvers.append(
                SectorEtfResolver(
                    CompositeClassificationProvider(providers),
                    sector_map=settings.get("sector_etf_map"),
                    industry_map=settings.get("industry_etf_map"),
                )
            )

    if settings.get("enable_correlation", True) and db_manager is not None:
        resolvers.append(
            CorrelationBenchmarkResolver(
                returns=DatabaseReturnsProvider(db_manager),
                candidates=settings.get("candidate_pool") or DEFAULT_CANDIDATE_POOL,
                timeframe=settings.get("correlation_timeframe", "1 day"),
                lookback=int(settings.get("correlation_lookback", 250)),
                min_observations=int(settings.get("correlation_min_observations", 60)),
                min_correlation=float(settings.get("correlation_min_threshold", 0.3)),
            )
        )

    resolvers.append(StaticBenchmarkResolver(default_symbol))
    return CachingBenchmarkResolver(ChainedBenchmarkResolver(resolvers))


def distinct_benchmarks(
    resolver: BenchmarkResolver, symbols: Sequence[str]
) -> Dict[str, str]:
    """
    Resolve many symbols at once; the download step needs the full set of benchmark ETFs.
    """
    mapping: Dict[str, str] = {}
    for symbol in symbols:
        chosen = resolver.resolve(symbol)
        if chosen:
            mapping[symbol] = chosen
    return mapping
