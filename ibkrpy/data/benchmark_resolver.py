# ibkrpy/data/benchmark_resolver.py
# 每檔標的的 benchmark (基準 ETF) 選擇邏輯

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
    """標的的產業分類。兩個欄位都可能是 None (資料來源沒有涵蓋該標的)。"""

    sector: Optional[str] = None
    industry: Optional[str] = None

    def is_empty(self) -> bool:
        return not self.sector and not self.industry


# 埠 (Ports) —— 抽象依賴，具體實作在下方的 Adapters 區


class ClassificationProvider(ABC):
    """提供標的的產業分類。單一方法，避免強迫實作者處理用不到的介面 (ISP)。"""

    @abstractmethod
    def classify(self, symbol: str) -> AssetClassification: ...


class ReturnsProvider(ABC):
    """提供對數報酬序列，供相關性計算使用。回傳 None 代表資料不足。"""

    @abstractmethod
    def get_returns(
        self, symbol: str, timeframe: str, lookback: int
    ) -> Optional[pd.Series]: ...


class BenchmarkReader(ABC):
    """只讀介面。實盤路徑只需要這個 (ISP)。"""

    @abstractmethod
    def get(self, symbol: str) -> Optional[str]: ...


class BenchmarkWriter(ABC):
    """只寫介面。只有訓練路徑需要這個 (ISP)。"""

    @abstractmethod
    def set(self, symbol: str, benchmark: str) -> None: ...


class BenchmarkResolver(ABC):
    """
    契約：
      - 回傳一個 benchmark 代碼，或回傳 None 表示「無法決定，請往下一位詢問」
      - 絕不拋例外。任何內部錯誤都必須降級為 None 並記錄日誌。
      - 絕不回傳 symbol 自己 (自我參照的 benchmark 沒有資訊量)
    """

    @abstractmethod
    def resolve(self, symbol: str) -> Optional[str]: ...

    @property
    def name(self) -> str:
        return type(self).__name__


class PinnedBenchmarkResolver(BenchmarkResolver):
    """
    讀取訓練階段鎖定的選擇，優先級最高。

    這一層是正確性的關鍵，不只是效能優化：模型吃的 bench_return /
    bench_correlation 兩個特徵是相對於「訓練時那一檔 benchmark」算出來的。
    實盤若換成另一檔 ETF，特徵分布就變了，而模型不會報錯，只會安靜地變差。
    因此只要訓練時做過決定，實盤就必須沿用。
    """

    def __init__(self, reader: BenchmarkReader):
        self._reader = reader

    def resolve(self, symbol: str) -> Optional[str]:
        try:
            pinned = self._reader.get(symbol)
        except Exception as e:
            logger.warning(f"[{symbol}] 讀取已鎖定的 benchmark 失敗: {e}")
            return None
        if pinned and pinned != symbol:
            return pinned
        return None


class ExplicitMapResolver(BenchmarkResolver):
    """使用者在 config.yaml 明確指定的對應，優先於任何自動推導。"""

    def __init__(self, mapping: Dict[str, str] = None):
        self._map = {str(k).upper(): str(v).upper() for k, v in (mapping or {}).items()}

    def resolve(self, symbol: str) -> Optional[str]:
        chosen = self._map.get(str(symbol).upper())
        return chosen if chosen and chosen != symbol else None


class SectorEtfResolver(BenchmarkResolver):
    """
    依產業分類挑選對應的板塊 ETF。

    先比對 industry 再比對 sector —— 半導體股跟著 SMH 走的程度遠高於
    跟著整個 XLK 走，這個粒度差異在特徵上是有意義的。
    """

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
            logger.warning(f"[{symbol}] 產業分類查詢失敗: {e}")
            return None

        if cls.is_empty():
            return None

        if cls.industry:
            hit = self._industry_map.get(cls.industry.strip().upper())
            if hit and hit != symbol:
                logger.debug(
                    f"[{symbol}] 依 industry='{cls.industry}' 選定 benchmark {hit}"
                )
                return hit

        if cls.sector:
            hit = self._sector_map.get(cls.sector.strip().upper())
            if hit and hit != symbol:
                logger.debug(
                    f"[{symbol}] 依 sector='{cls.sector}' 選定 benchmark {hit}"
                )
                return hit

        return None


class CorrelationBenchmarkResolver(BenchmarkResolver):
    """
    從候選池中挑出與該標的歷史報酬相關性最高的一檔。

    這是最貼近「benchmark 隨標的調整」原意的做法：不依賴任何人工分類，
    直接讓資料說話。代價是需要候選池的歷史資料已經在資料庫裡，
    因此它在 Chain 中排在分類法之後 —— 冷啟動時自然降級。

    刻意加上 min_correlation 門檻：若最佳候選的相關性也只有 0.2，
    那這個 benchmark 對模型是雜訊而非資訊，寧可回 None 交給下一位。
    """

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
            logger.warning(f"[{symbol}] 讀取報酬序列失敗: {e}")
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
                f"[{symbol}] 候選池中最佳相關性僅 {best_corr:.2f} "
                f"(低於門檻 {self._min_corr:.2f})，放棄以相關性選擇。"
            )
            return None

        logger.debug(
            f"[{symbol}] 依相關性選定 benchmark {best_symbol} (ρ={best_corr:.2f})"
        )
        return best_symbol


class StaticBenchmarkResolver(BenchmarkResolver):
    """
    終端 resolver：永遠給得出答案，讓 Chain 保證有結果。
    若預設值恰好就是該標的自己 (例如 QQQ 本身也在資產池中)，
    改用備援代碼，避免 bench_correlation 恆等於 1。
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
    Composite / Chain of Responsibility。依序詢問，第一個給出答案的獲勝。

    這是 OCP 的落腳處：要調整優先順序或加入新規則，只需要改組裝清單，
    既有的 resolver 一行都不用動。
    """

    def __init__(self, resolvers: Sequence[BenchmarkResolver]):
        self._resolvers = list(resolvers)
        if not self._resolvers:
            raise ValueError("ChainedBenchmarkResolver 至少需要一個 resolver")

    def resolve(self, symbol: str) -> Optional[str]:
        for resolver in self._resolvers:
            try:
                result = resolver.resolve(symbol)
            except Exception as e:
                # 契約規定不得拋例外，但仍防禦性處理，避免單一 resolver 拖垮整條鏈
                logger.error(f"[{symbol}] {resolver.name} 違反契約拋出例外: {e}")
                continue
            if result:
                logger.debug(
                    f"[{symbol}] benchmark = {result} (由 {resolver.name} 決定)"
                )
                return result
        return None


class CachingBenchmarkResolver(BenchmarkResolver):
    """
    Decorator。快取不是「選擇」的職責，因此獨立成一層 (SRP)。
    相關性計算會掃資料庫，沒有快取的話每個 tick 都會重算。
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


# Adapters —— 把既有的具體元件接到上面的埠上


class StaticClassificationProvider(ClassificationProvider):
    """由 config.yaml 直接提供分類。也是單元測試最方便的替身。"""

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
    讀取 pipeline_manager 已經在抓的 weights/fmp_cache.json。
    這些欄位本來就存在 (best_params['fmp_sector'])，只是從未被用來選 benchmark。
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
                logger.warning(f"FMP 快取讀取失敗 ({self._path}): {e}")
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
    """
    從 config.yaml 的 assets[].tags 推導分類。

    ConfigManager.AssetProfile 一直保留著 tags 欄位，只是沒有消費者。
    這一層讓舊的 tags 設計重新可用，而不必修改 ConfigManager。
    """

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
    """依序詢問多個來源，補齊彼此缺漏的欄位。"""

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
    把 DatabaseManager 接到 ReturnsProvider 埠上。
    CorrelationBenchmarkResolver 因此完全不需要知道底層是 SQLite。
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
    讀出訓練時實際使用的 benchmark。
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
    把訓練階段的決定落盤到 weights/benchmark_map.json。
    同時實作讀寫兩個介面，但消費端各自只依賴自己需要的那一個。
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
                logger.warning(f"benchmark 對應表讀取失敗: {e}")
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
            logger.error(f"benchmark 對應表寫入失敗: {e}")

    def as_dict(self) -> Dict[str, str]:
        return dict(self._load())


class NullBenchmarkStore(BenchmarkReader, BenchmarkWriter):
    """空實作。回測或測試時不希望污染 weights/ 目錄。"""

    def get(self, symbol: str) -> Optional[str]:
        return None

    def set(self, symbol: str, benchmark: str) -> None:
        return None


# 組裝 (Composition Root 使用)

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
    Composition Root 的輔助函式：依 config.yaml 組出完整的 resolver 鏈。

    刻意做成模組層級的函式而不是類別方法 —— 組裝是應用層的職責，
    不屬於任何一個 resolver。這樣每個 resolver 都能單獨被測試與重用。

    優先順序：
      1. 訓練時寫進特徵清單的選擇 (最高權威：權重就是配著它訓練的)
      2. _benchmark_map.json 的紀錄
      3. config 明確指定
      4. 產業 / 板塊 ETF 對應
      5. 歷史報酬相關性
      6. 全域預設值 (終端，保證有答案)
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
    批次解析。資料下載階段需要知道「總共要多抓哪幾檔 ETF」。
    放在模組層級而非塞進介面，是為了讓 BenchmarkResolver 維持單一方法 (ISP)。
    """
    mapping: Dict[str, str] = {}
    for symbol in symbols:
        chosen = resolver.resolve(symbol)
        if chosen:
            mapping[symbol] = chosen
    return mapping
