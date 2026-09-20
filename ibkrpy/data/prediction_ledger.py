from __future__ import annotations

import datetime as dt
import json
import logging
import math
import os
import sqlite3
from dataclasses import dataclass, field
from typing import Dict, List, Optional

logger = logging.getLogger("ibkrpy.prediction_ledger")

TERM_BARS = {
    "short_term": ("5 mins", 78),
    "mid_term": ("1 hour", 6.5),
    "long_term": ("1 day", 1),
}


@dataclass
class PredictionRecord:
    """一次掃描產生的完整快照。欄位刻意齊全 —— 事後想切哪個維度都切得動。"""

    symbol: str
    timestamp: str
    term: str
    timeframe: str
    horizon_bars: int
    price_at_prediction: float
    predicted_return: float
    model_predictions: Dict[str, float] = field(default_factory=dict)
    sigma: Optional[float] = None
    sigma_source: Optional[str] = None
    sigma_realized: Optional[float] = None
    regime: Optional[str] = None
    threshold: Optional[float] = None
    threshold_source: Optional[str] = None
    reason_code: Optional[str] = None
    action: Optional[str] = None
    benchmark: Optional[str] = None
    reversal_risk: Optional[float] = None
    model_generation: Optional[str] = None


_SCHEMA = """
CREATE TABLE IF NOT EXISTS prediction_ledger (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    symbol TEXT NOT NULL,
    timestamp DATETIME NOT NULL,
    term TEXT,
    timeframe TEXT,
    horizon_bars INTEGER,
    price_at_prediction REAL,
    predicted_return REAL,
    model_predictions TEXT,
    sigma REAL,
    sigma_source TEXT,
    sigma_realized REAL,
    regime TEXT,
    threshold REAL,
    threshold_source TEXT,
    reason_code TEXT,
    action TEXT,
    benchmark TEXT,
    reversal_risk REAL,
    model_generation TEXT,
    -- 以下由 resolve() 事後回填
    realized_return REAL,
    realized_price REAL,
    resolved_at DATETIME,
    UNIQUE (symbol, timestamp, timeframe)
)
"""

_INDEXES = [
    "CREATE INDEX IF NOT EXISTS idx_ledger_unresolved "
    "ON prediction_ledger (realized_return, timestamp)",
    "CREATE INDEX IF NOT EXISTS idx_ledger_symbol ON prediction_ledger (symbol, timestamp)",
]


class PredictionLedger:
    """
    契約: record() 永不拋例外、永不阻塞交易迴圈。

    帳本壞掉不該讓系統停止交易；反過來，交易迴圈的例外也不該讓帳本漏記。
    因此所有寫入都包在 try 裡並只記 log。
    """

    def __init__(self, db_path: str, enabled: bool = True):
        self.db_path = db_path
        self.enabled = bool(enabled)
        if self.enabled:
            self._init()

    def _conn(self) -> sqlite3.Connection:
        conn = sqlite3.connect(self.db_path, timeout=15.0)
        conn.execute("PRAGMA journal_mode=WAL;")
        conn.execute("PRAGMA synchronous=NORMAL;")
        conn.row_factory = sqlite3.Row
        return conn

    def _init(self) -> None:
        try:
            with self._conn() as conn:
                conn.execute(_SCHEMA)
                for col, decl in (("model_generation", "TEXT"),):
                    try:
                        conn.execute(
                            f"ALTER TABLE prediction_ledger ADD COLUMN {col} {decl}"
                        )
                    except sqlite3.OperationalError:
                        pass
                for idx in _INDEXES:
                    conn.execute(idx)
                conn.commit()
        except Exception as e:
            logger.error(f"預測帳本初始化失敗，已停用: {e}")
            self.enabled = False

    _INSERT = """
        INSERT OR IGNORE INTO prediction_ledger
        (symbol, timestamp, term, timeframe, horizon_bars,
         price_at_prediction, predicted_return, model_predictions,
         sigma, sigma_source, sigma_realized, regime, threshold,
         threshold_source, reason_code, action, benchmark, reversal_risk,
         model_generation)
        VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)
    """

    @staticmethod
    def _row(rec: "PredictionRecord") -> tuple:
        return (
            rec.symbol,
            rec.timestamp,
            rec.term,
            rec.timeframe,
            int(rec.horizon_bars),
            float(rec.price_at_prediction),
            float(rec.predicted_return),
            json.dumps(rec.model_predictions or {}),
            rec.sigma,
            rec.sigma_source,
            rec.sigma_realized,
            rec.regime,
            rec.threshold,
            rec.threshold_source,
            rec.reason_code,
            rec.action,
            rec.benchmark,
            rec.reversal_risk,
            rec.model_generation,
        )

    def record_many(self, recs: List["PredictionRecord"]) -> None:
        """
        一次寫入多筆。交易迴圈每輪掃 39 檔，逐筆開連線等於每輪開 39 次;
        累積一輪後一次寫入，連線成本降到 1/39。回填與匯入也用這條路徑。
        """
        if not self.enabled or not recs:
            return
        try:
            with self._conn() as conn:
                conn.executemany(self._INSERT, [self._row(r) for r in recs])
                conn.commit()
        except Exception as e:
            logger.warning(f"預測帳本批次寫入失敗 (不影響交易): {e}")

    def record(self, rec: PredictionRecord) -> None:
        """
        掃描時呼叫。HOLD 也要記 —— 被門檻擋掉的預測正是用來判斷
        「門檻是否設得太高」的樣本，只記成交過的單會產生嚴重的選擇偏差。
        """
        if not self.enabled:
            return
        try:
            with self._conn() as conn:
                conn.execute(self._INSERT, self._row(rec))
                conn.commit()
        except Exception as e:
            logger.warning(f"[{rec.symbol}] 預測帳本寫入失敗 (不影響交易): {e}")

    def resolve(self, max_rows: int = 20000) -> Dict[str, int]:
        """
        用 market_data 的 K 線把 realized_return 補上。可重複執行，只處理未回填的列。

        對每一筆未回填的預測，取「預測時刻之後第 horizon_bars 根」的收盤價。
        若那根還沒出現 (時間未到、或資料尚未下載) 就先跳過，下次再試。
        """
        if not self.enabled:
            return {"resolved": 0, "pending": 0, "expired": 0}

        stats = {"resolved": 0, "pending": 0, "expired": 0}
        try:
            with self._conn() as conn:
                rows = conn.execute(
                    """
                    SELECT id, symbol, timestamp, timeframe, horizon_bars,
                           price_at_prediction
                    FROM prediction_ledger
                    WHERE realized_return IS NULL
                    ORDER BY timestamp ASC LIMIT ?
                    """,
                    (int(max_rows),),
                ).fetchall()

                now = dt.datetime.now(dt.timezone.utc)
                updates = []
                for r in rows:
                    bar = conn.execute(
                        """
                        SELECT timestamp, close FROM market_data
                        WHERE symbol = ? AND timeframe = ? AND timestamp > ?
                        ORDER BY timestamp ASC LIMIT 1 OFFSET ?
                        """,
                        (
                            r["symbol"],
                            r["timeframe"],
                            r["timestamp"],
                            max(int(r["horizon_bars"]) - 1, 0),
                        ),
                    ).fetchone()

                    if bar is None:
                        age = self._age_days(r["timestamp"], now)
                        if age is not None and age > 30:
                            updates.append((float("nan"), None, r["id"]))
                            stats["expired"] += 1
                        else:
                            stats["pending"] += 1
                        continue

                    p0 = float(r["price_at_prediction"] or 0.0)
                    p1 = float(bar["close"] or 0.0)
                    if p0 <= 0 or p1 <= 0:
                        stats["pending"] += 1
                        continue

                    updates.append((math.log(p1 / p0), p1, r["id"]))
                    stats["resolved"] += 1

                if updates:
                    conn.executemany(
                        "UPDATE prediction_ledger SET realized_return = ?, "
                        "realized_price = ?, resolved_at = CURRENT_TIMESTAMP WHERE id = ?",
                        updates,
                    )
                    conn.commit()
        except Exception as e:
            logger.error(f"預測帳本回填失敗: {e}")
        return stats

    @staticmethod
    def _age_days(ts: str, now: dt.datetime) -> Optional[float]:
        try:
            parsed = dt.datetime.fromisoformat(str(ts).replace("Z", "+00:00"))
            if parsed.tzinfo is None:
                parsed = parsed.replace(tzinfo=dt.timezone.utc)
            return (now - parsed).total_seconds() / 86400.0
        except Exception:
            return None

    def fetch_resolved(self, symbol: str = None, min_rows: int = 0) -> List[dict]:
        if not self.enabled:
            return []
        try:
            with self._conn() as conn:
                q = (
                    "SELECT * FROM prediction_ledger WHERE realized_return IS NOT NULL "
                    "AND realized_return = realized_return"
                )
                params = ()
                if symbol:
                    q += " AND symbol = ?"
                    params = (symbol,)
                q += " ORDER BY timestamp ASC"
                rows = [dict(r) for r in conn.execute(q, params).fetchall()]
            return rows if len(rows) >= min_rows else rows
        except Exception as e:
            logger.error(f"預測帳本讀取失敗: {e}")
            return []


def measure(rows: List[dict], min_samples: int = 30) -> Optional[dict]:
    """
    給定已回填的樣本，算出邊際的三個關鍵數字。

    hit_rate  方向命中率。0.5 代表擲硬幣。
    ic        預測與實際的相關係數 (Information Coefficient)。
              量化界的經驗值: 0.03-0.05 已算可用，0.10 以上很強。
    slope     以實際對預測做迴歸的斜率。這是校準係數 ——
              斜率 0.4 表示「實際只走了預測的 4 成」，方向有訊息但幅度被壓縮，
              把預測乘以 1/0.4 才是無偏估計。斜率 ~0 表示沒有邊際。

    另外回傳 slope 的標準誤與 t 值，因為斜率為正但不顯著等於沒有結論。
    """
    pred = [float(r["predicted_return"]) for r in rows]
    real = [float(r["realized_return"]) for r in rows]
    pairs = [
        (p, a) for p, a in zip(pred, real) if math.isfinite(p) and math.isfinite(a)
    ]
    n = len(pairs)
    if n < min_samples:
        return None

    xs = [p for p, _ in pairs]
    ys = [a for _, a in pairs]
    mx, my = sum(xs) / n, sum(ys) / n
    sxx = sum((x - mx) ** 2 for x in xs)
    syy = sum((y - my) ** 2 for y in ys)
    sxy = sum((x - mx) * (y - my) for x, y in pairs)

    if sxx <= 0 or syy <= 0:
        return {"n": n, "degenerate": True}

    ic = sxy / math.sqrt(sxx * syy)
    slope = sxy / sxx
    intercept = my - slope * mx

    resid = sum((y - (intercept + slope * x)) ** 2 for x, y in pairs)
    se = math.sqrt(resid / (n - 2) / sxx) if n > 2 and sxx > 0 else float("nan")
    t = slope / se if se and math.isfinite(se) and se > 0 else float("nan")

    directional = [(p, a) for p, a in pairs if p != 0]
    hits = sum(1 for p, a in directional if (p > 0) == (a > 0))
    hit_rate = hits / len(directional) if directional else float("nan")

    return {
        "n": n,
        "hit_rate": hit_rate,
        "ic": ic,
        "slope": slope,
        "intercept": intercept,
        "slope_se": se,
        "slope_t": t,
        "mean_abs_pred": sum(abs(x) for x in xs) / n,
        "mean_abs_real": sum(abs(y) for y in ys) / n,
        "degenerate": False,
    }


def weight_fingerprint(weights_dir: str, symbols=None) -> str:
    """
    權重檔的指紋 (檔名 + 大小 + mtime 的雜湊前 12 碼)。

    為什麼要記這個
    --------------
    收集樣本期間模型必須不變，否則同一份樣本會混入兩個世代，
    匯總斜率就沒有意義。凍結開關是第一道防線，但它擋不住手動
    --mode train、還原備份、或另一台機器同步過來的權重。
    把指紋記進每一筆預測，事後至少能把樣本切開 ——
    「悄悄混在一起」比「分成兩段」危險得多。
    """
    import hashlib

    try:
        if not os.path.isdir(weights_dir):
            return "no-weights-dir"
        parts = []
        for name in sorted(os.listdir(weights_dir)):
            if not name.endswith((".keras", ".pkl")):
                continue
            if symbols and not any(name.startswith(f"{sym}_") for sym in symbols):
                continue
            st = os.stat(os.path.join(weights_dir, name))
            parts.append(f"{name}:{st.st_size}:{int(st.st_mtime)}")
        if not parts:
            return "no-weights"
        return hashlib.sha256("|".join(parts).encode()).hexdigest()[:12]
    except Exception as e:
        logger.debug(f"權重指紋計算失敗: {e}")
        return "unknown"


def required_samples(
    pred_std: float, resid_std: float, slope: float, t_target: float = 2.0
) -> Optional[int]:
    """
    要在 |t| ≥ t_target 的水準上偵測到給定斜率，需要幾筆樣本。

    這個函式的存在是因為「沒偵測到」和「不存在」是兩回事。
    以本系統的實際數量級 (預測 std ≈ 0.4%、日報酬 std ≈ 1.8%)，
    真斜率 0.40 需要約 500 筆才看得出來 —— 單一標的每日一筆要等兩年，
    但 37 檔匯總只要 14 個交易日。所以分析必須「先匯總、後分檔」，
    否則會把樣本不足誤讀成沒有邊際，然後做出錯誤的重訓決定。
    """
    try:
        if pred_std <= 0 or slope <= 0 or resid_std <= 0:
            return None
        return int(math.ceil((t_target * resid_std / (pred_std * slope)) ** 2))
    except Exception:
        return None


def power_note(stats: Optional[dict], slope_of_interest: float = 0.3) -> str:
    """在斜率不顯著時，說明「還需要多少樣本」而不是逕自宣告沒有邊際。"""
    if not stats or stats.get("degenerate"):
        return ""
    se, n = stats.get("slope_se"), stats.get("n", 0)
    if not se or not math.isfinite(se) or se <= 0:
        return ""
    need = int(math.ceil(n * (2.0 * se / slope_of_interest) ** 2))
    if need <= n:
        return ""
    return (
        f"\n  -> 檢定力: 目前 n={n} 只能偵測到 |斜率| ≥ {2 * se:.2f}。"
        f"若真實斜率是 {slope_of_interest:.2f}，需約 {need:,} 筆才會顯著"
        f" (再等 {need - n:,} 筆)。「沒偵測到」不等於「不存在」。"
    )


def verdict(stats: Optional[dict]) -> str:
    """把數字翻成一句可以據以行動的話。"""
    if stats is None:
        return "樣本不足，尚無法判斷。"
    if stats.get("degenerate"):
        return f"n={stats['n']}，但預測或實際完全沒有變異 —— 模型可能已塌陷成常數。"

    t = stats.get("slope_t", float("nan"))
    slope, ic, hr, n = stats["slope"], stats["ic"], stats["hit_rate"], stats["n"]
    head = (
        f"n={n} · 命中率 {hr * 100:.1f}% · IC {ic:+.3f} · "
        f"斜率 {slope:+.3f} (t={t:+.2f})"
    )

    if not math.isfinite(t) or abs(t) < 2.0:
        return (
            f"{head}\n  -> 斜率與 0 無法區分 (|t| < 2)。目前沒有證據顯示存在邊際。"
            + power_note(stats)
        )
    if slope <= 0:
        return (
            f"{head}\n  -> 斜率顯著為負。預測方向與實際相反 —— "
            f"先確認目標定義與資料對齊有無錯誤，不要急著反向操作。"
        )
    shrink = 1.0 / slope if slope > 0 else float("inf")
    return (
        f"{head}\n  -> 斜率顯著為正: 方向有訊息，幅度被壓縮約 {shrink:.1f} 倍。"
        f"把預測乘以 {shrink:.2f} 才是無偏估計；門檻應據此重新設定。"
    )


def build_prediction_ledger(db_path: str, config=None) -> PredictionLedger:
    """Composition Root 使用。"""
    enabled = True
    if config is not None:
        try:
            val = config.get("ledger_settings.enabled")
            if val is not None:
                enabled = bool(val)
        except Exception:
            pass
    return PredictionLedger(db_path, enabled=enabled)
