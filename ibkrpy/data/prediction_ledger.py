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
    """Full snapshot of one scan. Fields are deliberately complete so any dimension can be sliced later."""

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
    -- Filled in later by resolve()
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
    Contract: record() never raises and never blocks the trading loop.
    A broken ledger must not stop trading, and trading-loop errors must not drop ledger
    rows, so every write is wrapped in try and only logged.
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
            logger.error(f"Prediction ledger init failed; disabled: {e}")
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
        Write several rows in one connection (one per loop instead of one per symbol).
        Backfill and import use the same path.
        """
        if not self.enabled or not recs:
            return
        try:
            with self._conn() as conn:
                conn.executemany(self._INSERT, [self._row(r) for r in recs])
                conn.commit()
        except Exception as e:
            logger.warning(
                f"Prediction ledger batch write failed (trading unaffected): {e}"
            )

    def record(self, rec: PredictionRecord) -> None:
        """
        Called on every scan. HOLDs are recorded too: predictions blocked by the threshold
        are the samples that show whether it is too high; logging only fills is biased.
        """
        if not self.enabled:
            return
        try:
            with self._conn() as conn:
                conn.execute(self._INSERT, self._row(rec))
                conn.commit()
        except Exception as e:
            logger.warning(
                f"[{rec.symbol}] Prediction ledger write failed (trading unaffected): {e}"
            )

    def resolve(self, max_rows: int = 20000) -> Dict[str, int]:
        """
        Fill realized_return from market_data bars. Idempotent; only unresolved rows are touched.

        Each prediction uses the close horizon_bars bars after the prediction time. If that
        bar is not there yet (too early or not downloaded) the row is skipped until next time.
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
            logger.error(f"Prediction ledger backfill failed: {e}")
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
            logger.error(f"Prediction ledger read failed: {e}")
            return []


def measure(rows: List[dict], min_samples: int = 30) -> Optional[dict]:
    """
    The three key edge statistics for resolved samples.

    hit_rate  Directional hit rate; 0.5 is a coin flip.
    ic        Correlation between predicted and realized (information coefficient).
                        Rule of thumb: 0.03-0.05 is usable, above 0.10 is strong.
    slope     Slope of realized on predicted, i.e. the calibration coefficient.
                        0.4 means realized moves are 40% of predicted: direction is informative
                        but magnitude is compressed. A slope near 0 means no edge.

    Also returns the slope's standard error and t-stat; a positive but insignificant slope is no conclusion.
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
    Fingerprint of the weight files (first 12 hex chars of a hash of name, size, mtime).

    Stored with every prediction so samples from different model generations can be
    separated. The freeze switch cannot stop a manual --mode train, a restored backup or
    synced weights, and silently pooling two generations makes the slope meaningless.
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
        logger.debug(f"Weight fingerprint failed: {e}")
        return "unknown"


def required_samples(
    pred_std: float, resid_std: float, slope: float, t_target: float = 2.0
) -> Optional[int]:
    """
    Samples needed to detect a given slope at |t| ≥ t_target.

    Not detecting an edge is not the same as no edge. With prediction std ≈ 0.4% and
    daily return std ≈ 1.8%, a true slope of 0.40 needs about 500 samples: two years for
    one symbol at one per day, but about 14 trading days pooled over 37 symbols. Analyse
    pooled first, then per symbol.
    """
    try:
        if pred_std <= 0 or slope <= 0 or resid_std <= 0:
            return None
        return int(math.ceil((t_target * resid_std / (pred_std * slope)) ** 2))
    except Exception:
        return None


def power_note(stats: Optional[dict], slope_of_interest: float = 0.3) -> str:
    """When the slope is insignificant, say how many more samples are needed instead of declaring no edge."""
    if not stats or stats.get("degenerate"):
        return ""
    se, n = stats.get("slope_se"), stats.get("n", 0)
    if not se or not math.isfinite(se) or se <= 0:
        return ""
    need = int(math.ceil(n * (2.0 * se / slope_of_interest) ** 2))
    if need <= n:
        return ""
    return (
        f"\n  -> Power: with n={n} only |slope| ≥ {2 * se:.2f} is detectable. "
        f"If the true slope is {slope_of_interest:.2f}, about {need:,} samples are needed"
        f" ({need - n:,} more). Not detected does not mean absent."
    )


def verdict(stats: Optional[dict]) -> str:
    """Turn the numbers into one actionable sentence."""
    if stats is None:
        return "Not enough samples to judge yet."
    if stats.get("degenerate"):
        return f"n={stats['n']}, but predictions or outcomes have no variance; the model may have collapsed to a constant."

    t = stats.get("slope_t", float("nan"))
    slope, ic, hr, n = stats["slope"], stats["ic"], stats["hit_rate"], stats["n"]
    head = (
        f"n={n} · hit rate {hr * 100:.1f}% · IC {ic:+.3f} · "
        f"slope {slope:+.3f} (t={t:+.2f})"
    )

    if not math.isfinite(t) or abs(t) < 2.0:
        return (
            f"{head}\n  -> Slope is indistinguishable from 0 (|t| < 2). No evidence of edge yet."
            + power_note(stats)
        )
    if slope <= 0:
        return (
            f"{head}\n  -> Slope is significantly negative: predictions point the wrong way. "
            f"Check the target definition and data alignment before trading the reverse."
        )
    shrink = 1.0 / slope if slope > 0 else float("inf")
    return (
        f"{head}\n  -> Slope is significantly positive: direction is informative, magnitude is compressed about {shrink:.1f}x."
        f" Multiply predictions by {shrink:.2f} for an unbiased estimate and reset thresholds accordingly."
    )


def build_prediction_ledger(db_path: str, config=None) -> PredictionLedger:
    """Used by the composition root."""
    enabled = True
    if config is not None:
        try:
            val = config.get("ledger_settings.enabled")
            if val is not None:
                enabled = bool(val)
        except Exception:
            pass
    return PredictionLedger(db_path, enabled=enabled)
