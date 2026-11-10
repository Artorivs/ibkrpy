# ibkrpy/manager/pipeline_manager.py
# Download, database and training pipeline

import os
import math
import asyncio
import pandas as pd
import json
import joblib
import numpy as np
import time
from typing import Tuple, Dict

from ib_insync import Stock

from ibkrpy.shared.config_manager import ConfigManager
from ibkrpy.shared.db_manager import DatabaseManager
from ibkrpy.data import bar_sync
from ibkrpy.data.data_pipeline import DataPipeline
from ibkrpy.data.ibkr_data_manager import IBKRDataManager
from ibkrpy.data.external_data import ExternalDataFetcher, align_daily_series
from ibkrpy.evaluation.model_tuner import ModelTuner
from ibkrpy.shared.horizon import forecast_horizon
import logging

logger = logging.getLogger(__name__)

PROJECT_ROOT = os.path.dirname(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
)
WEIGHTS_DIR = os.path.join(PROJECT_ROOT, "weights")
DATA_DIR = os.path.join(PROJECT_ROOT, "data")


class PipelineManager:
    """Pipeline manager for data fetching, feature engineering and model retraining."""

    def __init__(
        self,
        config: ConfigManager,
        db: DatabaseManager,
        pipeline: DataPipeline,
        ib_data: IBKRDataManager,
        ext_fetcher: ExternalDataFetcher,
        target_symbol: str = None,
        benchmark_resolver=None,
        benchmark_store=None,
    ):
        self.config = config
        self.db = db
        self.pipeline = pipeline
        self.ib_data = ib_data
        self.ext = ext_fetcher

        self.benchmark_symbol = self.config.get(
            "general_settings.benchmark_symbol", "SPY"
        )

        if target_symbol:
            self.symbols = [target_symbol]
        else:
            self.symbols = (
                [p.symbol for p in self.config.asset_profiles]
                if self.config.asset_profiles
                else ["AAPL"]
            )

        if benchmark_resolver is None:
            from ibkrpy.data.benchmark_resolver import StaticBenchmarkResolver

            benchmark_resolver = StaticBenchmarkResolver(self.benchmark_symbol)
        self.benchmark_resolver = benchmark_resolver
        self.benchmark_store = benchmark_store

        self._benchmark_for: Dict[str, str] = {}
        self._tradable = list(self.symbols)
        self._refresh_benchmark_map()

        self.symbol_terms = {}
        param_path = os.path.join(WEIGHTS_DIR, "global_best_params.json")
        if os.path.exists(param_path):
            try:
                with open(param_path, "r", encoding="utf-8") as f:
                    global_params = json.load(f)
                    for sym, params in global_params.items():
                        if "term" in params:
                            self.symbol_terms[sym] = params["term"]
            except Exception:
                pass

        for sym in self.symbols:
            if sym not in self.symbol_terms:
                self.symbol_terms[sym] = "long_term"

    def _refresh_benchmark_map(self):
        """
        Resolve each symbol's benchmark and add every benchmark used to the download list.

        With per-symbol benchmarks several ETFs (SMH / XLF / XLV ...) may be in use. Each needs
        history first, or engineer_advanced_features gets an empty benchmark_df and the
        bench_return / bench_correlation columns disappear.
        """
        self._benchmark_for = {}
        for symbol in self._tradable:
            chosen = self.benchmark_resolver.resolve(symbol) or self.benchmark_symbol
            self._benchmark_for[symbol] = chosen

        needed = sorted(set(self._benchmark_for.values()))
        self._reference_only = {b for b in needed if b not in self._tradable}

        self.symbols = sorted(self._reference_only) + list(self._tradable)

        pairs = ", ".join(f"{s}→{b}" for s, b in list(self._benchmark_for.items())[:8])
        logger.info(
            f"📐 Benchmark map resolved ({len(set(self._benchmark_for.values()))} unique): "
            f"{pairs}{' ...' if len(self._benchmark_for) > 8 else ''}"
        )
        if self._reference_only:
            logger.info(
                f"\t-> Extra reference symbols to download: {', '.join(sorted(self._reference_only))}"
            )

    def benchmark_for(self, symbol: str) -> str:
        return self._benchmark_for.get(symbol, self.benchmark_symbol)

    def _correlation_candidates(self) -> list:
        """
        Candidate pool for correlation-based selection.

        CorrelationBenchmarkResolver needs candidate history, but a candidate only enters the
        download list once chosen, and it cannot be chosen without data. So the pool is
        downloaded on its own, for the single timeframe the correlation uses (daily by
        default) to keep API cost low.
        """
        settings = self.config.get("benchmark_settings") or {}
        if not settings.get("enable_correlation", True):
            return []
        from ibkrpy.data.benchmark_resolver import DEFAULT_CANDIDATE_POOL

        pool = settings.get("candidate_pool") or DEFAULT_CANDIDATE_POOL
        return [s for s in pool if s not in self._tradable]

    async def _ingest_benchmark_candidates(self):
        """Fill daily bars for the correlation pool. Up-to-date symbols are skipped by _sync_symbol_term_data."""
        candidates = self._correlation_candidates()
        if not candidates:
            return

        settings = self.config.get("benchmark_settings") or {}
        timeframe = settings.get("correlation_timeframe", "1 day")
        term = next(
            (
                t
                for t in ("long_term", "mid_term", "short_term")
                if self._get_term_settings(t)["bar_size"] == timeframe
            ),
            "long_term",
        )

        logger.info(
            f"📐 Filling benchmark candidate pool ({len(candidates)} symbols, {timeframe} only)..."
        )
        for symbol in candidates:
            contract = await self.ib_data.qualify(symbol)
            if contract is None:
                continue
            try:
                await self._sync_symbol_term_data(symbol, term, contract)
            except Exception as e:
                logger.warning(f"\t⚠️ [{symbol}] Candidate pool sync failed: {e}")

    def _get_term_settings(self, term: str) -> dict:
        """Bar size, total download days and per-request chunk days for a term."""
        if term == "short_term":
            return {
                "bar_size": self.config.get(
                    "general_settings.short_term_bar_size", "5 mins"
                ),
                "total_days": 60,
                "chunk_days": 30,
            }
        elif term == "mid_term":
            return {
                "bar_size": self.config.get(
                    "general_settings.mid_term_bar_size", "1 hour"
                ),
                "total_days": 180,
                "chunk_days": 90,
            }
        else:
            return {
                "bar_size": self.config.get(
                    "general_settings.long_term_bar_size", "1 day"
                ),
                "total_days": 730,
                "chunk_days": 365,
            }

    async def _sync_symbol_term_data(self, symbol: str, term: str, contract: Stock):
        """
        Bring stored bars up to date, on demand.

        Fetches backwards from now until a chunk reaches the oldest bar that needs
        refreshing, so a long outage never leaves a hole. That bar is the last stored one,
        or an earlier one when the stored series has a hole of ≥2 weekdays or a split-like
        jump inside the download window. If the refetched closes disagree with stored ones
        (the provider re-adjusted for a split), older stored bars are rescaled to match.
        """
        settings = self._get_term_settings(term)
        bar_size = settings["bar_size"]
        total_days = settings["total_days"]
        chunk_days = settings["chunk_days"]

        logger.debug(f"\t-> 🔄 Syncing {term} ({bar_size}) data...")
        df_existing = self.db.get_market_data_sync(symbol, timeframe=bar_size)
        now = pd.Timestamp.now(tz="UTC")
        window_start = now - pd.Timedelta(days=total_days)

        if df_existing.empty or len(df_existing) < 50:
            df_existing = pd.DataFrame()
            target = window_start
        else:
            df_existing = bar_sync.normalize_bars(df_existing)
            last = df_existing.index[-1]
            repairs = {
                "hole": bar_sync.find_gap_start(df_existing.index, window_start),
                "split-like jump": bar_sync.find_split_jump(
                    df_existing["Close"], window_start
                ),
            }
            repairs = {k: v for k, v in repairs.items() if v is not None}
            behind = np.busday_count(
                bar_sync.session_dates(df_existing.index[-1:])[0],
                now.tz_convert(bar_sync.NY).date(),
            )
            if not repairs and behind <= 0:
                logger.debug("\t✅ Data is up to date; nothing to sync.")
                return
            target = min([last, *repairs.values()])
            for why, at in repairs.items():
                logger.warning(
                    f"\t-> [{symbol}] {bar_size}: stored series has a {why} after "
                    f"{at:%Y-%m-%d}; refetching from there."
                )

        chunks = []
        end = None
        max_requests = math.ceil(total_days / chunk_days) + 2
        for _ in range(max_requests):
            end_ts = end if end is not None else now
            span = math.ceil((end_ts - target).total_seconds() / 86400) + 1
            duration = bar_sync.duration_str(min(chunk_days, max(span, 1)), bar_size)
            end_str = "" if end is None else end.strftime("%Y%m%d-%H:%M:%S")

            df_chunk = await self.ib_data.fetch_historical_data(
                contract=contract,
                end_datetime=end_str,
                duration=duration,
                bar_size=bar_size,
                what_to_show="TRADES",
            )
            if df_chunk.empty:
                logger.warning(
                    f"\t-> [{symbol}] {duration}/{bar_size} request returned no data; retrying in 20s..."
                )
                await asyncio.sleep(20)
                df_chunk = await self.ib_data.fetch_historical_data(
                    contract=contract,
                    end_datetime=end_str,
                    duration=duration,
                    bar_size=bar_size,
                    what_to_show="TRADES",
                    allow_cache=False,
                )
            if df_chunk.empty:
                break

            df_chunk = bar_sync.normalize_bars(df_chunk)
            chunks.append(df_chunk)
            first = df_chunk.index[0]
            if first <= target or (end is not None and first >= end):
                break
            end = first
            await asyncio.sleep(1)

        if not chunks:
            logger.error(f"❌ [{symbol}] Every chunk for this term failed.")
            return

        df_new = bar_sync.normalize_bars(pd.concat(chunks))
        if df_new.index[0] > target and not df_existing.empty:
            logger.warning(
                f"\t-> [{symbol}] {bar_size}: download stopped at {df_new.index[0]:%Y-%m-%d}, "
                f"short of {target:%Y-%m-%d}; the hole stays until the next sync."
            )

        if not df_existing.empty:
            ratio = bar_sync.adjustment_ratio(df_existing["Close"], df_new["Close"])
            if bar_sync.needs_rescale(ratio):
                older = df_existing[df_existing.index < df_new.index[0]]
                if not older.empty:
                    self.db.save_bulk_market_data(
                        symbol, bar_sync.rescale_bars(older, ratio), timeframe=bar_size
                    )
                logger.warning(
                    f"\t-> [{symbol}] {bar_size}: stored prices differ from the provider's "
                    f"by ×{ratio:.3f} (split or adjustment); rescaled {len(older)} older bars."
                )

        self.db.save_bulk_market_data(symbol, df_new, timeframe=bar_size)
        logger.info(f"\t✅ [{symbol}] Merged {len(df_new)} {bar_size} bars.")

    def _train_dl_models(
        self,
        symbol: str,
        df: pd.DataFrame,
        bench_df: pd.DataFrame,
        macro_data: dict,
        benchmark_symbol: str = None,
        horizon: int = 1,
    ):
        """Train the deep learning models (LSTM, Transformer) and save .keras files."""
        logger.info(f"🔍 [{symbol}] Training deep learning models...")
        logger.info(f"[{symbol}] Rows: {len(df)}, features: {df.shape[1]}")
        if df.empty or len(df) < 60:
            logger.warning(
                f"⚠️ [{symbol}] Far too little data ({len(df)} rows); skipping DL training."
            )
            return

        os.makedirs(WEIGHTS_DIR, exist_ok=True)

        df_adv = self.pipeline.engineer_advanced_features(df, bench_df, macro_data)
        df_adv = df_adv.ffill().bfill().fillna(0)

        scale_cols = self.pipeline.select_model_features(df_adv)
        price_rel = self.pipeline.classify_price_relative(df_adv, scale_cols)
        minmax_cols = [c for c in scale_cols if c not in price_rel]

        logger.info(
            f"\t=> Feature columns: {len(scale_cols)} "
            f"(price-relative {len(price_rel)} / Min-Max {len(minmax_cols)}), "
            f"target: log return over {horizon} bar(s)"
        )

        split = int(len(df_adv) * 0.8)
        df_train = df_adv.iloc[:split]

        self.pipeline.save_training_artifacts(
            symbol,
            scaler=self.pipeline._compute_scaler(df_train, minmax_cols),
            features=scale_cols,
            price_relative=price_rel,
            target_mode="log_return",
            target_scale=100.0,
            benchmark=benchmark_symbol or self.benchmark_for(symbol),
            horizon_bars=horizon,
        )
        df_scaled = self.pipeline.transform_scale(df_adv, minmax_cols, symbol)

        look_back = 60
        X, y = self.pipeline.create_sequences(
            df_scaled,
            scale_cols,
            look_back,
            raw_close=df_adv["Close"],
            price_relative=price_rel,
            target_mode="log_return",
            target_scale=100.0,
            horizon=horizon,
        )

        logger.info(
            f"\t=> Total bars: {len(df_adv)}, valid training sequences: {len(X)}"
        )

        if len(X) < 16:
            logger.warning(
                f"⚠️ [{symbol}] Too few valid sequences for gradient descent ({len(X)}); skipping DL training."
            )
            return

        dynamic_batch_size = min(32, max(8, len(X) // 4))

        try:
            from keras.callbacks import EarlyStopping
            from ibkrpy.models.lstm import LSTMModel
            from ibkrpy.models.transformer import TransformerModel

            callbacks = [
                EarlyStopping(monitor="val_loss", patience=5, restore_best_weights=True)
            ]
            fit_kwargs = dict(
                epochs=25,
                batch_size=dynamic_batch_size,
                verbose=0,
                callbacks=callbacks,
                validation_split=0.2,
                shuffle=False,
            )

            logger.info(f"\t-> 🚀 Fitting LSTM (batch size: {dynamic_batch_size})...")
            lstm = LSTMModel(
                look_back=look_back, feature_cols=scale_cols, weights_dir=WEIGHTS_DIR
            )
            lstm.model = lstm._build_model()
            lstm.model.fit(X, y, **fit_kwargs)
            lstm_path = os.path.join(WEIGHTS_DIR, f"{symbol}_LSTM.keras")
            lstm.model.save(lstm_path)

            if os.path.exists(lstm_path):
                logger.info(f"\t✅ LSTM weights written: {lstm_path}")
            else:
                logger.error("\t❌ LSTM weights were not written.")

            logger.info(
                f"\t-> 🚀 Fitting Transformer (batch size: {dynamic_batch_size})..."
            )
            transformer = TransformerModel(
                look_back=look_back, feature_cols=scale_cols, weights_dir=WEIGHTS_DIR
            )
            transformer.model = transformer._build_model()
            transformer.model.fit(X, y, **fit_kwargs)
            tf_path = os.path.join(WEIGHTS_DIR, f"{symbol}_Transformer.keras")
            transformer.model.save(tf_path)

            if os.path.exists(tf_path):
                logger.info(f"\t✅ Transformer weights written: {tf_path}")
            else:
                logger.error("\t❌ Transformer weights were not written.")

            logger.info(f"✅ [{symbol}] Deep learning training finished.\n")
        except ImportError:
            logger.warning(
                "⚠️ TensorFlow/Keras not installed; skipping deep learning training."
            )
        except Exception as e:
            logger.error(f"❌ [{symbol}] DL training failed: {e}")

    def _train_safe_models(self, symbol: str, df: pd.DataFrame, horizon: int = 1):
        """Train the statistical and regime models (ARIMA, GARCH, HMM) into one .pkl bundle."""
        logger.info(f"🔍 [{symbol}] Training statistical and regime models...")
        if df.empty or len(df) < 50:
            logger.warning(
                f"⚠️ [{symbol}] Not enough data to train; skipping statistical models."
            )
            return

        weights_dir = WEIGHTS_DIR
        os.makedirs(weights_dir, exist_ok=True)

        classical_bundle = {"horizon_bars": int(horizon)}

        logger.info("\t-> Fitting ARIMA...")
        try:
            from statsmodels.tsa.arima.model import ARIMA

            series = df["Close"].dropna().values
            model_arima = ARIMA(series, order=(5, 1, 0))
            res_arima = model_arima.fit()
            classical_bundle["arima"] = res_arima
            logger.info("\t✅ ARIMA trained")
        except Exception as e:
            logger.warning(f"\t⚠️ ARIMA training failed: {e}")

        logger.info("\t-> Fitting GARCH...")
        try:
            from arch import arch_model

            returns = np.log(df["Close"] / df["Close"].shift(1)).dropna() * 100.0
            if len(returns) > 20:
                am = arch_model(returns, vol="Garch", p=1, q=1, dist="normal")
                res_garch = am.fit(disp="off")
                classical_bundle["garch"] = res_garch.params
                logger.info("\t✅ GARCH trained")
        except Exception as e:
            logger.warning(f"\t⚠️ GARCH training failed: {e}")

        bundle_path = os.path.join(weights_dir, f"{symbol}_classical.pkl")
        joblib.dump(classical_bundle, bundle_path)

        if os.path.exists(bundle_path):
            logger.info(f"\t✅ [{symbol}] Classical model bundle written.")

    def _vectorised_regimes(self, df: pd.DataFrame) -> pd.Series:
        """
        Market regime for every bar of the series in one pass.

        Same rules as MarketRegimeDetector.detect(), vectorised instead of called per bar
        (which recomputes indicators over the whole series each time); orders of magnitude faster.
        """
        from ibkrpy.shared import indicators as ta
        from ibkrpy.strategy.regime_detector import MarketRegimeDetector

        d = MarketRegimeDetector()
        out = pd.Series("SIDEWAYS_QUIET", index=df.index, dtype=object)

        try:
            adx = ta.adx(df["High"], df["Low"], df["Close"], length=d.adx_period)
            adx_val = (
                adx.iloc[:, 0]
                if adx is not None and not adx.empty
                else pd.Series(0.0, index=df.index)
            )
            sma_s = ta.sma(df["Close"], length=d.ma_short)
            sma_l = ta.sma(df["Close"], length=d.ma_long)
            atr = ta.atr(df["High"], df["Low"], df["Close"], length=d.atr_period)
            atr_pct = (atr / df["Close"]).fillna(0.0)

            trending = adx_val.fillna(0.0) > d.adx_threshold
            volatile = atr_pct > d.vol_threshold
            bull = sma_s > sma_l

            out[trending & bull] = "BULL_TREND"
            out[trending & ~bull] = "BEAR_TREND"
            out[~trending & volatile] = "SIDEWAYS_VOLATILE"
            out[~trending & ~volatile] = "SIDEWAYS_QUIET"
        except Exception as e:
            logger.warning(
                f"\t⚠️ Vectorised regime calculation failed; using SIDEWAYS_QUIET throughout: {e}"
            )

        return out

    def _walk_forward_predictions(
        self, symbol: str, df: pd.DataFrame, term: str
    ) -> pd.DataFrame:
        """
        Generate genuine model predictions over the out-of-sample segment.

          - First 70% is in-sample, last 30% out-of-sample
          - Models are fitted on in-sample only, then forecast the term's horizon over OOS
          - Row t holds the forecast made at bar t's close for bar t+h, as live does, so
            prediction / Close[t] - 1 is the predicted move from the price live would see
          - Regimes come from MarketRegimeDetector
          - Volatility is the rolling per-bar std of log returns scaled by √h, the same
            horizon σ the live strategy receives

        ARIMA only by default: it fits in seconds, and the tournament chooses the bar size and
        risk parameters rather than validating the networks. For an ensemble closer to live,
        set tuning_settings.walk_forward_models in config.yaml.
        """
        wf_models = self.config.get("tuning_settings.walk_forward_models", ["ARIMA"])
        oos_frac = float(self.config.get("tuning_settings.oos_fraction", 0.3))
        max_hist = int(self.config.get("tuning_settings.trailing_window", 500))
        horizon = forecast_horizon(self.config, term)

        n = len(df)
        split = int(n * (1 - oos_frac))
        if n < 150 or split < 100:
            logger.warning(
                f"\t⚠️ {term} has too little data for a walk-forward split ({n} bars)."
            )
            return pd.DataFrame()

        oos = df.iloc[split:].copy()
        close = df["Close"].astype(float).values

        preds = np.full(len(oos), np.nan)

        if "ARIMA" in wf_models:
            try:
                from statsmodels.tsa.arima.model import ARIMA

                base = ARIMA(close[:split], order=(5, 1, 0)).fit()

                for k in range(len(oos)):
                    t = split + k
                    lo = max(0, t + 1 - max_hist)
                    hist = close[lo : t + 1]
                    if len(hist) < 20:
                        continue
                    try:
                        path = base.apply(hist).forecast(steps=horizon)
                        preds[k] = float(path[horizon - 1])
                    except Exception:
                        preds[k] = close[t]
            except Exception as e:
                logger.warning(f"\t⚠️ ARIMA walk-forward failed: {e}")

        dl_wanted = [m for m in wf_models if m in ("LSTM", "Transformer")]
        if dl_wanted:
            dl_preds = self._walk_forward_dl(symbol, df, split, dl_wanted, horizon)
            if dl_preds is not None:
                stacked = [
                    p for p in ([preds] if "ARIMA" in wf_models else []) + [dl_preds]
                ]
                preds = np.nanmean(np.vstack(stacked), axis=0)

        oos["prediction"] = preds
        oos = oos[np.isfinite(oos["prediction"])]
        if oos.empty:
            logger.warning(f"\t⚠️ {term} produced no valid OOS predictions.")
            return pd.DataFrame()

        log_ret = np.log(df["Close"] / df["Close"].shift(1))
        bar_vol = (
            log_ret.rolling(20).std().reindex(oos.index).fillna(0.005).replace(0, 0.005)
        )
        oos["volatility"] = bar_vol * np.sqrt(horizon)

        oos["regime"] = (
            self._vectorised_regimes(df).reindex(oos.index).fillna("SIDEWAYS_QUIET")
        )

        edge = (oos["prediction"] / oos["Close"] - 1).abs()
        logger.info(
            f"\t=> OOS samples {len(oos)} bars (total {n}), tuning models {wf_models}, "
            f"horizon {horizon} bar(s); median predicted edge {edge.median()*100:.3f}%, "
            f"median per-bar vol {bar_vol.median()*100:.3f}% "
            f"(over the horizon {bar_vol.median() * np.sqrt(horizon) * 100:.3f}%)"
        )

        return oos

    def _walk_forward_dl(
        self,
        symbol: str,
        df: pd.DataFrame,
        split: int,
        model_types: list,
        horizon: int = 1,
    ):
        """Train networks in-sample and infer window by window over OOS (expensive, not the default)."""
        try:
            from keras.callbacks import EarlyStopping
            from ibkrpy.models.lstm import LSTMModel
            from ibkrpy.models.transformer import TransformerModel
        except ImportError:
            logger.debug(
                "    ⚠️ TensorFlow/Keras not installed; walk-forward skips the networks."
            )
            return None

        try:
            df_adv = (
                self.pipeline.engineer_advanced_features(df).ffill().bfill().fillna(0)
            )
            feats = self.pipeline.select_model_features(df_adv)
            price_rel = self.pipeline.classify_price_relative(df_adv, feats)
            minmax = [c for c in feats if c not in price_rel]

            adv_split = min(split, len(df_adv) - 1)
            tmp_key = f"__wf_{symbol}"
            self.pipeline.fit_scale(df_adv.iloc[:adv_split], minmax, tmp_key)
            df_s = self.pipeline.transform_scale(df_adv, minmax, tmp_key)

            look_back = 60
            X, y = self.pipeline.create_sequences(
                df_s,
                feats,
                look_back,
                raw_close=df_adv["Close"],
                price_relative=price_rel,
                target_mode="log_return",
                target_scale=100.0,
                horizon=horizon,
            )
            if len(X) < 100:
                return None

            # Sequence i is anchored on bar seq_pos - 1 and labelled with bar seq_pos - 1 + h.
            # Train only on labels inside the in-sample segment; predict from anchors after it.
            seq_pos = np.arange(len(X)) + look_back
            in_mask = seq_pos - 1 + horizon < adv_split
            out_mask = seq_pos >= adv_split

            cb = [
                EarlyStopping(monitor="val_loss", patience=3, restore_best_weights=True)
            ]
            outputs = []
            for mt in model_types:
                cls = LSTMModel if mt == "LSTM" else TransformerModel
                m = cls(
                    look_back=look_back, feature_cols=feats, weights_dir=WEIGHTS_DIR
                )
                m.model = m._build_model()
                m.model.fit(
                    X[in_mask],
                    y[in_mask],
                    epochs=12,
                    batch_size=32,
                    verbose=0,
                    validation_split=0.2,
                    shuffle=False,
                    callbacks=cb,
                )
                raw = m.model.predict(X[out_mask], verbose=0).reshape(-1)
                anchors = df_adv["Close"].values[seq_pos[out_mask] - 1]
                outputs.append(anchors * np.exp(raw / 100.0))

            oos_idx = df_adv.index[seq_pos[out_mask] - 1]
            series = pd.Series(np.mean(outputs, axis=0), index=oos_idx)
            return series.reindex(df.index[split:]).values
        except Exception as e:
            logger.warning(f"\t⚠️ Network walk-forward failed: {e}")
            return None
        finally:
            self.pipeline.invalidate(f"__wf_{symbol}")

    def _run_optuna_optimization(
        self,
        symbol: str,
        df: pd.DataFrame,
        bench_df: pd.DataFrame,
        macro_data: dict,
        term: str,
    ) -> Tuple[dict, float]:
        """Tune parameters on genuine out-of-sample predictions; returns (best params, composite score)."""
        default_params = {
            "volatility_stop_loss_multiplier": 2.0,
            "volatility_take_profit_multiplier": 3.0,
        }

        oos = self._walk_forward_predictions(symbol, df, term)
        if oos.empty:
            return default_params, -999.0

        tuner = ModelTuner(config=self.config)
        try:
            return tuner.optimize_strategy_params(
                symbol, df.loc[oos.index], oos, n_trials=20, term=term
            )
        except Exception as e:
            logger.warning(f"\t⚠️ [{symbol}] {term} Optuna tuning error: {e}")
            return default_params, -999.0

    async def run_data_ingestion(self):
        """Stage 1: incremental daily download into the database (on demand)."""
        logger.info(
            "[Pipeline] Starting incremental download and database sync (daily sync)"
        )

        all_terms = ["long_term", "mid_term", "short_term"]

        await self._ingest_benchmark_candidates()

        if hasattr(self.benchmark_resolver, "invalidate"):
            self.benchmark_resolver.invalidate()
            self._refresh_benchmark_map()

        skipped = []
        for symbol in self.symbols:
            logger.debug(f"[{symbol}] Checking and syncing latest market data...")

            contract = await self.ib_data.qualify(symbol)
            if contract is None:
                skipped.append(symbol)
                continue

            for term in all_terms:
                try:
                    await self._sync_symbol_term_data(symbol, term, contract)
                except Exception as e:
                    logger.warning(f"\t⚠️ [{symbol}] {term} data update raised: {e}")

            await asyncio.sleep(0.5)

        if skipped:
            logger.error(
                f"❌ Skipped {len(skipped)} symbols whose contracts could not be resolved: "
                f"{', '.join(skipped)}. Check the spelling of assets in config.yaml."
            )

        stats = getattr(self.ib_data, "pacing_stats", None)
        if callable(stats):
            s = stats()
            logger.info(
                f"📊 [Data sync done] {s['cache_misses']} historical requests / "
                f"{s['cache_hits']} cache hits (hit rate {s['cache_hit_rate']:.0%})."
            )

    async def run_training_and_tuning(self):
        """Stage 2: term tournament. Drop weak terms, then tune and train the best one."""
        logger.info("[Pipeline] Starting term tournament and model training")

        if hasattr(self.benchmark_resolver, "invalidate"):
            self.benchmark_resolver.invalidate()
            self._refresh_benchmark_map()

        all_terms = ["long_term", "mid_term", "short_term"]
        os.makedirs(DATA_DIR, exist_ok=True)

        global_vix_series = None
        fred_cache_path = os.path.join(DATA_DIR, "fred_vix_cache.csv")
        need_fetch_fred = True

        if os.path.exists(fred_cache_path):
            mod_time = os.path.getmtime(fred_cache_path)
            if (time.time() - mod_time) < 43200:
                try:
                    global_vix_series = pd.read_csv(
                        fred_cache_path, index_col=0, parse_dates=True
                    ).squeeze("columns")
                    need_fetch_fred = False
                    logger.debug(
                        "   -> 🌍 Reading cached FRED VIX history from data/..."
                    )
                except Exception:
                    pass

        if need_fetch_fred and self.ext:
            logger.debug("   -> 🌍 Requesting latest macro data from FRED (VIXCLS)...")
            try:
                global_vix_series = await self.ext.fetch_fred_series("VIXCLS")
                if global_vix_series is not None and not global_vix_series.empty:
                    global_vix_series.to_csv(fred_cache_path)
                    logger.info(
                        f"\t✅ FRED VIX data fetched and saved to {fred_cache_path}."
                    )
            except Exception as e:
                logger.warning(f"\t⚠️ FRED API request failed: {e}")
                if os.path.exists(fred_cache_path):
                    global_vix_series = pd.read_csv(
                        fred_cache_path, index_col=0, parse_dates=True
                    ).squeeze("columns")

        fmp_cache_path = os.path.join(WEIGHTS_DIR, "fmp_cache.json")
        fmp_cache = {}
        if os.path.exists(fmp_cache_path):
            try:
                with open(fmp_cache_path, "r", encoding="utf-8") as f:
                    fmp_cache = json.load(f)
            except Exception:
                pass

        trained_any = False
        for symbol in self.symbols:
            if symbol in getattr(self, "_reference_only", set()):
                logger.info(
                    f"⏭️ Skipping {symbol}: benchmark reference data only. "
                    f"Add it to assets in config.yaml to trade it."
                )
                continue

            trained_any = True
            logger.info(f"🔥 Starting term tournament and training: {symbol} 🔥")

            fmp_data = {}
            if self.ext and self.ext.fmp_api_key:
                if symbol in fmp_cache:
                    logger.info(
                        "\t-> 🏢 Reading FMP company fundamentals from cache..."
                    )
                    fmp_data = fmp_cache[symbol]
                else:
                    logger.info("\t-> 🏢 Requesting company fundamentals from FMP...")
                    try:
                        fmp_profile = await self.ext.fetch_fmp_profile(symbol)
                        if fmp_profile:
                            fmp_data = fmp_profile
                            fmp_cache[symbol] = fmp_data
                            os.makedirs(WEIGHTS_DIR, exist_ok=True)
                            with open(fmp_cache_path, "w", encoding="utf-8") as f:
                                json.dump(fmp_cache, f, indent=4)
                    except Exception as e:
                        logger.warning(f"\t=> ⚠️ FMP API request raised: {e}")

            best_term = None
            best_score = -float("inf")
            best_params = {}
            best_df = pd.DataFrame()
            best_macro = {}

            for term in all_terms:
                settings = self._get_term_settings(term)
                bar_size = settings["bar_size"]

                df = self.db.get_market_data_sync(symbol, timeframe=bar_size)

                last_date = (
                    pd.Timestamp(df.index[-1]).normalize() if not df.empty else None
                )
                now_date = pd.Timestamp.now().normalize()
                stale_days = (
                    np.busday_count(last_date.date(), now_date.date())
                    if last_date
                    else 999
                )

                if df.empty or len(df) < 100 or stale_days > 5:
                    logger.info(
                        f"\t-> 🔍 Data for {term} is missing or stale; downloading now..."
                    )
                    try:
                        contract = await self.ib_data.qualify(symbol)
                        if contract is None:
                            logger.error(
                                f"\t-> ❌ [{symbol}] Contract cannot be resolved; skipping {term}."
                            )
                            continue
                        await self._sync_symbol_term_data(symbol, term, contract)
                        df = self.db.get_market_data_sync(symbol, timeframe=bar_size)
                    except Exception as e:
                        logger.warning(f"\t-> ⚠️ On-demand download failed: {e}")

                if df.empty or len(df) < 100:
                    logger.warning(
                        f"\t-> ⚠️ Still not enough {term} data; skipping this term."
                    )
                    continue

                bench_df = self.db.get_market_data_sync(
                    self.benchmark_for(symbol), timeframe=bar_size
                )
                if not bench_df.empty:
                    bench_df = bench_df.reindex(df.index, method="ffill").bfill()
                else:
                    logger.warning(
                        f"\t-> ⚠️ Benchmark {self.benchmark_for(symbol)} has no {bar_size} data; "
                        f"bench_return / bench_correlation are missing this round."
                    )

                macro_data = {}
                if global_vix_series is not None and not global_vix_series.empty:
                    macro_data["VIX"] = align_daily_series(global_vix_series, df.index)

                logger.info(f"\t-> ⏳ Evaluating {term}...")
                params, score = self._run_optuna_optimization(
                    symbol, df, bench_df, macro_data, term
                )
                logger.info(f"\t=> {term} expected composite score: {score:.2f}")

                if score > best_score:
                    best_score = score
                    best_term = term
                    best_params = params
                    best_df = df
                    best_macro = macro_data

            MIN_VIABLE_SCORE = self.config.get(
                "tuning_settings.min_viable_score", -900.0
            )

            if best_term is None or best_score <= MIN_VIABLE_SCORE:
                logger.error(
                    f"❌ [{symbol}] No term passed evaluation (best score {best_score:.2f}); "
                    f"stopping training for this symbol. No weights or parameters were produced, so it will not trade live."
                )
                continue

            logger.info(
                f"🎉 [{symbol}] Tournament finished. Winning term: {best_term} (score: {best_score:.2f})"
            )

            best_params["term"] = best_term
            best_horizon = forecast_horizon(self.config, best_term)
            best_params["horizon_bars"] = best_horizon
            if fmp_data:
                best_params["fmp_sector"] = fmp_data.get("sector")
                best_params["fmp_industry"] = fmp_data.get("industry")
                best_params["fmp_beta"] = fmp_data.get("beta")
                best_params["fmp_mktCap"] = fmp_data.get("mktCap")

            chosen_benchmark = self.benchmark_for(symbol)
            best_params["benchmark"] = chosen_benchmark

            bench_df = self.db.get_market_data_sync(
                chosen_benchmark,
                timeframe=self._get_term_settings(best_term)["bar_size"],
            )
            if not bench_df.empty:
                bench_df = bench_df.reindex(best_df.index, method="ffill").bfill()

            logger.info(f"\t-> 📐 Benchmark used for this training: {chosen_benchmark}")
            self._train_dl_models(
                symbol,
                best_df,
                bench_df,
                best_macro,
                benchmark_symbol=chosen_benchmark,
                horizon=best_horizon,
            )
            self._train_safe_models(symbol, best_df, horizon=best_horizon)

            if self.benchmark_store is not None:
                try:
                    self.benchmark_store.set(symbol, chosen_benchmark)
                except Exception as e:
                    logger.warning(f"\t⚠️ Failed to write benchmark map: {e}")

            weights_dir = WEIGHTS_DIR
            os.makedirs(weights_dir, exist_ok=True)

            param_path = os.path.join(weights_dir, "global_best_params.json")
            global_params = {}
            if os.path.exists(param_path):
                try:
                    with open(param_path, "r", encoding="utf-8") as f:
                        global_params = json.load(f)
                except Exception:
                    pass

            global_params[symbol] = best_params

            with open(param_path, "w", encoding="utf-8") as f:
                json.dump(global_params, f, indent=4)

            logger.info(f"\t✅ Best strategy parameters saved to: {param_path}")
            self.symbol_terms[symbol] = best_term
            logger.info(f"🏆 {symbol} ({best_term}) training and tuning complete.")

        if not trained_any:
            logger.error(
                f"❌ No symbol entered training this run. Pending list: {self.symbols}; "
                f"reference-only: {sorted(getattr(self, '_reference_only', set()))}."
            )

    async def run_autopilot(self):
        """Run the whole pipeline."""
        await self.run_data_ingestion()
        await self.run_training_and_tuning()
