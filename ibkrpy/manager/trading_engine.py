# ibkrpy/manager/trading_engine.py
# Trading engine. One-way flow: data -> macro context -> prediction -> strategy -> execution

import asyncio
import os
import math
from datetime import datetime, timezone
import time
from typing import Dict, Any
from ib_insync import Order, LimitOrder, MarketOrder, Stock, TagValue
from ibkrpy.strategy.volatility_estimator import build_volatility_estimator
from ibkrpy.shared.session_clock import build_session_clock
from ibkrpy.data.external_data import align_daily_series
from ibkrpy.strategy.execution_costs import build_execution_cost_model
from ibkrpy.data.prediction_ledger import (
    TERM_BARS,
    PredictionRecord,
    build_prediction_ledger,
    weight_fingerprint,
)
import pandas as pd
import numpy as np
import logging

logger = logging.getLogger("ibkrpy.trading_engine")
_TERM_POLL_SECONDS = {
    "short_term": 300,
    "mid_term": 900,
    "long_term": 3600,
}


class TradingEngine:
    """Main loop with one-way data flow, adaptive data fetching per term, driven by config."""

    COLLAPSE_RATIO = 0.05

    def __init__(
        self,
        data_manager,
        model_orchestrator,
        risk_controller,
        strategy_map: Dict[str, Any],
        db_manager=None,
        ext_fetcher=None,
        market_analyzer=None,
        data_pipeline=None,
        regime_detector=None,
        dry_run: bool = False,
        symbol_terms: Dict[str, str] = None,
        config_manager=None,
        benchmark_resolver=None,
    ):
        self.data = data_manager
        self.models = model_orchestrator
        self.risk = risk_controller
        self.strategies = strategy_map
        self.db = db_manager
        self.ext = ext_fetcher
        self.market_analyzer = market_analyzer
        self.pipeline = data_pipeline
        self.regime_detector = regime_detector
        self.dry_run = dry_run
        self.symbol_terms = symbol_terms or {}

        self._cycle_decisions: Dict[str, int] = {}
        self._cycle_started_at = time.time()

        if config_manager is None:
            from ibkrpy.shared.config_manager import ConfigManager

            self.config = ConfigManager()
        else:
            self.config = config_manager

        if benchmark_resolver is None:
            from ibkrpy.data.benchmark_resolver import StaticBenchmarkResolver

            benchmark_resolver = StaticBenchmarkResolver(
                self.config.get("general_settings.benchmark_symbol", "SPY")
            )
        self.benchmark_resolver = benchmark_resolver
        self._benchmark_logged = set()

        self.vol_estimator = build_volatility_estimator(self.config)

        self.session_clock = build_session_clock(self.config)

        self.cost_model = build_execution_cost_model(self.config)
        self._term_lifted: set = set()

        self.ledger = build_prediction_ledger(
            getattr(self.db, "db_path", None) or "trading_data.db", self.config
        )
        self._ledger_buffer: list = []
        from ibkrpy.data.data_pipeline import WEIGHTS_DIR

        self._model_generation = weight_fingerprint(WEIGHTS_DIR)
        logger.info(f"🔖 Model generation fingerprint: {self._model_generation}")

        self.global_context = {}
        self.cached_funds = 0.0
        self.cached_net_liq = 0.0
        self.cached_positions = {}
        self._position_avg_cost: Dict[str, float] = {}
        # Notional of entries sent since positions were last fetched.
        self._pending_gross = 0.0

        from ibkrpy.strategy.strategy_components import EquityWatermark
        from ibkrpy.data.data_pipeline import PROJECT_ROOT

        self.equity_watermark = EquityWatermark(
            os.path.join(PROJECT_ROOT, "data", "risk_state.json")
        )
        self.cached_vix_series = None
        self.vix_last_fetch_time = 0.0

        self.cached_benchmarks = {}
        self._last_prices = {}
        self.qualified_contracts = {}

        self._last_tick_at: Dict[str, float] = {}

        from collections import defaultdict, deque

        self._model_pred_hist = defaultdict(
            lambda: defaultdict(lambda: deque(maxlen=120))
        )
        self._collapse_warned = set()

        self.max_prediction_deviation = float(
            self.config.get("strategy_settings.max_prediction_deviation", 0.10)
        )

    def _update_model_liveness(
        self, symbol: str, preds_pct: Dict[str, float]
    ) -> Dict[str, float]:
        """
        Yes/no collapse check per model: 1.0 if alive, 0.0 if collapsed.

        Prediction scales differ by orders of magnitude across terms, so the test is
        relative: a model whose prediction std is below COLLAPSE_RATIO of the most
        dispersed model's std counts as collapsed. Dispersion is not used as a weight,
        since that would favour the noisiest model.
        """
        hist = self._model_pred_hist[symbol]
        for name, pct in preds_pct.items():
            hist[name].append(float(pct))

        stds = {
            name: float(np.std(values))
            for name, values in hist.items()
            if len(values) >= 8
        }
        peak = max(stds.values(), default=0.0)
        if peak <= 1e-12:
            return {name: 1.0 for name in preds_pct}

        return {
            name: (
                0.0 if name in stds and stds[name] / peak < self.COLLAPSE_RATIO else 1.0
            )
            for name in preds_pct
        }

    _TERM_ORDER = ("short_term", "mid_term", "long_term")

    def _effective_term(self, symbol: str) -> str:
        """
        Term actually used after applying the min_term floor.

        Costs are paid once per round trip, while predictive edge grows roughly with the
        square root of holding time. A 5-minute bar moves about 1/8.8 of a daily bar, but the
        spread and commission stay the same. Sizing cannot change that ratio (commission
        amortises, spread does not); only a longer holding period does. So short terms chosen
        by walk-forward are lifted to this floor.
        """
        term = self.symbol_terms.get(symbol, "long_term")
        floor = str(self.config.get("general_settings.min_term", "long_term"))
        try:
            if self._TERM_ORDER.index(term) < self._TERM_ORDER.index(floor):
                if symbol not in self._term_lifted:
                    self._term_lifted.add(symbol)
                    logger.info(
                        f"[{symbol}] ⏫ Term lifted from {term} to {floor} "
                        f"(general_settings.min_term): short-term edge does not "
                        f"cover fixed trading costs."
                    )
                return floor
        except ValueError:
            logger.error(
                f"Unknown term setting {term!r} or min_term {floor!r}; keeping the original."
            )
        return term

    def _cost_floor_pct(self, price: float, net_liquidation: float) -> float:
        """
        Round-trip cost as a fraction of the notional this loop intends to trade.

        Share count is unknown at the threshold stage, so base_position_pct × net liquidation
        stands in for notional. This is a coarse screen; the real check is before ordering
        (see _execute_signal). Feeding it to min_edge_pct still matters: exit geometry uses
        it as the take-profit floor, so a target is never set inside the cost band.
        """
        try:
            base_pct = float(
                self.config.get("strategy_settings.base_position_pct", 0.08)
            )
            notional = max(float(net_liquidation) * base_pct, float(price))
            shares = max(notional / float(price), 1.0)
            cost = self.cost_model.estimate(price, shares)
            safety = float(
                self.config.get("execution_cost_settings.safety_multiple", 1.5)
            )
            return cost.pct * safety
        except Exception as e:
            logger.warning(f"Cost floor estimate failed; using configured value: {e}")
            return float(self.config.get("strategy_settings.min_edge_pct", 0.0010))

    def _record_to_ledger(
        self, symbol, term, context, ensemble_preds, vol, decision, reversal_risk
    ):
        """
        Buffer a full snapshot of this scan. Rows are written in one batch in
        log_cycle_summary() rather than one connection per symbol.

        Never raises: a broken ledger must not stop trading.
        """
        try:
            timeframe, _ = TERM_BARS.get(term, ("1 day", 1))
            price = float(context.get("current_price") or 0.0)
            if price <= 0:
                return

            preds = {
                str(k): float(v)
                for k, v in (ensemble_preds or {}).items()
                if isinstance(v, (int, float)) and math.isfinite(float(v))
            }
            ensemble = decision.get("expected_move")
            if ensemble is None and preds:
                ensemble = sum(preds.values()) / len(preds)

            self._ledger_buffer.append(
                PredictionRecord(
                    symbol=symbol,
                    timestamp=datetime.now(timezone.utc).isoformat(timespec="seconds"),
                    term=term,
                    timeframe=timeframe,
                    horizon_bars=1,
                    price_at_prediction=price,
                    predicted_return=float(ensemble or 0.0),
                    model_predictions=preds,
                    sigma=getattr(vol, "value", None),
                    sigma_source=getattr(vol, "source", None),
                    sigma_realized=getattr(vol, "realized", None),
                    regime=str(context.get("regime")),
                    threshold=decision.get("threshold"),
                    threshold_source=decision.get("threshold_source"),
                    reason_code=decision.get("reason_code"),
                    action=decision.get("action", "HOLD"),
                    benchmark=context.get("benchmark"),
                    reversal_risk=reversal_risk,
                    model_generation=self._model_generation,
                )
            )
        except Exception as e:
            logger.debug(f"[{symbol}] Ledger snapshot failed (trading unaffected): {e}")

    def flush_ledger(self):
        """Write this loop's buffered predictions at once. Called by log_cycle_summary."""
        if not self._ledger_buffer:
            return
        try:
            self.ledger.record_many(self._ledger_buffer)
        finally:
            self._ledger_buffer = []

    @property
    def _extended_poll_multiplier(self) -> float:
        return float(self.config.get("session_settings.extended_poll_multiplier", 3.0))

    def _poll_interval(self, symbol: str) -> int:
        term = self._effective_term(symbol)
        default = _TERM_POLL_SECONDS.get(term, 3600)
        return int(self.config.get(f"general_settings.poll_seconds_{term}", default))

    def should_tick(self, symbol: str, session=None) -> bool:
        """
        Decide from the bar size whether to query IBKR this loop. Symbols with a position are
        exempt: their risk is live and cannot wait for the next bar.

        Outside regular hours the interval is longer: bars are sparse, and polling at the
        regular-hours rate only returns the same unfinished bar and wastes API quota.
        """
        if session is not None and not session.is_tradable:
            return False

        if self.cached_positions.get(symbol):
            return True
        last = self._last_tick_at.get(symbol)
        if last is None:
            return True

        interval = self._poll_interval(symbol)
        if session is not None and session.is_extended:
            interval = int(interval * self._extended_poll_multiplier)
        return (time.time() - last) >= interval

    @staticmethod
    def _normalise_ohlcv(frame: pd.DataFrame) -> pd.DataFrame:
        """Normalise column names, dtypes and timezone in one place."""
        if frame is None or frame.empty:
            return pd.DataFrame()

        frame = frame.rename(
            columns={
                "open": "Open",
                "high": "High",
                "low": "Low",
                "close": "Close",
                "volume": "Volume",
            }
        )
        columns = [
            c for c in ("Open", "High", "Low", "Close", "Volume") if c in frame.columns
        ]
        if not columns:
            return pd.DataFrame()

        frame = frame[columns].apply(pd.to_numeric, errors="coerce")
        frame.index = pd.to_datetime(frame.index, utc=True)
        return frame

    async def _fetch_benchmark_history(
        self, benchmark_symbol: str, bar_size: str, duration: str
    ) -> pd.DataFrame:
        """Fetch benchmark history (database plus live increment) and write it back."""
        contract = await self._get_qualified_contract(benchmark_symbol)
        if contract is None:
            logger.warning(f"Benchmark {benchmark_symbol} contract cannot be resolved.")
            return pd.DataFrame()

        recent = self._normalise_ohlcv(
            await self.data.fetch_historical_data(
                contract=contract,
                duration=duration,
                bar_size=bar_size,
                what_to_show="TRADES",
            )
        )

        stored = pd.DataFrame()
        if self.db:
            stored = self._normalise_ohlcv(
                self.db.get_market_data_sync(benchmark_symbol, timeframe=bar_size)
            )
            if not recent.empty:
                self.db.save_bulk_market_data(
                    benchmark_symbol, recent, timeframe=bar_size
                )

        if stored.empty:
            return recent
        if recent.empty:
            return stored

        merged = pd.concat([stored, recent])
        return merged[~merged.index.duplicated(keep="last")].sort_index()

    async def _load_benchmark_frame(
        self,
        symbol: str,
        benchmark_symbol: str,
        df: pd.DataFrame,
        bar_size: str,
        duration: str,
        cols_to_keep,
    ) -> pd.DataFrame:
        """Benchmark data aligned to df.index, or an empty frame with the reason logged.

        An empty frame means bench_return and bench_correlation are missing; align_to_manifest
        fills them with 0 and warns.
        """
        if benchmark_symbol == symbol:
            return df.copy()

        cache_key = f"{benchmark_symbol}_{bar_size}"
        try:
            if cache_key not in self.cached_benchmarks:
                self.cached_benchmarks[cache_key] = await self._fetch_benchmark_history(
                    benchmark_symbol, bar_size, duration
                )

            raw = self.cached_benchmarks[cache_key]
            if raw is None or raw.empty:
                logger.warning(
                    f"⚠️ [{symbol}] Benchmark {benchmark_symbol} has no data; "
                    f"bench_return / bench_correlation are missing this round."
                )
                return pd.DataFrame()

            return raw.reindex(df.index, method="ffill").bfill()

        except Exception as exc:
            logger.warning(
                f"⚠️ [{symbol}] Failed to get benchmark ({benchmark_symbol}): {exc}"
            )
            return pd.DataFrame()

    def _predict_one(
        self, symbol: str, model_type: str, df_scaled, df_adv, current_price: float
    ):
        """One model's prediction. Returns (predicted price, rejection reason); exactly one is None.

        Every return carries an explicit reason, so a silent model and a rejected prediction
        can be told apart in the logs.
        Errors should never pass silently.
        """
        is_ready = getattr(self.models, "is_ready", None)
        if callable(is_ready) and not is_ready(symbol, model_type):
            return None, "untrained"

        frame = df_adv if model_type == "ARIMA" else df_scaled
        pred_raw, _ = self.models.predict(symbol, frame, model_type=model_type)

        if isinstance(pred_raw, (list, np.ndarray)):
            if len(pred_raw) == 0:
                return None, "empty output"
            pred_raw = float(pred_raw[0])

        if pred_raw is None or pd.isna(pred_raw):
            return None, "NaN"

        if float(pred_raw) == 0.0 and model_type in ("LSTM", "Transformer"):
            return None, "output is 0 (weights missing)"

        if self.pipeline and model_type in ("LSTM", "Transformer"):
            pred_real = self.pipeline.decode_prediction(pred_raw, current_price, symbol)
            if pred_real is None:
                return None, "decode failed"
        else:
            pred_real = float(pred_raw)

        if pred_real <= 0:
            return None, "not positive"

        deviation = abs(pred_real - current_price) / current_price
        if deviation > self.max_prediction_deviation:
            return None, f"deviation {deviation * 100:.1f}%"

        return pred_real, None

    def _collect_predictions(
        self, symbol: str, df_scaled, df_adv, current_price: float
    ):
        """Collect predictions from every model. Rejection logic lives in _predict_one."""
        accepted, rejected = {}, {}
        for model_type in ("LSTM", "Transformer", "ARIMA"):
            prediction, reason = self._predict_one(
                symbol, model_type, df_scaled, df_adv, current_price
            )
            if prediction is None:
                rejected[model_type] = reason
            else:
                accepted[model_type] = prediction
        return accepted, rejected

    def _log_predictions(
        self,
        symbol: str,
        term: str,
        bar_size: str,
        current_price: float,
        accepted: Dict[str, float],
        rejected: Dict[str, str],
    ):
        """Summarise the prediction stage in one log line."""
        if accepted:
            detail = " · ".join(
                f"{name} {(price / current_price - 1) * 100:+.2f}%"
                for name, price in accepted.items()
            )
        else:
            detail = "none"

        if rejected:
            detail += " || rejected: " + " · ".join(
                f"{k}({v})" for k, v in rejected.items()
            )

        logger.info(
            f"[{symbol}] 🧠 Model predictions ({term}/{bar_size}) "
            f"price {current_price:.2f} -> {detail}"
        )

    def _regime_size_scale(
        self, regime, reversal_risk: float, is_reducing: bool
    ) -> float:
        """Scale position size by regime and reversal risk.

        Reducing and closing always return 1.0: shrinking an exit leaves a tail that cannot
        be handled.

        A confirmed bull regime scales up, so the system is not too conservative in bull
        markets; rising reversal risk scales down linearly, so positions exit while a top
        forms instead of waiting for the regime to turn bearish.
        """
        if is_reducing:
            return 1.0

        from ibkrpy.strategy.strategy_components import MarketRegime

        scales = {
            MarketRegime.BULL_TREND: self.config.get(
                "strategy_settings.size_scale_bull", 1.25
            ),
            MarketRegime.BEAR_TREND: self.config.get(
                "strategy_settings.size_scale_bear", 0.60
            ),
            MarketRegime.SIDEWAYS_VOLATILE: self.config.get(
                "strategy_settings.size_scale_volatile", 0.70
            ),
            MarketRegime.SIDEWAYS_QUIET: self.config.get(
                "strategy_settings.size_scale_quiet", 0.90
            ),
        }
        scale = float(scales.get(regime, 1.0))
        scale *= max(1.0 - reversal_risk * 0.75, 0.25)
        return scale

    def _record_decision(self, code: str):
        self._cycle_decisions[code] = self._cycle_decisions.get(code, 0) + 1

    def log_cycle_summary(self):
        """
        Log a one-line summary at the end of each scan, showing how many symbols stopped at
        each gate even on days with no orders.

        Also writes this loop's buffered prediction snapshots to the ledger.
        """
        self.flush_ledger()
        if not self._cycle_decisions:
            logger.info(
                "📊 [Loop summary] No symbol passed polling throttle (all cooling down)."
            )
            return

        total = sum(self._cycle_decisions.values())
        breakdown = " | ".join(
            f"{k}×{v}"
            for k, v in sorted(self._cycle_decisions.items(), key=lambda kv: -kv[1])
        )
        elapsed = time.time() - self._cycle_started_at
        line = f"📊 [Loop summary] Scanned {total} / {elapsed:.0f}s -> {breakdown}"

        stats = getattr(self.data, "pacing_stats", None)
        if callable(stats):
            s = stats()
            line += (
                f" || API: 10-min window {s['requests_in_window']}/{s['window_limit']}, "
                f"cache hit rate {s['cache_hit_rate']:.0%}"
            )
        logger.info(line)

        self._cycle_decisions = {}
        self._cycle_started_at = time.time()

    async def _get_qualified_contract(self, symbol: str):
        """
        Get and cache the fully qualified contract via IBKRDataManager.qualify().
        Returns None if the symbol cannot be resolved, so an empty conId=0 contract is never
        cached.
        """
        if symbol in self.qualified_contracts:
            return self.qualified_contracts[symbol]

        contract = None
        if hasattr(self.data, "qualify"):
            contract = await self.data.qualify(symbol)
        else:
            contract = Stock(symbol, "SMART", "USD")
            resolved = await self.data.ib.qualifyContractsAsync(contract)
            if not resolved or not contract.conId:
                logger.error(
                    f"[{symbol}] ❌ Contract resolution failed; skipping this symbol this loop."
                )
                contract = None

        if contract is not None:
            self.qualified_contracts[symbol] = contract
        return contract

    def _get_dynamic_benchmark(self, symbol: str) -> str:
        """
        The symbol's benchmark, chosen by the injected BenchmarkResolver
        (rules live in benchmark_resolver.py).
        """
        chosen = self.benchmark_resolver.resolve(symbol)
        if not chosen:
            chosen = self.config.get("general_settings.benchmark_symbol", "SPY")

        if symbol not in self._benchmark_logged:
            self._benchmark_logged.add(symbol)
            logger.info(f"[{symbol}] 📐 Benchmark = {chosen}")
        return chosen

    async def update_system_state(self):
        """Called before each main loop: refresh account cache and global market context."""
        self.cached_benchmarks.clear()
        try:
            positions = await self.data.ib.reqPositionsAsync()
            account_summary = await self.data.ib.accountSummaryAsync()

            available_funds = 0.0
            net_liquidation = 0.0

            for item in account_summary:
                if item.tag == "AvailableFunds" and item.currency in ("BASE", "USD"):
                    available_funds = float(item.value)
                elif item.tag == "NetLiquidation" and item.currency in ("BASE", "USD"):
                    net_liquidation = float(item.value)

            pos_dict = {
                p.contract.symbol: p.position for p in positions if p.position != 0
            }

            self.cached_funds = available_funds
            self.cached_net_liq = net_liquidation
            self.cached_positions = pos_dict
            self._position_avg_cost = {
                p.contract.symbol: float(p.avgCost or 0.0)
                for p in positions
                if p.position != 0
            }
            self._pending_gross = 0.0

            try:
                ny_day = self.session_clock.state().now.strftime("%Y-%m-%d")
            except Exception:
                ny_day = datetime.now(timezone.utc).strftime("%Y-%m-%d")
            self.equity_watermark.update(net_liquidation, ny_day)

            if self.db:
                await self.db.update_account_info(
                    net_liquidation, available_funds, pos_dict
                )

            await self._protect_unhedged_positions()

        except Exception as e:
            logger.warning(
                f"⚠️ Failed to get account state: {e}; using last cached data."
            )

        if self.market_analyzer:
            self.global_context = await asyncio.to_thread(
                self.market_analyzer.get_global_context
            )

    def _current_gross_exposure(self) -> float:
        """
        Current gross exposure (sum of absolute notional, long and short).
        Prices at the live price or average cost, plus orders placed earlier this loop.
        """
        total = 0.0
        for sym, qty in (self.cached_positions or {}).items():
            if not qty:
                continue
            price = self._live_market_price(sym) or self._position_avg_cost.get(sym)
            if price:
                total += abs(qty) * price
        return total + self._pending_gross

    async def _protect_unhedged_positions(self):
        """Find held positions without stop/take-profit orders (e.g. manual buys) and attach OCA protection."""
        if self.dry_run:
            return

        positions = self.cached_positions
        if not positions:
            return

        open_trades = self.data.ib.openTrades()

        for symbol, pos_qty in positions.items():
            if pos_qty == 0:
                continue

            protect_action = "SELL" if pos_qty > 0 else "BUY"
            protected_qty = 0.0
            for t in open_trades:
                if (
                    t.contract.symbol == symbol
                    and t.order.action == protect_action
                    and t.order.orderType in ("STP", "STP LMT")
                ):
                    protected_qty += float(t.order.totalQuantity or 0)
            has_protection = protected_qty >= abs(pos_qty)

            if not has_protection:
                logger.warning(
                    f"[Guard] 🛡️ {symbol} has an unprotected position ({pos_qty} shares); attaching OCA stop/take-profit..."
                )
                # Clear leftover exit orders first so they can't fill alongside the new OCA.
                if not await self._cancel_open_orders(symbol):
                    logger.critical(
                        f"❌ [{symbol}] Could not confirm cancellation of existing orders; "
                        "not attaching protection to avoid duplicate exits. Check manually."
                    )
                    continue
                await self._attach_oca_protection(symbol, pos_qty)

    def _live_market_price(self, symbol: str):
        """
        Live market price. Prefers the portfolio marketPrice streamed by IBKR through
        updatePortfolio (no historical request quota), then the price seen this run_tick.
        Returns None if neither exists.
        """
        try:
            for item in self.data.ib.portfolio():
                if (
                    item.contract.symbol == symbol
                    and item.marketPrice
                    and item.marketPrice > 0
                ):
                    return float(item.marketPrice)
        except Exception as e:
            logger.warning(f"[{symbol}] Failed to read portfolio market price: {e}")

        p = self._last_prices.get(symbol)
        return float(p) if p else None

    async def _attach_oca_protection(self, symbol: str, pos_qty: float):
        contract = await self._get_qualified_contract(symbol)
        try:
            df = pd.DataFrame()
            if self.db:
                df = self.db.get_market_data_sync(symbol, timeframe="1 day")

            if df.empty or len(df) < 10:
                df = await self.data.fetch_historical_data(
                    contract, duration="60 D", bar_size="1 day"
                )

            if df.empty:
                return

            if "close" in df.columns:
                df = df.rename(
                    columns={
                        "open": "Open",
                        "high": "High",
                        "low": "Low",
                        "close": "Close",
                        "volume": "Volume",
                    }
                )

            live_price = self._live_market_price(symbol)
            db_price = float(df["Close"].iloc[-1])

            if live_price is None:
                logger.critical(
                    f"❌ [{symbol}] No live price; refusing to attach protection (position still unprotected, needs manual action). "
                    f"\tDatabase reference price is {db_price:.2f}."
                )
                return

            drift = abs(db_price / live_price - 1) if live_price > 0 else 1.0
            if drift > 0.03:
                logger.warning(
                    f"❌ [{symbol}] Local data is stale by {drift*100:.1f}% "
                    f"(DB {db_price:.2f} vs market {live_price:.2f}); anchoring on market price."
                )

            current_price = live_price
            returns = np.log(df["Close"] / df["Close"].shift(1)).dropna()
            annual_vol = returns.std() * np.sqrt(252) if len(returns) > 10 else 0.20

            term = self._effective_term(symbol)
            if term == "short_term":
                periods_per_year = 252 * 78
            elif term == "mid_term":
                periods_per_year = 252 * 6.5
            else:
                periods_per_year = 252
            daily_vol = annual_vol / math.sqrt(periods_per_year)

            strategy = self.strategies.get(symbol)
            sl_mult = strategy.sl_multiplier if strategy else 2.0
            tp_mult = strategy.tp_multiplier if strategy else 3.0

            if pos_qty > 0:
                action = "SELL"
                sl_price = current_price * (1 - daily_vol * sl_mult)
                tp_price = current_price * (1 + daily_vol * tp_mult)
            else:
                action = "BUY"
                sl_price = current_price * (1 + daily_vol * sl_mult)
                tp_price = current_price * (1 - daily_vol * tp_mult)

            sl_price, tp_price = round(sl_price, 2), round(tp_price, 2)
            if pos_qty > 0:
                bad = sl_price >= current_price or tp_price <= current_price
            else:
                bad = sl_price <= current_price or tp_price >= current_price

            if bad:
                logger.critical(
                    f"❌ [{symbol}] Protection prices on the wrong side; refusing to send: "
                    f"market {current_price:.2f} / stop {sl_price:.2f} / take-profit {tp_price:.2f} "
                    f"(position {pos_qty:+.0f} shares). They would fill at once and close the position by mistake."
                )
                return

            oca_group = f"OCA_PROTECT_{symbol}_{int(time.time())}"

            sl_order = Order(
                action=action,
                totalQuantity=abs(pos_qty),
                orderType="STP",
                auxPrice=sl_price,
                tif="GTC",
                ocaGroup=oca_group,
                ocaType=1,
            )
            tp_order = LimitOrder(
                action=action,
                totalQuantity=abs(pos_qty),
                lmtPrice=tp_price,
                tif="GTC",
                ocaGroup=oca_group,
                ocaType=1,
            )

            self.data.ib.placeOrder(contract, sl_order)
            self.data.ib.placeOrder(contract, tp_order)

            logger.info(
                f"✅ [{symbol}] OCA protection attached -> stop (STP): {sl_price:.2f}, take-profit (LMT): {tp_price:.2f}"
            )
        except Exception as e:
            logger.critical(f"❌ [{symbol}] Failed to attach protection: {e}")

    CANCEL_CONFIRM_SECONDS = 5.0

    async def _wait_done(self, trades, timeout: float) -> bool:
        """Wait until every trade is Filled/Cancelled/Inactive; False on timeout."""
        loop = asyncio.get_running_loop()
        deadline = loop.time() + timeout
        while True:
            if all(t.isDone() for t in trades):
                return True
            if loop.time() >= deadline:
                return False
            await asyncio.sleep(0.25)

    async def _cancel_open_orders(self, symbol: str) -> bool:
        """Cancel all open orders for the symbol and wait for confirmation.

        Returns True only when nothing is left working for that symbol.
        """
        if self.dry_run:
            return True

        targets = [
            t
            for t in self.data.ib.openTrades()
            if t.contract.symbol == symbol and not t.isDone()
        ]
        if not targets:
            return True

        for trade in targets:
            self.data.ib.cancelOrder(trade.order)

        confirmed = await self._wait_done(targets, self.CANCEL_CONFIRM_SECONDS)
        if confirmed:
            logger.info(f"[{symbol}] 🧹 Cancelled {len(targets)} open order(s).")
        else:
            pending = [t.order.orderId for t in targets if not t.isDone()]
            logger.critical(
                f"❌ [{symbol}] {len(pending)} order(s) not confirmed cancelled within "
                f"{self.CANCEL_CONFIRM_SECONDS:g}s: {pending}"
            )
        return confirmed

    async def _close_position(
        self, contract, symbol: str, action: str, quantity: int, current_price: float
    ) -> float:
        """Close a position while keeping at most one exit order set working.

        1. Cancel the existing stop/target and wait for confirmation; abort if it fails.
        2. Adaptive market order in RTH, outsideRth limit in extended hours.
        3. Wait up to close_fill_timeout_sec.
        4. If not fully filled, cancel the rest and re-protect the remaining shares.

        Returns the number of shares closed.
        """
        exits_before = [
            t
            for t in self.data.ib.openTrades()
            if t.contract.symbol == symbol
            and t.order.action == action
            and not t.isDone()
        ]
        if not await self._cancel_open_orders(symbol):
            logger.critical(
                f"❌ [{symbol}] Existing stop/target not confirmed cancelled; "
                "skipping the close so it can't fill alongside the stop."
            )
            return 0.0

        # A stop can fill while we cancel it; don't close those shares twice.
        already_out = sum(float(t.orderStatus.filled or 0.0) for t in exits_before)
        if already_out > 0:
            prev = float(self.cached_positions.get(symbol, 0.0) or 0.0)
            self.cached_positions[symbol] = prev + (
                already_out if action == "BUY" else -already_out
            )
            logger.warning(
                f"[{symbol}] {already_out:g} shares exited while cancelling; reducing the close."
            )
            quantity = int(max(quantity - already_out, 0))
            if quantity <= 0:
                return already_out

        try:
            is_ext = bool(self.session_clock.classify().is_extended)
        except Exception:
            is_ext = False

        if is_ext:
            buffer = 0.002
            limit_price = (
                current_price * (1 + buffer)
                if action == "BUY"
                else current_price * (1 - buffer)
            )
            order = LimitOrder(action, quantity, round(limit_price, 2))
            order.outsideRth = True
        else:
            order = MarketOrder(action, quantity)
            order.algoStrategy = "Adaptive"
            order.algoParams = [TagValue("adaptivePriority", "Normal")]
        order.tif = "DAY"

        trade = self.data.ib.placeOrder(contract, order)

        timeout = float(
            self.config.get("strategy_settings.close_fill_timeout_sec", 60) or 60
        )
        await self._wait_done([trade], timeout)

        filled = float(trade.orderStatus.filled or 0.0)
        remaining = quantity - filled

        if remaining > 0:
            if not trade.isDone():
                self.data.ib.cancelOrder(trade.order)
                cancelled = await self._wait_done([trade], self.CANCEL_CONFIRM_SECONDS)
                filled = float(trade.orderStatus.filled or 0.0)
                remaining = quantity - filled
                if not cancelled:
                    logger.critical(
                        f"❌ [{symbol}] Close order {trade.order.orderId} neither filled nor cancelled; "
                        f"{remaining:g} shares are unprotected. Check manually."
                    )
                    return filled

            if remaining > 0:
                still_held = remaining if action == "SELL" else -remaining
                logger.critical(
                    f"⚠️ [{symbol}] Close filled {filled:g}/{quantity}; "
                    f"cancelled the rest and re-protecting {still_held:+g} shares."
                )
                await self._attach_oca_protection(symbol, still_held)

        prev = float(self.cached_positions.get(symbol, 0.0) or 0.0)
        signed_fill = filled if action == "BUY" else -filled
        self.cached_positions[symbol] = prev + signed_fill
        if filled > 0:
            logger.info(f"[{symbol}] ✅ Closed {filled:g} shares.")
        return filled

    async def run_tick(self, symbol: str, session=None):
        self._last_tick_at[symbol] = time.time()

        available_funds = self.cached_funds
        net_liquidation = self.cached_net_liq
        current_pos = self.cached_positions.get(symbol, 0.0)

        contract = await self._get_qualified_contract(symbol)
        if contract is None:
            self._record_decision("CONTRACT_UNRESOLVED")
            return

        term = self._effective_term(symbol)

        if term == "short_term":
            bar_size_str = self.config.get(
                "general_settings.short_term_bar_size", "5 mins"
            )
        elif term == "mid_term":
            bar_size_str = self.config.get(
                "general_settings.mid_term_bar_size", "1 hour"
            )
        else:
            bar_size_str = self.config.get(
                "general_settings.long_term_bar_size", "1 day"
            )

        live_duration = "3 D"
        df_recent = await self.data.fetch_historical_data(
            contract=contract,
            duration=live_duration,
            bar_size=bar_size_str,
            what_to_show="TRADES",
        )

        df_db = pd.DataFrame()
        if self.db:
            df_db = self.db.get_market_data_sync(symbol, timeframe=bar_size_str)
            if not df_db.empty:
                df_db = df_db[["Open", "High", "Low", "Close", "Volume"]].astype(float)
                df_db.index = pd.to_datetime(df_db.index, utc=True)

        if not df_recent.empty:
            df_recent.index = pd.to_datetime(df_recent.index, utc=True)
            df_recent = df_recent.rename(
                columns={
                    "open": "Open",
                    "high": "High",
                    "low": "Low",
                    "close": "Close",
                    "volume": "Volume",
                }
            )
            df_recent = df_recent[["Open", "High", "Low", "Close", "Volume"]].astype(
                float
            )

            if self.db:
                self.db.save_bulk_market_data(symbol, df_recent, timeframe=bar_size_str)

        if not df_db.empty and not df_recent.empty:
            df = pd.concat([df_db, df_recent])

            df = df[~df.index.duplicated(keep="last")].sort_index()
        elif not df_recent.empty:
            df = df_recent
        else:
            df = df_db

        cols_to_keep = ["Open", "High", "Low", "Close", "Volume"]
        for col in cols_to_keep:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors="coerce")

        if df.empty or len(df) < 60:
            self._record_decision("NO_DATA")
            logger.warning(
                f"⚠️ [{symbol}] {term} ({bar_size_str}) not enough data: "
                f"only {len(df)} bars (need ≥60). Run --mode download to fill the database."
            )
            return

        current_price = float(df["Close"].iloc[-1])
        self._last_prices[symbol] = current_price

        benchmark_symbol = self._get_dynamic_benchmark(symbol)
        bench_df = await self._load_benchmark_frame(
            symbol, benchmark_symbol, df, bar_size_str, live_duration, cols_to_keep
        )

        macro_dict = {}
        if self.ext:
            current_time = time.time()

            if (
                self.cached_vix_series is None
                or (current_time - self.vix_last_fetch_time) > 28800
            ):
                try:
                    vix_series = await self.ext.fetch_fred_series("VIXCLS")
                    if vix_series is not None and not vix_series.empty:
                        self.cached_vix_series = vix_series
                        self.vix_last_fetch_time = current_time
                        logger.info(
                            "🌍 [System] Macro data (VIXCLS) updated from FRED and cached locally (valid 8 hours)."
                        )
                except Exception as e:
                    logger.warning(f"⚠️ FRED fetch failed; keeping local cache: {e}")

            if self.cached_vix_series is not None and not self.cached_vix_series.empty:
                macro_dict["VIX"] = align_daily_series(self.cached_vix_series, df.index)

        if self.pipeline:
            bench_data = bench_df if not bench_df.empty else None
            df_adv = self.pipeline.engineer_advanced_features(
                df, bench_data, macro_dict
            )
            df_adv = df_adv.ffill().bfill().fillna(0)

            df_adv, scale_cols = self.pipeline.align_to_manifest(df_adv, symbol)
            df_scaled, scale_cols = self.pipeline.transform_for_inference(
                df_adv, symbol
            )
        else:
            df_adv = df.ffill().bfill().fillna(0)
            df_scaled = df_adv

        from ibkrpy.strategy.strategy_components import MarketRegime

        regime = MarketRegime.SIDEWAYS_QUIET
        reversal_risk = 0.0

        if self.regime_detector:
            assess = getattr(self.regime_detector, "assess", None)
            if callable(assess):
                assessment = assess(df_adv, key=symbol)
                regime = assessment.regime
                reversal_risk = assessment.reversal_risk
                if assessment.is_turning:
                    logger.warning(
                        f"[{symbol}] ⚠️ Trend weakening: {assessment.describe()}"
                    )
            else:
                regime = self.regime_detector.detect(df_adv)

        context = {
            "vix_series": macro_dict.get("VIX"),
            "current_price": current_price,
            "regime": regime,
            "reversal_risk": reversal_risk,
            "session": (
                session if session is not None else self.session_clock.classify()
            ),
            **self.equity_watermark.snapshot(self.cached_net_liq),
        }
        check = getattr(
            self.risk, "check_entries_allowed", self.risk.check_trade_allowed
        )
        entries_allowed, risk_reason = check(context)
        if not entries_allowed:
            self._record_decision("RISK_ENTRIES_BLOCKED")
            if abs(current_pos) > 0:
                logger.warning(
                    f"[{symbol}] 🛡️ {risk_reason} blocks new exposure; "
                    f"existing {current_pos:+.0f} shares are still evaluated for exit."
                )
            else:
                logger.info(f"[{symbol}] 🛡️ {risk_reason} blocks new exposure.")

        ensemble_preds, rejected_models = self._collect_predictions(
            symbol, df_scaled, df_adv, current_price
        )
        self._log_predictions(
            symbol, term, bar_size_str, current_price, ensemble_preds, rejected_models
        )

        if not ensemble_preds:
            self._record_decision("NO_USABLE_MODEL")
            logger.warning(
                f"⚠️ [{symbol}] No usable model; holding. "
                f"If the reason is 'untrained', run python ibkrpy/core/main.py --mode train {symbol}."
            )
            return

        preds_pct = {k: (v / current_price - 1) for k, v in ensemble_preds.items()}
        liveness = self._update_model_liveness(symbol, preds_pct)

        dead = [k for k, v in liveness.items() if v <= 0]
        if dead and symbol not in self._collapse_warned:
            self._collapse_warned.add(symbol)
            logger.error(
                f"🧟 [{symbol}] Model output collapsed to a constant: {', '.join(dead)}. "
                f"They are left out of the ensemble. "
                f"Retrain to fix this."
            )

        _, model_volatility = self.models.predict(symbol, df_adv, model_type="GARCH")
        if isinstance(model_volatility, (list, np.ndarray)):
            model_volatility = (
                float(model_volatility[0])
                if len(model_volatility) > 0
                else float("nan")
            )

        # GARCH already returns σ per bar of this term.
        try:
            model_vol_per_bar = float(model_volatility)
        except (TypeError, ValueError):
            model_vol_per_bar = None

        vol = self.vol_estimator.estimate(symbol, df_adv, model_vol_per_bar)
        adjusted_volatility = vol.value

        logger.debug(
            f"[{symbol}] σ={adjusted_volatility * 100:.3f}% "
            f"(source {vol.source}; realized {vol.realized * 100:.3f}%"
            + (f"; model/realized ×{vol.ratio:.2f}" if vol.ratio is not None else "")
            + ")"
        )

        strategy = self.strategies.get(symbol)
        if not strategy:
            self._record_decision("NO_STRATEGY")
            logger.error(f"[{symbol}] ❌ No strategy object found; skipping this loop.")
            return

        if hasattr(strategy, "set_model_liveness"):
            strategy.set_model_liveness(liveness)

        dynamic_floor = self._cost_floor_pct(
            context["current_price"], getattr(self, "cached_net_liq", 0.0) or 0.0
        )
        configured = float(self.config.get("strategy_settings.min_edge_pct", 0.0010))
        strategy.min_edge_pct = max(dynamic_floor, configured)

        signal = strategy.generate_signal(
            current_price=context["current_price"],
            volatility=adjusted_volatility,
            regime=context["regime"],
            ensemble_predictions=ensemble_preds,
            current_position=current_pos,
            reversal_risk=reversal_risk,
        )

        decision = getattr(strategy, "last_decision", {}) or {}
        self._record_decision(decision.get("reason_code", "UNKNOWN"))
        self._record_to_ledger(
            symbol, term, context, ensemble_preds, vol, decision, reversal_risk
        )
        describe = getattr(strategy, "describe_last_decision", None)
        trace = describe() if callable(describe) else str(decision)

        if signal:
            logger.info(f"[{symbol}] ✅ Strategy decision: {trace}")
        else:
            logger.info(f"[{symbol}] ⏸️ Strategy decision: {trace}")
            return

        is_reducing = (current_pos > 0 and signal["action"] == "SELL") or (
            current_pos < 0 and signal["action"] == "BUY"
        )
        if not entries_allowed and not is_reducing:
            self._record_decision("RISK_ENTRIES_BLOCKED")
            logger.info(
                f"[{symbol}] 🛡️ Signal valid but risk gate closed; skipping new exposure."
            )
            return

        if signal:
            conviction = 1.0
            target_weight = self.config.get("strategy_settings.base_position_pct", 0.08)
            if self.market_analyzer:
                analysis = self.market_analyzer.analyze_stock_risk(
                    symbol=symbol,
                    context=self.global_context,
                    action=signal["action"],
                    current_positions=self.cached_positions,
                )
                conviction = analysis["conviction_multiplier"]
                target_weight = analysis.get("target_weight", 0.10)

                if analysis["warnings"]:
                    logger.warning(
                        f"[{symbol}] 🌍 Macro warning: {' | '.join(analysis['warnings'])}"
                    )

            regime_scale = self._regime_size_scale(regime, reversal_risk, is_reducing)
            target_weight *= regime_scale
            logger.info(
                f"[{symbol}] 📏 Sizing: base×tilt {target_weight / max(regime_scale, 1e-9) * 100:.2f}% "
                f"× regime {regime_scale:.2f} × conviction {conviction:.2f} "
                f"= {target_weight * conviction * 100:.2f}% of net liquidation"
            )

            await self._execute_signal(
                symbol,
                signal,
                current_price,
                available_funds,
                net_liquidation,
                current_pos,
                conviction,
                target_weight,
            )

    async def _execute_signal(
        self,
        symbol: str,
        signal: Dict[str, Any],
        current_price: float,
        available_funds: float,
        net_liquidation: float,
        current_pos: float,
        conviction: float = 1.0,
        target_weight: float = 0.10,
    ):
        action = signal["action"]
        sl_price = signal["stop_loss_price"]
        tp_price = signal["take_profit_price"]
        regime_name = signal["regime"]
        term_name = signal.get("term", "unknown")

        is_closing_only = False
        trade_quantity = 0

        allow_shorting = self.config.get("strategy_settings.allow_shorting", False)
        final_weight = min(target_weight * conviction, 0.35)
        max_gross = self.config.get("strategy_settings.max_gross_exposure", 1.0)

        def _affordable_qty() -> int:
            target_cash = net_liquidation * final_weight
            budget = min(target_cash, available_funds * 0.95)

            if net_liquidation > 0:
                room = (max_gross * net_liquidation) - self._current_gross_exposure()
                if room <= 0:
                    logger.warning(
                        f"⚠️ [{symbol}] Portfolio gross exposure cap reached ({max_gross*100:.0f}%); no new position."
                    )
                    return 0
                budget = min(budget, room)

            return int(budget / current_price) if current_price > 0 else 0

        if action == "BUY":
            if current_pos > 0:
                self._record_decision("ALREADY_LONG")
                logger.info(
                    f"[{symbol}] ⏸️ Already long {current_pos:+.0f} shares; BUY signal does not add."
                )
                return
            elif current_pos < 0:
                trade_quantity = int(abs(current_pos))
                is_closing_only = True
            else:
                trade_quantity = _affordable_qty()

        elif action == "SELL":
            if current_pos < 0:
                self._record_decision("ALREADY_SHORT")
                logger.info(
                    f"[{symbol}] ⏸️ Already short {current_pos:+.0f} shares; SELL signal does not add."
                )
                return
            elif current_pos > 0:
                trade_quantity = int(current_pos)
                is_closing_only = True
            else:
                if not allow_shorting:
                    self._record_decision("SHORTING_DISABLED")
                    logger.info(
                        f"[{symbol}] ⏸️ SELL signal with no position and allow_shorting=False; "
                        f"short blocked."
                    )
                    return
                trade_quantity = _affordable_qty()

        if trade_quantity <= 0:
            self._record_decision("NO_BUYING_POWER")
            budget = min(net_liquidation * final_weight, available_funds * 0.95)
            logger.warning(
                f"[{symbol}] ⚠️ {action} signal but 0 shares affordable. "
                f"Target weight {final_weight * 100:.1f}% / net liq ${net_liquidation:,.0f} / "
                f"available ${available_funds:,.0f} / budget ${budget:,.0f} / price {current_price:.2f}."
            )
            return

        min_trade_usd = self.config.get("strategy_settings.min_trade_usd", 500.0)
        trade_value = trade_quantity * current_price

        if not is_closing_only:
            try:
                gross_edge = abs(float(tp_price) - current_price) / current_price
            except (TypeError, ValueError, ZeroDivisionError):
                gross_edge = 0.0

            try:
                session_now = self.session_clock.classify()
                is_ext = bool(session_now.is_extended)
            except Exception:
                is_ext = False

            viable, cost, why = self.cost_model.is_viable(
                edge_pct=gross_edge,
                price=current_price,
                shares=trade_quantity,
                extended_session=is_ext,
                safety_multiple=float(
                    self.config.get("execution_cost_settings.safety_multiple", 1.5)
                ),
            )
            if not viable:
                self._record_decision("COST_NOT_VIABLE")
                logger.info(
                    f"[{symbol}] ⏸️ {action} signal cancelled: costs exceed the edge. {why}"
                )
                return

        if not is_closing_only and trade_value < min_trade_usd:
            self._record_decision("BELOW_MIN_TRADE")
            logger.info(
                f"[{symbol}] ⏸️ {action} signal cancelled: trade value ${trade_value:.2f} "
                f"is below the minimum ${min_trade_usd:.2f}, to avoid fee drag."
            )
            return

        logger.info(
            f"[{symbol}] 🎯 Executing ({term_name}): {action} {trade_quantity} shares @ ~{current_price:.2f} (allocated weight: {final_weight*100:.1f}%)"
        )
        logger.info(
            f"\t=> [Protection] stop (STP) at {sl_price:.2f} | take-profit (LMT) at {tp_price:.2f}"
        )

        contract = await self._get_qualified_contract(symbol)
        if contract is None:
            self._record_decision("CONTRACT_UNRESOLVED")
            logger.error(
                f"[{symbol}] ❌ Contract resolution failed before ordering; trade aborted."
            )
            return

        self._record_decision(f"ORDER_{action}")
        try:
            if self.dry_run:
                if not is_closing_only:
                    self.cached_funds = max(0.0, self.cached_funds - trade_value)
                logger.info(f"[{symbol}] 🛡️ [Dry-run] Simulated order placed.")
            elif is_closing_only:
                filled = await self._close_position(
                    contract, symbol, action, trade_quantity, current_price
                )
                if filled <= 0:
                    self._record_decision("CLOSE_NOT_FILLED")
                    return
                trade_quantity = int(filled)
            else:
                if not await self._cancel_open_orders(symbol):
                    self._record_decision("CANCEL_UNCONFIRMED")
                    logger.critical(
                        f"❌ [{symbol}] Old orders not confirmed cancelled; skipping entry."
                    )
                    return

                self.cached_funds = max(0.0, self.cached_funds - trade_value)
                self._pending_gross += trade_value

                algo_params = [TagValue("adaptivePriority", "Normal")]

                slippage_buffer = 0.002
                limit_entry_price = (
                    current_price * (1 + slippage_buffer)
                    if action == "BUY"
                    else current_price * (1 - slippage_buffer)
                )

                parent_id = self.data.ib.client.getReqId()
                parent = LimitOrder(action, trade_quantity, round(limit_entry_price, 2))
                parent.algoStrategy = "Adaptive"
                parent.algoParams = algo_params
                parent.orderId = parent_id
                parent.tif = "DAY"
                parent.transmit = False

                rev_action = "SELL" if action == "BUY" else "BUY"
                sl_order = Order(
                    action=rev_action,
                    totalQuantity=trade_quantity,
                    orderType="STP",
                    auxPrice=round(sl_price, 2),
                    parentId=parent_id,
                    tif="GTC",
                    transmit=False,
                )
                tp_order = LimitOrder(
                    action=rev_action,
                    totalQuantity=trade_quantity,
                    lmtPrice=round(tp_price, 2),
                    parentId=parent_id,
                    tif="GTC",
                    transmit=True,
                )

                self.data.ib.placeOrder(contract, parent)
                self.data.ib.placeOrder(contract, sl_order)
                self.data.ib.placeOrder(contract, tp_order)

            if self.db:
                reason = (
                    "AI close (reverse signal)"
                    if is_closing_only
                    else f"AI entry ({term_name} | Alloc:{final_weight*100:.1f}%)"
                )
                await self.db.log_trade(
                    {
                        "symbol": symbol,
                        "action": action,
                        "quantity": trade_quantity,
                        "price": current_price,
                        "regime": regime_name,
                        "reason": ("[Simulated] " if self.dry_run else "") + reason,
                    }
                )
        except Exception as e:
            logger.critical(f"❌ [{symbol}] Order placement error: {e}")
