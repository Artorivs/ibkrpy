# ibkrpy/core/main.py
# Command-line entry point

import argparse
import sys
import os
import asyncio
import subprocess
import json
import logging
import warnings
import caffeine

os.environ["OMP_NUM_THREADS"] = "1"
os.environ["KMP_DUPLICATE_LIB_OK"] = "True"
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"
warnings.filterwarnings("ignore")

project_root = os.path.dirname(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
)
sys.path.append(project_root)
caffeine.on(display=False)

logger = logging.getLogger("ibkrpy.main")

MAX_DEEP_SLEEP_SECONDS = 6 * 3600

core_dir_name = os.path.basename(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
)

from ibkrpy.shared.config_manager import ConfigManager
from ibkrpy.shared.db_manager import DatabaseManager
from ibkrpy.shared.system_log import setup_logger
from ibkrpy.data.ibkr_data_manager import IBKRDataManager
from ibkrpy.data.external_data import ExternalDataFetcher
from ibkrpy.data.data_pipeline import DataPipeline
from ibkrpy.data.benchmark_resolver import build_benchmark_resolver, JsonBenchmarkStore
from ibkrpy.strategy.prediction_calibrator import (
    build_threshold_policy,
    build_prediction_history,
    CollapseDetector,
)
from ibkrpy.manager.model_orchestrator import ModelOrchestrator
from ibkrpy.manager.trading_engine import TradingEngine
from ibkrpy.manager.pipeline_manager import PipelineManager
from ibkrpy.strategy.core_strategy import CoreStrategy
from ibkrpy.strategy.strategy_components import (
    RiskController,
    VIXHaltRule,
    ReversalRiskRule,
    DrawdownHaltRule,
    SessionEntryRule,
)
from ibkrpy.strategy.regime_detector import MarketRegimeDetector
from ibkrpy.strategy.market_analyzer import MarketAnalyzer
from ibkrpy.core.system_daemon import SystemDaemon
from ibkrpy.core.retrain_scheduler import RetrainScheduler

from ibkrpy.models.lstm import LSTMModel
from ibkrpy.models.transformer import TransformerModel
from ibkrpy.models.arima import ARIMAModel
from ibkrpy.models.garch import GARCHModel


class AutomatedModelFactory:
    """
    Build neural-network models from the feature manifest saved at training time.
    """

    def __init__(self, data_pipeline=None):
        self.weights_dir = os.path.join(project_root, "weights")
        os.makedirs(self.weights_dir, exist_ok=True)
        self.pipeline = data_pipeline

    def _features_for(self, symbol):
        if symbol is None or self.pipeline is None:
            return None
        return self.pipeline.load_feature_manifest(symbol)

    def create_model(self, model_type, symbol=None):
        if model_type == "LSTM":
            return LSTMModel(
                feature_cols=self._features_for(symbol), weights_dir=self.weights_dir
            )
        if model_type == "Transformer":
            return TransformerModel(
                feature_cols=self._features_for(symbol), weights_dir=self.weights_dir
            )
        if model_type == "ARIMA":
            return ARIMAModel(weights_dir=self.weights_dir)
        if model_type == "GARCH":
            return GARCHModel(weights_dir=self.weights_dir)
        raise ValueError(f"Unknown model type: {model_type}")


def launch_dashboard():
    """
    Launch the dashboard.
    """
    candidates = [
        os.path.join(project_root, core_dir_name, "core", "trading_dashboard.py"),
        os.path.join(project_root, core_dir_name, "ui", "trading_dashboard.py"),
    ]
    for ui_path in candidates:
        if os.path.exists(ui_path):
            subprocess.Popen([sys.executable, ui_path])
            logger.info(f"Dashboard launched: {ui_path}")
            return
    logger.error(f"trading_dashboard.py not found; tried: {candidates}")


def _build_benchmark_stack(config, db_manager, data_pipeline):
    """
    Composition root: the only place that decides which resolvers run, and in what order.

    TradingEngine and PipelineManager depend only on the BenchmarkResolver abstraction,
    so changing the selection rules only touches this function.
    """
    weights_dir = os.path.join(project_root, "weights")
    store = JsonBenchmarkStore(os.path.join(weights_dir, "benchmark_map.json"))
    resolver = build_benchmark_resolver(
        config=config,
        db_manager=db_manager,
        asset_profiles=config.asset_profiles,
        store=store,
        weights_dir=weights_dir,
        data_pipeline=data_pipeline,
    )
    return resolver, store


def _build_ib_manager(config, client_id: int = None) -> IBKRDataManager:
    """--client-id overrides the config so the retrain subprocess uses its own connection."""
    ib_manager = IBKRDataManager(
        host=config.get("ib_settings.host", "127.0.0.1"),
        port=config.get("ib_settings.port", 7497),
        client_id=(
            client_id
            if client_id is not None
            else config.get("ib_settings.client_id", 1)
        ),
    )
    print(
        f"Connecting to IBKR (host {ib_manager.host}:{ib_manager.port}, client ID {ib_manager.client_id})..."
    )
    return ib_manager


async def run_pipeline_mode(
    mode: str, target_symbol: str = None, client_id: int = None
) -> bool:
    """
    Returns whether it succeeded.
    """
    config = ConfigManager()
    db_manager = DatabaseManager()
    data_pipeline = DataPipeline(config=config)
    ext_fetcher = ExternalDataFetcher(
        fred_api_key=config.get("api_keys_settings.fred_api_key")
    )

    ib_manager = _build_ib_manager(config, client_id)

    try:
        await ib_manager.connect()
    except ConnectionError as e:
        logger.error(f"❌ Could not connect to IBKR; aborting: {e}")
        return False

    benchmark_resolver, benchmark_store = _build_benchmark_stack(
        config, db_manager, data_pipeline
    )

    pipeline = PipelineManager(
        config=config,
        db=db_manager,
        pipeline=data_pipeline,
        ib_data=ib_manager,
        ext_fetcher=ext_fetcher,
        target_symbol=target_symbol,
        benchmark_resolver=benchmark_resolver,
        benchmark_store=benchmark_store,
    )
    try:
        if mode == "download":
            await pipeline.run_data_ingestion()
        elif mode == "train":
            await pipeline.run_training_and_tuning()
        elif mode == "autopilot":
            await pipeline.run_autopilot()
    finally:
        if ib_manager.ib.isConnected():
            ib_manager.ib.disconnect()
    return True


async def live_trading_loop(
    engine: TradingEngine, symbols: list, retrain: RetrainScheduler = None
):
    """
    engine.should_tick() throttles by bar size: daily symbols hourly, hourly bars every
    15 minutes, 5-minute bars every 5 minutes. Symbols with a position aren't throttled.

    When the market is closed, sleep until the next session starts instead of waking
    every minute.
    """
    offset = 0
    loop_seconds = 60
    clock = engine.session_clock
    last_session = None

    try:
        while True:
            state = clock.state()
            if retrain is not None:
                retrain.poll(state.now)

            if not state.session.is_tradable:
                nap = min(state.seconds_until_change, MAX_DEEP_SLEEP_SECONDS)
                if retrain is not None:
                    retrain.maybe_start(state.now)
                    nap = min(nap, retrain.seconds_until_recheck(state.now))
                logger.info(
                    f"😴 {state.session.value}: sleeping {nap / 3600:.2f} h, "
                    f"{state.next_session.value} starts {state.next_change:%m-%d %H:%M}"
                )
                await asyncio.sleep(max(nap, 1.0))
                continue

            if retrain is not None and retrain.running:
                logger.warning(
                    "⚠️ Retrain still running; staying connected without trading this loop."
                )
                await asyncio.sleep(
                    min(loop_seconds, max(state.seconds_until_change, 1.0))
                )
                continue

            if state.session is not last_session:
                logger.info(f"🔔 Entering {state.session.value} session")
                last_session = state.session

            await engine.update_system_state()
            order = symbols[offset:] + symbols[:offset]
            offset = (offset + 1) % max(len(symbols), 1)

            due = [s for s in order if engine.should_tick(s, state.session)]
            if due:
                logger.info(
                    f"🔁 Scanning {len(due)}/{len(symbols)} symbols: {', '.join(due[:12])}"
                    + (" ..." if len(due) > 12 else "")
                )
            for symbol in due:
                await engine.run_tick(symbol, session=state.session)
                await asyncio.sleep(0.5)

            engine.log_cycle_summary()

            await asyncio.sleep(min(loop_seconds, max(state.seconds_until_change, 1.0)))
    except asyncio.CancelledError:
        if retrain is not None:
            await retrain.stop()


async def run_live_mode(args):
    config = ConfigManager()
    db_manager = DatabaseManager()
    ext_fetcher = ExternalDataFetcher(
        fred_api_key=config.get("api_keys_settings.fred_api_key")
    )
    market_analyzer = MarketAnalyzer(db_manager=db_manager, config_manager=config)
    data_pipeline = DataPipeline(config=config)
    regime_detector = MarketRegimeDetector(config.get("regime_settings") or {})

    ib_manager = _build_ib_manager(config, args.client_id)

    try:
        await ib_manager.connect()
    except ConnectionError as e:
        logger.error(f"❌ Could not connect to IBKR; aborting startup: {e}")
        return

    model_orchestrator = ModelOrchestrator(
        model_factory=AutomatedModelFactory(data_pipeline=data_pipeline),
        data_pipeline=data_pipeline,
    )
    risk_controller = RiskController(
        rules=[
            VIXHaltRule(
                threshold=float(
                    config.get("strategy_settings.vix_halt_threshold", 35.0)
                )
            ),
            ReversalRiskRule(
                threshold=float(
                    config.get("strategy_settings.reversal_block_level", 0.75)
                )
            ),
            DrawdownHaltRule(
                max_daily_loss_pct=float(
                    config.get("risk_settings.max_daily_loss_pct", 0.02)
                ),
                max_drawdown_pct=float(
                    config.get("risk_settings.max_drawdown_pct", 0.10)
                ),
            ),
            SessionEntryRule(
                allow_extended_entries=bool(
                    config.get("session_settings.allow_extended_entries", False)
                )
            ),
        ]
    )

    symbols = (
        [p.symbol for p in config.asset_profiles] if config.asset_profiles else ["AAPL"]
    )
    if args.symbol:
        symbols = [args.symbol]

    weights_dir = os.path.join(project_root, "weights")
    threshold_policy = build_threshold_policy(config)
    prediction_history = build_prediction_history(config, weights_dir)
    _cs = config.get("threshold_settings") or {}
    collapse_detector = (
        CollapseDetector(
            min_samples=int(_cs.get("collapse_min_samples", 25)),
            dispersion_floor=float(_cs.get("collapse_dispersion_floor", 1e-5)),
        )
        if _cs.get("enable_collapse_guard", True)
        else None
    )
    logger.info(f"📏 Entry threshold policy: {threshold_policy.name}")

    strategy_map = {}
    symbol_terms = {}

    global_params_path = os.path.join(
        project_root, "weights", "global_best_params.json"
    )
    global_params = {}
    if os.path.exists(global_params_path):
        try:
            with open(global_params_path, "r", encoding="utf-8") as f:
                global_params = json.load(f)
        except Exception:
            pass

    for sym in symbols:
        cfg = dict(config.get("strategy_settings") or {})

        if sym in global_params:
            cfg.update(global_params[sym])
            if "term" in global_params[sym]:
                symbol_terms[sym] = global_params[sym]["term"]

        strategy_map[sym] = CoreStrategy(
            sym,
            cfg,
            threshold_policy=threshold_policy,
            prediction_history=prediction_history,
            collapse_detector=collapse_detector,
        )

    benchmark_resolver, _ = _build_benchmark_stack(config, db_manager, data_pipeline)

    engine = TradingEngine(
        data_manager=ib_manager,
        model_orchestrator=model_orchestrator,
        risk_controller=risk_controller,
        strategy_map=strategy_map,
        db_manager=db_manager,
        ext_fetcher=ext_fetcher,
        market_analyzer=market_analyzer,
        data_pipeline=data_pipeline,
        regime_detector=regime_detector,
        dry_run=args.dry_run,
        symbol_terms=symbol_terms,
        config_manager=config,
        benchmark_resolver=benchmark_resolver,
    )

    def _on_retrained():
        model_orchestrator.invalidate()
        logger.info("Model cache cleared; new weights load on the next loop.")

    retrain = RetrainScheduler(config, on_success=_on_retrained)

    try:
        if args.mode == "live":
            await live_trading_loop(engine, symbols, retrain=retrain)
        elif args.mode == "daemon":
            await SystemDaemon(
                ib_manager, engine, symbols, retrain_scheduler=retrain
            ).run_24_7()
    except KeyboardInterrupt:
        pass
    finally:
        if ib_manager.ib.isConnected():
            ib_manager.ib.disconnect()


def main():
    _is_subprocess = "--client-id" in sys.argv

    _boot_config = ConfigManager()
    setup_logger(_boot_config.get("log_settings") or {}, enable_file=not _is_subprocess)

    parser = argparse.ArgumentParser(
        description="IBKR AI trading system",
        formatter_class=argparse.RawTextHelpFormatter,
    )
    parser.add_argument(
        "--mode",
        type=str,
        required=True,
        choices=["download", "train", "autopilot", "live", "daemon", "ui"],
    )
    parser.add_argument(
        "symbol",
        nargs="?",
        type=str,
        default=None,
        help="Run for a single symbol (optional, e.g. MRVL)",
    )
    parser.add_argument("--ui", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument(
        "--client-id",
        type=int,
        default=None,
        help="Override the IBKR client_id from config. Used to give the retrain subprocess\n"
        "its own ID so it doesn't clash with the 24/7 main process.",
    )
    args = parser.parse_args()

    if args.ui or args.mode == "ui":
        launch_dashboard()
        if args.mode == "ui":
            sys.exit(0)

    match args.mode:
        case "download" | "train" | "autopilot":
            ok = asyncio.run(run_pipeline_mode(args.mode, args.symbol, args.client_id))
            if not ok:
                sys.exit(1)
        case "live" | "daemon":
            asyncio.run(run_live_mode(args))


if __name__ == "__main__":
    main()
