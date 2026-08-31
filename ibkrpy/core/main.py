# ibkrpy/core/main.py
# 系統總樞紐 (Command Center)

import argparse
import sys
import os
import asyncio
import subprocess
import json
import logging
import warnings
import caffeine

# ========== macOS 基礎防禦 ==========
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["KMP_DUPLICATE_LIB_OK"] = "True"
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"
warnings.filterwarnings("ignore")
# ====================================

project_root = os.path.dirname(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
)
sys.path.append(project_root)
caffeine.on(display=False)

logger = logging.getLogger("ibkrpy.main")

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
)
from ibkrpy.strategy.regime_detector import MarketRegimeDetector
from ibkrpy.strategy.market_analyzer import MarketAnalyzer
from ibkrpy.core.system_daemon import SystemDaemon

from ibkrpy.models.lstm import LSTMModel
from ibkrpy.models.transformer import TransformerModel
from ibkrpy.models.arima import ARIMAModel
from ibkrpy.models.garch import GARCHModel


class AutomatedModelFactory:
    """
    依訓練時保存的特徵清單建立神經網路模型。
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
        raise ValueError(f"未知的模型類型: {model_type}")


def launch_dashboard():
    """
    啟動視覺化儀表板。
    """
    candidates = [
        os.path.join(project_root, core_dir_name, "core", "trading_dashboard.py"),
        os.path.join(project_root, core_dir_name, "ui", "trading_dashboard.py"),
    ]
    for ui_path in candidates:
        if os.path.exists(ui_path):
            subprocess.Popen([sys.executable, ui_path])
            logger.info(f"已啟動儀表板: {ui_path}")
            return
    logger.error(f"找不到 trading_dashboard.py，已嘗試: {candidates}")


def _build_benchmark_stack(config, db_manager, data_pipeline):
    """
    Composition Root：唯一知道「要用哪些 resolver、以什麼順序」的地方。

    把組裝集中在這裡，TradingEngine 與 PipelineManager 都只依賴
    BenchmarkResolver 抽象，兩者不需要知道任何一條選擇規則 (DIP)。
    要改規則只改這一個函式。
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


async def run_pipeline_mode(
    mode: str, target_symbol: str = None, client_id: int = None
):
    config = ConfigManager()
    db_manager = DatabaseManager()
    data_pipeline = DataPipeline()
    ext_fetcher = ExternalDataFetcher(
        fred_api_key=config.get("api_keys_settings.fred_api_key")
    )

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
        f"嘗試連線至 IBKR (Host: {ib_manager.host}:{ib_manager.port}, Client ID: {ib_manager.client_id})..."
    )

    try:
        await ib_manager.connect()
    except ConnectionError as e:
        logger.error(f"❌ 無法連線至 IBKR，中止本次作業: {e}")
        return

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


async def live_trading_loop(
    engine: TradingEngine, symbols: list, interval_minutes: int = 5
):
    """
    由 engine.should_tick() 依 bar size 決定該不該問：日 K 標的每小時一次、
    小時 K 每 15 分鐘一次、5 分 K 才每 5 分鐘一次。有持倉的標的不受節流限制。

    休市時進入 deepsleep —— 直接睡到下一個時段開始，而不是每分鐘醒來確認一次。
    原本的迴圈完全不知道市場是否開盤，週末與假日照樣每 60 秒喚醒並呼叫
    update_system_state()，一個週末就是 3,330 次無用的喚醒與 IBKR 請求。
    """
    offset = 0
    loop_seconds = 60
    clock = engine.session_clock
    last_session = None

    try:
        while True:
            state = clock.state()

            if not state.session.is_tradable:
                # deepsleep。上限 6 小時是為了讓長睡眠仍能定期確認連線與日曆，
                # 跨越長週末時會分成幾段睡完，而不是一次睡 79 小時。
                nap = min(state.seconds_until_change, 6 * 3600)
                logger.info(
                    f"😴 {state.session.value} —— 深度休眠 {nap / 3600:.2f} 小時，"
                    f"{state.next_change:%m-%d %H:%M} 進入 {state.next_session.value}"
                )
                await asyncio.sleep(max(nap, 1.0))
                continue

            if state.session is not last_session:
                logger.info(f"🔔 進入 {state.session.value} 時段")
                last_session = state.session

            await engine.update_system_state()
            # 輪替起點，避免資金耗盡時清單後段的標的長期被跳過
            order = symbols[offset:] + symbols[:offset]
            offset = (offset + 1) % max(len(symbols), 1)

            due = [s for s in order if engine.should_tick(s, state.session)]
            if due:
                logger.info(
                    f"🔁 本輪待掃描 {len(due)}/{len(symbols)} 檔: {', '.join(due[:12])}"
                    + (" ..." if len(due) > 12 else "")
                )
            for symbol in due:
                await engine.run_tick(symbol, session=state.session)
                await asyncio.sleep(0.5)

            engine.log_cycle_summary()

            # 不要睡過時段邊界，否則開盤瞬間會遲到最多 60 秒。
            await asyncio.sleep(min(loop_seconds, max(state.seconds_until_change, 1.0)))
    except asyncio.CancelledError:
        pass


async def run_live_mode(args):
    config = ConfigManager()
    db_manager = DatabaseManager()
    ext_fetcher = ExternalDataFetcher(
        fred_api_key=config.get("api_keys_settings.fred_api_key")
    )
    market_analyzer = MarketAnalyzer(db_manager=db_manager, config_manager=config)
    data_pipeline = DataPipeline()
    regime_detector = MarketRegimeDetector(config.get("regime_settings") or {})

    ib_manager = IBKRDataManager(
        host=config.get("ib_settings.host", "127.0.0.1"),
        port=config.get("ib_settings.port", 7497),
        client_id=(
            args.client_id
            if args.client_id is not None
            else config.get("ib_settings.client_id", 1)
        ),
    )
    print(
        f"嘗試連線至 IBKR (Host: {ib_manager.host}:{ib_manager.port}, Client ID: {ib_manager.client_id})..."
    )

    try:
        await ib_manager.connect()
    except ConnectionError as e:
        logger.error(f"❌ 無法連線至 IBKR，中止啟動: {e}")
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
        ]
    )

    symbols = (
        [p.symbol for p in config.asset_profiles] if config.asset_profiles else ["AAPL"]
    )
    # 覆蓋為單一標的 (若有提供)
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
    logger.info(f"📏 進場門檻政策: {threshold_policy.name}")

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

    benchmark_resolver, benchmark_store = _build_benchmark_stack(
        config, db_manager, data_pipeline
    )

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

    try:
        if args.mode == "live":
            await live_trading_loop(engine, symbols, interval_minutes=5)
        elif args.mode == "daemon":
            await SystemDaemon(
                ib_manager,
                engine,
                PipelineManager(
                    config,
                    db_manager,
                    data_pipeline,
                    ib_manager,
                    ext_fetcher,
                    target_symbol=args.symbol,
                    benchmark_resolver=benchmark_resolver,
                    benchmark_store=benchmark_store,
                ),
                symbols,
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
        description="IBKR AI 量化交易系統 總樞紐",
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
        help="指定單一股票代碼 (選填，例如 MRVL)",
    )
    parser.add_argument("--ui", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument(
        "--client-id",
        type=int,
        default=None,
        help="覆寫 config 的 IBKR client_id。SystemDaemon 以此讓重訓子行程\n"
        "使用不同號碼，避免與 24/7 主行程的連線衝突。",
    )
    args = parser.parse_args()

    if args.ui or args.mode == "ui":
        launch_dashboard()
        if args.mode == "ui":
            sys.exit(0)

    match args.mode:
        case "download":
            asyncio.run(run_pipeline_mode("download", args.symbol, args.client_id))
        case "train":
            asyncio.run(run_pipeline_mode("train", args.symbol, args.client_id))
        case "autopilot":
            asyncio.run(run_pipeline_mode("autopilot", args.symbol, args.client_id))
        case "live":
            asyncio.run(run_live_mode(args))
        case "daemon":
            asyncio.run(run_live_mode(args))


if __name__ == "__main__":
    main()
