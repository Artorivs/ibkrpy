# ibkrpy/core/main.py
# 系統總樞紐 (Command Center)

import argparse
import sys
import os
import asyncio
import subprocess
import json
import warnings
import caffeine
import datetime

# ========== macOS 基礎防禦 ==========
os.environ['OMP_NUM_THREADS'] = '1'
os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
warnings.filterwarnings("ignore")
# ====================================

project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.append(project_root)
caffeine.on(display=False)

core_dir_name = os.path.basename(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from ibkrpy.shared.config_manager import ConfigManager
from ibkrpy.shared.db_manager import DatabaseManager
from ibkrpy.shared.system_log import setup_logger
from ibkrpy.data.ibkr_data_manager import IBKRDataManager
from ibkrpy.data.external_data import ExternalDataFetcher
from ibkrpy.data.data_pipeline import DataPipeline
from ibkrpy.manager.model_orchestrator import ModelOrchestrator
from ibkrpy.manager.trading_engine import TradingEngine
from ibkrpy.manager.pipeline_manager import PipelineManager
from ibkrpy.strategy.core_strategy import CoreStrategy
from ibkrpy.strategy.strategy_components import RiskController, VIXHaltRule, MarketRegimeDetector
from ibkrpy.strategy.market_analyzer import MarketAnalyzer
from ibkrpy.core.system_daemon import SystemDaemon

from ibkrpy.models.lstm import LSTMModel
from ibkrpy.models.transformer import TransformerModel
from ibkrpy.models.arima import ARIMAModel
from ibkrpy.models.garch import GARCHModel
from ibkrpy.models.hmm import HMMModel

class AutomatedModelFactory:
    def __init__(self):
        self.weights_dir = os.path.join(project_root, "weights")
        os.makedirs(self.weights_dir, exist_ok=True)

    def create_model(self, model_type):
        if model_type == "LSTM": return LSTMModel(weights_dir=self.weights_dir)
        if model_type == "Transformer": return TransformerModel(weights_dir=self.weights_dir)
        if model_type == "ARIMA": return ARIMAModel(weights_dir=self.weights_dir)
        if model_type == "GARCH": return GARCHModel(weights_dir=self.weights_dir)
        if model_type == "HMM": return HMMModel(weights_dir=self.weights_dir)
        raise ValueError(f"未知的模型類型: {model_type}")

def launch_dashboard():
    ui_path = os.path.join(project_root, core_dir_name, "ui", "trading_dashboard.py")
    subprocess.Popen([sys.executable, ui_path])

async def run_pipeline_mode(mode: str):
    config = ConfigManager()
    db_manager = DatabaseManager()
    data_pipeline = DataPipeline()
    ext_fetcher = ExternalDataFetcher(fred_api_key=config.get("api_keys_settings.fred_api_key"))
    
    ib_manager = IBKRDataManager(host=config.get("ib_settings.host", "127.0.0.1"), port=config.get("ib_settings.port", 7497), client_id=config.get("ib_settings.client_id", 1))
    print(f"嘗試連線至 IBKR (Host: {ib_manager.host}:{ib_manager.port}, Client ID: {ib_manager.client_id})...")
    
    await ib_manager.connect()

    pipeline = PipelineManager(config=config, db=db_manager, pipeline=data_pipeline, ib_data=ib_manager, ext_fetcher=ext_fetcher)
    try:
        if mode == "download": await pipeline.run_data_ingestion()
        elif mode == "train": await pipeline.run_training_and_tuning()
        elif mode == "autopilot": await pipeline.run_autopilot()
    finally:
        if ib_manager.ib.isConnected(): ib_manager.ib.disconnect()

async def live_trading_loop(engine: TradingEngine, symbols: list, interval_minutes: int = 5):
    logger = setup_logger()
    try:
        while True:
            await engine.update_system_state()
            for symbol in symbols:
                await engine.run_tick(symbol)
                await asyncio.sleep(2)
            await asyncio.sleep(interval_minutes * 60)
    except asyncio.CancelledError: pass

async def run_live_mode(args):
    config = ConfigManager()
    db_manager = DatabaseManager()
    ext_fetcher = ExternalDataFetcher(fred_api_key=config.get("api_keys_settings.fred_api_key"))
    market_analyzer = MarketAnalyzer(db_manager=db_manager, config_manager=config)
    data_pipeline = DataPipeline()  
    regime_detector = MarketRegimeDetector() 
    
    ib_manager = IBKRDataManager(host=config.get("ib_settings.host", "127.0.0.1"), port=config.get("ib_settings.port", 7497), client_id=config.get("ib_settings.client_id", 1))
    print(f"嘗試連線至 IBKR (Host: {ib_manager.host}:{ib_manager.port}, Client ID: {ib_manager.client_id})...")
    
    await ib_manager.connect()
        
    if not ib_manager.ib.isConnected(): return

    model_orchestrator = ModelOrchestrator(model_factory=AutomatedModelFactory())
    risk_controller = RiskController(rules=[VIXHaltRule(threshold=35.0)])
    
    symbols = [p.symbol for p in config.asset_profiles] if config.asset_profiles else ["AAPL"]
    strategy_map = {}
    symbol_terms = {}
    
    global_params_path = os.path.join(project_root, "weights", "global_best_params.json")
    global_params = {}
    if os.path.exists(global_params_path):
        try:
            with open(global_params_path, 'r', encoding='utf-8') as f:
                global_params = json.load(f)
        except Exception: pass
    
    for sym in symbols:
        cfg = config.get("strategy_settings") or {}
        
        if sym in global_params:
            cfg.update(global_params[sym])
            if 'term' in global_params[sym]:
                symbol_terms[sym] = global_params[sym]['term']
            
        strategy_map[sym] = CoreStrategy(sym, cfg)

    engine = TradingEngine(
        data_manager=ib_manager, model_orchestrator=model_orchestrator,
        risk_controller=risk_controller, strategy_map=strategy_map, db_manager=db_manager,
        ext_fetcher=ext_fetcher, market_analyzer=market_analyzer, data_pipeline=data_pipeline,
        regime_detector=regime_detector, dry_run=args.dry_run, symbol_terms=symbol_terms
    )

    try:
        if args.mode == "live": await live_trading_loop(engine, symbols, interval_minutes=5)
        elif args.mode == "daemon": await SystemDaemon(ib_manager, engine, PipelineManager(config, db_manager, data_pipeline, ib_manager, ext_fetcher), symbols).run_24_7()
    except KeyboardInterrupt: pass
    finally:
        if ib_manager.ib.isConnected(): ib_manager.ib.disconnect()

def main():
    parser = argparse.ArgumentParser(description="IBKR AI 量化交易系統 總樞紐", formatter_class=argparse.RawTextHelpFormatter)
    parser.add_argument("--mode", type=str, required=True, choices=["download", "train", "autopilot", "live", "daemon", "ui"])
    parser.add_argument("--ui", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    if args.ui or args.mode == "ui":
        launch_dashboard()
        if args.mode == "ui": sys.exit(0)
    
    match args.mode:
        case "download": asyncio.run(run_pipeline_mode("download"))
        case "train": asyncio.run(run_pipeline_mode("train"))
        case "autopilot": asyncio.run(run_pipeline_mode("autopilot"))
        case "live": asyncio.run(run_live_mode(args))
        case "daemon": asyncio.run(run_live_mode(args))

if __name__ == "__main__": main()