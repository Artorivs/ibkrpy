# ibkrpy/manager/trading_engine.py
# 系統交易引擎：量化系統的主心臟。單向處理 數據 -> 宏觀全局 -> 預測 -> 策略 -> 執行

import asyncio
import math
import time
from typing import Dict, Any
from ib_insync import Order, LimitOrder, Stock, TagValue
import pandas as pd
import numpy as np
import logging

# 掛在 "ibkrpy" 樹下，確保 system_log 的 QueueHandler 會收到本模組的日誌
logger = logging.getLogger("ibkrpy.trading_engine")
_TERM_POLL_SECONDS = {
    "short_term": 300,      # 5 分 K -> 每 5 分鐘
    "mid_term": 900,        # 1 小時 K -> 每 15 分鐘 (盤中提早看到未收完的當根)
    "long_term": 3600,      # 日 K -> 每小時
}


class TradingEngine:
    """單向資料流的主迴圈，具備長短期數據自適應抓取能力，全面落實配置驅動。"""
    
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
        config_manager=None
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
        
        if config_manager is None:
            from ibkrpy.shared.config_manager import ConfigManager
            self.config = ConfigManager("config.yaml")
        else:
            self.config = config_manager
            
        self.global_context = {}
        self.cached_funds = 0.0
        self.cached_net_liq = 0.0
        self.cached_positions = {}
        self.cached_vix_series = None
        self.vix_last_fetch_time = 0.0

        self.cached_benchmarks = {}
        self._last_prices = {}
        self.qualified_contracts = {}

        # 每檔標的上次真正跑完 run_tick 的時間，用來做 per-term 節流
        self._last_tick_at: Dict[str, float] = {}
        # 本輪掃描的決策統計，供 log_cycle_summary 輸出
        self._cycle_decisions: Dict[str, int] = {}
        self._cycle_started_at = time.time()

    def _poll_interval(self, symbol: str) -> int:
        term = self.symbol_terms.get(symbol, "long_term")
        default = _TERM_POLL_SECONDS.get(term, 3600)
        return int(self.config.get(f"general_settings.poll_seconds_{term}", default))

    def should_tick(self, symbol: str) -> bool:
        """
        由 bar size 決定該不該真的去問 IBKR。持倉中的標的例外 —— 有部位時
        風險是實時的，不能等下一根 K 線。
        """
        if self.cached_positions.get(symbol):
            return True
        last = self._last_tick_at.get(symbol)
        if last is None:
            return True
        return (time.time() - last) >= self._poll_interval(symbol)

    def _record_decision(self, code: str):
        self._cycle_decisions[code] = self._cycle_decisions.get(code, 0) + 1

    def log_cycle_summary(self):
        """
        每輪掃描結束時印一行總結。
        """
        if not self._cycle_decisions:
            logger.info("📊 [本輪總結] 沒有任何標的通過輪詢節流 (皆在冷卻期內)。")
            return

        total = sum(self._cycle_decisions.values())
        breakdown = " | ".join(
            f"{k}×{v}" for k, v in sorted(self._cycle_decisions.items(), key=lambda kv: -kv[1])
        )
        elapsed = time.time() - self._cycle_started_at
        line = f"📊 [本輪總結] 掃描 {total} 檔 / 耗時 {elapsed:.0f}s -> {breakdown}"

        stats = getattr(self.data, "pacing_stats", None)
        if callable(stats):
            s = stats()
            line += (f" || API: 10分鐘窗 {s['requests_in_window']}/{s['window_limit']}，"
                     f"快取命中率 {s['cache_hit_rate']:.0%}")
        logger.info(line)

        self._cycle_decisions = {}
        self._cycle_started_at = time.time()

    async def _get_qualified_contract(self, symbol: str):
        """
        獲取並緩存「完整」合約。
        必然失敗。現在改為委派給 IBKRDataManager.qualify()，解析不到就回 None。
        """
        if symbol in self.qualified_contracts:
            return self.qualified_contracts[symbol]

        contract = None
        if hasattr(self.data, "qualify"):
            contract = await self.data.qualify(symbol)
        else:  # 相容舊版 data manager
            contract = Stock(symbol, "SMART", "USD")
            resolved = await self.data.ib.qualifyContractsAsync(contract)
            if not resolved or not contract.conId:
                logger.error(f"[{symbol}] ❌ 合約解析失敗，本輪跳過此標的。")
                contract = None

        if contract is not None:
            self.qualified_contracts[symbol] = contract
        return contract

    def _get_dynamic_benchmark(self, symbol: str) -> str:
        """
        回傳該標的使用的 benchmark
        """
        return self.config.get("general_settings.benchmark_symbol", "QQQ")

    async def update_system_state(self):
        """每一輪大迴圈開始前統一調用，更新帳戶快取與全局市場上下文"""
        self.cached_benchmarks.clear() # 每一輪迴圈開始前清空大盤快取
        try:
            positions = await self.data.ib.reqPositionsAsync()
            account_summary = await self.data.ib.accountSummaryAsync()
            
            available_funds = 0.0
            net_liquidation = 0.0
            
            for item in account_summary:
                if item.tag == 'AvailableFunds' and item.currency in ('BASE', 'USD'):
                    available_funds = float(item.value)
                elif item.tag == 'NetLiquidation' and item.currency in ('BASE', 'USD'):
                    net_liquidation = float(item.value)
                    
            pos_dict = {p.contract.symbol: p.position for p in positions if p.position != 0}
            
            self.cached_funds = available_funds
            self.cached_net_liq = net_liquidation
            self.cached_positions = pos_dict
            
            if self.db:
                await self.db.update_account_info(net_liquidation, available_funds, pos_dict)
                
            await self._protect_unhedged_positions()
                
        except Exception as e:
            logger.warning(f"⚠️ 獲取帳戶狀態失敗: {e}，將使用前次快取數據。")

        if self.market_analyzer:
            self.global_context = await asyncio.to_thread(self.market_analyzer.get_global_context)

    def _current_gross_exposure(self) -> float:
        """
        目前的總曝險 (多空取絕對值後加總的名目金額)。
        以最近一次 run_tick 看到的價格估算；沒有價格紀錄時退回以持股數 x 最後已知價。
        """
        total = 0.0
        for sym, qty in (self.cached_positions or {}).items():
            if not qty:
                continue
            price = self._last_prices.get(sym)
            if price:
                total += abs(qty) * price
        return total

    async def _protect_unhedged_positions(self):
        """掃描帳戶中的持倉，若發現沒有掛出停損/停利單的持股（例如手動買入），自動補上 OCA 保護傘。"""
        if self.dry_run: return
        
        positions = self.cached_positions
        if not positions: return
        
        open_trades = self.data.ib.openTrades()
        
        for symbol, pos_qty in positions.items():
            if pos_qty == 0: continue
            
            # 檢查該標的是否有反向的未決訂單 (SELL單若持倉為正，BUY單若持倉為負)
            protect_action = "SELL" if pos_qty > 0 else "BUY"
            protected_qty = 0.0
            for t in open_trades:
                if (t.contract.symbol == symbol
                        and t.order.action == protect_action
                        and t.order.orderType in ("STP", "STP LMT")):
                    protected_qty += float(t.order.totalQuantity or 0)
            has_protection = protected_qty >= abs(pos_qty)
                    
            if not has_protection:
                logger.warning(f"[守護] 🛡️ 偵測到 {symbol} 存在無保護持倉 ({pos_qty} 股)，準備自動掛載 OCA 停損停利單...")
                await self._attach_oca_protection(symbol, pos_qty)

    def _live_market_price(self, symbol: str):
        """
        取得即時市價。優先使用 IBKR 已串流的 portfolio marketPrice ——
        它由 updatePortfolio 事件持續推送，不額外消耗歷史資料請求配額。
        其次退回本輪 run_tick 記錄的價格。兩者皆無則回傳 None。
        """
        try:
            for item in self.data.ib.portfolio():
                if item.contract.symbol == symbol and item.marketPrice and item.marketPrice > 0:
                    return float(item.marketPrice)
        except Exception as e:
            logger.warning(f"[{symbol}] 讀取 portfolio 市價失敗: {e}")

        p = self._last_prices.get(symbol)
        return float(p) if p else None

    async def _attach_oca_protection(self, symbol: str, pos_qty: float):
        contract = await self._get_qualified_contract(symbol)
        try:
            # 優先從 DB 讀取日 K 計算真實波動率，避免每 5 分鐘因未平倉而狂刷 60 天歷史請求
            df = pd.DataFrame()
            if self.db:
                df = self.db.get_market_data_sync(symbol, timeframe='1 day')
                
            if df.empty or len(df) < 10:
                df = await self.data.fetch_historical_data(contract, duration='60 D', bar_size='1 day')
                
            if df.empty: return
            
            if 'close' in df.columns:
                df.rename(columns={'open': 'Open', 'high': 'High', 'low': 'Low', 'close': 'Close', 'volume': 'Volume'}, inplace=True)
            
            live_price = self._live_market_price(symbol)
            db_price = float(df['Close'].iloc[-1])

            if live_price is None:
                logger.critical(
                    f"❌ [{symbol}]取不到即時市價，拒絕掛載保護傘（部位仍無保護，需人工處理）。"
                    f"	資料庫參考價為 {db_price:.2f}。"
                )
                return

            drift = abs(db_price / live_price - 1) if live_price > 0 else 1.0
            if drift > 0.03:
                logger.warning(
                    f"❌ [{symbol}] 本地行情已過期 {drift*100:.1f}%"
                    f"（DB {db_price:.2f} vs 市價 {live_price:.2f}），改以市價為錨。"
                )

            current_price = live_price
            returns = np.log(df['Close'] / df['Close'].shift(1)).dropna()
            annual_vol = returns.std() * np.sqrt(252) if len(returns) > 10 else 0.20

            term = self.symbol_terms.get(symbol, "long_term")
            if term == "short_term":
                periods_per_year = 252 * 78     # 5 分 K
            elif term == "mid_term":
                periods_per_year = 252 * 6.5    # 1 小時 K
            else:
                periods_per_year = 252          # 日 K
            daily_vol = annual_vol / math.sqrt(periods_per_year)
            
            # 從策略抓取乘數設定，若無則預設停損 2.0 倍 / 停利 3.0 倍波動
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
                    f"❌ [{symbol}] 保護傘價位方向錯誤，已拒絕送出："
                    f"市價 {current_price:.2f} / 停損 {sl_price:.2f} / 停利 {tp_price:.2f}"
                    f"（部位 {pos_qty:+.0f} 股）。掛出去會立即成交，等同誤平倉。"
                )
                return

            # 建立 OCA 群組標籤 (加上時間戳確保唯一性)
            oca_group = f"OCA_PROTECT_{symbol}_{int(time.time())}"
            
            # 建立獨立的 STP 與 LMT 單，並透過 ocaGroup 綁定。ocaType=1 代表觸發其一即取消另一
            sl_order = Order(action=action, totalQuantity=abs(pos_qty), orderType="STP", auxPrice=sl_price, tif='GTC', ocaGroup=oca_group, ocaType=1)
            tp_order = LimitOrder(action=action, totalQuantity=abs(pos_qty), lmtPrice=tp_price, tif='GTC', ocaGroup=oca_group, ocaType=1)
            
            self.data.ib.placeOrder(contract, sl_order)
            self.data.ib.placeOrder(contract, tp_order)
            
            logger.info(f"✅ [{symbol}] 掛載保護傘 (OCA) -> 停損(STP): {sl_price:.2f}, 停利(LMT): {tp_price:.2f}")
        except Exception as e:
            logger.critical(f"❌ [{symbol}] 掛載保護傘失敗: {e}")

    async def _cancel_open_orders(self, symbol: str):
        """[關鍵防護] 取消該標的目前所有未成交的委託單，防範舊的停損/停利單變成「孤兒單」導致裸空"""
        if self.dry_run:
            return
            
        open_trades = self.data.ib.openTrades()
        canceled_count = 0
        for trade in open_trades:
            if trade.contract.symbol == symbol:
                self.data.ib.cancelOrder(trade.order)
                canceled_count += 1
                
        if canceled_count > 0:
            logger.info(f"[{symbol}] 🧹 已清除 {canceled_count} 筆歷史未決訂單 (防範孤兒單衝突)。")
            # 稍微等待 IBKR 系統同步取消狀態
            await asyncio.sleep(0.5)

    async def run_tick(self, symbol: str):
        self._last_tick_at[symbol] = time.time()

        available_funds = self.cached_funds
        net_liquidation = self.cached_net_liq
        current_pos = self.cached_positions.get(symbol, 0.0)

        contract = await self._get_qualified_contract(symbol)
        if contract is None:
            self._record_decision("CONTRACT_UNRESOLVED")
            return

        term = self.symbol_terms.get(symbol, "long_term")
        
        if term == "short_term":
            bar_size_str = self.config.get("general_settings.short_term_bar_size", "5 mins")
        elif term == "mid_term":
            bar_size_str = self.config.get("general_settings.mid_term_bar_size", "1 hour")
        else:
            bar_size_str = self.config.get("general_settings.long_term_bar_size", "1 day")
            
        is_short_term = (term == "short_term")
            
        # 實盤模式下，只向 IBKR 請求最近「3 天」的輕量數據
        live_duration = "3 D" 
        df_recent = await self.data.fetch_historical_data(
            contract=contract, duration=live_duration, bar_size=bar_size_str, what_to_show='TRADES'
        )
        
        # 取得資料庫中的歷史長線資料，並與剛抓到的最新輕量資料合併 (Stitching)
        df_db = pd.DataFrame()
        if self.db:
            df_db = self.db.get_market_data_sync(symbol, timeframe=bar_size_str)
            if not df_db.empty:
                df_db = df_db[['Open', 'High', 'Low', 'Close', 'Volume']].astype(float)
                df_db.index = pd.to_datetime(df_db.index, utc=True)

        if not df_recent.empty:
            df_recent.index = pd.to_datetime(df_recent.index, utc=True)
            df_recent.rename(columns={'open': 'Open', 'high': 'High', 'low': 'Low', 'close': 'Close', 'volume': 'Volume'}, inplace=True)
            df_recent = df_recent[['Open', 'High', 'Low', 'Close', 'Volume']].astype(float)
            

            # 順手將這 3 天的新資料寫入 DB，讓資料庫保持最新
            if self.db:
                self.db.save_bulk_market_data(symbol, df_recent, timeframe=bar_size_str)

        if not df_db.empty and not df_recent.empty:
            df = pd.concat([df_db, df_recent])

            df = df[~df.index.duplicated(keep='last')].sort_index()
        elif not df_recent.empty:
            df = df_recent
        else:
            df = df_db
            
        # 確保型態正確 (防護 TA-Lib / Pandas TA 報錯)
        cols_to_keep = ['Open', 'High', 'Low', 'Close', 'Volume']
        for col in cols_to_keep:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors='coerce')

        
        if df.empty or len(df) < 60:
            self._record_decision("NO_DATA")
            logger.warning(
                f"⚠️ [{symbol}] {term} ({bar_size_str}) 資料不足："
                f"僅 {len(df)} 根 K 線 (需要 ≥60)。請執行 --mode download 補齊資料庫。"
            )
            return

        current_price = float(df['Close'].iloc[-1])
        self._last_prices[symbol] = current_price

        benchmark_symbol = self._get_dynamic_benchmark(symbol)
        bench_df = pd.DataFrame()
        try:
            if benchmark_symbol == symbol:
                bench_df = df.copy()
            else:
                cache_key = f"{benchmark_symbol}_{bar_size_str}"
                if cache_key in self.cached_benchmarks:
                    bench_df_raw = self.cached_benchmarks[cache_key]
                    if not bench_df_raw.empty:
                        bench_df = bench_df_raw.reindex(df.index, method='ffill').bfill()
                else:
                    bench_contract = await self._get_qualified_contract(benchmark_symbol)
                    
                    bench_recent = await self.data.fetch_historical_data(
                        contract=bench_contract, duration=live_duration, bar_size=bar_size_str, what_to_show='TRADES'
                    )
                    
                    bench_db = pd.DataFrame()
                    if self.db:
                        bench_db = self.db.get_market_data_sync(benchmark_symbol, timeframe=bar_size_str)
                        if not bench_db.empty:
                            bench_db = bench_db[['Open', 'High', 'Low', 'Close', 'Volume']].astype(float)
                            bench_db.index = pd.to_datetime(bench_db.index, utc=True)
                        
                    if not bench_recent.empty:
                        bench_recent.index = pd.to_datetime(bench_recent.index, utc=True)
                        bench_recent.rename(columns={'open': 'Open', 'high': 'High', 'low': 'Low', 'close': 'Close', 'volume': 'Volume'}, inplace=True)
                        bench_recent = bench_recent[['Open', 'High', 'Low', 'Close', 'Volume']].astype(float)
                        

                        if self.db:
                            self.db.save_bulk_market_data(benchmark_symbol, bench_recent, timeframe=bar_size_str)

                    if not bench_db.empty and not bench_recent.empty:
                        bench_df_raw = pd.concat([bench_db, bench_recent])
                        bench_df_raw = bench_df_raw[~bench_df_raw.index.duplicated(keep='last')].sort_index()
                    elif not bench_recent.empty:
                        bench_df_raw = bench_recent
                    else:
                        bench_df_raw = bench_db
                        
                    for col in cols_to_keep:
                        if col in bench_df_raw.columns:
                            bench_df_raw[col] = pd.to_numeric(bench_df_raw[col], errors='coerce')

                    # 寫入快取，供同一迴圈的下一檔股票使用
                    self.cached_benchmarks[cache_key] = bench_df_raw
                    
                    if not bench_df_raw.empty:
                        bench_df = bench_df_raw.reindex(df.index, method='ffill').bfill()
                        
        except Exception as e:
            logger.warning(f"⚠️ [{symbol}] 獲取基準指標 Benchmark ({benchmark_symbol}) 失敗: {e}")

        macro_dict = {}
        if self.ext:
            current_time = time.time()
            
            if self.cached_vix_series is None or (current_time - self.vix_last_fetch_time) > 28800:
                try:
                    vix_series = await self.ext.fetch_fred_series("VIXCLS")
                    if vix_series is not None and not vix_series.empty:
                        self.cached_vix_series = vix_series
                        self.vix_last_fetch_time = current_time
                        logger.info(f"🌍 [系統] 成功從 FRED 更新宏觀數據 (VIXCLS)，已寫入本地快取 (有效期限 8 小時)。")
                except Exception as e:
                    logger.warning(f"⚠️ 獲取 FRED 數據失敗，將維持使用本地快取: {e}")

            # 直接使用快取的數據進行特徵對齊
            if self.cached_vix_series is not None and not self.cached_vix_series.empty:
                vix_daily = self.cached_vix_series.copy()
                vix_daily.index = vix_daily.index.normalize()
                
                df_idx_naive = df.index
                vix_aligned_values = df_idx_naive.normalize().map(vix_daily)
                vix_aligned = pd.Series(vix_aligned_values, index=df.index).ffill().bfill()
                
                if vix_aligned.isna().all():
                    vix_aligned = pd.Series(20.0, index=df.index)
                macro_dict['VIX'] = vix_aligned

        if self.pipeline:
            bench_data = bench_df if not bench_df.empty else None
            df_adv = self.pipeline.engineer_advanced_features(df, bench_data, macro_dict)
            df_adv = df_adv.ffill().bfill().fillna(0)
            
            df_adv, scale_cols = self.pipeline.align_to_manifest(df_adv, symbol)
            df_scaled, scale_cols = self.pipeline.transform_for_inference(df_adv, symbol)
        else:
            df_adv = df.ffill().bfill().fillna(0)
            df_scaled = df_adv

        # 情境偵測
        from ibkrpy.strategy.strategy_components import MarketRegime
        regime = MarketRegime.SIDEWAYS_QUIET
        if self.regime_detector:
            regime = self.regime_detector.detect(df_adv)

        context = {"vix_series": macro_dict.get("VIX"), "current_price": current_price, "regime": regime}
        is_allowed, reason = self.risk.check_trade_allowed(context)
        if not is_allowed:
            self._record_decision("RISK_BLOCKED")
            logger.info(f"[{symbol}] 🛡️ 交易系統拒絕進場: {reason}")
            return

        ensemble_preds = {}
        rejected_models = {}
        target_models = ["LSTM", "Transformer", "ARIMA"]

        for m_type in target_models:
            if hasattr(self.models, "is_ready") and not self.models.is_ready(symbol, m_type):
                rejected_models[m_type] = "未訓練"
                continue

            pred_raw, _ = self.models.predict(
                symbol, df_scaled if m_type != "ARIMA" else df_adv, model_type=m_type
            )

            if isinstance(pred_raw, (list, np.ndarray)):
                if len(pred_raw) == 0:
                    rejected_models[m_type] = "空輸出"
                    continue
                pred_raw = float(pred_raw[0])

            if pred_raw is None or pd.isna(pred_raw):
                rejected_models[m_type] = "NaN"
                continue

            # 未訓練時 orchestrator 回傳 0.0；例外時回傳 current_price。兩者都必須擋掉。
            if float(pred_raw) == 0.0 and m_type in ("LSTM", "Transformer"):
                rejected_models[m_type] = "輸出為 0 (權重缺失)"
                continue

            if self.pipeline and m_type in ("LSTM", "Transformer"):
                pred_real = self.pipeline.decode_prediction(pred_raw, current_price, symbol)
                if pred_real is None:
                    rejected_models[m_type] = "解碼失敗"
                    continue
            else:
                pred_real = float(pred_raw)

            deviation = abs(pred_real - current_price) / current_price
            if pred_real <= 0 or deviation > 0.10:
                rejected_models[m_type] = f"偏離 {deviation * 100:.1f}%"
                logger.warning(
                    f"[{symbol}] 🛡️ 剔除 {m_type}：預測 {pred_real:.2f} vs 現價 "
                    f"{current_price:.2f} (偏差 {deviation * 100:.1f}%)"
                )
                continue

            ensemble_preds[m_type] = pred_real

        # 決策過程的完整留痕：每個模型各自預測了多少、誰被剔除、為什麼
        if ensemble_preds:
            detail = "·".join(
                f"{k} {(v / current_price - 1) * 100:+.2f}%" for k, v in ensemble_preds.items()
            )
        else:
            detail = "無"
        if rejected_models:
            detail += " || 剔除: " + "·".join(f"{k}({v})" for k, v in rejected_models.items())
        logger.debug(f"[{symbol}] 🧠 模型預測 ({term}/{bar_size_str}) 現價 {current_price:.2f} -> {detail}")

        if not ensemble_preds:
            self._record_decision("NO_USABLE_MODEL")
            logger.warning(
                f"⚠️ [{symbol}] 無任何可用模型，維持觀望。"
                f"若 reason 為「未訓練」，請執行 python ibkrpy/core/main.py --mode train {symbol}。"
            )
            return

        _, annual_volatility = self.models.predict(symbol, df_adv, model_type="GARCH")
        if isinstance(annual_volatility, (list, np.ndarray)):
            annual_volatility = float(annual_volatility[0]) if len(annual_volatility) > 0 else 0.15
            
        if pd.isna(annual_volatility) or annual_volatility <= 0:
            annual_volatility = 0.15

        if is_short_term:
            adjusted_volatility = annual_volatility / math.sqrt(252 * 78)
        elif term == "mid_term":
            adjusted_volatility = annual_volatility / math.sqrt(252 * 6.5)
        else:
            adjusted_volatility = annual_volatility / math.sqrt(252)

        strategy = self.strategies.get(symbol)
        if not strategy:
            self._record_decision("NO_STRATEGY")
            logger.error(f"[{symbol}] ❌ 找不到對應的策略物件，本輪跳過。")
            return

        signal = strategy.generate_signal(
            current_price=context["current_price"],
            volatility=adjusted_volatility,  
            regime=context["regime"],
            ensemble_predictions=ensemble_preds
        )

        # 不論成交與否，都把策略的判斷依據寫進日誌。
        decision = getattr(strategy, "last_decision", {}) or {}
        self._record_decision(decision.get("reason_code", "UNKNOWN"))
        describe = getattr(strategy, "describe_last_decision", None)
        trace = describe() if callable(describe) else str(decision)

        if signal:
            logger.info(f"[{symbol}] ✅ 策略決策: {trace}")
        else:
            logger.info(f"[{symbol}] ⏸️ 策略決策: {trace}")
            return

        if signal:
            conviction = 1.0
            target_weight = 0.10 
            if self.market_analyzer:
                analysis = self.market_analyzer.analyze_stock_risk(
                    symbol=symbol, context=self.global_context, action=signal["action"], current_positions=self.cached_positions
                )
                conviction = analysis["conviction_multiplier"]
                target_weight = analysis.get("target_weight", 0.10)
                
                if analysis["warnings"]:
                    logger.warning(f"[{symbol}] 🌍 宏觀警告: {' | '.join(analysis['warnings'])}")

            await self._execute_signal(symbol, signal, current_price, available_funds, net_liquidation, current_pos, conviction, target_weight)

    async def _execute_signal(self, symbol: str, signal: Dict[str, Any], current_price: float, available_funds: float, net_liquidation: float, current_pos: float, conviction: float = 1.0, target_weight: float = 0.10):
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
                    logger.warning(f"⚠️ [{symbol}] 已達組合總曝險上限 ({max_gross*100:.0f}%)，本次不建立新倉。")
                    return 0
                budget = min(budget, room)

            return int(budget / current_price) if current_price > 0 else 0

        if action == "BUY":
            if current_pos > 0:
                self._record_decision("ALREADY_LONG")
                logger.info(f"[{symbol}] ⏸️ 已持有多單 {current_pos:+.0f} 股，BUY 訊號不重複加倉。")
                return
            elif current_pos < 0:
                trade_quantity = int(abs(current_pos))
                is_closing_only = True
            else:
                trade_quantity = _affordable_qty()

        elif action == "SELL":
            if current_pos < 0:
                self._record_decision("ALREADY_SHORT")
                logger.info(f"[{symbol}] ⏸️ 已持有空單 {current_pos:+.0f} 股，SELL 訊號不重複加倉。")
                return
            elif current_pos > 0:
                trade_quantity = int(current_pos)
                is_closing_only = True
            else:
                if not allow_shorting:
                    self._record_decision("SHORTING_DISABLED")
                    logger.info(
                        f"[{symbol}] ⏸️ 產生 SELL 訊號但無持倉，且 allow_shorting=False，"
                        f"已阻擋做空。"
                    )
                    return
                trade_quantity = _affordable_qty()

        if trade_quantity <= 0:
            self._record_decision("NO_BUYING_POWER")
            budget = min(net_liquidation * final_weight, available_funds * 0.95)
            logger.warning(
                f"[{symbol}] ⚠️ 有 {action} 訊號但可下單股數為 0。"
                f"目標權重 {final_weight * 100:.1f}% / 淨值 ${net_liquidation:,.0f} / "
                f"可用資金 ${available_funds:,.0f} / 預算 ${budget:,.0f} / 現價 {current_price:.2f}。"
            )
            return

        # 預設最低建倉門檻
        min_trade_usd = self.config.get("strategy_settings.min_trade_usd", 500.0) 
        trade_value = trade_quantity * current_price
        
        # 注意：若是「平倉單 (is_closing_only)」，就算僅剩 1 股也必須無條件出清，因此排除在此檢查外。
        if not is_closing_only and trade_value < min_trade_usd:
            self._record_decision("BELOW_MIN_TRADE")
            logger.info(
                f"[{symbol}] ⏸️ 有 {action} 訊號但建倉總值 ${trade_value:.2f} "
                f"低於最小經濟門檻 ${min_trade_usd:.2f}，取消以免手續費耗損。"
            )
            return
            
        logger.info(f"[{symbol}] 🎯 準備執行 ({term_name}): {action} {trade_quantity} 股 @ 市價約 {current_price:.2f} (動態分配權重: {final_weight*100:.1f}%)")
        logger.info(f"	=> [安全防護] 停損單(STP)設定於: {sl_price:.2f} | 停利單(LMT)設定於: {tp_price:.2f}")
        
        contract = await self._get_qualified_contract(symbol)
        if contract is None:
            self._record_decision("CONTRACT_UNRESOLVED")
            logger.error(f"[{symbol}] ❌ 下單前合約解析失敗，已中止本次交易。")
            return

        self._record_decision(f"ORDER_{action}")
        try:
            # 1. 下單前強制清理歷史孤兒單
            await self._cancel_open_orders(symbol)
            
            if not is_closing_only:
                self.cached_funds = max(0.0, self.cached_funds - trade_value)

            if self.dry_run:
                logger.info(f"[{symbol}] 🛡️ [Dry-Run] 虛擬下單成功！")
            else:
                algo_params = [TagValue('adaptivePriority', 'Normal')]
                
                # 2. 定義最大容忍滑價 (0.2%)
                slippage_buffer = 0.002
                limit_entry_price = current_price * (1 + slippage_buffer) if action == "BUY" else current_price * (1 - slippage_buffer)

                if is_closing_only:
                    # 提早平倉
                    order = LimitOrder(action, trade_quantity, round(limit_entry_price, 2))
                    order.algoStrategy = 'Adaptive'
                    order.algoParams = algo_params
                    order.tif = 'DAY'
                    self.data.ib.placeOrder(contract, order)
                else:
                    # 全新開倉
                    parent_id = self.data.ib.client.getReqId()
                    parent = LimitOrder(action, trade_quantity, round(limit_entry_price, 2))
                    parent.algoStrategy = 'Adaptive'
                    parent.algoParams = algo_params
                    parent.orderId = parent_id
                    parent.tif = 'DAY'
                    parent.transmit = False
                    
                    rev_action = "SELL" if action == "BUY" else "BUY"
                    sl_order = Order(action=rev_action, totalQuantity=trade_quantity, orderType="STP", auxPrice=round(sl_price, 2), parentId=parent_id, tif='GTC', transmit=False)
                    tp_order = LimitOrder(action=rev_action, totalQuantity=trade_quantity, lmtPrice=round(tp_price, 2), parentId=parent_id, tif='GTC', transmit=True)
                    
                    self.data.ib.placeOrder(contract, parent)
                    self.data.ib.placeOrder(contract, sl_order)
                    self.data.ib.placeOrder(contract, tp_order)

            if self.db:
                reason = "AI反向平倉" if is_closing_only else f"AI建倉 ({term_name} | Alloc:{final_weight*100:.1f}%)"
                await self.db.log_trade({"symbol": symbol, "action": action, "quantity": trade_quantity, "price": current_price, "regime": regime_name, "reason": ("[虛擬] " if self.dry_run else "") + reason})
        except Exception as e:
            logger.critical(f"❌ [{symbol}] 下單過程發生錯誤: {e}")