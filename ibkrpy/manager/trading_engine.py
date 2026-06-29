# ibkrpy/manager/trading_engine.py
# 系統交易引擎：量化系統的主心臟。單向處理 數據 -> 宏觀全局 -> 預測 -> 策略 -> 執行

import asyncio
import math
import time
from typing import Dict, Any, List
from ib_insync import Order, MarketOrder, LimitOrder, Stock, TagValue
import pandas as pd
import numpy as np

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

    def _get_dynamic_benchmark(self, symbol: str) -> str:
        """根據資產的標籤 (Tags) 動態選擇最適合的基準大盤 (Sector Benchmark)"""
        default_bench = self.config.get("general_settings.benchmark_symbol", "SPY")
        
        # 尋找該資產的設定設定
        asset_profile = next((p for p in self.config.asset_profiles if p.symbol == symbol), None)
        if not asset_profile or not asset_profile.tags:
            return default_bench
            
        tags = [t.upper() for t in asset_profile.tags]
        
        # 根據特性標籤匹配最佳大盤 ETF
        if any(t in tags for t in ["TECH", "SOFTWARE", "SEMICONDUCTOR", "CLOUD", "AI", "CYBERSECURITY"]):
            return "QQQ"  # 納斯達克 100 (科技成長股)
        elif "FINANCIALS" in tags or "BANKING" in tags:
            return "XLF"  # 金融板塊 ETF
        elif "ENERGY" in tags or "OIL" in tags:
            return "XLE"  # 能源板塊 ETF
        elif "HEALTHCARE" in tags or "PHARMACEUTICALS" in tags:
            return "XLV"  # 醫療保健板塊 ETF
        elif "CONSUMER_STAPLES" in tags or "RETAIL" in tags:
            return "XLP"  # 必需消費品板塊 ETF
        elif "UTILITIES" in tags:
            return "XLU"  # 公用事業板塊 ETF
        elif "INDUSTRIALS" in tags:
            return "XLI"  # 工業板塊 ETF
            
        return default_bench

    async def update_system_state(self):
        """每一輪大迴圈開始前統一調用，更新帳戶快取與全局市場上下文"""
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
            print(f"⚠️ 獲取帳戶狀態失敗: {e}，將使用前次快取數據。")

        if self.market_analyzer:
            self.global_context = await asyncio.to_thread(self.market_analyzer.get_global_context)

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
            has_protection = False
            for t in open_trades:
                if t.contract.symbol == symbol and t.order.action == protect_action:
                    has_protection = True
                    break
                    
            if not has_protection:
                print(f"\n[守護神] 🛡️ 偵測到 {symbol} 存在無保護持倉 ({pos_qty} 股)，準備自動掛載 OCA 停損停利單...")
                await self._attach_oca_protection(symbol, pos_qty)

    async def _attach_oca_protection(self, symbol: str, pos_qty: float):
        contract = Stock(symbol, "SMART", "USD")
        try:
            await self.data.ib.qualifyContractsAsync(contract)
            
            # 獲取日 K 以計算真實波動率
            df = await self.data.fetch_historical_data(contract, duration='60 D', bar_size='1 day')
            if df.empty: return
            
            df.rename(columns={'open': 'Open', 'high': 'High', 'low': 'Low', 'close': 'Close', 'volume': 'Volume'}, inplace=True)
            
            current_price = float(df['Close'].iloc[-1])
            returns = np.log(df['Close'] / df['Close'].shift(1)).dropna()
            annual_vol = returns.std() * np.sqrt(252) if len(returns) > 10 else 0.20
            daily_vol = annual_vol / math.sqrt(252)
            
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
                
            # 建立 OCA 群組標籤 (加上時間戳確保唯一性)
            oca_group = f"OCA_PROTECT_{symbol}_{int(time.time())}"
            
            # 建立獨立的 STP 與 LMT 單，並透過 ocaGroup 綁定。ocaType=1 代表觸發其一即取消另一
            sl_order = Order(action=action, totalQuantity=abs(pos_qty), orderType="STP", auxPrice=round(sl_price, 2), tif='GTC', ocaGroup=oca_group, ocaType=1)
            tp_order = LimitOrder(action=action, totalQuantity=abs(pos_qty), lmtPrice=round(tp_price, 2), tif='GTC', ocaGroup=oca_group, ocaType=1)
            
            self.data.ib.placeOrder(contract, sl_order)
            self.data.ib.placeOrder(contract, tp_order)
            
            print(f"[{symbol}] ✅ 成功掛載手動保護傘 (OCA) -> 停損(STP): {sl_price:.2f}, 停利(LMT): {tp_price:.2f}")
        except Exception as e:
            print(f"[{symbol}] ❌ 掛載保護傘失敗: {e}")

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
            print(f"[{symbol}] 🧹 已清除 {canceled_count} 筆歷史未決訂單 (防範孤兒單衝突)。")
            # 稍微等待 IBKR 系統同步取消狀態
            await asyncio.sleep(0.5)

    async def run_tick(self, symbol: str):
        print(f"\n[{symbol}] 啟動實盤決策迴圈...")
        
        available_funds = self.cached_funds
        net_liquidation = self.cached_net_liq
        current_pos = self.cached_positions.get(symbol, 0.0)
        
        contract = Stock(symbol, "SMART", "USD")
        await self.data.ib.qualifyContractsAsync(contract)
        
        term = self.symbol_terms.get(symbol, "long_term")
        
        if term == "short_term":
            bar_size_str = self.config.get("general_settings.short_term_bar_size", "5 mins")
            duration_str = self.config.get("general_settings.short_term_duration", "60 D")
        elif term == "mid_term":
            bar_size_str = self.config.get("general_settings.mid_term_bar_size", "1 hour")
            duration_str = self.config.get("general_settings.mid_term_duration", "180 D")
        else:
            bar_size_str = self.config.get("general_settings.long_term_bar_size", "1 day")
            duration_str = self.config.get("general_settings.long_term_duration", "2 Y")
            
        is_short_term = (term == "short_term")
            
        df = await self.data.fetch_historical_data(
            contract=contract, duration=duration_str, bar_size=bar_size_str, what_to_show='TRADES'
        )
        
        if df.empty or len(df) < 60:
            print(f"[{symbol}] ⚠️ 獲取 {term} ({bar_size_str}) 實時 K 線失敗或數據量不足。")
            return

        df.index = pd.to_datetime(df.index)

        df.rename(columns={'open': 'Open', 'high': 'High', 'low': 'Low', 'close': 'Close', 'volume': 'Volume'}, inplace=True)
        if self.db:
            self.db.save_bulk_market_data(symbol, df.tail(300))
        current_price = float(df['Close'].iloc[-1])
        
        # 自動為資產配對對應板塊的大盤指標 (Alpha 萃取)
        benchmark_symbol = self._get_dynamic_benchmark(symbol)
        bench_df = pd.DataFrame()
        try:
            if benchmark_symbol == symbol:
                # 標的本身就是大盤 (例如 QQQ)，避免重複抓取浪費 API 配額
                bench_df = df.copy()
            else:
                bench_contract = Stock(benchmark_symbol, "SMART", "USD")
                await self.data.ib.qualifyContractsAsync(bench_contract)
                bench_df_raw = await self.data.fetch_historical_data(
                    contract=bench_contract, duration=duration_str, bar_size=bar_size_str, what_to_show='TRADES'
                )
                if not bench_df_raw.empty:
                    bench_df_raw.index = pd.to_datetime(bench_df_raw.index)
                    bench_df_raw.rename(columns={'open': 'Open', 'high': 'High', 'low': 'Low', 'close': 'Close', 'volume': 'Volume'}, inplace=True)
                    bench_df = bench_df_raw.reindex(df.index, method='ffill').bfill()
        except Exception as e:
            print(f"[{symbol}] ⚠️ 獲取基準指標 Benchmark ({benchmark_symbol}) 失敗: {e}")

        macro_dict = {}
        if self.ext:
            vix_series = await self.ext.fetch_fred_series("VIXCLS")
            if vix_series is not None and not vix_series.empty:
                vix_daily = vix_series.copy()
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
            
            scale_cols = ['Open', 'High', 'Low', 'Close', 'Volume']
            df_scaled = self.pipeline.transform_scale(df_adv, columns=scale_cols, symbol=symbol)
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
            print(f"[{symbol}] 🛡️ 交易系統拒絕進場: {reason}")
            return

        ensemble_preds = {}
        target_models = ["LSTM", "Transformer", "ARIMA"]
        
        for m_type in target_models:
            pred_raw, _ = self.models.predict(symbol, df_scaled if m_type != "ARIMA" else df_adv, model_type=m_type)
            
            if isinstance(pred_raw, (list, np.ndarray)):
                if len(pred_raw) > 0:
                    pred_raw = float(pred_raw[0])
                else:
                    continue
                    
            if pd.isna(pred_raw):
                continue
                
            pred_real = None
            if self.pipeline and m_type in ["LSTM", "Transformer"]:
                try:
                    pred_real_arr = self.pipeline.inverse_transform_scale(pred_raw, "Close", symbol)
                    if isinstance(pred_real_arr, (list, np.ndarray)):
                        pred_real = float(pred_real_arr[0])
                    else:
                        pred_real = float(pred_real_arr)
                except Exception:
                    try:
                        scaler = self.pipeline.scalers.get(symbol)
                        if scaler:
                            c_min = scaler.data_min_[3]
                            c_max = scaler.data_max_[3]
                            pred_real = float(pred_raw) * (c_max - c_min) + c_min
                    except Exception:
                        pass
            else:
                pred_real = float(pred_raw)
                
            if pred_real is None:
                continue
                
            deviation = abs(pred_real - current_price) / current_price
            if pred_real <= 0 or deviation > 0.10:
                print(f"[{symbol}] 🛡️ 剔除 {m_type} 嚴重偏離或失效之預測 (預測: {pred_real:.2f} | 現價: {current_price:.2f} | 偏差: {deviation*100:.1f}%)")
                continue
                
            ensemble_preds[m_type] = pred_real

        if not ensemble_preds:
            print(f"[{symbol}] ⚠️ 所有模型預測皆失效或觸發 10% 偏差安全網，強制維持觀望 (HOLD)。")
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
        if not strategy: return

        signal = strategy.generate_signal(
            current_price=context["current_price"],
            volatility=adjusted_volatility,  
            regime=context["regime"],
            ensemble_predictions=ensemble_preds
        )

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
                    print(f"[{symbol}] 🌍 宏觀警告: {' | '.join(analysis['warnings'])}")

            await self._execute_signal(symbol, signal, current_price, available_funds, net_liquidation, current_pos, conviction, target_weight)
        else:
            print(f"[{symbol}] ⏸️ 策略判斷維持觀望 (HOLD)。")

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
        
        if action == "BUY":
            if current_pos > 0: return
            elif current_pos < 0:
                trade_quantity = int(abs(current_pos))
                is_closing_only = True
            else:
                target_cash = net_liquidation * final_weight
                trade_quantity = int(min(target_cash, available_funds * 0.95) / current_price)
                
        elif action == "SELL":
            if current_pos < 0: return
            elif current_pos > 0:
                trade_quantity = int(current_pos)
                is_closing_only = True
            else:
                if not allow_shorting:
                    print(f"[{symbol}] ⚠️ 已阻擋做空指令，維持觀望。")
                    return
                target_cash = net_liquidation * final_weight
                trade_quantity = int(min(target_cash, available_funds * 0.95) / current_price)

        if trade_quantity <= 0: return
        
        # 預設最低建倉門檻為 $500 美元 (可透過 config.yaml 的 min_trade_usd 配置調整)
        min_trade_usd = self.config.get("strategy_settings.min_trade_usd", 500.0) 
        trade_value = trade_quantity * current_price
        
        # 注意：若是「平倉單 (is_closing_only)」，就算僅剩 1 股也必須無條件出清，因此排除在此檢查外。
        if not is_closing_only and trade_value < min_trade_usd:
            print(f"[{symbol}] ⚠️ 預期建倉總值 (${trade_value:.2f}) 低於最小經濟門檻 (${min_trade_usd:.2f})，為防範手續費耗損，取消本次交易。")
            return
            
        print(f"[{symbol}] 🎯 準備執行 ({term_name}): {action} {trade_quantity} 股 @ 市價約 {current_price:.2f} (動態分配權重: {final_weight*100:.1f}%)")
        print(f"      => [安全防護] 停損單(STP)設定於: {sl_price:.2f} | 停利單(LMT)設定於: {tp_price:.2f}")
        
        contract = Stock(symbol, "SMART", "USD")
        try:
            # 1. 下單前強制清理歷史孤兒單
            await self._cancel_open_orders(symbol)
            
            if self.dry_run:
                print(f"[{symbol}] 🛡️ [Dry-Run] 虛擬下單成功！")
            else:
                algo_params = [TagValue('adaptivePriority', 'Normal')]
                
                # 2. 定義最大容忍滑價 (0.2%)，取代無底線的市價單
                slippage_buffer = 0.002
                limit_entry_price = current_price * (1 + slippage_buffer) if action == "BUY" else current_price * (1 - slippage_buffer)

                if is_closing_only:
                    # 提早平倉：僅發送單一限價單即可 (舊的停損停利已在上方被清除)
                    order = LimitOrder(action, trade_quantity, round(limit_entry_price, 2))
                    order.algoStrategy = 'Adaptive'
                    order.algoParams = algo_params
                    order.tif = 'DAY'
                    self.data.ib.placeOrder(contract, order)
                else:
                    # 全新開倉：發送帶有防護機制的 Bracket Order
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
            print(f"[{symbol}] ❌ 下單過程發生錯誤: {e}")