# ibkrpy/manager/pipeline_manager.py
# 負責將原本 scripts/ 中的下載、資料庫封裝與訓練邏輯模組化

import os
import asyncio
import pandas as pd
import json
import joblib
import numpy as np
from datetime import datetime
from typing import Tuple

from ib_insync import Stock

from ibkrpy.shared.config_manager import ConfigManager
from ibkrpy.shared.db_manager import DatabaseManager
from ibkrpy.data.data_pipeline import DataPipeline
from ibkrpy.data.ibkr_data_manager import IBKRDataManager
from ibkrpy.data.external_data import ExternalDataFetcher
from ibkrpy.evaluation.model_tuner import ModelTuner

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
WEIGHTS_DIR = os.path.join(PROJECT_ROOT, "weights")

class PipelineManager:
    """整合資料抓取、特徵工程與 AI 模型重訓的管線管理器"""
    
    def __init__(self, config: ConfigManager, db: DatabaseManager, pipeline: DataPipeline, ib_data: IBKRDataManager, ext_fetcher: ExternalDataFetcher):
        self.config = config
        self.db = db
        self.pipeline = pipeline
        self.ib_data = ib_data
        self.ext = ext_fetcher
        
        self.benchmark_symbol = self.config.get("general_settings.benchmark_symbol", "SPY")
        self.symbols = [p.symbol for p in self.config.asset_profiles] if self.config.asset_profiles else ["AAPL"]
        if self.benchmark_symbol not in self.symbols:
            self.symbols.insert(0, self.benchmark_symbol)
            
        self.symbol_terms = {p.symbol: p.term for p in self.config.asset_profiles} if self.config.asset_profiles else {}
        if self.benchmark_symbol not in self.symbol_terms:
            self.symbol_terms[self.benchmark_symbol] = "long_term"

    def _get_term_settings(self, term: str) -> dict:
        """根據交易週期 (term) 返回最適合的 K線級別、總下載天數、與單次分批天數"""
        if term == "short_term":
            return {"bar_size": self.config.get("general_settings.short_term_bar_size", "5 mins"), "total_days": 150, "chunk_days": 15}
        elif term == "mid_term":
            return {"bar_size": self.config.get("general_settings.mid_term_bar_size", "1 hour"), "total_days": 360, "chunk_days": 60}
        else: 
            return {"bar_size": self.config.get("general_settings.long_term_bar_size", "1 day"), "total_days": 730, "chunk_days": 365}

    def _train_dl_models(self, symbol: str, df: pd.DataFrame, bench_df: pd.DataFrame, macro_data: dict):
        """訓練深度學習模型（LSTM、Transformer）並儲存 .keras"""
        print(f"🔍 [{symbol}] 開始訓練深度學習模型...")
        print(f"[{symbol}] 資料量: {len(df)} 筆，特徵數量: {df.shape[1]} 欄")
        print(f"[{symbol}] 基準資料量: {len(bench_df)} 筆， 宏觀數據量: {len(macro_data)} 筆")
        if df.empty or len(df) < 60:
            print(f"⚠️ [{symbol}] 資料量極度不足 (僅 {len(df)} 筆)，跳過 DL 訓練。")
            return

        os.makedirs(WEIGHTS_DIR, exist_ok=True)
        
        df_adv = self.pipeline.engineer_advanced_features(df, bench_df, macro_data)
        df_adv = df_adv.ffill().bfill().fillna(0)
        
        scale_cols = ['Open', 'High', 'Low', 'Close', 'Volume']
        self.pipeline.fit_scale(df_adv, scale_cols, symbol)
        df_scaled = self.pipeline.transform_scale(df_adv, scale_cols, symbol)
        
        look_back = 60
        X, y = self.pipeline.create_sequences(df_scaled, scale_cols, 'Close', look_back)
        
        print(f"      => 總 K 線數: {len(df_adv)} 筆，產出有效訓練序列: {len(X)} 組")
        
        if len(X) < 16:
            print(f"⚠️ [{symbol}] 有效序列數量不足以支撐梯度下降 (僅 {len(X)} 組)，跳過 DL 訓練。")
            return

        dynamic_batch_size = min(32, max(8, len(X) // 4))

        try:
            from keras.callbacks import EarlyStopping
            from ibkrpy.models.lstm import LSTMModel
            from ibkrpy.models.transformer import TransformerModel

            callbacks = [EarlyStopping(monitor='loss', patience=5, restore_best_weights=True)]

            print(f"   -> 🚀 擬合 LSTM 模型 (Batch Size: {dynamic_batch_size})...")
            lstm = LSTMModel(look_back=look_back, feature_cols=scale_cols, weights_dir=WEIGHTS_DIR)
            lstm.model = lstm._build_model()
            lstm.model.fit(X, y, epochs=25, batch_size=dynamic_batch_size, verbose=0, callbacks=callbacks)
            lstm_path = os.path.join(WEIGHTS_DIR, f"{symbol}_LSTM.keras")
            lstm.model.save(lstm_path)
            
            if os.path.exists(lstm_path): print(f"      ✅ [成功] LSTM 權重已實體寫入: {lstm_path}")
            else: print(f"      ❌ [失敗] LSTM 寫入異常！")

            print(f"   -> 🚀 擬合 Transformer 模型 (Batch Size: {dynamic_batch_size})...")
            transformer = TransformerModel(look_back=look_back, feature_cols=scale_cols, weights_dir=WEIGHTS_DIR)
            transformer.model = transformer._build_model()
            transformer.model.fit(X, y, epochs=25, batch_size=dynamic_batch_size, verbose=0, callbacks=callbacks)
            tf_path = os.path.join(WEIGHTS_DIR, f"{symbol}_Transformer.keras")
            transformer.model.save(tf_path)
            
            if os.path.exists(tf_path): print(f"      ✅ [成功] Transformer 權重已實體寫入: {tf_path}")
            else: print(f"      ❌ [失敗] Transformer 寫入異常！")
            
            print(f"✅ [{symbol}] 深度學習模型訓練完畢。\n")
        except ImportError:
            print(f"⚠️ 尚未安裝 TensorFlow/Keras，跳過深度學習訓練。")
        except Exception as e:
            print(f"❌ [{symbol}] DL 模型訓練失敗: {e}")

    def _train_safe_models(self, symbol: str, df: pd.DataFrame):
        """訓練統計與狀態模型（ARIMA、GARCH、HMM）並打包成單一 .pkl 檔案"""
        print(f"🔍 [{symbol}] 開始訓練統計與狀態模型...")
        if df.empty or len(df) < 50:
            print(f"⚠️ [{symbol}] 資料量不足以進行有效訓練，跳過統計模型。")
            return
            
        weights_dir = os.path.abspath("weights")
        os.makedirs(weights_dir, exist_ok=True)
        
        classical_bundle = {}

        print(f"   -> 擬合 ARIMA 模型...")
        try:
            from statsmodels.tsa.arima.model import ARIMA
            series = df['Close'].dropna().values
            model_arima = ARIMA(series, order=(5, 1, 0))
            res_arima = model_arima.fit()
            classical_bundle['arima'] = res_arima
            print(f"      ✅ ARIMA 模型訓練完成")
        except Exception as e: print(f"   ⚠️ ARIMA 訓練失敗: {e}")

        print(f"   -> 擬合 GARCH 模型...")
        try:
            from arch import arch_model
            returns = np.log(df['Close'] / df['Close'].shift(1)).dropna() * 100.0
            if len(returns) > 20:
                am = arch_model(returns, vol='Garch', p=1, q=1, dist='normal')
                res_garch = am.fit(disp='off')
                classical_bundle['garch'] = res_garch.params
                print(f"      ✅ GARCH 模型訓練完成")
        except Exception as e: print(f"   ⚠️ GARCH 訓練失敗: {e}")

        print(f"   -> 擬合 HMM 模型...")
        try:
            from hmmlearn.hmm import GaussianHMM
            df_features = pd.DataFrame(index=df.index)
            df_features['log_return'] = np.log(df['Close'] / df['Close'].shift(1)) * 100.0
            df_features['volatility'] = df_features['log_return'].rolling(window=5).std()
            df_features = df_features.dropna()
            X_hmm = df_features[['log_return', 'volatility']].values
            if len(X_hmm) > 20:
                hmm = GaussianHMM(n_components=2, covariance_type="full", n_iter=100)
                hmm.fit(X_hmm)
                classical_bundle['hmm'] = hmm
                print(f"      ✅ HMM 模型訓練完成")
        except Exception as e: print(f"   ⚠️ HMM 訓練失敗: {e}")
        
        # 將三大傳統模型打包為一個檔案，大幅降低輸出檔案數量
        bundle_path = os.path.join(weights_dir, f"{symbol}_classical.pkl")
        joblib.dump(classical_bundle, bundle_path)
        
        if os.path.exists(bundle_path):
            print(f"✅ [{symbol}] 統計模型整合包 (Classical Bundle) 寫入完成。")

    def _run_optuna_optimization(self, symbol: str, df: pd.DataFrame, bench_df: pd.DataFrame, macro_data: dict, term: str) -> Tuple[dict, float]:
        """使用 Optuna 進行參數最佳化，並回傳 (最佳參數, 複合評分)"""
        tuner = ModelTuner(model_orchestrator=None, data_manager=None)
        precomputed_data = df.copy()
        if not precomputed_data.empty:
            np.random.seed(42)
            precomputed_data['prediction'] = precomputed_data['Close'] * (1 + np.random.normal(0, 0.005, len(precomputed_data)))
            
            # [修正 1] 波動率必須使用 Bar 級別，不可乘上 np.sqrt(252) 年化，否則短線停損跨度會嚴重失真
            bar_volatility = precomputed_data['Close'].pct_change().rolling(20).std().fillna(0.005)
            precomputed_data['volatility'] = bar_volatility.replace(0, 0.005)
            
            # [修正 2] 移除寫死的 'SIDEWAYS_QUIET'，改以真實機率分佈隨機生成市場狀態，解開短線封殺令
            regimes = ['BULL_TREND', 'BEAR_TREND', 'SIDEWAYS_VOLATILE', 'SIDEWAYS_QUIET']
            precomputed_data['regime'] = np.random.choice(regimes, len(precomputed_data), p=[0.3, 0.3, 0.3, 0.1])
            
        try:
            best_params, best_score = tuner.optimize_strategy_params(symbol, df, precomputed_data, n_trials=20, term=term)
            return best_params, best_score
        except Exception as e:
            print(f"⚠️ [{symbol}] {term} Optuna 尋優過程發生錯誤: {e}")
            default_params = {
                "min_prediction_threshold_pct": 0.005,
                "volatility_stop_loss_multiplier": 2.0,
                "volatility_take_profit_multiplier": 3.0
            }
            return default_params, -999.0
    
    async def run_data_ingestion(self):
        """階段一：增量下載資料並寫入資料庫，實現資料集分批次遞增擴充 (Data Lake 模式)"""
        print("\n" + "="*60)
        print(" [Pipeline] 啟動資料增量下載與資料庫同步")
        print("="*60)
        
        all_terms = ["long_term", "mid_term", "short_term"]
        
        for symbol in self.symbols:
            print(f"\n[{symbol}] 檢核並同步最新市場資料...")
            try:
                contract = Stock(symbol, "SMART", "USD")
                await self.ib_data.ib.qualifyContractsAsync(contract)
                
                for term in all_terms:
                    settings = self._get_term_settings(term)
                    bar_size = settings["bar_size"]
                    total_days = settings["total_days"]
                    chunk_days = settings["chunk_days"]
                    
                    print(f"   -> 🔄 正在同步 {term} ({bar_size}) 資料...")
                    df_existing = self.db.get_market_data_sync(symbol, timeframe=bar_size)
                    days_to_fetch = 0
                    
                    if df_existing.empty or len(df_existing) < 50:
                        days_to_fetch = total_days
                    else:
                        last_date = pd.Timestamp(df_existing.index[-1])
                        now_date = pd.Timestamp.now()
                        
                        bus_days_diff = np.busday_count(last_date.date(), now_date.date())
                        
                        if bus_days_diff <= 0:
                            print(f"      ✅ 資料已是最新狀態，無須同步。")
                            continue
                            
                        days_to_fetch = min(bus_days_diff + 2, total_days)

                    df_new_list = []
                    remaining_days = days_to_fetch
                    current_end_date = datetime.now()
                    
                    while remaining_days > 0:
                        fetch_days = min(remaining_days, chunk_days)
                        duration_str = f"{fetch_days} D"
                        end_date_str = current_end_date.strftime('%Y%m%d %H:%M:%S')
                        
                        df_chunk = pd.DataFrame()
                        attempts = 0
                        
                        while attempts < 3 and df_chunk.empty:
                            if attempts > 0:
                                await asyncio.sleep(5)
                                
                            df_chunk = await self.ib_data.fetch_historical_data(
                                contract=contract,
                                end_datetime=end_date_str,
                                duration=duration_str,
                                bar_size=bar_size,
                                what_to_show='TRADES'
                            )
                            attempts += 1
                            
                        if not df_chunk.empty:
                            df_chunk.index = pd.to_datetime(df_chunk.index, utc=True)
                            df_new_list.append(df_chunk)
                            first_dt = df_chunk.index[0]
                            if isinstance(first_dt, str): first_dt = pd.to_datetime(first_dt, utc=True)
                            current_end_date = first_dt
                        else:
                            break
                            
                        remaining_days -= fetch_days
                        await asyncio.sleep(2) 
                    
                    if df_new_list:
                        df_new = pd.concat(df_new_list)
                        df_new.index = pd.to_datetime(df_new.index, utc=True)
                        df_new.sort_index(inplace=True)
                        df_new = df_new[~df_new.index.duplicated(keep='last')]
                        df_new.rename(columns={'open': 'Open', 'high': 'High', 'low': 'Low', 'close': 'Close', 'volume': 'Volume'}, inplace=True)
                        cols_to_keep = ['Open', 'High', 'Low', 'Close', 'Volume']
                        df_new = df_new[[c for c in cols_to_keep if c in df_new.columns]]

                        if not df_existing.empty:
                            df_combined = pd.concat([df_existing, df_new])
                            df_combined = df_combined[~df_combined.index.duplicated(keep='last')]
                            df_combined.sort_index(inplace=True)
                            self.db.save_bulk_market_data(symbol, df_combined, timeframe=bar_size)
                        else:
                            self.db.save_bulk_market_data(symbol, df_new, timeframe=bar_size)
                            
                        print(f"      ✅ 成功合併並寫入 {len(df_new)} 筆 {bar_size} K線。")
                    else:
                        print(f"      ❌ [{symbol}] 該週期所有分批資料獲取皆失敗。")
                    
            except Exception as e:
                print(f"⚠️ [{symbol}] 資料更新發生例外錯誤: {e}")
            
            await asyncio.sleep(2)

    async def run_training_and_tuning(self):
        """階段二：多週期選拔 (Tournament-based Selection)，淘汰弱勢週期，適應並訓練最佳模型"""
        print("\n" + "="*60)
        print(" [Pipeline] 啟動多週期選拔 (Term Tournament) 與 AI 訓練")
        print("="*60)
        
        all_terms = ["long_term", "mid_term", "short_term"]
        
        for symbol in self.symbols:
            if symbol == self.benchmark_symbol:
                continue
                
            print(f"\n🔥 啟動週期選拔與訓練任務: {symbol} 🔥")
            
            best_term = None
            best_score = -float('inf')
            best_params = {}
            best_df = pd.DataFrame()
            best_macro = {}
            
            for term in all_terms:
                settings = self._get_term_settings(term)
                bar_size = settings["bar_size"]

                df = self.db.get_market_data_sync(symbol, timeframe=bar_size)
                if df.empty or len(df) < 100:
                    continue
                    
                last_date = pd.Timestamp(df.index[-1]).normalize()
                now_date = pd.Timestamp.now().normalize()
                
                stale_days = np.busday_count(last_date.date(), now_date.date())
                if stale_days > 5:
                    continue
                
                bench_df = self.db.get_market_data_sync(self.benchmark_symbol, timeframe=bar_size)
                if not bench_df.empty:
                    bench_df = bench_df.reindex(df.index, method='ffill').bfill()
                
                macro_data = {}
                if self.ext:
                    vix_series = await self.ext.fetch_fred_series("VIXCLS")
                    if not vix_series.empty:
                        if getattr(vix_series.index, 'tz', None) is not None:
                            vix_series.index = vix_series.index
                        vix_daily = vix_series.copy()
                        vix_daily.index = vix_daily.index.normalize()
                        
                        df_idx_naive = df.index if getattr(df.index, 'tz', None) is not None else df.index
                        vix_aligned_values = df_idx_naive.normalize().map(vix_daily)
                        
                        vix_aligned = pd.Series(vix_aligned_values, index=df.index).ffill().bfill()
                        
                        if vix_aligned.isna().all():
                            vix_aligned = pd.Series(20.0, index=df.index)
                            
                        macro_data['VIX'] = vix_aligned

                print(f"\n   -> ⏳ 正在評估 {term} 策略潛力...")
                params, score = self._run_optuna_optimization(symbol, df, bench_df, macro_data, term)
                print(f"      => {term} 複合評分預期: {score:.2f}")
                
                if score > best_score:
                    best_score = score
                    best_term = term
                    best_params = params
                    best_df = df
                    best_macro = macro_data

            if best_term is None:
                print(f"❌ [{symbol}] 所有週期皆無法通過評估，強制終止此標的之訓練。")
                continue
            
            print(f"\n🎉 [{symbol}] 選拔結束！冠軍週期為: {best_term} (得分: {best_score:.2f})")
            
            best_params['term'] = best_term
            
            bench_df = self.db.get_market_data_sync(self.benchmark_symbol, timeframe=self._get_term_settings(best_term)["bar_size"])
            if not bench_df.empty:
                bench_df = bench_df.reindex(best_df.index, method='ffill').bfill()
            
            self._train_dl_models(symbol, best_df, bench_df, best_macro)
            self._train_safe_models(symbol, best_df)
            
            weights_dir = os.path.abspath("weights")
            os.makedirs(weights_dir, exist_ok=True)
            
            # 將最佳參數整合至 global_best_params.json
            param_path = os.path.join(weights_dir, "global_best_params.json")
            global_params = {}
            if os.path.exists(param_path):
                try:
                    with open(param_path, 'r', encoding='utf-8') as f:
                        global_params = json.load(f)
                except Exception: pass
                
            global_params[symbol] = best_params
            
            with open(param_path, 'w', encoding='utf-8') as f:
                json.dump(global_params, f, indent=4)
                
            print(f"      ✅ 最佳策略參數已更新至全域檔案: {param_path}")
            self.symbol_terms[symbol] = best_term
            print(f"🏆 {symbol} ({best_term}) 模型訓練與參數尋優徹底完成。")

    async def run_autopilot(self):
        """一鍵全自動執行"""
        await self.run_data_ingestion()
        await self.run_training_and_tuning()