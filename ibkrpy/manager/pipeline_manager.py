# ibkrpy/manager/pipeline_manager.py
# 負責將原本 scripts/ 中的下載、資料庫封裝與訓練邏輯模組化

import os
import asyncio
import pandas as pd
import json
import joblib
import numpy as np
import time
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
DATA_DIR = os.path.join(PROJECT_ROOT, "data")

class PipelineManager:
    """整合資料抓取、特徵工程與 AI 模型重訓的管線管理器"""
    
    def __init__(self, config: ConfigManager, db: DatabaseManager, pipeline: DataPipeline, ib_data: IBKRDataManager, ext_fetcher: ExternalDataFetcher, target_symbol: str = None):
        self.config = config
        self.db = db
        self.pipeline = pipeline
        self.ib_data = ib_data
        self.ext = ext_fetcher
        
        self.benchmark_symbol = self.config.get("general_settings.benchmark_symbol", "SPY")
        
        # 如果有指定單一標的，則直接覆蓋 self.symbols
        if target_symbol:
            self.symbols = [target_symbol]
        else:
            self.symbols = [p.symbol for p in self.config.asset_profiles] if self.config.asset_profiles else ["AAPL"]
            
        # 確保基準大盤必定存在於下載與訓練列表中 (放置於首位優先下載)
        if self.benchmark_symbol not in self.symbols:
            self.symbols.insert(0, self.benchmark_symbol)
            
        self.symbol_terms = {}
        param_path = os.path.join(WEIGHTS_DIR, "global_best_params.json")
        if os.path.exists(param_path):
            try:
                with open(param_path, 'r', encoding='utf-8') as f:
                    global_params = json.load(f)
                    for sym, params in global_params.items():
                        if "term" in params:
                            self.symbol_terms[sym] = params["term"]
            except Exception: pass
            
        for sym in self.symbols:
            if sym not in self.symbol_terms:
                self.symbol_terms[sym] = "long_term"

    def _get_term_settings(self, term: str) -> dict:
        """根據交易週期 (term) 返回最適合的 K線級別、總下載天數、與單次分批天數"""
        if term == "short_term":
            return {"bar_size": self.config.get("general_settings.short_term_bar_size", "5 mins"), "total_days": 60, "chunk_days": 30}
        elif term == "mid_term":
            return {"bar_size": self.config.get("general_settings.mid_term_bar_size", "1 hour"), "total_days": 180, "chunk_days": 90}
        else: 
            return {"bar_size": self.config.get("general_settings.long_term_bar_size", "1 day"), "total_days": 730, "chunk_days": 365}

    async def _sync_symbol_term_data(self, symbol: str, term: str, contract: Stock):
        """核心資料抓取模組，支援動態按需調用"""
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
            if last_date.tz is None:
                last_date = last_date.tz_localize('UTC')
            last_date_ny = last_date.tz_convert('America/New_York')
            now_ny = pd.Timestamp.now(tz='America/New_York')
            
            bus_days_diff = np.busday_count(last_date_ny.date(), now_ny.date())
            
            if bus_days_diff <= 0:
                print(f"      ✅ 資料已是最新狀態，無須同步。")
                return
                
            days_to_fetch = min(bus_days_diff + 1, total_days)

        df_new_list = []
        remaining_days = days_to_fetch
        current_end_date = datetime.now()
        
        while remaining_days > 0:
            fetch_days = min(remaining_days, chunk_days)
            
            # 針對大跨度轉換為 Y (年) 或 M (月) 的格式，提升 IBKR 接受度
            if fetch_days >= 365:
                duration_str = f"{fetch_days // 365} Y"
            elif fetch_days >= 30 and bar_size != "1 day":
                duration_str = f"{fetch_days // 30} M"
            else:
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
                df_combined = df_new
                df_combined.sort_index(inplace=True)
                self.db.save_bulk_market_data(symbol, df_combined, timeframe=bar_size)
            else:
                self.db.save_bulk_market_data(symbol, df_new, timeframe=bar_size)
                
            print(f"      ✅ 成功合併並寫入 {len(df_new)} 筆 {bar_size} K線。")
        else:
            if days_to_fetch > 0:
                print(f"      ❌ [{symbol}] 該週期所有分批資料獲取皆失敗。")

    def _train_dl_models(self, symbol: str, df: pd.DataFrame, bench_df: pd.DataFrame, macro_data: dict):
        """訓練深度學習模型（LSTM、Transformer）並儲存 .keras"""
        print(f"🔍 [{symbol}] 開始訓練深度學習模型...")
        print(f"[{symbol}] 資料量: {len(df)} 筆，特徵數量: {df.shape[1]} 欄")
        if df.empty or len(df) < 60:
            print(f"⚠️ [{symbol}] 資料量極度不足 (僅 {len(df)} 筆)，跳過 DL 訓練。")
            return

        os.makedirs(WEIGHTS_DIR, exist_ok=True)
        
        df_adv = self.pipeline.engineer_advanced_features(df, bench_df, macro_data)
        df_adv = df_adv.ffill().bfill().fillna(0)
        
        scale_cols = self.pipeline.select_model_features(df_adv)

        price_rel = self.pipeline.classify_price_relative(df_adv, scale_cols)
        minmax_cols = [c for c in scale_cols if c not in price_rel]

        self.pipeline.save_feature_manifest(
            symbol, scale_cols, price_relative=price_rel,
            target_mode="log_return", target_scale=100.0,
        )
        print(f"      => 特徵欄位: {len(scale_cols)} 個 "
              f"(價格相對 {len(price_rel)} / Min-Max {len(minmax_cols)})，目標: 對數報酬率")

        split = int(len(df_adv) * 0.8)
        df_train = df_adv.iloc[:split]

        self.pipeline.fit_scale(df_train, minmax_cols, symbol)
        df_scaled = self.pipeline.transform_scale(df_adv, minmax_cols, symbol)

        look_back = 60
        X, y = self.pipeline.create_sequences(
            df_scaled, scale_cols, look_back,
            raw_close=df_adv['Close'],
            price_relative=price_rel,
            target_mode="log_return", target_scale=100.0,
        )
        
        print(f"      => 總 K 線數: {len(df_adv)} 筆，產出有效訓練序列: {len(X)} 組")
        
        if len(X) < 16:
            print(f"⚠️ [{symbol}] 有效序列數量不足以支撐梯度下降 (僅 {len(X)} 組)，跳過 DL 訓練。")
            return

        dynamic_batch_size = min(32, max(8, len(X) // 4))

        try:
            from keras.callbacks import EarlyStopping
            from ibkrpy.models.lstm import LSTMModel
            from ibkrpy.models.transformer import TransformerModel

            callbacks = [EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)]
            fit_kwargs = dict(
                epochs=25, batch_size=dynamic_batch_size, verbose=0,
                callbacks=callbacks, validation_split=0.2, shuffle=False,  # 時序資料不可打亂
            )

            print(f"   -> 🚀 擬合 LSTM 模型 (Batch Size: {dynamic_batch_size})...")
            lstm = LSTMModel(look_back=look_back, feature_cols=scale_cols, weights_dir=WEIGHTS_DIR)
            lstm.model = lstm._build_model()
            lstm.model.fit(X, y, **fit_kwargs)
            lstm_path = os.path.join(WEIGHTS_DIR, f"{symbol}_LSTM.keras")
            lstm.model.save(lstm_path)
            
            if os.path.exists(lstm_path): print(f"      ✅ [成功] LSTM 權重已實體寫入: {lstm_path}")
            else: print(f"      ❌ [失敗] LSTM 寫入異常！")

            print(f"   -> 🚀 擬合 Transformer 模型 (Batch Size: {dynamic_batch_size})...")
            transformer = TransformerModel(look_back=look_back, feature_cols=scale_cols, weights_dir=WEIGHTS_DIR)
            transformer.model = transformer._build_model()
            transformer.model.fit(X, y, **fit_kwargs)
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
            
        weights_dir = WEIGHTS_DIR
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

        bundle_path = os.path.join(weights_dir, f"{symbol}_classical.pkl")
        joblib.dump(classical_bundle, bundle_path)
        
        if os.path.exists(bundle_path):
            print(f"✅ [{symbol}] 統計模型整合包 (Classical Bundle) 寫入完成。")

    # ------------------------------------------------------------------
    # Walk-forward 尋優
    # ------------------------------------------------------------------

    def _vectorised_regimes(self, df: pd.DataFrame) -> pd.Series:
        """
        對整段資料一次算出各根 K 線的市場情境。

        與 MarketRegimeDetector.detect() 的判定邏輯完全相同，只是把逐根呼叫
        (每次重算全序列指標) 改為向量化計算一次 —— 結果一致但快上數個量級。
        """
        import pandas_ta as ta
        from ibkrpy.strategy.strategy_components import MarketRegimeDetector

        d = MarketRegimeDetector()
        out = pd.Series("SIDEWAYS_QUIET", index=df.index, dtype=object)

        try:
            adx = ta.adx(df['High'], df['Low'], df['Close'], length=d.adx_period)
            adx_val = adx.iloc[:, 0] if adx is not None and not adx.empty else pd.Series(0.0, index=df.index)
            sma_s = ta.sma(df['Close'], length=d.ma_short)
            sma_l = ta.sma(df['Close'], length=d.ma_long)
            atr = ta.atr(df['High'], df['Low'], df['Close'], length=d.atr_period)
            atr_pct = (atr / df['Close']).fillna(0.0)

            trending = adx_val.fillna(0.0) > d.adx_threshold
            volatile = atr_pct > d.vol_threshold
            bull = sma_s > sma_l

            out[trending & bull] = "BULL_TREND"
            out[trending & ~bull] = "BEAR_TREND"
            out[~trending & volatile] = "SIDEWAYS_VOLATILE"
            out[~trending & ~volatile] = "SIDEWAYS_QUIET"
        except Exception as e:
            print(f"      ⚠️ 情境向量化計算失敗，全段退回 SIDEWAYS_QUIET: {e}")

        return out

    def _walk_forward_predictions(self, symbol: str, df: pd.DataFrame, term: str) -> pd.DataFrame:
        """
        在 out-of-sample 區段產生「真實的」模型預測。

          - 切分前 70% 為 in-sample，後 30% 為 out-of-sample
          - 模型只在 in-sample 上擬合，逐根對 OOS 產生真正的一步預測
          - 情境改用 MarketRegimeDetector 的實際判定
          - 波動率改用實際的滾動對數報酬標準差 (與實盤送進策略的量綱一致)

        預設只用 ARIMA 產生尋優預測 —— 它秒級可擬合，而錦標賽要決定的是
        「K 線週期與風控參數」，不是驗證神經網路。若要更貼近實盤的 Ensemble，
        可在 config.yaml 設定 tuning_settings.walk_forward_models。
        """
        wf_models = self.config.get("tuning_settings.walk_forward_models", ["ARIMA"])
        oos_frac = float(self.config.get("tuning_settings.oos_fraction", 0.3))
        max_hist = int(self.config.get("tuning_settings.trailing_window", 500))

        n = len(df)
        split = int(n * (1 - oos_frac))
        if n < 150 or split < 100:
            print(f"      ⚠️ {term} 資料量不足以做 walk-forward 切分 (共 {n} 根)。")
            return pd.DataFrame()

        oos = df.iloc[split:].copy()
        close = df['Close'].astype(float).values

        preds = np.full(len(oos), np.nan)

        if "ARIMA" in wf_models:
            try:
                from statsmodels.tsa.arima.model import ARIMA
                # 只在 in-sample 擬合一次，取得參數
                base = ARIMA(close[:split], order=(5, 1, 0)).fit()

                for k in range(len(oos)):
                    t = split + k                      # 要預測 close[t]
                    lo = max(0, t - max_hist)
                    hist = close[lo:t]                 # 只用 t 之前的資料，無前視
                    if len(hist) < 20:
                        continue
                    try:
                        preds[k] = float(base.apply(hist).forecast(steps=1)[0])
                    except Exception:
                        preds[k] = close[t - 1]
            except Exception as e:
                print(f"      ⚠️ ARIMA walk-forward 失敗: {e}")

        # 需要神經網路參與時，只在 in-sample 訓練一次，再對 OOS 逐窗推論
        dl_wanted = [m for m in wf_models if m in ("LSTM", "Transformer")]
        if dl_wanted:
            dl_preds = self._walk_forward_dl(symbol, df, split, dl_wanted)
            if dl_preds is not None:
                stacked = [p for p in ([preds] if "ARIMA" in wf_models else []) + [dl_preds]]
                preds = np.nanmean(np.vstack(stacked), axis=0)

        oos['prediction'] = preds
        oos = oos[np.isfinite(oos['prediction'])]
        if oos.empty:
            print(f"      ⚠️ {term} 未能產生任何有效的 OOS 預測。")
            return pd.DataFrame()

        log_ret = np.log(df['Close'] / df['Close'].shift(1))
        bar_vol = log_ret.rolling(20).std().reindex(oos.index).fillna(0.005).replace(0, 0.005)
        oos['volatility'] = bar_vol

        oos['regime'] = self._vectorised_regimes(df).reindex(oos.index).fillna("SIDEWAYS_QUIET")

        edge = (oos['prediction'] / oos['Close'] - 1).abs()
        print(f"      => OOS 樣本 {len(oos)} 根 (總計 {n})，尋優模型 {wf_models}；"
              f"預測邊際中位數 {edge.median()*100:.3f}%，每根波動中位數 {bar_vol.median()*100:.3f}%")

        return oos

    def _walk_forward_dl(self, symbol: str, df: pd.DataFrame, split: int, model_types: list):
        """在 in-sample 訓練神經網路，對 OOS 逐窗推論 (成本高，非預設路徑)"""
        try:
            from keras.callbacks import EarlyStopping
            from ibkrpy.models.lstm import LSTMModel
            from ibkrpy.models.transformer import TransformerModel
        except ImportError:
            print("      ⚠️ 未安裝 TensorFlow/Keras，walk-forward 略過神經網路。")
            return None

        try:
            df_adv = self.pipeline.engineer_advanced_features(df).ffill().bfill().fillna(0)
            feats = self.pipeline.select_model_features(df_adv)
            price_rel = self.pipeline.classify_price_relative(df_adv, feats)
            minmax = [c for c in feats if c not in price_rel]

            adv_split = min(split, len(df_adv) - 1)
            tmp_key = f"__wf_{symbol}"
            self.pipeline.fit_scale(df_adv.iloc[:adv_split], minmax, tmp_key)
            df_s = self.pipeline.transform_scale(df_adv, minmax, tmp_key)

            look_back = 60
            X, y = self.pipeline.create_sequences(
                df_s, feats, look_back, raw_close=df_adv['Close'],
                price_relative=price_rel, target_mode="log_return", target_scale=100.0,
            )
            if len(X) < 100:
                return None

            # 序列 i 的目標落在 df_adv 的第 i+look_back 根
            seq_pos = np.arange(len(X)) + look_back
            in_mask = seq_pos < adv_split

            cb = [EarlyStopping(monitor='val_loss', patience=3, restore_best_weights=True)]
            outputs = []
            for mt in model_types:
                cls = LSTMModel if mt == "LSTM" else TransformerModel
                m = cls(look_back=look_back, feature_cols=feats, weights_dir=WEIGHTS_DIR)
                m.model = m._build_model()
                m.model.fit(X[in_mask], y[in_mask], epochs=12, batch_size=32,
                            verbose=0, validation_split=0.2, shuffle=False, callbacks=cb)
                raw = m.model.predict(X[~in_mask], verbose=0).reshape(-1)
                anchors = df_adv['Close'].values[seq_pos[~in_mask] - 1]
                outputs.append(anchors * np.exp(raw / 100.0))

            oos_idx = df_adv.index[seq_pos[~in_mask]]
            series = pd.Series(np.mean(outputs, axis=0), index=oos_idx)
            return series.reindex(df.index[split:]).values
        except Exception as e:
            print(f"      ⚠️ 神經網路 walk-forward 失敗: {e}")
            return None
        finally:
            self.pipeline.invalidate(f"__wf_{symbol}")

    def _run_optuna_optimization(self, symbol: str, df: pd.DataFrame, bench_df: pd.DataFrame, macro_data: dict, term: str) -> Tuple[dict, float]:
        """在真實的 out-of-sample 預測上做參數尋優，並回傳 (最佳參數, 複合評分)"""
        default_params = {
            "min_prediction_threshold_pct": 0.005,
            "volatility_stop_loss_multiplier": 2.0,
            "volatility_take_profit_multiplier": 3.0,
        }

        oos = self._walk_forward_predictions(symbol, df, term)
        if oos.empty:
            return default_params, -999.0

        tuner = ModelTuner(model_orchestrator=None, data_manager=None)
        try:
            # 回測區間必須與預測區間一致，否則權益曲線會混入沒有訊號的 in-sample 段
            return tuner.optimize_strategy_params(
                symbol, df.loc[oos.index], oos, n_trials=20, term=term
            )
        except Exception as e:
            print(f"⚠️ [{symbol}] {term} Optuna 尋優過程發生錯誤: {e}")
            return default_params, -999.0

    async def run_data_ingestion(self):
        """階段一：日常增量下載資料並寫入資料庫 (按需下載模式)"""
        print("\n" + "="*60)
        print(" [Pipeline] 啟動資料增量下載與資料庫同步 (Daily Sync)")
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
                        # 將資料庫的最後時間與系統當前時間，強制統一對齊到美東時間 (America/New_York)
                        last_date = pd.Timestamp(df_existing.index[-1])
                        if last_date.tz is None:
                            last_date = last_date.tz_localize('UTC')
                        last_date_ny = last_date.tz_convert('America/New_York')
                        now_ny = pd.Timestamp.now(tz='America/New_York')
                        
                        bus_days_diff = np.busday_count(last_date_ny.date(), now_ny.date())
                        
                        if bus_days_diff <= 0:
                            print(f"      ✅ 資料已是最新狀態，無須同步。")
                            continue
                            
                        # 如果差 1 天，抓 1+1=2 天緩衝即可，避免每次都盲目抓 3 天
                        days_to_fetch = min(bus_days_diff + 1, total_days)

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
        os.makedirs(DATA_DIR, exist_ok=True)
        
        # 1. 永久儲存 FRED 數據至 data/
        global_vix_series = None
        fred_cache_path = os.path.join(DATA_DIR, "fred_vix_cache.csv")
        need_fetch_fred = True
        
        if os.path.exists(fred_cache_path):
            mod_time = os.path.getmtime(fred_cache_path)
            if (time.time() - mod_time) < 43200:
                try:
                    global_vix_series = pd.read_csv(fred_cache_path, index_col=0, parse_dates=True).squeeze("columns")
                    need_fetch_fred = False
                    print("   -> 🌍 從本地 data/ 讀取 FRED VIX 歷史快取...")
                except Exception: pass
                
        if need_fetch_fred and self.ext:
            print("   -> 🌍 正在向 FRED 請求最新全局宏觀數據 (VIXCLS)...")
            try:
                global_vix_series = await self.ext.fetch_fred_series("VIXCLS")
                if global_vix_series is not None and not global_vix_series.empty:
                    global_vix_series.to_csv(fred_cache_path)
                    print(f"      ✅ FRED VIX 數據獲取成功，已永久儲存至 {fred_cache_path}。")
            except Exception as e:
                print(f"      ⚠️ FRED API 請求失敗: {e}")
                if os.path.exists(fred_cache_path):
                    global_vix_series = pd.read_csv(fred_cache_path, index_col=0, parse_dates=True).squeeze("columns")

        # 2. 讀取 FMP 基本面本地快取
        fmp_cache_path = os.path.join(WEIGHTS_DIR, "fmp_cache.json")
        fmp_cache = {}
        if os.path.exists(fmp_cache_path):
            try:
                with open(fmp_cache_path, 'r', encoding='utf-8') as f:
                    fmp_cache = json.load(f)
            except Exception: pass
        
        for symbol in self.symbols:
            if symbol == self.benchmark_symbol:
                continue
                
            print(f"\n🔥 啟動週期選拔與訓練任務: {symbol} 🔥")
            
            fmp_data = {}
            if self.ext and self.ext.fmp_api_key:
                if symbol in fmp_cache:
                    print(f"   -> 🏢 從本地快取讀取 FMP 公司基本面數據...")
                    fmp_data = fmp_cache[symbol]
                else:
                    print(f"   -> 🏢 正在向 FMP 請求公司基本面數據...")
                    try:
                        fmp_profile = await self.ext.fetch_fmp_profile(symbol)
                        if fmp_profile:
                            fmp_data = fmp_profile
                            fmp_cache[symbol] = fmp_data
                            os.makedirs(WEIGHTS_DIR, exist_ok=True)
                            with open(fmp_cache_path, 'w', encoding='utf-8') as f:
                                json.dump(fmp_cache, f, indent=4)
                    except Exception as e:
                        print(f"      => ⚠️ FMP API 請求發生例外錯誤: {e}")
            
            best_term = None
            best_score = -float('inf')
            best_params = {}
            best_df = pd.DataFrame()
            best_macro = {}
            
            for term in all_terms:
                settings = self._get_term_settings(term)
                bar_size = settings["bar_size"]

                df = self.db.get_market_data_sync(symbol, timeframe=bar_size)
                
                last_date = pd.Timestamp(df.index[-1]).normalize() if not df.empty else None
                now_date = pd.Timestamp.now().normalize()
                stale_days = np.busday_count(last_date.date(), now_date.date()) if last_date else 999

                if df.empty or len(df) < 100 or stale_days > 5:
                    print(f"   -> 🔍 為了評估 {term}，發現資料空缺或過期，啟動即時下載...")
                    try:
                        contract = Stock(symbol, "SMART", "USD")
                        await self.ib_data.ib.qualifyContractsAsync(contract)
                        await self._sync_symbol_term_data(symbol, term, contract)
                        df = self.db.get_market_data_sync(symbol, timeframe=bar_size)
                    except Exception as e:
                        print(f"   -> ⚠️ 即時下載失敗: {e}")

                if df.empty or len(df) < 100:
                    print(f"   -> ⚠️ {term} 資料仍不足，跳過該週期評估。")
                    continue
                
                bench_df = self.db.get_market_data_sync(self.benchmark_symbol, timeframe=bar_size)
                if not bench_df.empty:
                    bench_df = bench_df.reindex(df.index, method='ffill').bfill()
                
                macro_data = {}
                if global_vix_series is not None and not global_vix_series.empty:
                    if getattr(global_vix_series.index, 'tz', None) is not None:
                        global_vix_series.index = global_vix_series.index
                    vix_daily = global_vix_series.copy()
                    vix_daily.index = vix_daily.index.normalize()
                    
                    df_idx_naive = df.index if getattr(df.index, 'tz', None) is not None else df.index
                    vix_aligned_values = df_idx_naive.normalize().map(vix_daily)
                    vix_aligned = pd.Series(vix_aligned_values, index=df.index).ffill().bfill()
                    if vix_aligned.isna().all(): vix_aligned = pd.Series(20.0, index=df.index)
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
            if fmp_data:
                best_params['fmp_sector'] = fmp_data.get('sector')
                best_params['fmp_industry'] = fmp_data.get('industry')
                best_params['fmp_beta'] = fmp_data.get('beta')
                best_params['fmp_mktCap'] = fmp_data.get('mktCap')
            
            bench_df = self.db.get_market_data_sync(self.benchmark_symbol, timeframe=self._get_term_settings(best_term)["bar_size"])
            if not bench_df.empty:
                bench_df = bench_df.reindex(best_df.index, method='ffill').bfill()
            
            self._train_dl_models(symbol, best_df, bench_df, best_macro)
            self._train_safe_models(symbol, best_df)
            
            weights_dir = WEIGHTS_DIR
            os.makedirs(weights_dir, exist_ok=True)

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