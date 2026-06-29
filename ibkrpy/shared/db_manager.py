# ibkrpy/shared/db_manager.py
# 負責交易紀錄、市場數據、資金與持倉的 SQLite 持久化儲存

import os
import sqlite3
import asyncio
import pandas as pd
from typing import Dict, Any, Optional
from .system_log import global_logger

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
DEFAULT_DB_PATH = os.path.join(PROJECT_ROOT, "data", "trading_data.db")

class DatabaseManager:
    """輕量化 SQLite 非同步/同步雙軌管理器 (具備 WAL 併發防護與 Upsert 增量寫入)"""

    def __init__(self, db_path: str = None):
        # 統一將資料庫路徑綁定在專案目錄下
        self.db_path = db_path or DEFAULT_DB_PATH
        os.makedirs(os.path.dirname(self.db_path), exist_ok=True)
        
        self.logger = global_logger
        self._init_tables()

    def _get_connection(self) -> sqlite3.Connection:
        """獲取連線並開啟 WAL 模式，防範 UI 讀取與交易迴圈寫入造成的 Database Locked"""
        conn = sqlite3.connect(self.db_path, timeout=15.0)
        conn.execute("PRAGMA journal_mode=WAL;")
        conn.execute("PRAGMA synchronous=NORMAL;")
        return conn

    def _execute_sync(self, query: str, params: tuple = ()) -> None:
        """內部同步執行 (寫入)"""
        with self._get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute(query, params)
            conn.commit()

    def _fetch_sync(self, query: str, params: tuple = ()) -> list:
        """內部同步讀取"""
        with self._get_connection() as conn:
            conn.row_factory = sqlite3.Row
            cursor = conn.cursor()
            cursor.execute(query, params)
            return [dict(row) for row in cursor.fetchall()]

    def _init_tables(self):
        """初始化必要之資料表"""
        query_trades = """
        CREATE TABLE IF NOT EXISTS trade_logs (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            timestamp DATETIME DEFAULT CURRENT_TIMESTAMP,
            symbol TEXT,
            action TEXT,
            quantity REAL,
            price REAL,
            regime TEXT,
            reason TEXT
        )
        """
        query_market = """
        CREATE TABLE IF NOT EXISTS market_data (
            timestamp DATETIME,
            symbol TEXT,
            timeframe TEXT,
            open REAL,
            high REAL,
            low REAL,
            close REAL,
            volume REAL,
            PRIMARY KEY (symbol, timeframe, timestamp)
        )
        """
        query_account = """
        CREATE TABLE IF NOT EXISTS account_state (
            id INTEGER PRIMARY KEY CHECK (id = 1),
            timestamp DATETIME DEFAULT CURRENT_TIMESTAMP,
            net_liquidation REAL,
            available_funds REAL
        )
        """
        query_positions = """
        CREATE TABLE IF NOT EXISTS portfolio_positions (
            symbol TEXT PRIMARY KEY,
            position REAL,
            timestamp DATETIME DEFAULT CURRENT_TIMESTAMP
        )
        """
        query_equity_history = """
        CREATE TABLE IF NOT EXISTS equity_history (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            timestamp DATETIME DEFAULT CURRENT_TIMESTAMP,
            net_liquidation REAL
        )
        """
        self._execute_sync(query_trades)
        self._execute_sync(query_market)
        self._execute_sync(query_account)
        self._execute_sync(query_positions)
        self._execute_sync(query_equity_history)
        
        self._execute_sync("CREATE INDEX IF NOT EXISTS idx_trade_logs_timestamp ON trade_logs(timestamp DESC)")
        self._execute_sync("CREATE INDEX IF NOT EXISTS idx_equity_history_timestamp ON equity_history(timestamp DESC)")

    async def update_account_info(self, net_liq: float, avail_funds: float, positions: dict):
        """非同步更新帳戶資金與持倉狀態，並記錄收益歷史"""
        def _sync_update():
            with self._get_connection() as conn:
                cursor = conn.cursor()
                cursor.execute("""
                    INSERT INTO account_state (id, net_liquidation, available_funds, timestamp)
                    VALUES (1, ?, ?, CURRENT_TIMESTAMP)
                    ON CONFLICT(id) DO UPDATE SET
                    net_liquidation=excluded.net_liquidation,
                    available_funds=excluded.available_funds,
                    timestamp=CURRENT_TIMESTAMP
                """, (net_liq, avail_funds))
                
                cursor.execute("DELETE FROM portfolio_positions")
                for sym, pos in positions.items():
                    if pos != 0:
                        cursor.execute("""
                            INSERT INTO portfolio_positions (symbol, position, timestamp)
                            VALUES (?, ?, CURRENT_TIMESTAMP)
                        """, (sym, pos))
                        
                cursor.execute("""
                    INSERT INTO equity_history (net_liquidation, timestamp)
                    VALUES (?, CURRENT_TIMESTAMP)
                """, (net_liq,))
                conn.commit()

        try:
            await asyncio.to_thread(_sync_update)
        except Exception as e:
            self.logger.error(f"更新帳戶狀態失敗: {e}")

    async def log_trade(self, trade_data: Dict[str, Any]):
        query = """
        INSERT INTO trade_logs (symbol, action, quantity, price, regime, reason)
        VALUES (?, ?, ?, ?, ?, ?)
        """
        params = (
            trade_data.get("symbol"), trade_data.get("action"),
            trade_data.get("quantity", 0), trade_data.get("price", 0.0),
            trade_data.get("regime", "UNKNOWN"), trade_data.get("reason", "")
        )
        try:
            await asyncio.to_thread(self._execute_sync, query, params)
        except Exception as e:
            self.logger.error(f"記錄交易日誌失敗: {e}")

    async def get_recent_trades(self, limit: int = 50) -> pd.DataFrame:
        query = "SELECT * FROM trade_logs ORDER BY timestamp DESC LIMIT ?"
        try:
            results = await asyncio.to_thread(self._fetch_sync, query, (limit,))
            return pd.DataFrame(results)
        except Exception as e:
            self.logger.error(f"獲取交易紀錄失敗: {e}")
            return pd.DataFrame()

    def save_bulk_market_data(self, symbol: str, df: pd.DataFrame, timeframe: str = "1 day"):
        """儲存市場資料，採用 Upsert 達成無損增量寫入"""
        if df.empty: return
        
        df_db = df.copy()
        df_db['symbol'] = symbol
        df_db['timeframe'] = timeframe
        
        if df_db.index.name != 'timestamp':
            df_db.index.name = 'timestamp'
        df_db = df_db.reset_index()
        
        df_db['timestamp'] = pd.to_datetime(df_db['timestamp'], utc=True).dt.strftime('%Y-%m-%d %H:%M:%S')
        
        df_db.rename(columns={'Open': 'open', 'High': 'high', 'Low': 'low', 'Close': 'close', 'Volume': 'volume'}, inplace=True)
        cols = ['timestamp', 'symbol', 'timeframe', 'open', 'high', 'low', 'close', 'volume']
        df_db = df_db[cols]
        
        data_tuples = list(df_db.itertuples(index=False, name=None))
        
        query = """
        INSERT OR REPLACE INTO market_data (timestamp, symbol, timeframe, open, high, low, close, volume)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?)
        """
        
        with self._get_connection() as conn:
            cursor = conn.cursor()
            cursor.executemany(query, data_tuples)
            conn.commit()

    def get_market_data_sync(self, symbol: str, timeframe: str = "1 day") -> pd.DataFrame:
        """同步提取完整歷史市場資料"""
        query = "SELECT * FROM market_data WHERE symbol = ? AND timeframe = ? ORDER BY timestamp ASC"
        results = self._fetch_sync(query, (symbol, timeframe))
        
        if not results:
            return pd.DataFrame()
            
        df = pd.DataFrame(results)
        
        df['timestamp'] = pd.to_datetime(df['timestamp'], utc=True, errors='coerce')
        
        df.set_index('timestamp', inplace=True)
        df = df[~df.index.duplicated(keep='last')]
        df.sort_index(inplace=True)
        
        df.rename(columns={'open': 'Open', 'high': 'High', 'low': 'Low', 'close': 'Close', 'volume': 'Volume'}, inplace=True)
        return df