# ibkrpy/ui/trading_dashboard.py
# 系統視覺化監控儀表板

import os
import sys
import tkinter as tk
from tkinter import ttk
import pandas as pd

import matplotlib
matplotlib.use("TkAgg")
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import matplotlib.dates as mdates

# 取得專案根目錄
project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.append(project_root)

from ibkrpy.shared.db_manager import DatabaseManager

class TradingDashboard:
    def __init__(self, root):
        self.root = root
        self.root.title("IBKR 量化交易監控中心")
        self.root.geometry("1300x800")
        
        self.style = ttk.Style()
        self.style.theme_use('clam')
        self.style.configure("TFrame", background="#1E1E1E")
        self.style.configure("TLabel", background="#1E1E1E", foreground="#FFFFFF", font=("Helvetica", 11))
        self.style.configure("Header.TLabel", font=("Helvetica", 14, "bold"), foreground="#00E5FF")
        self.style.configure("Treeview", background="#2D2D2D", foreground="#FFFFFF", fieldbackground="#2D2D2D", rowheight=25)
        self.style.map('Treeview', background=[('selected', '#00E5FF')])
        
        self.root.configure(bg="#1E1E1E")
        
        self.db = DatabaseManager()
        
        self._build_ui()
        self._update_data() 

    def _build_ui(self):
        # 頂部狀態列
        top_frame = ttk.Frame(self.root, padding=10)
        top_frame.pack(side=tk.TOP, fill=tk.X)
        title_lbl = ttk.Label(top_frame, text="🟢 系統狀態: 實盤/模擬盤自動運行中", style="Header.TLabel")
        title_lbl.pack(side=tk.LEFT)

        main_frame = ttk.Frame(self.root, padding=10)
        main_frame.pack(fill=tk.BOTH, expand=True)
        
        # --- 左側：AI 交易紀錄 ---
        left_frame = ttk.Frame(main_frame)
        left_frame.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=(0, 10))
        ttk.Label(left_frame, text="📝 實盤 交易紀錄", style="Header.TLabel").pack(anchor=tk.W, pady=(0, 5))
        
        columns = ("Time", "Symbol", "Action", "Qty", "Price", "Regime", "Reason")
        self.tree = ttk.Treeview(left_frame, columns=columns, show="headings")
        for col in columns:
            self.tree.heading(col, text=col)
            self.tree.column(col, width=90 if col != "Time" else 150, anchor=tk.CENTER)
            
        self.tree.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        scrollbar = ttk.Scrollbar(left_frame, orient=tk.VERTICAL, command=self.tree.yview)
        self.tree.configure(yscroll=scrollbar.set)
        scrollbar.pack(side=tk.RIGHT, fill=tk.Y)

        # --- 右側：資金與圖表 ---
        right_frame = ttk.Frame(main_frame, width=480)
        right_frame.pack(side=tk.RIGHT, fill=tk.Y)
        
        # 右上：帳戶狀態
        ttk.Label(right_frame, text="💼 帳戶資金與實盤庫存", style="Header.TLabel").pack(anchor=tk.W, pady=(0, 5))
        self.account_info_text = tk.Text(right_frame, width=50, height=10, bg="#2D2D2D", fg="#FFD700", font=("Consolas", 12), borderwidth=0, padx=5, pady=5)
        self.account_info_text.pack(fill=tk.X, pady=(0, 15))

        # 圖表控制區
        chart_ctrl_frame = ttk.Frame(right_frame)
        chart_ctrl_frame.pack(fill=tk.X, pady=(0, 5))
        ttk.Label(chart_ctrl_frame, text="📉 選擇圖表: ", font=("Helvetica", 12, "bold"), foreground="#00E5FF").pack(side=tk.LEFT)
        
        self.chart_selector = ttk.Combobox(chart_ctrl_frame, state="readonly", width=20, font=("Helvetica", 11))
        self.chart_selector.pack(side=tk.LEFT, padx=5)
        self.chart_selector.set("💰 帳戶收益曲線 (Equity)")
        
        # 右下：圖表繪製區 (只建立一次 ax 以防止記憶體洩漏)
        self.figure = plt.Figure(figsize=(5, 3.5), dpi=100, facecolor='#1E1E1E')
        self.ax = self.figure.add_subplot(111)
        self.ax2 = None  # 預留副座標軸參考
            
        self.canvas = FigureCanvasTkAgg(self.figure, right_frame)
        self.canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)

        # --- 底部：實時終端機 ---
        bottom_frame = ttk.Frame(self.root, padding=10)
        bottom_frame.pack(side=tk.BOTTOM, fill=tk.X)
        ttk.Label(bottom_frame, text="📡 底層監控終端 (API 防護 / MAD 剔除 / 宏觀警報)", style="Header.TLabel").pack(anchor=tk.W, pady=(0, 5))
        
        self.log_terminal = tk.Text(bottom_frame, height=8, bg="#0C0C0C", fg="#00FF41", font=("Consolas", 10), borderwidth=1, relief="solid")
        self.log_terminal.pack(fill=tk.X)

    def _tail_system_logs(self):
        """讀取最新的 log，透過差異對比防範 macOS Mach Port 報錯"""
        log_path = os.path.join(project_root, "logs", "trading_bot.log")
        if not os.path.exists(log_path):
            return

        try:
            with open(log_path, "r", encoding="utf-8") as f:
                lines = f.readlines()[-10:]
                new_text = "".join(lines)
                current_text = self.log_terminal.get(1.0, tk.END)
                
                if new_text.strip() != current_text.strip():
                    self.log_terminal.delete(1.0, tk.END)
                    self.log_terminal.insert(tk.END, new_text)
                    self.log_terminal.see(tk.END)
        except Exception:
            pass

    def _update_data(self):
        """每 5 秒自動讀取資料庫刷新畫面"""
        try:
            # 1. 更新交易紀錄
            df_trades = self.db._fetch_sync("SELECT * FROM trade_logs ORDER BY timestamp DESC LIMIT 20")
            for item in self.tree.get_children():
                self.tree.delete(item)
            for row in df_trades:
                time_str = row['timestamp'][:19]
                reason_str = row.get('reason', '')
                values = (time_str, row['symbol'], row['action'], row['quantity'], f"${row['price']:.2f}", row['regime'], reason_str)
                tag = 'buy' if row['action'] == 'BUY' else 'sell'
                self.tree.insert("", tk.END, values=values, tags=(tag,))
                
            self.tree.tag_configure('buy', foreground='#00FF7F') 
            self.tree.tag_configure('sell', foreground='#FF3366') 

            # 2. 更新帳戶狀態 (加入差異對比)
            account_data = self.db._fetch_sync("SELECT * FROM account_state WHERE id=1")
            position_data = self.db._fetch_sync("SELECT * FROM portfolio_positions ORDER BY symbol")
            
            new_acc_text = ""
            if account_data:
                acc = account_data[0]
                new_acc_text += f"💰 總淨值(NLV): ${acc['net_liquidation']:,.2f}\n"
                new_acc_text += f"💵 可用現金:   ${acc['available_funds']:,.2f}\n"
                new_acc_text += "-"*40 + "\n"
                new_acc_text += "📦 當前實盤持倉:\n"
                if position_data:
                    for p in position_data:
                        color_prefix = "+" if p['position'] > 0 else ""
                        new_acc_text += f"  ➤ {p['symbol']:<6}: {color_prefix}{p['position']} 股\n"
                else:
                    new_acc_text += "  (目前帳戶無持倉)\n"
            else:
                new_acc_text = "等待實盤引擎掃描市場與同步資金...\n"

            current_acc_text = self.account_info_text.get(1.0, tk.END)
            if new_acc_text.strip() != current_acc_text.strip():
                self.account_info_text.delete(1.0, tk.END)
                self.account_info_text.insert(tk.END, new_acc_text)

            # 3. 動態更新下拉選單選項
            symbols_data = self.db._fetch_sync("SELECT DISTINCT symbol FROM market_data")
            available_symbols = [f"📊 標的走勢: {row['symbol']}" for row in symbols_data]
            options = ["💰 帳戶收益曲線 (Equity)"] + available_symbols
            
            current_selection = self.chart_selector.get()
            self.chart_selector['values'] = options
            if current_selection not in options and options:
                self.chart_selector.set(options[0])

            # 4. 根據選單繪製指定的圖表 (安全清除 Axes，防範洩漏)
            selection = self.chart_selector.get()
            
            # 清除副座標軸
            if self.ax2 is not None:
                self.ax2.remove()
                self.ax2 = None
                
            self.ax.clear()
            self.ax.set_facecolor('#2D2D2D') 
            self.ax.tick_params(colors='white', labelsize=9)
            for spine in self.ax.spines.values():
                spine.set_color('#4DA8DA')
            
            if selection == "💰 帳戶收益曲線 (Equity)":
                df_eq = pd.DataFrame(self.db._fetch_sync("SELECT timestamp, net_liquidation FROM equity_history ORDER BY timestamp DESC LIMIT 300"))
                if not df_eq.empty:
                    df_eq['timestamp'] = pd.to_datetime(df_eq['timestamp'])
                    df_eq.sort_values('timestamp', inplace=True)
                    self.ax.plot(df_eq['timestamp'], df_eq['net_liquidation'], color='#FFD700', linewidth=2, label='Equity')
                    self.ax.set_title("Account Net Liquidation Curve", color='#FFFFFF')
                    self.ax.set_ylabel("Net Liq (USD)", color='white')
                    self.ax.legend(loc="upper left", facecolor='#1E1E1E', labelcolor='white')
            
            elif selection.startswith("📊 標的走勢:"):
                sym = selection.split(": ")[1]
                df_market = pd.DataFrame(self.db._fetch_sync("SELECT timestamp, close FROM market_data WHERE symbol = ? AND timeframe = '1 day' ORDER BY timestamp DESC LIMIT 300", (sym,)))
                if not df_market.empty:
                    df_market['timestamp'] = pd.to_datetime(df_market['timestamp'], format='mixed', utc=True)
                    df_market.sort_values('timestamp', inplace=True)
                    
                    df_market['SMA_20'] = df_market['close'].rolling(window=20).mean()
                    first_close = df_market['close'].iloc[0]
                    df_market['Return_Pct'] = ((df_market['close'] / first_close) - 1) * 100

                    self.ax.plot(df_market['timestamp'], df_market['close'], color='#00E5FF', linewidth=2, label='Price')
                    self.ax.plot(df_market['timestamp'], df_market['SMA_20'], color='#FF3366', linewidth=1.5, linestyle='--', label='20-SMA (Trend)')
                    self.ax.set_ylabel("Price (USD)", color='#00E5FF')
                    self.ax.tick_params(axis='y', labelcolor='#00E5FF')
                    self.ax.legend(loc="upper left", facecolor='#1E1E1E', labelcolor='white')
                    
                    self.ax2 = self.ax.twinx()
                    self.ax2.plot(df_market['timestamp'], df_market['Return_Pct'], color='#00FF41', linewidth=1.5, alpha=0.7, label='Cum. Return (%)')
                    self.ax2.set_ylabel("Momentum Return (%)", color='#00FF41')
                    self.ax2.tick_params(axis='y', labelcolor='#00FF41')
                    self.ax2.legend(loc="lower right", facecolor='#1E1E1E', labelcolor='white')

                    self.ax.set_title(f"{sym} - Price Trend & Momentum Potential", color='#FFFFFF')

            self.ax.xaxis.set_major_formatter(mdates.DateFormatter('%H:%M'))
            self.figure.autofmt_xdate(rotation=45)
            self.canvas.draw()

            # 5. 更新實時終端機日誌
            self._tail_system_logs()

        except Exception as e:
            print(f"UI 更新錯誤: {e}")

        self.root.after(5000, self._update_data)

def run_ui():
    root = tk.Tk()
    app = TradingDashboard(root)
    root.mainloop()

if __name__ == "__main__":
    run_ui()