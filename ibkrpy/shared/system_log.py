# ibkrpy/shared/system_log.py
# 統一處理系統日誌輸出與嚴重錯誤警報 (支援 Zero-Blocking 非阻塞架構)

import logging
import os
import queue
import atexit
from logging.handlers import RotatingFileHandler, SMTPHandler, QueueHandler, QueueListener
from typing import Dict, Any

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# 宣告全域的 Listener 以便在系統關閉時優雅結束
_log_listener = None

def setup_logger(config: Dict[str, Any] = None) -> logging.Logger:
    """
    初始化全域日誌系統。
    導入 QueueListener 實現非阻塞架構，確保寫檔與寄送 Email 不會卡死主程式的 asyncio 迴圈。
    """
    global _log_listener
    cfg = config or {}
    logger = logging.getLogger("ibkrpy")
    
    # 避免重複綁定 Handler
    if logger.handlers:
        return logger
        
    logger.setLevel(logging.DEBUG)
    formatter = logging.Formatter('%(asctime)s [%(levelname)s] %(filename)s:%(lineno)d - %(message)s')

    # 用於收集所有實際執行 I/O 的 Handlers
    io_handlers = []

    # 1. 控制台輸出 (INFO 及以上)
    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.INFO)
    console_handler.setFormatter(formatter)
    io_handlers.append(console_handler)

    # 2. 檔案輸出 (DEBUG 及以上，帶有自動輪轉機制)
    log_dir = os.path.join(PROJECT_ROOT, "logs")
    os.makedirs(log_dir, exist_ok=True)
    file_handler = RotatingFileHandler(
        os.path.join(log_dir, "trading_bot.log"),
        maxBytes=10 * 1024 * 1024, # 10 MB
        backupCount=5,
        encoding="utf-8"
    )
    file_handler.setLevel(logging.DEBUG)
    file_handler.setFormatter(formatter)
    io_handlers.append(file_handler)

    # 3. Email 警報機制 (僅 ERROR 與 CRITICAL) 
    smtp_settings = cfg.get("alert_settings", {})
    if smtp_settings.get("enable_email_alerts"):
        mail_handler = SMTPHandler(
            mailhost=(smtp_settings.get("smtp_server"), smtp_settings.get("smtp_port", 587)),
            fromaddr=smtp_settings.get("sender_email"),
            toaddrs=[smtp_settings.get("receiver_email")],
            subject="[Trading Bot Alert] 系統發生嚴重異常",
            credentials=(smtp_settings.get("sender_email"), smtp_settings.get("sender_password")),
            secure=() # 啟用 TLS
        )
        mail_handler.setLevel(logging.ERROR) 
        mail_handler.setFormatter(formatter)
        io_handlers.append(mail_handler)

    
    # 建立一個無上限的執行緒安全 Queue
    log_queue = queue.Queue(-1)
    
    # 主 logger 只需要掛載一個極速的 QueueHandler
    queue_handler = QueueHandler(log_queue)
    logger.addHandler(queue_handler)

    _log_listener = QueueListener(log_queue, *io_handlers, respect_handler_level=True)
    _log_listener.start()
    
    atexit.register(_log_listener.stop)

    return logger

# 提供一個全域實例供其他模組導入
global_logger = logging.getLogger("ibkrpy")