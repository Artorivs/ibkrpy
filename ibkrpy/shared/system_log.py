# ibkrpy/shared/system_log.py
# 統一處理系統日誌輸出與嚴重錯誤警報 (支援 Zero-Blocking 非阻塞架構)

import logging
import os
import queue
import atexit
from logging.handlers import RotatingFileHandler, SMTPHandler, QueueHandler, QueueListener
from typing import Dict, Any

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

_log_listener = None


class RateLimitFilter(logging.Filter):
    """
    同一個發生點在時間窗內只放行一次，並在下次放行時附上被抑制的次數。

    email 警報最常見的失效方式不是漏發，而是轟炸：某個每 5 分鐘執行一次的
    迴圈開始持續失敗，一夜之間寄出上百封相同內容的信，收件人就會設規則
    全部歸檔 —— 接著真正的問題發生時也沒人看見。

    以 (模組, 行號, 等級) 為鍵，而非訊息內容，因為訊息通常含有變動的例外文字。
    """

    def __init__(self, interval_seconds: int = 1800):
        super().__init__()
        self.interval = interval_seconds
        self._last = {}      # key -> 上次放行的時間
        self._suppressed = {}  # key -> 期間被抑制的次數

    def filter(self, record: logging.LogRecord) -> bool:
        import time
        key = (record.module, record.lineno, record.levelno)
        now = time.time()
        last = self._last.get(key)

        if last is not None and (now - last) < self.interval:
            self._suppressed[key] = self._suppressed.get(key, 0) + 1
            return False

        n = self._suppressed.pop(key, 0)
        if n:
            record.msg = f"{record.getMessage()}\n\n[同一位置在過去 {self.interval // 60} 分鐘內另有 {n} 次相同等級的事件被抑制]"
            record.args = ()

        self._last[key] = now
        return True


def setup_logger(config: Dict[str, Any] = None, enable_file: bool = True) -> logging.Logger:
    """
    初始化全域日誌系統。

    :param config: config.yaml 中 log_settings 區塊的內容 (dict)。
                   例如 ConfigManager().get("log_settings")
    :param enable_file: 是否掛載檔案 handler。

    [修正] RotatingFileHandler 不是多行程安全的 —— 輪替時是「重新命名再開新檔」，
    兩個行程同時觸發，其中一方會繼續寫入已被改名的 inode，那段日誌就此消失。
    daemon 主行程與重訓子行程都會呼叫本函式，因此子行程必須傳 enable_file=False。
    子行程的 stdout 本來就由 SystemDaemon 逐行轉發進日誌，重複寫檔既冗餘又危險。

    導入 QueueListener 實現非阻塞架構，確保寫檔與寄送 Email 不會卡死 asyncio 迴圈。
    """
    global _log_listener
    cfg = config or {}
    logger = logging.getLogger("ibkrpy")

    if logger.handlers:
        return logger

    logger.setLevel(logging.DEBUG)
    # 不向 root 傳遞，避免第三方套件呼叫 basicConfig() 後造成重複輸出
    logger.propagate = False

    formatter = logging.Formatter('%(asctime)s [%(levelname)s] %(filename)s:%(lineno)d - %(message)s')
    io_handlers = []

    # 1. 控制台輸出 —— 等級改由 log_settings.level 決定
    console_level = getattr(logging, str(cfg.get("level", "INFO")).upper(), logging.INFO)
    console_handler = logging.StreamHandler()
    console_handler.setLevel(console_level)
    console_handler.setFormatter(formatter)
    io_handlers.append(console_handler)

    # 2. 檔案輸出 (DEBUG 及以上，帶有自動輪轉機制)
    if enable_file:
        log_dir = os.path.join(PROJECT_ROOT, "logs")
        os.makedirs(log_dir, exist_ok=True)
        file_handler = RotatingFileHandler(
            os.path.join(log_dir, "trading_bot.log"),
            maxBytes=10 * 1024 * 1024,
            backupCount=5,
            encoding="utf-8",
        )
        file_handler.setLevel(logging.DEBUG)
        file_handler.setFormatter(formatter)
        io_handlers.append(file_handler)

    # 3. Email 警報機制 (僅 ERROR 與 CRITICAL)
    if cfg.get("enable_email_alerts"):
        required = ("smtp_server", "sender_email", "receiver_email", "sender_password")
        missing = [k for k in required if not cfg.get(k)]
        if missing:
            # 明確報錯，而不是靜默跳過 —— 使用者以為開啟了警報，實際上沒有
            logging.getLogger("ibkrpy").error(
                f"已啟用 email 警報，但 log_settings 缺少必要欄位: {missing}，警報未生效。"
            )
        else:
            mail_handler = SMTPHandler(
                mailhost=(cfg["smtp_server"], cfg.get("smtp_port", 587)),
                fromaddr=cfg["sender_email"],
                toaddrs=[cfg["receiver_email"]],
                subject="[Trading Bot Alert] 系統發生嚴重異常",
                credentials=(cfg["sender_email"], cfg["sender_password"]),
                secure=(),
            )
            mail_handler.setLevel(logging.ERROR)
            mail_handler.setFormatter(formatter)
            # 只對 email 套用節流；控制台與檔案仍完整記錄每一次事件
            mail_handler.addFilter(RateLimitFilter(
                interval_seconds=int(cfg.get("alert_throttle_minutes", 30)) * 60
            ))
            io_handlers.append(mail_handler)

    log_queue = queue.Queue(-1)
    logger.addHandler(QueueHandler(log_queue))

    _log_listener = QueueListener(log_queue, *io_handlers, respect_handler_level=True)
    _log_listener.start()
    atexit.register(_log_listener.stop)

    return logger


# 提供一個全域實例供其他模組導入
global_logger = logging.getLogger("ibkrpy")