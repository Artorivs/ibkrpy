# ibkrpy/shared/system_log.py
# Logging setup and throttled critical-error email alerts

import logging
import os
import queue
import atexit
from logging.handlers import (
    RotatingFileHandler,
    SMTPHandler,
    QueueHandler,
    QueueListener,
)
from typing import Dict, Any

from datetime import datetime

PROJECT_ROOT = os.path.dirname(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
)

_log_listener = None


class RateLimitFilter(logging.Filter):
    """
    Lets each call site through once per time window, and reports how many were suppressed.

    Email alerts usually fail by flooding, not by silence: a loop that fails every
    5 minutes sends hundreds of identical emails overnight, the recipient filters
    them, and the real incident goes unseen.

    Keyed on (module, line, level), not message text, which often varies.
    """

    def __init__(self, interval_seconds: int = 1800):
        super().__init__()
        self.interval = interval_seconds
        self._last = {}
        self._suppressed = {}

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
            record.msg = f"{record.getMessage()}\n\n[{n} more events at this location and level suppressed in the last {self.interval // 60} min]"
            record.args = ()

        self._last[key] = now
        return True


def setup_logger(
    config: Dict[str, Any] = None, enable_file: bool = True
) -> logging.Logger:
    """
    Initialize logging.

    :param config: the log_settings block of config.yaml (dict),
                                  e.g. ConfigManager().get("log_settings")
    :param enable_file: whether to attach the file handler.
    """
    global _log_listener
    cfg = config or {}
    logger = logging.getLogger("ibkrpy")

    if logger.handlers:
        return logger

    logger.setLevel(logging.DEBUG)
    logger.propagate = False

    formatter = logging.Formatter(
        "%(asctime)s [%(levelname)s] %(filename)s:%(lineno)d - %(message)s"
    )
    io_handlers = []

    console_level = getattr(
        logging, str(cfg.get("level", "INFO")).upper(), logging.INFO
    )
    console_handler = logging.StreamHandler()
    console_handler.setLevel(console_level)
    console_handler.setFormatter(formatter)
    io_handlers.append(console_handler)

    if enable_file:
        log_dir = os.path.join(PROJECT_ROOT, "logs")
        os.makedirs(log_dir, exist_ok=True)
        file_handler = RotatingFileHandler(
            os.path.join(
                log_dir, f"trading_bot_{datetime.now().strftime('%Y-%m-%d')}.log"
            ),
            maxBytes=10 * 1024 * 1024,
            backupCount=5,
            encoding="utf-8",
        )
        file_handler.setLevel(logging.DEBUG)
        file_handler.setFormatter(formatter)
        io_handlers.append(file_handler)

    if cfg.get("enable_email_alerts"):
        required = ("smtp_server", "sender_email", "receiver_email", "sender_password")
        missing = [k for k in required if not cfg.get(k)]
        if missing:
            logging.getLogger("ibkrpy").error(
                f"Email alerts are enabled but log_settings is missing: {missing}; alerts are off."
            )
        else:
            mail_handler = SMTPHandler(
                mailhost=(cfg["smtp_server"], cfg.get("smtp_port", 587)),
                fromaddr=cfg["sender_email"],
                toaddrs=[cfg["receiver_email"]],
                subject="[Trading Bot Alert] Critical error",
                credentials=(cfg["sender_email"], cfg["sender_password"]),
                secure=(),
            )
            mail_handler.setLevel(logging.ERROR)
            mail_handler.setFormatter(formatter)
            mail_handler.addFilter(
                RateLimitFilter(
                    interval_seconds=int(cfg.get("alert_throttle_minutes", 30)) * 60
                )
            )
            io_handlers.append(mail_handler)

    log_queue = queue.Queue(-1)
    logger.addHandler(QueueHandler(log_queue))

    _log_listener = QueueListener(log_queue, *io_handlers, respect_handler_level=True)
    _log_listener.start()
    atexit.register(_log_listener.stop)

    return logger


global_logger = logging.getLogger("ibkrpy")
