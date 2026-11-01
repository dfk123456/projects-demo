# ============================================================
# logging_setup.py —— 日志初始化（控制台 + 落盘轮转）
# ============================================================
import logging
import os
import sys
from logging.handlers import RotatingFileHandler

import config


def setup_logger(name: str = "contract_agent") -> logging.Logger:
    """创建并返回项目统一 logger，重复调用只会初始化一次。"""
    logger = logging.getLogger(name)
    if getattr(logger, "_inited", False):
        return logger

    logger.setLevel(getattr(logging, config.LOG_LEVEL.upper(), logging.INFO))
    logger.propagate = False
    fmt = logging.Formatter(
        "%(asctime)s | %(levelname)-7s | %(threadName)-10s | %(message)s",
        datefmt="%H:%M:%S")

    # 控制台
    sh = logging.StreamHandler(stream=sys.stdout)
    sh.setFormatter(fmt)
    logger.addHandler(sh)

    # 文件（轮转）
    try:
        os.makedirs(config.LOG_DIR, exist_ok=True)
        fh = RotatingFileHandler(config.LOG_FILE, maxBytes=config.LOG_MAX_BYTES,
                                 backupCount=config.LOG_BACKUP_COUNT, encoding="utf-8")
        fh.setFormatter(fmt)
        logger.addHandler(fh)
    except Exception:
        logger.warning("日志文件初始化失败，仅输出到控制台")

    logger._inited = True
    return logger
