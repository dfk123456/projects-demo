# -*- coding: utf-8 -*-
# ============================================================
# startup.py —— 服务启动引导：目录准备 + 知识库自动建库 + 环境自检
# ============================================================
import os
import time

import config
from core.logging_setup import setup_logger
from core.rag import LawKnowledgeBase

logger = setup_logger()


def bootstrap(auto_build_kb: bool = True) -> dict:
    """启动前准备动作，返回环境自检摘要。"""
    summary = {}
    for d in (config.DATA_DIR, config.LOG_DIR, config.TASK_DB_DIR, config.EXPORT_DIR,
              config.LOCAL_VECTOR_DIR, config.LOCAL_STORAGE_DIR, config.KNOWLEDGE_DIR):
        os.makedirs(d, exist_ok=True)

    t0 = time.time()
    try:
        kb = LawKnowledgeBase()
        count = kb.count()
        if count <= 0 and auto_build_kb:
            res = kb.build(rebuild=False)
            count = res.get("total", 0)
            summary["kb_built"] = res
        summary["knowledge_count"] = count
    except Exception as exc:
        logger.warning("知识库初始化失败：%s", str(exc)[:160])
        summary["knowledge_count"] = -1

    summary["default_model"] = config.DEFAULT_MODEL
    summary["startup_sec"] = round(time.time() - t0, 2)
    logger.info("启动就绪：默认模型=%s，法条库=%s 条，耗时 %.2fs",
                summary["default_model"], summary.get("knowledge_count"), summary["startup_sec"])
    return summary
