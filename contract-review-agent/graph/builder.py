# ============================================================
# builder.py —— 构建并编译 LangGraph 循环工作流
#
# 图结构：
#
#        ┌─────────────────────────── ① 片段循环 ───────────────┐
#        │                                                      │
#   [parse] → [analyze_batch] → route_next_batch ──(继续循环)────┘
#                                      │
#                                      └──(片段处理完 / 预算耗尽)→ [enrich]
#                                                                    │
#        ┌──────────────────── ② 复核循环 ──────────────────────┐    │
#        │                                                     │    ▼
#        └── [recheck] ← route_after_enrich ──→ [aggregate] → END
#
# 说明：两个 conditional edge 构成真正的「循环」，不是简单线性管道。
# ============================================================
import time
from typing import Any, Callable, Dict, Optional

from langgraph.graph import END, StateGraph

import config
from core.logging_setup import setup_logger
from graph.nodes import (node_aggregate, node_analyze_batch, node_enrich,
                         node_parse, node_recheck, route_after_enrich,
                         route_next_batch)
from graph.state import ReviewState, push_log, set_progress

logger = setup_logger()

_graph = None
_graph_lock = None


def build_graph():
    """构建 StateGraph（含两个循环的循环 Agent 工作流）。"""
    builder = StateGraph(ReviewState)

    builder.add_node("parse", node_parse)
    builder.add_node("analyze_batch", node_analyze_batch)
    builder.add_node("enrich", node_enrich)
    builder.add_node("recheck", node_recheck)
    builder.add_node("aggregate", node_aggregate)

    builder.set_entry_point("parse")
    builder.add_edge("parse", "analyze_batch")

    # ① 片段循环：处理完一批后判断是否需要再分析下一批
    builder.add_conditional_edges(
        "analyze_batch", route_next_batch,
        {"analyze_batch": "analyze_batch", "enrich": "enrich", "aggregate": "aggregate"})

    # ② 复核循环：法条补充后判断是否需要定向复核
    builder.add_conditional_edges(
        "enrich", route_after_enrich, {"recheck": "recheck", "aggregate": "aggregate"})
    builder.add_edge("recheck", "enrich")

    builder.add_edge("aggregate", END)
    return builder.compile()


def get_graph():
    """获取（单例）编译后的图。"""
    global _graph, _graph_lock
    import threading
    if _graph_lock is None:
        _graph_lock = threading.Lock()
    if _graph is not None:
        return _graph
    with _graph_lock:
        if _graph is None:
            _graph = build_graph()
            logger.info("LangGraph 循环工作流已编译（片段循环 + 复核循环）")
        return _graph


def run_review(task_id: str, file_path: str, filename: str,
               model_id: str = "", embed_id: str = "",
               progress_cb: Optional[Callable] = None) -> Dict[str, Any]:
    """执行一次完整审查，返回最终报告 dict。

    progress_cb(event)：收到 {"type":"log"/"progress", ...}，用于实时推送到前端。
    """
    graph = get_graph()
    initial = ReviewState(
        task_id=task_id, file_path=file_path, filename=filename,
        model_id=model_id or config.DEFAULT_MODEL,
        embed_id=embed_id or config.DEFAULT_EMBED_MODEL,
        started_at=time.time(), deadline=time.time() + config.TIME_BUDGET_SEC,
        logs=[], candidate_risks=[], llm_calls=0, rag_hits=0,
        stage="初始化", progress=0.0, progress_cb=progress_cb,
    )
    set_progress(initial, "启动审查流程", 0.02)
    push_log(initial, f"开始审查《{filename}》，模型：{initial['model_id']}")

    try:
        result = graph.invoke(initial, config={"recursion_limit": 1000})
    except Exception as exc:
        logger.exception("审查工作流异常终止")
        push_log(initial, f"❌ 工作流异常：{str(exc)[:200]}")
        raise

    report = result.get("final_report") or {}
    if not report.get("errors") and not report:
        report = {"executive_summary": "审查失败，未生成报告"}
    return report


def describe_graph() -> Dict[str, Any]:
    """返回图形结构描述（用于前端/接口展示勘察情感可达性）。"""
    return {
        "entry": "parse",
        "nodes": ["parse", "analyze_batch", "enrich", "recheck", "aggregate"],
        "loops": [
            {"from": "analyze_batch", "condition": "route_next_batch",
             "back_to": "analyze_batch", "desc": "片段循环：逐批把大片段送入大模型识别风险"},
            {"from": "enrich", "condition": "route_after_enrich",
             "back_to": "recheck→enrich", "desc": "复核循环：对缺法条/低置信风险点定向复核"},
        ],
        "edges": [
            ["parse", "analyze_batch"], ["analyze_batch", "analyze_batch|enrich|aggregate"],
            ["enrich", "recheck|aggregate"], ["recheck", "enrich"], ["aggregate", "END"],
        ],
    }
