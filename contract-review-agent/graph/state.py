# ============================================================
# state.py —— LangGraph 工作流状态定义（长文档循环审查 Agent）
#
# 用 TypedDict 描述（LangGraph 原生支持），配套提供：
#   push_log()     —— 写过程日志并推送给前端
#   set_progress() —— 更新阶段与进度
# ============================================================
import time
from typing import Any, Callable, Dict, List, Optional

from typing_extensions import TypedDict


class ReviewState(TypedDict, total=False):
    """一次合同审查任务的图状态。"""

    # ---- 任务基本信息 ----
    task_id: str
    file_path: str
    filename: str
    model_id: str
    embed_id: str

    # ---- 文档 & 切片 ----
    pages: int
    chunks: List[Dict[str, Any]]
    cursor: int
    prev_summary: str
    parser: str

    # ---- 过程数据 ----
    candidate_risks: List[Dict[str, Any]]
    logs: List[str]
    llm_calls: int
    rag_hits: int
    enriched_ids: List[str]

    # ---- 控制字段 ----
    stage: str
    progress: float
    error: str
    recheck_count: int
    deadline: float
    truncated: bool
    started_at: float
    elapsed_sec: float

    # ---- 结果 ----
    final_report: Optional[Dict[str, Any]]
    progress_cb: Optional[Callable]


def _callback(state: ReviewState) -> Optional[Callable]:
    fn = state.get("progress_cb")
    return fn if callable(fn) else None


def push_log(state: ReviewState, message: str):
    """记录一行过程日志，并实时推送（SSE/轮询可见）。"""
    state.setdefault("logs", []).append(message)
    cb = _callback(state)
    if cb:
        try:
            cb({"type": "log", "message": message,
                "stage": state.get("stage", ""), "progress": state.get("progress", 0.0)})
        except Exception:
            pass


def set_progress(state: ReviewState, stage: str, progress: float):
    """更新阶段与完成度（0~1）。"""
    state["stage"] = stage
    state["progress"] = round(max(0.0, min(1.0, float(progress))), 4)
    cb = _callback(state)
    if cb:
        try:
            cb({"type": "progress", "stage": stage, "progress": state["progress"]})
        except Exception:
            pass


def elapsed(state: ReviewState) -> float:
    return max(0.0, time.time() - float(state.get("started_at") or time.time()))


def time_left(state: ReviewState) -> float:
    deadline = float(state.get("deadline") or 0)
    return deadline - time.time() if deadline else 1e9
