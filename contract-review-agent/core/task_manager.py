# ============================================================
# task_manager.py —— 审查任务管理（线程执行 + 进度推送 + 结果持久化）
# ============================================================
import json
import os
import queue
import threading
import time
import uuid
from typing import Any, Callable, Dict, List, Optional

import config
from core.logging_setup import setup_logger
from core.report import finalize_report

logger = setup_logger()

TASK_FILE = os.path.join(config.TASK_DB_DIR, "tasks.json")


class Task(dict):
    """任务记录（dict 形态，便于 JSON 序列化）。"""


class TaskManager:
    """进程内任务中心：提交任务 → 后台线程跑图 → 结果落盘 → 前端可轮询/SSE。"""

    def __init__(self):
        os.makedirs(config.TASK_DB_DIR, exist_ok=True)
        self.tasks: Dict[str, Dict[str, Any]] = {}
        self.subscribers: Dict[str, List[queue.Queue]] = {}
        self._lock = threading.RLock()
        self._load()

    # ---------------- 持久化 ----------------
    def _load(self):
        if os.path.exists(TASK_FILE):
            try:
                with open(TASK_FILE, "r", encoding="utf-8") as fh:
                    data = json.load(fh)
                for tid, task in data.items():
                    task.setdefault("logs", [])
                    self.tasks[tid] = task
                logger.info("已恢复历史任务 %s 条", len(self.tasks))
            except Exception as exc:
                logger.warning("历史任务加载失败：%s", str(exc)[:120])

    def _persist(self):
        try:
            os.makedirs(config.TASK_DB_DIR, exist_ok=True)
            snapshot = {}
            with self._lock:
                for tid, task in self.tasks.items():
                    copy = dict(task)
                    copy.pop("thread", None)
                    snapshot[tid] = copy
            with open(TASK_FILE, "w", encoding="utf-8") as fh:
                json.dump(snapshot, fh, ensure_ascii=False, indent=2)
        except Exception as exc:
            logger.warning("任务持久化失败：%s", str(exc)[:120])

    # ---------------- 订阅（SSE） ----------------
    def subscribe(self, task_id: str) -> queue.Queue:
        q: queue.Queue = queue.Queue(maxsize=500)
        with self._lock:
            self.subscribers.setdefault(task_id, []).append(q)
        return q

    def unsubscribe(self, task_id: str, q: queue.Queue):
        with self._lock:
            subs = self.subscribers.get(task_id) or []
            if q in subs:
                subs.remove(q)

    def _publish(self, task_id: str, event: Dict[str, Any]):
        with self._lock:
            subs = list(self.subscribers.get(task_id) or [])
        for q in subs:
            try:
                q.put_nowait(event)
            except Exception:
                pass

    # ---------------- 创建/查询 ----------------
    def create(self, filename: str, file_key: str, model_id: str,
               embed_id: str = "") -> Dict[str, Any]:
        task_id = time.strftime("%Y%m%d%H%M%S") + "-" + uuid.uuid4().hex[:6]
        task = Task(id=task_id, filename=filename, file_key=file_key,
                    model_id=model_id, embed_id=embed_id,
                    status="pending", stage="排队中", progress=0.0, logs=[],
                    report=None, error="", created_at=time.strftime("%Y-%m-%d %H:%M:%S"),
                    elapsed_sec=0.0)
        with self._lock:
            self.tasks[task_id] = task
        self._persist()
        return task

    def get(self, task_id: str) -> Optional[Dict[str, Any]]:
        with self._lock:
            task = self.tasks.get(task_id)
            return dict(task) if task else None

    def list(self, limit: int = 50) -> List[Dict[str, Any]]:
        with self._lock:
            items = [dict(t) for t in self.tasks.values()]
        items.sort(key=lambda x: x.get("created_at", ""), reverse=True)
        for it in items:
            it.pop("logs", None)
        return items[:limit]

    def update(self, task_id: str, **fields):
        with self._lock:
            task = self.tasks.get(task_id)
            if not task:
                return
            task.update(fields)
        self._persist()

    def save_report(self, task_id: str, report: Dict[str, Any]) -> Dict[str, Any]:
        """保存人工编辑后的报告（并重算统计）。"""
        report["task_id"] = task_id
        finalize_report(report)
        with self._lock:
            task = self.tasks.get(task_id)
            if task:
                task["report"] = report
        self._persist()
        return report

    # ---------------- 执行 ----------------
    def submit(self, task: Dict[str, Any], runner: Callable):
        """在后台线程执行审查 runner（负责把进度写回任务记录）。"""
        def _worker():
            task_id = task["id"]
            started = time.time()
            self.update(task_id, status="running", stage="启动中")

            def cb(event: Dict[str, Any]):
                if event.get("type") == "log" and event.get("message"):
                    with self._lock:
                        self.tasks[task_id].setdefault("logs", []).append(event["message"])
                elif event.get("type") == "progress":
                    with self._lock:
                        self.tasks[task_id]["stage"] = event.get("stage", "")
                        self.tasks[task_id]["progress"] = event.get("progress", 0.0)
                self._publish(task_id, event)

            try:
                report = runner(task, cb)
                self.update(task_id, status="done", stage="已完成", progress=1.0,
                            report=report, elapsed_sec=round(time.time() - started, 2))
                self._publish(task_id, {"type": "done", "report": report})
            except Exception as exc:
                logger.exception("任务 %s 执行失败", task_id)
                self.update(task_id, status="failed", stage="失败",
                            error=str(exc)[:300], elapsed_sec=round(time.time() - started, 2))
                self._publish(task_id, {"type": "error", "error": str(exc)[:300]})
            finally:
                self._persist()
                self._publish(task_id, {"type": "closed"})

        tid = task["id"]
        th = threading.Thread(target=_worker, name=f"task-{tid}", daemon=True)
        th.start()


_manager = None
_manager_lock = threading.Lock()


def get_task_manager() -> TaskManager:
    global _manager
    if _manager is None:
        with _manager_lock:
            if _manager is None:
                _manager = TaskManager()
    return _manager
