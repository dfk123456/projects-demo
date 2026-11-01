# -*- coding: utf-8 -*-
# ============================================================
# main_api.py —— FastAPI 服务入口（长文档深度分析 Agent · 合同审查系统）
#
# 接口一览：
#   GET  /api/health                     健康检查（后端可用性/存储/向量库/模型）
#   GET  /api/models                     可选大模型清单（模型切换）
#   GET  /api/models/ping                测试某个模型连通性
#   POST /api/models/reload              热加载 config.MODELS（改配置即时生效）
#   GET  /api/embed-models               可选向量模型清单
#   POST /api/upload                     上传合同 PDF
#   GET  /api/tasks                      任务列表
#   POST /api/review/start               启动审查任务
#   GET  /api/review/{task_id}           任务状态 + 报告
#   GET  /api/review/{task_id}/stream    SSE 实时进度
#   PUT  /api/review/{task_id}/report    保存人工编辑后的报告
#   POST /api/review/{task_id}/export    导出 Word / Markdown
#   GET  /api/download/{filename}        下载导出文件
#   POST /api/knowledge/rebuild          重建法律法规知识库
#   GET  /api/knowledge/info             知识库状态
#   GET  /api/graph                      查看 LangGraph 工作流结构
# ============================================================
import importlib
import json
import os
import time
from typing import Any, Dict

from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.responses import (FileResponse, HTMLResponse, JSONResponse,
                               PlainTextResponse, StreamingResponse)
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

import config
from core.llm_client import list_embed_models, list_models, resolve_model
from core.logging_setup import setup_logger
from core.rag import LawKnowledgeBase
from core.report import export_docx, export_markdown
from core.storage import get_storage
from core.task_manager import get_task_manager
from core.vectorstore import get_vector_store
from graph.builder import describe_graph, run_review

logger = setup_logger()
app = FastAPI(title="长文档深度分析 Agent · 合同风险审查系统", version="1.0.0")

_storage = get_storage()
_tasks = get_task_manager()


def _task_or_404(task_id: str) -> Dict[str, Any]:
    task = _tasks.get(task_id)
    if not task:
        raise HTTPException(status_code=404, detail=f"任务不存在：{task_id}")
    return task


# ------------------------------------------------------------
# 静态页面
# ------------------------------------------------------------
@app.get("/", response_class=HTMLResponse)
async def index():
    html_path = os.path.join(config.STATIC_DIR, "index.html")
    if not os.path.exists(html_path):
        return PlainTextResponse("静态页面缺失，请确认 static/index.html")
    with open(html_path, "r", encoding="utf-8") as fh:
        return HTMLResponse(fh.read())


app.mount("/static", StaticFiles(directory=config.STATIC_DIR), name="static")
os.makedirs(config.EXPORT_DIR, exist_ok=True)


# ------------------------------------------------------------
# 健康检查 / 模型清单
# ------------------------------------------------------------
@app.get("/api/health")
async def health():
    store = get_vector_store()
    try:
        kb_count = store.count()
    except Exception:
        kb_count = -1
    return {
        "ok": True,
        "service": "contract-review-agent",
        "time": time.strftime("%Y-%m-%d %H:%M:%S"),
        "storage_backend": type(_storage).__name__,
        "vector_backend": type(store).__name__,
        "knowledge_count": kb_count,
        "default_model": config.DEFAULT_MODEL,
        "models": len(config.MODELS),
        "max_upload_mb": config.MAX_UPLOAD_MB,
    }


@app.get("/api/models")
async def api_models():
    return list_models()


@app.get("/api/models/ping")
async def api_ping(model_id: str = ""):
    from core.llm_client import get_llm_client
    return get_llm_client().ping(model_id=model_id or config.DEFAULT_MODEL)


@app.post("/api/models/reload")
async def api_reload_models():
    """热加载模型配置：改完 config.MODELS 后调用，不改服务立即生效。"""
    try:
        importlib.reload(config)
        return {"ok": True, "models": list_models()}
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"重载失败：{str(exc)[:200]}")


@app.get("/api/embed-models")
async def api_embed_models():
    return list_embed_models()


@app.get("/api/graph")
async def api_graph():
    return describe_graph()


# ------------------------------------------------------------
# 上传
# ------------------------------------------------------------
@app.post("/api/upload")
async def api_upload(file: UploadFile = File(...)):
    ext = os.path.splitext(file.filename or "")[1].lower()
    if ext not in config.SUPPORTED_EXT:
        raise HTTPException(status_code=400,
                            detail=f"不支持的文件类型 {ext}，仅支持 {', '.join(config.SUPPORTED_EXT)}")
    data = await file.read()
    size_mb = len(data) / (1024 * 1024)
    if size_mb > config.MAX_UPLOAD_MB:
        raise HTTPException(status_code=400,
                            detail=f"文件过大 {size_mb:.1f}MB（上限 {config.MAX_UPLOAD_MB}MB）")
    key = f"{config.MINIO_BUCKET_CONTRACT}/{time.strftime('%Y%m%d')}/{int(time.time())}_{file.filename}"
    _storage.save(key, data, content_type=file.content_type or "application/pdf")
    logger.info("文件上传成功：%s（%.2fMB）→ %s", file.filename, size_mb, key)
    return {"ok": True, "filename": file.filename, "file_key": key,
            "size_mb": round(size_mb, 3)}


# ------------------------------------------------------------
# 审查任务
# ------------------------------------------------------------
class ReviewStartRequest(BaseModel):
    file_key: str
    filename: str = ""
    model_id: str = ""
    embed_id: str = ""


@app.post("/api/review/start")
async def api_review_start(req: ReviewStartRequest):
    model_id = req.model_id or config.DEFAULT_MODEL
    if not any(m.get("id") == model_id for m in config.MODELS):
        raise HTTPException(status_code=400, detail=f"未知模型 id：{model_id}")
    filename = req.filename or os.path.basename(req.file_key)
    task = _tasks.create(filename, req.file_key, model_id, req.embed_id)

    def runner(t, cb):
        return run_review(t["id"], t["file_key"], t["filename"],
                          model_id=t.get("model_id"), embed_id=t.get("embed_id"),
                          progress_cb=cb)

    _tasks.submit(task, runner)
    return {"ok": True, "task_id": task["id"], "model": resolve_model(model_id).get("name")}


@app.get("/api/tasks")
async def api_tasks(limit: int = 30):
    return {"tasks": _tasks.list(limit=limit)}


@app.get("/api/review/{task_id}")
async def api_task_status(task_id: str, full: int = 0):
    task = _task_or_404(task_id)
    data = dict(task)
    data.pop("thread", None)
    if not full:
        data["logs"] = data.get("logs", [])[-200:]
    return data


@app.get("/api/review/{task_id}/stream")
async def api_task_stream(task_id: str):
    """SSE 实时推送：日志 / 进度 / 完成 / 失败。"""
    task = _task_or_404(task_id)
    q = _tasks.subscribe(task_id)

    def gen():
        try:
            # 先补发当前已有日志，避免前端连接晚于任务启动
            for line in (task.get("logs") or [])[-50:]:
                yield f"data: {json.dumps({'type': 'log', 'message': line}, ensure_ascii=False)}\n\n"
            yield f"data: {json.dumps({'type': 'progress', 'stage': task.get('stage', ''), 'progress': task.get('progress', 0.0)}, ensure_ascii=False)}\n\n"
            if task.get("status") in ("done", "failed"):
                payload = {"type": "done" if task.get("status") == "done" else "error"}
                if task.get("report"):
                    payload["report"] = task["report"]
                yield f"data: {json.dumps(payload, ensure_ascii=False)}\n\n"
                yield "data: {\"type\":\"closed\"}\n\n"
                return
            while True:
                try:
                    event = q.get(timeout=25)
                except Exception:
                    yield ": keep-alive\n\n"
                    continue
                yield f"data: {json.dumps(event, ensure_ascii=False)}\n\n"
                if event.get("type") in ("done", "error", "closed"):
                    break
        finally:
            _tasks.unsubscribe(task_id, q)

    return StreamingResponse(gen(), media_type="text/event-stream",
                             headers={"Cache-Control": "no-cache",
                                      "X-Accel-Buffering": "no"})


class ReportUpdate(BaseModel):
    report: Dict[str, Any]


@app.put("/api/review/{task_id}/report")
async def api_save_report(task_id: str, body: ReportUpdate):
    """保存 Web 页面上人工编辑后的报告。"""
    _task_or_404(task_id)
    report = _tasks.save_report(task_id, body.report)
    return {"ok": True, "report": report}


class ExportRequest(BaseModel):
    fmt: str = "docx"          # docx | md | json
    filename: str = ""


@app.post("/api/review/{task_id}/export")
async def api_export(task_id: str, body: ExportRequest):
    task = _task_or_404(task_id)
    report = task.get("report")
    if not report:
        raise HTTPException(status_code=400, detail="该任务暂无报告可导出")

    base = os.path.splitext(os.path.basename(task.get("filename") or "report"))[0]
    stamp = time.strftime("%H%M%S")
    fmt = (body.fmt or "docx").lower()
    if fmt == "docx":
        name = f"审查报告_{base}_{stamp}.docx"
        path = os.path.join(config.EXPORT_DIR, name)
        export_docx(report, path)
    elif fmt == "md":
        name = f"审查报告_{base}_{stamp}.md"
        path = os.path.join(config.EXPORT_DIR, name)
        with open(path, "w", encoding="utf-8") as fh:
            fh.write(export_markdown(report))
    elif fmt == "json":
        name = f"审查报告_{base}_{stamp}.json"
        path = os.path.join(config.EXPORT_DIR, name)
        with open(path, "w", encoding="utf-8") as fh:
            json.dump(report, fh, ensure_ascii=False, indent=2)
    else:
        raise HTTPException(status_code=400, detail=f"不支持的导出格式：{fmt}")
    logger.info("导出 %s → %s", fmt, name)
    return {"ok": True, "filename": name, "url": f"/api/download/{name}", "format": fmt}


@app.get("/api/download/{filename}")
async def api_download(filename: str, as_attachment: int = 1):
    path = os.path.join(config.EXPORT_DIR, os.path.basename(filename))
    if not os.path.exists(path):
        raise HTTPException(status_code=404, detail="导出文件不存在")
    if not as_attachment:
        return FileResponse(path)
    return FileResponse(path, filename=filename,
                        media_type="application/vnd.openxmlformats-officedocument.wordprocessingml.document")


@app.delete("/api/review/{task_id}")
async def api_delete_task(task_id: str):
    _task_or_404(task_id)
    _tasks.update(task_id, status="deleted", report=None, logs=[])
    return {"ok": True}


# ------------------------------------------------------------
# 法律法规知识库
# ------------------------------------------------------------
@app.post("/api/knowledge/rebuild")
async def api_knowledge_rebuild(rebuild: int = 0, embed_id: str = ""):
    kb = LawKnowledgeBase(embed_id=embed_id or config.DEFAULT_EMBED_MODEL)
    try:
        result = kb.build(rebuild=bool(rebuild))
        return {"ok": True, **result}
    except Exception as exc:
        logger.exception("知识库构建失败")
        raise HTTPException(status_code=500, detail=f"知识库构建失败：{str(exc)[:200]}")


@app.get("/api/knowledge/info")
async def api_knowledge_info():
    kb = LawKnowledgeBase()
    files = []
    if os.path.isdir(config.KNOWLEDGE_DIR):
        files = [f for f in os.listdir(config.KNOWLEDGE_DIR)
                 if f.lower().endswith((".md", ".txt"))]
    try:
        count = kb.count()
    except Exception:
        count = -1
    return {"count": count, "files": files, "top_k": config.RAG_TOP_K,
            "embed_default": config.DEFAULT_EMBED_MODEL}


# ------------------------------------------------------------
# 启动入口
# ------------------------------------------------------------
def ensure_dirs():
    for d in (config.DATA_DIR, config.LOG_DIR, config.TASK_DB_DIR,
              config.EXPORT_DIR, config.LOCAL_VECTOR_DIR,
              config.LOCAL_STORAGE_DIR, config.KNOWLEDGE_DIR):
        os.makedirs(d, exist_ok=True)


ensure_dirs()


def main():
    import uvicorn
    from api.startup import bootstrap

    bootstrap()
    logger.info("服务启动：http://%s:%s", config.WEB_HOST, config.WEB_PORT)
    uvicorn.run(app, host=config.WEB_HOST, port=config.WEB_PORT, log_level="info")


if __name__ == "__main__":
    main()
