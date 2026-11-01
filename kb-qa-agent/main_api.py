# ============================================================
# main_api.py —— FastAPI 后端入口（含可选控制台测试模式）
#
# 两种运行模式：
#   1) Web 模式  ：python main_api.py            （默认）
#                  启动 FastAPI + 静态页面，浏览器访问 http://127.0.0.1:8000
#   2) 控制台模式：python main_api.py --console
#                  不启动 Web 服务，直接在终端多轮问答调试
#
# 对外接口：
#   POST /chat          一次性问答 {question, enable_online_search, session_id}
#   POST /chat_stream   流式问答（SSE）：逐条推送推理日志与回答增量
#   POST /clear_history 清空指定会话（或全部会话）的多轮记忆
#   POST /upload_file   multipart(file, kb) -> PDF/TXT 增量入库
#   GET  /             静态网页
#
# 并发与隔离说明（重要）：
#   * 多轮记忆按 session_id 隔离，避免多用户 / 多标签页互相串扰；
#   * 问答是 CPU 密集型（本地 LLM 推理，单轮 30~70 秒），统一交给线程池
#     并用信号量限流，避免慢请求打满线程池拖垮整个服务；
#   * 单轮超过 config.CHAT_TIMEOUT_SEC 直接返回友好提示，不让前端无限等待。
#
# 说明：本文件不做任何 Ollama 启动逻辑，仅对接 localhost:11434。
# ============================================================
import argparse
import asyncio
import json
import logging
import os
import sys
from logging.handlers import RotatingFileHandler
import threading
import time
import uuid
from contextlib import asynccontextmanager

# 保证从任意目录运行时项目根都在导入路径里
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

# Windows 控制台默认 GBK，中文日志/输出需切到 UTF-8，避免打印时崩溃
for _stream in (sys.stdout, sys.stderr):
    if hasattr(_stream, "reconfigure"):
        _stream.reconfigure(encoding="utf-8", errors="replace")

import anyio.to_thread
import uvicorn
from fastapi import FastAPI, File, Form, HTTPException, Request, UploadFile
from fastapi.concurrency import run_in_threadpool
from fastapi.responses import FileResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

import auth
import config
import graph_builder
import llm_client
from vector_store import document_loader, math_kb, tech_kb

logger = logging.getLogger("kbqa")


# ------------------------------------------------------------
# 日志：同时输出到控制台与 logs/app.log，便于排障
# ------------------------------------------------------------
def _setup_logging():
    """日志同时输出到控制台与文件；文件按大小轮转，避免长期运行撑爆磁盘。"""
    os.makedirs(config.LOG_DIR, exist_ok=True)
    file_handler = RotatingFileHandler(
        config.LOG_FILE,
        maxBytes=config.LOG_MAX_BYTES,
        backupCount=config.LOG_BACKUP_COUNT,
        encoding="utf-8",
    )
    logging.basicConfig(
        level=getattr(logging, str(config.LOG_LEVEL).upper(), logging.INFO),
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        handlers=[file_handler, logging.StreamHandler()],
        force=True,
    )
    logging.getLogger("kbqa").setLevel(
        getattr(logging, str(config.LOG_LEVEL).upper(), logging.INFO))


# ------------------------------------------------------------
# 并发限制：问答信号量（在 lifespan 中创建）
# ------------------------------------------------------------
_chat_semaphore: asyncio.Semaphore = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    """启动时初始化日志、固定线程池上限并创建问答并发信号量。

    线程池额度比问答并发多留几个，避免文档入库等同步任务把额度占满。
    """
    global _chat_semaphore
    _setup_logging()
    auth.init_db()          # 建用户表（幂等）
    anyio.to_thread.current_default_thread_limiter().total_tokens = (
        config.WEB_MAX_CONCURRENCY + 4)
    _chat_semaphore = asyncio.Semaphore(config.WEB_MAX_CONCURRENCY)
    logger.info("服务启动：并发上限 %s，单轮超时 %ss，登录要求=%s，已注册用户 %s",
                config.WEB_MAX_CONCURRENCY, config.CHAT_TIMEOUT_SEC,
                config.AUTH_REQUIRED, auth.user_count())
    yield
    logger.info("服务停止")


app = FastAPI(
    title="离线专业知识库问答系统",
    description="LangGraph + Ollama 多智能体问答：FastAPI 后端 + Web 前端（支持流式输出）",
    version="1.1.0",
    lifespan=lifespan,
)


# ------------------------------------------------------------
# 请求/响应模型
# ------------------------------------------------------------
class ChatRequest(BaseModel):
    question: str                                   # 用户问题
    enable_online_search: bool = config.DEFAULT_ENABLE_ONLINE_SEARCH  # 本次是否允许联网
    session_id: str = ""                            # 会话标识（多轮记忆隔离用，可留空）
    token: str = ""                                 # 登录凭证（开启鉴权时必填）
    provider: str = ""                              # 模型来源：local / external（留空用默认）


class RegisterRequest(BaseModel):
    username: str
    password: str
    nickname: str = ""


class LoginRequest(BaseModel):
    username: str = ""
    password: str = ""
    token: str = ""                                 # 登录/注销复用：注销时只传 token


class ClearHistoryRequest(BaseModel):
    session_id: str = ""                            # 留空表示清空全部会话


class DeleteFileRequest(BaseModel):
    kb: str = ""                                    # tech（大模型技术）/ math（深度学习数学）
    filename: str = ""                              # 入库时使用的文件名


# ------------------------------------------------------------
# 多轮对话记忆：按 session_id 隔离（进程内，无需数据库）
# ------------------------------------------------------------
_sessions = {}          # session_id -> {"history": [...], "last": 最后活跃时间戳}
_sessions_lock = threading.Lock()


def _new_session_id() -> str:
    return uuid.uuid4().hex


def _get_session(session_id: str) -> dict:
    """取（必要时创建）会话，并顺带淘汰过期 / 超量的会话。"""
    now = time.time()
    with _sessions_lock:
        # 1) 淘汰过期会话
        expired = [sid for sid, s in _sessions.items()
                   if now - s["last"] > config.SESSION_TTL_SEC]
        for sid in expired:
            _sessions.pop(sid, None)
        # 2) 超量时淘汰最久未使用的
        if len(_sessions) > config.SESSION_MAX:
            overflow = sorted(_sessions, key=lambda k: _sessions[k]["last"])[
                :len(_sessions) - config.SESSION_MAX]
            for sid in overflow:
                _sessions.pop(sid, None)
        # 3) 取/建会话并刷新活跃时间
        sess = _sessions.get(session_id)
        if sess is None:
            sess = {"history": [], "last": now}
            _sessions[session_id] = sess
        sess["last"] = now
        return sess


def remember(question: str, answer: str, session_id: str):
    """把本轮问答写入指定会话的历史，并裁剪到最大轮数。

    话题拦截的回答不入历史（保持原有业务逻辑）。
    """
    if not (question.strip() and answer.strip()):
        return
    if answer.startswith("【话题拦截】"):
        return
    sess = _get_session(session_id)
    history = sess["history"]
    history.append({"role": "user", "content": question})
    history.append({"role": "assistant", "content": answer})
    max_items = config.HISTORY_MAX_TURNS * 2
    while len(history) > max_items:
        history.pop(0)


def handle_chat(question: str, enable_online_search: bool, session_id: str,
                provider: str = "") -> dict:
    """一次完整问答：交给 LangGraph 图执行，返回回答与日志（一次性返回）。

    provider：模型来源（local / external），None 表示用配置默认值。
    """
    result = graph_builder.run_chat(
        question=question,
        enable_online_search=enable_online_search,
        chat_history=list(_get_session(session_id)["history"]),
        provider=provider,
    )
    remember(question, result["answer"], session_id)
    return result


# ------------------------------------------------------------
# 路由：页面
# ------------------------------------------------------------
app.mount("/static", StaticFiles(directory=config.STATIC_DIR), name="static")


@app.get("/", include_in_schema=False)
def index():
    return FileResponse(os.path.join(config.STATIC_DIR, "index.html"))


# ------------------------------------------------------------
# 路由：POST /chat（一次性返回；异步 + 限流 + 超时）
# ------------------------------------------------------------
@app.post("/chat")
async def chat(req: ChatRequest):
    """执行一轮多智能体问答（一次性返回）。

    返回：answer（最终回答，含来源标注）、log（完整推理日志）、session_id。
    """
    _current_user(req.token)        # 开启鉴权时必须已登录
    question = (req.question or "").strip()
    session_id = (req.session_id or "").strip() or _new_session_id()
    if not question:
        return {"answer": "未收到有效问题，请输入你想了解的问题。",
                "log": "", "session_id": session_id}
    started = time.time()
    try:
        async with _chat_semaphore:      # 限流：避免慢请求打满线程池
            result = await asyncio.wait_for(
                run_in_threadpool(handle_chat, question,
                                  bool(req.enable_online_search), session_id,
                                  req.provider),
                timeout=config.CHAT_TIMEOUT_SEC,
            )
        logger.info("[chat] session=%s 完成，耗时 %.1fs，问题=%.40s",
                    session_id[:8], time.time() - started, question)
        return {"answer": result["answer"], "log": result["log_text"],
                "session_id": session_id}
    except asyncio.TimeoutError:
        logger.warning("[chat] session=%s 超时（%ss）",
                       session_id[:8], config.CHAT_TIMEOUT_SEC)
        return {"answer": (f"本次问答超过 {config.CHAT_TIMEOUT_SEC} 秒仍未完成，"
                           f"请稍后重试或换一个更具体的问题。"),
                "log": "[超时] 本地模型推理耗时过长，已中断本次等待。",
                "session_id": session_id}
    except Exception as exc:  # 最后防线：任何未预期异常都给友好提示
        logger.exception("[chat] session=%s 未预期异常：%s", session_id[:8], str(exc)[:200])
        return {
            "answer": f"服务端处理失败：{str(exc)[:200]}，请检查 Ollama 与日志。",
            "log": f"[1] (错误) 未预期异常\n     {str(exc)[:500]}",
            "session_id": session_id,
        }


# ------------------------------------------------------------
# 路由：POST /chat_stream（SSE 流式输出：推理日志 + 回答增量）
# ------------------------------------------------------------
def _sse(payload: dict) -> str:
    """把事件序列化成 SSE 数据帧（统一用 data 行，type 字段区分类型）。"""
    return "data: " + json.dumps(payload, ensure_ascii=False) + "\n\n"


@app.post("/chat_stream")
async def chat_stream(req: ChatRequest, request: Request):
    """流式问答：以 SSE 逐条推送推理过程与回答增量（打字机效果）。

    事件类型（data 中的 type 字段）：
      log    : {"kind", "title", "content"}  一条推理日志
      status : {"text"}                      阶段提示（如"正在生成答案"）
      token  : {"text"}                      回答增量文本
      done   : {"answer", "log_text", "session_id"}
      error  : {"text"}
    """
    _current_user(req.token)        # 开启鉴权时必须已登录
    question = (req.question or "").strip()
    session_id = (req.session_id or "").strip() or _new_session_id()

    async def event_generator():
        started = time.time()
        if not question:
            yield _sse({"type": "error", "text": "未收到有效问题。"})
            return
        logger.info("[stream] session=%s 开始，问题=%.40s", session_id[:8], question)

        # 图在独立线程执行，事件经队列回传；这里只负责非阻塞读取并推送，
        # 因此不会阻塞事件循环。
        events = graph_builder.iter_chat_events(
            question, bool(req.enable_online_search),
            list(_get_session(session_id)["history"]), req.provider)
        loop = asyncio.get_running_loop()

        def _next_event():
            """在线程中取下一个事件（同步阻塞，不占事件循环）。"""
            return next(events, None)

        try:
            async with _chat_semaphore:
                while True:
                    # 客户端已断开（用户点了"停止"或关闭页面）-> 提前结束并释放并发额度
                    if await request.is_disconnected():
                        logger.info("[stream] session=%s 客户端断开，提前结束推送",
                                    session_id[:8])
                        break
                    item = await asyncio.wait_for(
                        loop.run_in_executor(None, _next_event),
                        timeout=config.CHAT_TIMEOUT_SEC)
                    if item is None:
                        break
                    kind, payload = item
                    finished = False
                    if kind == "log":
                        frame = {"type": "log", **payload}
                    elif kind == "status":
                        frame = {"type": "status", "text": payload}
                    elif kind == "token":
                        frame = {"type": "token", "text": payload}
                    elif kind == "done":
                        remember(question, payload.get("answer", ""), session_id)
                        logger.info("[stream] session=%s 完成，耗时 %.1fs",
                                    session_id[:8], time.time() - started)
                        frame = {"type": "done",
                                 "answer": payload.get("answer", ""),
                                 "log_text": payload.get("log_text", ""),
                                 "session_id": session_id}
                        finished = True
                    elif kind == "error":
                        logger.error("[stream] session=%s 图执行错误：%s",
                                     session_id[:8], str(payload)[:200])
                        frame = {"type": "error", "text": payload}
                        finished = True
                    else:
                        continue

                    try:
                        yield _sse(frame)
                    except (asyncio.CancelledError, GeneratorExit):
                        raise                      # 取消信号需继续向外传递
                    except Exception as exc:       # 连接被客户端中断
                        logger.info("[stream] session=%s 连接已中断（%s），提前结束推送",
                                    session_id[:8], type(exc).__name__)
                        break

                    # 推送后立刻复查：用户点了"停止"时尽早结束，释放并发额度
                    if await request.is_disconnected():
                        logger.info("[stream] session=%s 客户端断开，提前结束推送",
                                    session_id[:8])
                        break
                    if finished:
                        break
        except asyncio.TimeoutError:
            logger.warning("[stream] session=%s 超时（%ss）",
                           session_id[:8], config.CHAT_TIMEOUT_SEC)
            yield _sse({"type": "error",
                        "text": (f"本次问答超过 {config.CHAT_TIMEOUT_SEC} 秒仍未完成，"
                                 f"请稍后重试。")})
        except Exception as exc:  # 最后防线
            logger.exception("[stream] session=%s 未预期异常：%s",
                             session_id[:8], str(exc)[:200])
            yield _sse({"type": "error", "text": f"服务端处理失败：{str(exc)[:200]}"})

    return StreamingResponse(
        event_generator(),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )


# ------------------------------------------------------------
# 路由：GET /model_status（可用模型来源，供前端切换本地 / 外部）
# ------------------------------------------------------------
@app.get("/model_status")
def model_status():
    """返回可用模型来源信息。

    本地=本机 Ollama（数据不出内网）；外部=OpenAI 兼容 API（需在配置中启用并填密钥）。
    """
    return llm_client.available_providers()


# ------------------------------------------------------------
# 登录鉴权：注册 / 登录 / 注销 / 当前用户
# ------------------------------------------------------------
def _current_user(token: str) -> str:
    """校验登录凭证并返回用户名；未开启鉴权时返回 anonymous。"""
    if not config.AUTH_REQUIRED:
        return "anonymous"
    username = auth.verify_token(token or "")
    if not username:
        raise HTTPException(status_code=401, detail="登录已失效，请重新登录。")
    return username


@app.post("/auth/register")
def register(req: RegisterRequest):
    """注册新用户：用户名 3~20 位（中英文数字下划线），密码至少 6 位。"""
    try:
        user = auth.register(req.username, req.password, req.nickname)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc))
    except Exception as exc:
        logger.error("注册失败 %s：%s", req.username, str(exc)[:200])
        raise HTTPException(status_code=500, detail="注册失败，请稍后重试。")
    logger.info("新用户注册：%s", user["username"])
    return {"ok": True, "message": "注册成功，请登录。", **user}


@app.post("/auth/login")
def login(req: LoginRequest):
    """登录成功返回凭证 token（前端保存后用于后续请求）。"""
    try:
        result = auth.login(req.username, req.password)
    except ValueError as exc:
        raise HTTPException(status_code=401, detail=str(exc))
    except Exception as exc:
        logger.error("登录失败 %s：%s", req.username, str(exc)[:200])
        raise HTTPException(status_code=500, detail="登录失败，请稍后重试。")
    logger.info("用户登录：%s", result["username"])
    return {"ok": True, "message": "登录成功。", **result}


@app.post("/auth/logout")
def logout(req: LoginRequest):
    """注销登录凭证。"""
    auth.logout(req.token)
    return {"ok": True}


@app.get("/auth/me")
def me(token: str = ""):
    """返回当前登录用户；凭证无效时返回 401。"""
    return {"username": _current_user(token),
            "auth_required": config.AUTH_REQUIRED}


@app.get("/auth/status")
def auth_status():
    """前端初始化用：是否需要登录、已注册用户数（决定首屏显示注册还是登录）。"""
    return {"auth_required": config.AUTH_REQUIRED, "users": auth.user_count()}


# ------------------------------------------------------------
# 路由：GET /health（健康检查，部署后用于探活）
# ------------------------------------------------------------
@app.get("/health")
def health():
    """健康检查：服务存活 + Ollama 连通性 + 两库文档块数 + 活跃会话数。

    status：ok=Ollama 在线；degraded=服务在但 Ollama 不可用（问答会走兜底）。
    """
    from llm_client import get_llm_client
    try:
        ollama_online = bool(get_llm_client().ping())
    except Exception as exc:
        ollama_online = False
        logger.warning("健康检查：Ollama 探测失败 %s", str(exc)[:120])

    try:
        tech_chunks = tech_kb.count()
        math_chunks = math_kb.count()
    except Exception:
        tech_chunks = math_chunks = 0

    return {
        "status": "ok" if ollama_online else "degraded",
        "version": app.version,
        "ollama": {
            "online": ollama_online,
            "base_url": config.OLLAMA_BASE_URL,
            "chat_model": config.CHAT_MODEL,
            "embedding_model": config.EMBEDDING_MODEL,
        },
        "kb": {"tech": tech_chunks, "math": math_chunks},
        "sessions": len(_sessions),
    }


# ------------------------------------------------------------
# 路由：GET /kb_status（知识库状态，供前端展示）
# ------------------------------------------------------------
def _kb_summary(module, manifest_file: str) -> dict:
    """读取某个知识库的文档块数与已入库文件清单。"""
    files = []
    try:
        with open(manifest_file, "r", encoding="utf-8") as fh:
            data = json.load(fh)
        for name, info in (data.get("files") or {}).items():
            files.append({"name": name, "chunks": info.get("chunks", 0)})
    except Exception:
        pass
    try:
        count = module.count()
    except Exception:
        count = 0
    return {"chunks": count, "files": files}


@app.get("/kb_status")
def kb_status():
    """返回两个知识库的文档块数与文件清单（前端上传后据此刷新展示）。"""
    return {
        "tech": _kb_summary(tech_kb, config.TECH_MANIFEST_FILE),
        "math": _kb_summary(math_kb, config.MATH_MANIFEST_FILE),
    }


# ------------------------------------------------------------
# 路由：DELETE /kb_file（删除某文件在知识库中的全部文档块）
# ------------------------------------------------------------
@app.delete("/kb_file")
def delete_kb_file(req: DeleteFileRequest):
    """删除指定知识库中某个文件的全部文档块，并同步移除入库清单记录。

    用途：上传错文件、或课件作废后清理知识库（需要时可重新上传同一文件）。
    """
    if req.kb not in ("tech", "math"):
        raise HTTPException(status_code=400,
                            detail="kb 参数必须为 tech（大模型技术）或 math（深度学习数学）。")
    filename = (req.filename or "").strip()
    if not filename:
        raise HTTPException(status_code=400, detail="filename 不能为空。")

    try:
        result = document_loader.delete_document(req.kb, filename)
    except Exception as exc:
        logger.error("删除失败 kb=%s file=%s：%s", req.kb, filename, str(exc)[:200])
        raise HTTPException(status_code=500, detail=f"删除失败：{str(exc)[:200]}")

    logger.info("文档删除 kb=%s file=%s 原记录存在=%s 剩余块数=%s",
                req.kb, filename, result["existed"], result["chunks"])
    return {"ok": True, **result,
            "message": f"已删除「{filename}」在知识库中的全部文档块，"
                       f"库内剩余 {result['chunks']} 块。"}


# ------------------------------------------------------------
# 路由：POST /clear_history（清空多轮记忆）
# ------------------------------------------------------------
@app.post("/clear_history")
def clear_history(req: ClearHistoryRequest):
    """清空指定会话（session_id 留空则清空全部会话）的多轮记忆。"""
    sid = (req.session_id or "").strip()
    with _sessions_lock:
        if sid:
            _sessions.pop(sid, None)
        else:
            _sessions.clear()
    return {"ok": True}


# ------------------------------------------------------------
# 路由：POST /upload_file（PDF/TXT 增量入库）
# ------------------------------------------------------------
def _validate_upload_content(raw: bytes, ext: str) -> str:
    """按文件内容校验（不只看扩展名），返回错误说明；通过则返回空字符串。"""
    if not raw:
        return "上传文件为空。"
    if ext == ".pdf":
        if not raw.startswith(b"%PDF-"):
            return "扩展名是 .pdf，但文件头不是 %PDF-，可能不是有效的 PDF 文件。"
        return ""
    for encoding in ("utf-8", "gb18030", "utf-16"):   # .txt
        try:
            raw.decode(encoding)
            return ""
        except (UnicodeDecodeError, ValueError):
            continue
    return "扩展名是 .txt，但内容无法按 UTF-8 / GB18030 解码，可能不是文本文件。"


@app.post("/upload_file")
async def upload_file(
        file: UploadFile = File(...),
        kb: str = Form("tech"),
        token: str = Form("")):
    """接收 PDF/TXT 文档并增量写入指定知识库（tech=大模型技术 / math=深度学习数学）。"""
    _current_user(token)            # 开启鉴权时必须已登录
    if kb not in ("tech", "math"):
        raise HTTPException(status_code=400,
                            detail="kb 参数必须为 tech（大模型技术）或 math（深度学习数学）。")

    filename = file.filename or "未命名文档"
    ext = os.path.splitext(filename)[1].lower()
    if ext not in (".pdf", ".txt"):
        raise HTTPException(status_code=400,
                            detail="仅支持 PDF / TXT 文档。")

    raw = await file.read()
    if len(raw) > config.MAX_UPLOAD_MB * 1024 * 1024:
        raise HTTPException(status_code=400,
                            detail=f"文件超过 {config.MAX_UPLOAD_MB}MB 限制。")

    # 内容校验：不只看扩展名，避免伪装文件入库
    invalid = _validate_upload_content(raw, ext)
    if invalid:
        logger.warning("上传被拒 kb=%s file=%s 原因=%s", kb, filename, invalid)
        raise HTTPException(status_code=400, detail=invalid)

    try:
        result = document_loader.add_document(kb, filename, raw)
    except Exception as exc:  # Ollama 离线 / 文档损坏等全部转成友好信息
        friendly = str(exc)
        if "ollama" in friendly.lower() or "连接" in friendly or "模型" in friendly:
            friendly = (f"入库失败：{friendly}。"
                        f"请确认 Ollama 已启动并已执行 ollama pull {config.EMBEDDING_MODEL}。")
        logger.error("入库失败 kb=%s file=%s：%s", kb, filename, str(exc)[:300])
        raise HTTPException(status_code=500, detail=friendly)

    logger.info("文档入库 kb=%s file=%s status=%s chunks=%s",
                kb, filename, result.get("status"), result.get("chunks"))
    return {
        "ok": True,
        "filename": filename,
        "status": result.get("status"),
        "message": result.get("message"),
        "chunks": result.get("chunks", 0),
    }


# ------------------------------------------------------------
# 控制台测试模式
# ------------------------------------------------------------
def console_mode():
    session_id = _new_session_id()
    print("=" * 60)
    print("  离线专业知识库问答系统 —— 控制台测试模式")
    print("  输入 exit / quit 退出；每次提问会打印完整推理日志。")
    print(f"  Ollama：{config.OLLAMA_BASE_URL} | 模型：{config.CHAT_MODEL}")
    print("=" * 60)

    while True:
        try:
            question = input("\n你 > ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\n再见。")
            break
        if question.lower() in ("exit", "quit", "q", "退出"):
            print("再见。")
            break

        online_flag = input("允许联网检索？(y/N，默认 N) > ").strip().lower() == "y"

        result = handle_chat(question, online_flag, session_id)
        print("\n【AI 回答】\n" + result["answer"])
        print("\n【完整推理日志】\n" + result["log_text"])


def main():
    parser = argparse.ArgumentParser(description="离线专业知识库问答系统")
    parser.add_argument("--console", action="store_true",
                        help="以控制台测试模式运行（不启动 Web 服务）")
    parser.add_argument("--host", default=config.WEB_HOST)
    parser.add_argument("--port", type=int, default=config.WEB_PORT)
    args = parser.parse_args()

    if args.console:
        console_mode()
    else:
        print(f"Web 服务启动：http://127.0.0.1:{args.port} （Ollama={config.OLLAMA_BASE_URL}）")
        uvicorn.run("main_api:app", host=args.host, port=args.port, reload=False)


if __name__ == "__main__":
    main()
