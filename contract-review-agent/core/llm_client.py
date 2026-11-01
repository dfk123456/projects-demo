# ============================================================
# llm_client.py —— 外部大模型 / Embedding 统一客户端（支持多模型热切换）
#
# 设计要点：
#   1) 全部走 OpenAI 兼容协议（chat/completions、embeddings），零厂商锁定；
#      另支持本机 Ollama（/api/chat）作为离线备选项。
#   2) 模型清单写在 config.MODELS / config.EMBED_MODELS，页面上可直接切换，
#      新增模型不需要改代码，改配置即可。
#   3) 统一提供 chat() / chat_json() / embed()，内部带重试、超时、友好异常。
#   4) 在线 Embedding 不可用时自动降级为本地词法向量，保证 RAG 不中断。
# ============================================================
import json
import os
import re
import threading
import time
from typing import Any, Dict, List, Optional

import requests

import config
from core.embeddings import LocalLexicalEmbedder
from core.logging_setup import setup_logger

logger = setup_logger()

BACKEND_OPENAI = "openai"
BACKEND_OLLAMA = "ollama"
BACKEND_LOCAL = "local"


class LLMError(Exception):
    """面向用户展示的模型调用错误（中文提示）。"""


# ------------------------------------------------------------
# 配置条目解析
# ------------------------------------------------------------
def get_model_entry(model_id: str = "") -> Optional[dict]:
    """按 id 取模型条目。"""
    wanted = (model_id or config.DEFAULT_MODEL or "").strip()
    for item in config.MODELS:
        if item.get("id") == wanted:
            return item
    return None


def resolve_model(model_id: str = "") -> dict:
    """确定本次实际使用的模型条目（找不到则回退默认，再不行取第一条可用项）。"""
    return (get_model_entry(model_id)
            or get_model_entry(config.DEFAULT_MODEL)
            or (config.MODELS[0] if config.MODELS else {}))


def get_embed_entry(embed_id: str = "") -> Optional[dict]:
    wanted = (embed_id or config.DEFAULT_EMBED_MODEL or "").strip()
    for item in config.EMBED_MODELS:
        if item.get("id") == wanted:
            return item
    return None


def resolve_embed(embed_id: str = "") -> dict:
    return (get_embed_entry(embed_id)
            or get_embed_entry(config.DEFAULT_EMBED_MODEL)
            or {"id": "local-bow", "backend": "local", "dim": 1024})


def list_models() -> Dict[str, Any]:
    """返回可选对话模型清单（供 API / 前端下拉框使用）。"""
    items = []
    for m in config.MODELS:
        backend = m.get("backend") or BACKEND_OPENAI
        available, note = True, ""
        if backend == BACKEND_OPENAI:
            if not config.EXTERNAL_API_ENABLED:
                available, note = False, "外部模型总开关已关闭"
            elif not m.get("key"):
                available, note = False, "未填密钥（见 config.py）"
        items.append({
            "id": m.get("id"), "name": m.get("name") or m.get("id"),
            "model": m.get("model") or "", "backend": backend,
            "remark": m.get("remark", ""), "available": available, "note": note,
        })
    return {"default": config.DEFAULT_MODEL, "models": items}


def list_embed_models() -> Dict[str, Any]:
    items = []
    for m in config.EMBED_MODELS:
        backend = m.get("backend") or BACKEND_LOCAL
        available, note = True, ""
        if backend == BACKEND_OPENAI and not m.get("key"):
            available, note = False, "未填密钥，将自动降级为本地向量"
        items.append({
            "id": m.get("id"), "name": m.get("name") or m.get("id"),
            "model": m.get("model") or "", "backend": backend,
            "available": available, "note": note,
        })
    return {"default": config.DEFAULT_EMBED_MODEL, "models": items}


# ------------------------------------------------------------
# JSON 抽取（模型不一定老实返回纯 JSON）
# ------------------------------------------------------------
def extract_json(text: str):
    """稳健抽取模型输出中的 JSON：支持 ```json 围栏、前后缀废话、数组/对象。"""
    if not text:
        return None
    cleaned = re.sub(r"```(?:json|JSON)?", "", text).strip()
    # 优先取首尾花括号
    for opener, closer in (("{", "}"), ("[", "]")):
        start, end = cleaned.find(opener), cleaned.rfind(closer)
        if start != -1 and end != -1 and end > start:
            try:
                return json.loads(cleaned[start:end + 1])
            except Exception:
                pass
    # 退化：逐行扫描首个可解析 JSON 块
    buf, depth = "", 0
    for ch in cleaned:
        if ch in "{[":
            depth += 1
        if depth:
            buf += ch
        if ch in "}]":
            depth -= 1
            if depth == 0:
                try:
                    return json.loads(buf)
                except Exception:
                    buf = ""
    return None


# ------------------------------------------------------------
# 客户端
# ------------------------------------------------------------
class LLMClient:
    """统一的模型调用入口（线程安全，进程级单例）。"""

    def __init__(self):
        self._local_embedder = None
        self._embed_lock = threading.Lock()

    # ---------------- Embedding ----------------
    def local_embedder(self) -> LocalLexicalEmbedder:
        if self._local_embedder is None:
            with self._embed_lock:
                if self._local_embedder is None:
                    self._local_embedder = LocalLexicalEmbedder(dim=config.MILVUS_DIM)
        return self._local_embedder

    def embed_texts(self, texts: List[str], embed_id: str = "") -> List[List[float]]:
        """批量向量化。在线模型失败（且允许降级）时用本地词法向量。"""
        entry = resolve_embed(embed_id)
        if (entry.get("backend") or BACKEND_LOCAL) == BACKEND_OPENAI and entry.get("key"):
            try:
                return self._openai_embed(texts, entry)
            except Exception as exc:
                if not config.EMBED_AUTO_FALLBACK:
                    raise LLMError(f"在线向量模型调用失败：{str(exc)[:200]}")
                logger.warning("在线向量模型 %s 不可用（%s），降级为本地词法向量",
                               entry.get("id"), str(exc)[:120])
        return self.local_embedder().embed_documents(texts)

    def embed_query(self, text: str, embed_id: str = "") -> List[float]:
        return self.embed_texts([text], embed_id=embed_id)[0]

    def _openai_embed(self, texts: List[str], entry: dict) -> List[List[float]]:
        url = (entry.get("base_url") or "").rstrip("/") + "/embeddings"
        payload = {"model": entry.get("model"), "input": texts}
        resp = requests.post(url, headers=self._headers(entry), json=payload,
                             timeout=config.LLM_TIMEOUT_SEC)
        if resp.status_code != 200:
            raise LLMError(f"Embedding 接口返回 HTTP {resp.status_code}：{resp.text[:160]}")
        data = resp.json().get("data") or []
        if len(data) != len(texts):
            raise LLMError("Embedding 返回条数与输入不一致")
        ordered = sorted(data, key=lambda d: d.get("index", 0))
        return [d.get("embedding") or [] for d in ordered]

    # ---------------- Chat ----------------
    @staticmethod
    def _headers(entry: dict) -> dict:
        return {"Authorization": "Bearer " + (entry.get("key") or ""),
                "Content-Type": "application/json"}

    @staticmethod
    def to_messages(messages) -> List[dict]:
        """把外部传入的消息形态（字符串 / tuple 列表 / 字典列表）规范化。"""
        if isinstance(messages, str):
            return [{"role": "user", "content": messages}]
        out = []
        for item in messages or []:
            if isinstance(item, dict):
                out.append({"role": item.get("role", "user"), "content": item.get("content", "")})
            elif isinstance(item, (tuple, list)) and len(item) == 2:
                out.append({"role": str(item[0]), "content": str(item[1])})
        return out

    def chat(self, messages, model_id: str = "", temperature: float = None,
             max_tokens: int = None, json_mode: bool = False) -> str:
        """调用大模型生成文本；带重试与友好异常。"""
        entry = resolve_model(model_id)
        backend = entry.get("backend") or BACKEND_OPENAI
        msgs = self.to_messages(messages)
        temp = config.LLM_TEMPERATURE if temperature is None else temperature
        use_json = json_mode and config.LLM_JSON_MODE
        last_err = None

        for attempt in range(config.LLM_MAX_RETRY + 1):
            try:
                if backend == BACKEND_OLLAMA:
                    return self._ollama_chat(msgs, entry, temp, max_tokens)
                return self._openai_chat(msgs, entry, temp, max_tokens, use_json)
            except LLMError as exc:
                last_err = exc
                # JSON 模式不被支持时，降级重试一次普通模式
                if use_json and "response_format" in str(exc):
                    logger.info("模型 %s 不支持 JSON 强制输出，降级为普通模式", entry.get("id"))
                    use_json = False
                    continue
                if attempt < config.LLM_MAX_RETRY:
                    wait = config.LLM_RETRY_BASE_WAIT * (2 ** attempt)
                    logger.warning("模型调用失败（第 %s 次）：%s，%.1fs 后重试",
                                   attempt + 1, str(exc)[:120], wait)
                    time.sleep(wait)
                    continue
        raise LLMError(f"模型 {entry.get('name') or entry.get('id')} 调用失败：{str(last_err)[:200]}")

    def _openai_chat(self, msgs: List[dict], entry: dict, temp: float,
                     max_tokens: int, json_mode: bool) -> str:
        url = (entry.get("base_url") or "").rstrip("/") + "/chat/completions"
        payload: Dict[str, Any] = {
            "model": entry.get("model"),
            "messages": msgs,
            "temperature": temp,
            "stream": False,
        }
        if max_tokens or entry.get("max_tokens"):
            payload["max_tokens"] = int(max_tokens or entry.get("max_tokens"))
        if json_mode:
            payload["response_format"] = {"type": "json_object"}

        try:
            resp = requests.post(url, headers=self._headers(entry), json=payload,
                                 timeout=config.LLM_TIMEOUT_SEC)
        except requests.RequestException as exc:
            raise LLMError(f"无法连接模型服务（{entry.get('base_url')}）：{str(exc)[:120]}")

        # 厂商不支持 response_format 时，去掉后抛给上层降级重试
        if resp.status_code != 200 and json_mode:
            body = resp.text.lower()
            if "response_format" in body or "unsupported" in body or resp.status_code == 400:
                raise LLMError("response_format unsupported by provider")
        if resp.status_code != 200:
            raise LLMError(f"HTTP {resp.status_code}：{resp.text[:200]}")
        data = resp.json()
        choices = data.get("choices") or [{}]
        return (choices[0].get("message") or {}).get("content") or ""

    def _ollama_chat(self, msgs: List[dict], entry: dict, temp: float, max_tokens: int) -> str:
        url = (entry.get("base_url") or "").rstrip("/") + "/api/chat"
        payload = {"model": entry.get("model"), "messages": msgs,
                   "stream": False, "options": {"temperature": temp}}
        try:
            resp = requests.post(url, json=payload, timeout=config.LLM_TIMEOUT_SEC)
        except requests.RequestException as exc:
            raise LLMError(f"无法连接本机 Ollama（{entry.get('base_url')}）：{str(exc)[:120]}")
        if resp.status_code != 200:
            raise LLMError(f"Ollama HTTP {resp.status_code}：{resp.text[:160]}")
        return ((resp.json().get("message") or {}).get("content") or "").strip()

    # ---------------- 结构化输出 ----------------
    def chat_json(self, prompt: str, fallback=None, model_id: str = "",
                  system: str = "", max_tokens: int = None):
        """要求模型输出 JSON；解析失败时返回 fallback（保证流程不中断）。"""
        messages = []
        if system:
            messages.append({"role": "system", "content": system})
        messages.append({"role": "user", "content": prompt})
        fallback = {} if fallback is None else fallback
        try:
            raw = self.chat(messages, model_id=model_id, json_mode=True, max_tokens=max_tokens)
            parsed = extract_json(raw)
        except LLMError as exc:
            logger.warning("chat_json 失败：%s", str(exc)[:160])
            return dict(fallback)
        if isinstance(parsed, dict):
            merged = dict(fallback)
            merged.update(parsed)
            return merged
        if isinstance(parsed, list):
            return parsed
        logger.warning("chat_json 未解析出 JSON，返回默认值，原始输出片段：%s",
                       (raw or "")[:120])
        return dict(fallback)

    # ---------------- 连通性自检 ----------------
    def ping(self, model_id: str = "") -> Dict[str, Any]:
        """快速探测某个模型是否可用（不产生实际推理成本）。"""
        entry = resolve_model(model_id)
        started = time.time()
        try:
            text = self.chat("回复两个字：可用", model_id=entry.get("id"), max_tokens=8)
            return {"ok": True, "model": entry.get("id"), "reply": text[:20],
                    "latency_ms": int((time.time() - started) * 1000)}
        except Exception as exc:
            return {"ok": False, "model": entry.get("id"), "error": str(exc)[:200],
                    "latency_ms": int((time.time() - started) * 1000)}


_client: Optional[LLMClient] = None
_client_lock = threading.Lock()


def get_llm_client() -> LLMClient:
    """进程级单例入口。"""
    global _client
    if _client is None:
        with _client_lock:
            if _client is None:
                _client = LLMClient()
    return _client
