# ============================================================
# llm_client.py —— 封装 Ollama 大模型与 Embedding 调用客户端
# 功能：
#   1) 懒加载 ChatOllama（qwen:7b）与 OllamaEmbeddings（bge-small）
#   2) chat() 统一文本对话 / ask_json() 稳定解析模型 JSON 输出
#   3) ping() 探测 Ollama 是否在线
#   4) 统一异常处理：把连接失败/模型缺失翻译成友好中文提示，
#      保证业务代码拿到异常后不会导致程序崩溃
# ============================================================
import json
import logging
import re
import threading

import config

logger = logging.getLogger("kbqa")

# 模型来源
PROVIDER_LOCAL = "local"          # 本地 Ollama：数据不出内网
PROVIDER_EXTERNAL = "external"    # 外部 OpenAI 兼容 API：能力更强，数据将外发


# ------------------------------------------------------------
# 自定义异常：向用户展示的友好错误
# ------------------------------------------------------------
class OllamaError(Exception):
    """Ollama 服务不可用 / 模型未拉取 等可预期错误，message 面向用户。"""


class MissingDependencyError(Exception):
    """缺少 pip 依赖（如 langchain_ollama 未安装）。"""


def _build_friendly(detail: str) -> str:
    """把底层异常详情加工成面向用户/日志的中文提示。"""
    if "Connection" in detail or "ConnectError" in detail or "connect" in detail.lower():
        return (f"无法连接本地 Ollama 服务（{config.OLLAMA_BASE_URL}）。"
                f"请确认 Ollama 已启动（命令：ollama serve）。")
    if "model" in detail.lower() and ("not found" in detail.lower() or "pull" in detail.lower()):
        return (f"Ollama 缺少所需模型。请先执行："
                f"ollama pull {config.CHAT_MODEL} 与 ollama pull {config.EMBEDDING_MODEL}。")
    if "Timeout" in detail or "timeout" in detail.lower():
        return "本地 Ollama 响应超时，可能正在加载模型，请稍后重试。"
    # 其余细节一并保留便于排障
    return f"调用本地 Ollama 时出现错误：{detail}"


def _extract_json_block(text: str):
    """从模型文本中稳健地抽取出第一个 {...} JSON 对象；找不到返回 None。"""
    if not text:
        return None
    # 先去掉可能的 ```json ... ``` 代码围栏
    text = re.sub(r"```(?:json)?", "", text)
    start = text.find("{")
    end = text.rfind("}")
    if start == -1 or end == -1 or end <= start:
        return None
    try:
        return json.loads(text[start:end + 1])
    except Exception:
        return None


# ------------------------------------------------------------
# LLM 客户端（单例懒加载）
# ------------------------------------------------------------
class LLMClient:
    """统一封装 ChatOllama / OllamaEmbeddings / 结果解析。"""

    def __init__(self):
        self._chat_models = {}      # (base_url, model) -> ChatOllama 实例
        self._embeddings = None     # OllamaEmbeddings 实例
        self._lock = threading.Lock()

    # ---------------- 底层实例懒加载 ----------------
    def get_chat_model(self, model: str = None, base_url: str = None):
        """返回 ChatOllama 实例（按「地址 + 模型」缓存，懒加载）。

        支持多个本地模型：不同的 model 会各自创建并缓存一个实例。
        """
        model = model or config.CHAT_MODEL
        base_url = base_url or config.OLLAMA_BASE_URL
        cache_key = (base_url, model)
        if self._chat_models.get(cache_key) is None:
            with self._lock:
                if self._chat_models.get(cache_key) is None:
                    try:
                        from langchain_ollama import ChatOllama
                    except ImportError as exc:
                        raise MissingDependencyError(
                            "未安装 langchain-ollama，请先执行 pip install -r requirements.txt") from exc
                    self._chat_models[cache_key] = ChatOllama(
                        base_url=base_url,
                        model=model,
                        temperature=config.LLM_TEMPERATURE,
                        num_ctx=config.LLM_NUM_CTX,
                        num_gpu=config.LLM_NUM_GPU,
                        # 底层 HTTP 超时：推理卡死时能中断等待，不让线程被无限挂住
                        sync_client_kwargs={"timeout": config.LLM_TIMEOUT_SEC},
                    )
        return self._chat_models[cache_key]

    def get_embeddings(self):
        """返回 OllamaEmbeddings 实例（进程内单例，懒加载）。"""
        if self._embeddings is None:
            try:
                from langchain_ollama import OllamaEmbeddings
            except ImportError as exc:
                raise MissingDependencyError(
                    "未安装 langchain-ollama，请先执行 pip install -r requirements.txt") from exc
            self._embeddings = OllamaEmbeddings(
                base_url=config.OLLAMA_BASE_URL,
                model=config.EMBEDDING_MODEL,
                num_gpu=config.LLM_NUM_GPU,
                sync_client_kwargs={"timeout": config.LLM_TIMEOUT_SEC},
            )
        return self._embeddings

    # ---------------- 对外方法 ----------------
    def ping(self) -> bool:
        """探测 Ollama /api/tags 是否可达（不做任何推理，速度快）。"""
        try:
            import requests
            resp = requests.get(config.OLLAMA_BASE_URL + "/api/tags", timeout=3)
            return resp.status_code == 200
        except Exception:
            return False

    # ---------------- 模型条目解析（对应 config.MODELS） ----------------
    @staticmethod
    def _find_entry(model_id: str):
        """按 id 查找模型条目；找不到返回 None。"""
        for item in config.MODELS or []:
            if item.get("id") == model_id:
                return item
        return None

    @classmethod
    def _local_entry(cls):
        """兜底本地条目：MODELS 里第一个 ollama 条目，没有则按默认配置构造。"""
        for item in config.MODELS or []:
            if (item.get("backend") or "ollama") == "ollama":
                return item
        return {"id": PROVIDER_LOCAL, "name": "本地 Ollama", "backend": "ollama",
                "base_url": config.OLLAMA_BASE_URL, "model": config.CHAT_MODEL}

    def _resolve_entry(self, provider: str = "") -> dict:
        """确定本次使用的模型条目。

        找不到、外部总开关关闭或未填密钥时，一律回退本地默认模型，保证可用。
        """
        wanted = (provider or config.DEFAULT_MODEL or "").strip()
        entry = (self._find_entry(wanted)
                 or self._find_entry(config.DEFAULT_MODEL)
                 or self._local_entry())
        if (entry.get("backend") or "ollama") == "openai":
            if not config.EXTERNAL_API_ENABLED:
                logger.warning("外部模型总开关已关闭，改用本地模型")
                return self._local_entry()
            if not entry.get("key"):
                logger.warning("模型 %s 未填写密钥，改用本地模型", entry.get("id"))
                return self._local_entry()
        return entry

    # ---------------- 外部 API（OpenAI 兼容协议，零新增依赖） ----------------
    @staticmethod
    def _to_api_messages(messages) -> list:
        """把内部消息形态（字符串 / 元组列表）转成 API 要求的字典列表。"""
        if isinstance(messages, str):
            return [{"role": "user", "content": messages}]
        result = []
        for item in messages or []:
            if isinstance(item, (tuple, list)) and len(item) == 2:
                result.append({"role": str(item[0]), "content": str(item[1])})
            elif isinstance(item, dict):
                result.append(item)
        return result

    def _api_request(self, messages, stream: bool = False, entry: dict = None):
        """发起外部 API 请求（OpenAI 兼容的 chat/completions）。

        entry 来自 config.MODELS，决定 base_url / model / key。
        """
        import requests
        entry = entry or {}
        url = (entry.get("base_url") or "").rstrip("/") + "/chat/completions"
        payload = {
            "model": entry.get("model"),
            "messages": self._to_api_messages(messages),
            "temperature": config.LLM_TEMPERATURE,
            "stream": stream,
        }
        return requests.post(
            url,
            headers={
                "Authorization": "Bearer " + (entry.get("key") or ""),
                "Content-Type": "application/json",
            },
            json=payload,
            timeout=config.EXTERNAL_API_TIMEOUT,
            stream=stream,
        )

    def _api_chat_once(self, messages, entry: dict) -> str:
        """外部 API 一次性调用，返回完整文本。"""
        resp = self._api_request(messages, stream=False, entry=entry)
        if resp.status_code != 200:
            raise OllamaError(f"外部 API 调用失败（HTTP {resp.status_code}）：{resp.text[:200]}")
        data = resp.json()
        choices = data.get("choices") or [{}]
        return (choices[0].get("message") or {}).get("content") or ""

    def _api_chat_stream(self, messages, entry: dict):
        """外部 API 流式调用，逐块 yield 文本（SSE 增量解析）。"""
        resp = self._api_request(messages, stream=True, entry=entry)
        if resp.status_code != 200:
            raise OllamaError(f"外部 API 调用失败（HTTP {resp.status_code}）：{resp.text[:200]}")
        for line in resp.iter_lines(decode_unicode=True):
            if not line:
                continue
            if not line.startswith("data:"):
                continue
            chunk = line[5:].strip()
            if chunk == "[DONE]":
                break
            try:
                data = json.loads(chunk)
            except Exception:
                continue
            choices = data.get("choices") or [{}]
            text = (choices[0].get("delta") or {}).get("content")
            if text:
                yield text

    def chat(self, messages, provider: str = "") -> str:
        """调用大模型（本地或外部）。

        provider：None 用配置默认；"local" 本地 Ollama；"external" 外部 API。
        外部调用失败时自动回退本地并在日志中记录。

        messages 支持两种形态：
          * 字符串               —— 视为单条 user 消息
          * [(role, content),...] —— LangChain 消息列表
        返回模型生成的文本；失败抛出 OllamaError（中文友好提示）。
        """
        entry = self._resolve_entry(provider)
        if entry.get("backend") == "openai":
            try:
                return self._api_chat_once(messages, entry).strip()
            except Exception as exc:
                logger.warning("外部模型 %s 调用失败，回退本地模型：%s",
                               entry.get("name"), str(exc)[:200])
                entry = self._local_entry()
        return self._ollama_chat(messages, entry)

    # ---------------- 本地 Ollama 调用 ----------------
    def _ollama_chat(self, messages, entry: dict) -> str:
        """用本地 Ollama 一次性生成。"""
        model = self.get_chat_model(entry.get("model"), entry.get("base_url"))
        if isinstance(messages, str):
            messages = [("user", messages)]
        try:
            result = model.invoke(messages)
            content = result.content
            if isinstance(content, list):  # 兼容部分版本返回 content 块数组
                parts = [c.get("text", "") if isinstance(c, dict) else str(c)
                         for c in content]
                return "".join(parts).strip()
            return str(content).strip()
        except (OllamaError, MissingDependencyError):
            raise
        except Exception as exc:  # 连接失败 / 模型缺失 / 超时等统一友好化
            raise OllamaError(_build_friendly(str(exc))) from exc

    def _ollama_stream(self, messages, entry: dict):
        """用本地 Ollama 流式生成，逐块 yield 文本。"""
        model = self.get_chat_model(entry.get("model"), entry.get("base_url"))
        if isinstance(messages, str):
            messages = [("user", messages)]
        try:
            for chunk in model.stream(messages):
                content = getattr(chunk, "content", "")
                if isinstance(content, list):
                    parts = [c.get("text", "") if isinstance(c, dict) else str(c)
                             for c in content]
                    content = "".join(parts)
                if content:
                    yield str(content)
        except (OllamaError, MissingDependencyError):
            raise
        except Exception as exc:
            raise OllamaError(_build_friendly(str(exc))) from exc

    def stream_chat(self, messages, provider: str = ""):
        """流式调用大模型（本地或外部），逐块 yield 文本（保持原始顺序）。

        用于前端打字机效果：调用方拿到一块就推给前端一块。
        messages 形态与 chat() 一致（字符串或 [(role, content), ...]）。
        外部调用在尚未产出任何内容时失败，会自动回退本地流式。
        """
        entry = self._resolve_entry(provider)
        if entry.get("backend") == "openai":
            produced = False
            try:
                for piece in self._api_chat_stream(messages, entry):
                    produced = True
                    yield piece
                return
            except Exception as exc:
                if produced:      # 已经吐了一部分，不能再回退，交给上层兜底
                    raise OllamaError(f"外部模型流式中断：{str(exc)[:200]}")
                logger.warning("外部模型 %s 流式失败，回退本地模型：%s",
                               entry.get("name"), str(exc)[:200])
                entry = self._local_entry()

        for piece in self._ollama_stream(messages, entry):
            yield piece

    def ask_json(self, prompt: str, fallback: dict, provider: str = "") -> dict:
        """让模型输出 JSON 并解析。

        fallback 为兜底默认值：解析失败或字段缺失时，调用方用 fallback
        覆盖缺省字段，保证任何情况下流程都能继续。
        """
        try:
            raw = self.chat(prompt, provider=provider)
            parsed = _extract_json_block(raw) or {}
        except (OllamaError, MissingDependencyError):
            parsed = {}
        # 用 fallback 补齐缺失字段，保留模型给出来的字段
        merged = dict(fallback)
        if isinstance(parsed, dict):
            for k, v in parsed.items():
                merged[k] = v
        return merged

    def embed_texts(self, texts):
        """批量向量化（入库使用）。"""
        try:
            emb = self.get_embeddings()
            return emb.embed_documents(texts)
        except (OllamaError, MissingDependencyError):
            raise
        except Exception as exc:
            raise OllamaError(_build_friendly(str(exc))) from exc

    def embed_query(self, text):
        """单条查询向量化（检索使用）。"""
        try:
            emb = self.get_embeddings()
            return emb.embed_query(text)
        except (OllamaError, MissingDependencyError):
            raise
        except Exception as exc:
            raise OllamaError(_build_friendly(str(exc))) from exc


# ------------------------------------------------------------
# 进程级单例入口（全项目共用同一个实例，避免重复建连）
# ------------------------------------------------------------
_client = None
_client_lock = threading.Lock()


def get_llm_client() -> LLMClient:
    global _client
    if _client is None:
        with _client_lock:
            if _client is None:
                _client = LLMClient()
    return _client


def available_providers() -> dict:
    """返回 config.MODELS 里的全部可选模型，供接口与前端展示/切换使用。

    本地(ollama) 始终可选；外部(openai) 需要填了密钥才可选。
    """
    providers = []
    for item in config.MODELS or []:
        backend = item.get("backend") or "ollama"
        available, note = True, ("数据不出内网" if backend == "ollama" else "")
        if backend == "openai":
            if not config.EXTERNAL_API_ENABLED:
                available, note = False, "外部模型总开关已关闭"
            elif not item.get("key"):
                available, note = False, "未填密钥（见 config.py）"
        providers.append({
            "id": item.get("id"),
            "name": item.get("name") or item.get("id"),
            "model": item.get("model") or "",
            "backend": backend,
            "available": available,
            "note": note,
        })
    return {
        "default": config.DEFAULT_MODEL,
        "providers": providers,
        "send_context": config.EXTERNAL_SEND_CONTEXT,
    }


def is_external(provider: str = "") -> bool:
    """判断某个模型 id 是否为外部模型。

    用于控制：外部模型 + EXTERNAL_SEND_CONTEXT=False 时，课件片段不外发。
    """
    entry = get_llm_client()._resolve_entry(provider)
    return (entry.get("backend") or "ollama") == "openai"
