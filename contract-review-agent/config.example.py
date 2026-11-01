# ============================================================
# config.example.py —— 配置模板
#
# 用法：
#   1) 复制本文件为 config.py：  cp config.example.py config.py
#   2) 在 config.py 中填入你自己的模型密钥（或设置同名环境变量）
#   3) config.py 已被 .gitignore 忽略，不会上传到 GitHub
#
# 所有参数均可用同名环境变量覆盖，例如：
#   export DEEPSEEK_API_KEY="sk-你的key"
#   export OLLAMA_NUM_CTX=4096
# ============================================================

# ============================================================
# config.py —— 全局配置（长文档深度分析 Agent · 法律合同审查系统）
#
# 集中管理全部可调参数：
#   1) 外部大模型清单（可在页面上随时切换；想加模型就往 MODELS 里加一条）
#   2) Embedding（向量化）方案：在线向量模型 / 本地词法向量
#   3) 向量库后端：Milvus（正式环境） / 本地文件向量库（免依赖开箱即用）
#   4) 文件存储：MinIO（正式环境） / 本地磁盘（默认）
#   5) 文档切分、并发、时间预算等 Agent 运行参数
#
# 所有参数均可被同名环境变量覆盖，方便部署时不动代码。
# ============================================================
import os

# 项目根目录（本文件所在目录）
BASE_DIR = os.path.dirname(os.path.abspath(__file__))


def _env_bool(name: str, default: bool) -> bool:
    """环境变量布尔值解析（'0'/'false'/'no' 视为 False）。"""
    val = os.environ.get(name)
    if val is None:
        return default
    return str(val).strip().lower() in ("1", "true", "yes", "on")


# ============================================================
# 1) 外部大模型清单 —— 【要加模型就改这里】
# ============================================================
# 页面上「模型来源」下拉 = 下面这个列表，支持任意 OpenAI 兼容接口。
#
# 每个条目字段：
#   id        唯一标识（英文，别重复）
#   name      下拉里显示的名字
#   backend   openai = OpenAI 兼容接口（DeepSeek / 通义 / Kimi / 智谱 / vLLM / Ollama-兼容模式均可）
#             ollama = 本机 Ollama（数据不出内网，需本机已启动 ollama serve）
#   base_url  接口地址（必须带 http:// 或 https://）
#   model     模型名（决定实际调用哪个模型）
#   key       密钥（backend=openai 时需要）
#   max_tokens / ctx / remark 可选辅助字段
#
# 想加几个模型就加几条；同一家想用不同模型，复制一条改 model 即可。
MODELS = [
    # ---- DeepSeek：主力，速度快、中文法律文本理解好（默认）----
    {"id": "deepseek-flash", "name": "DeepSeek · 通用对话（快）", "backend": "openai",
     "base_url": "https://api.deepseek.com/v1", "model": "deepseek-chat",
     "key": os.environ.get("DEEPSEEK_API_KEY", ""),
     "max_tokens": 4096, "remark": "性价比高，适合大批量分片审查"},

    # ---- 小米 MiMo ----
    {"id": "mimo-flash", "name": "小米 MiMo · v2.6-flash", "backend": "openai",
     "base_url": "https://api.xiaomimimo.com/v1", "model": "mimo-v2.6-flash",
     "key": os.environ.get("MIMO_API_KEY", ""),
     "max_tokens": 4096},
    {"id": "mimo-pro", "name": "小米 MiMo · v2.6-pro（更准）", "backend": "openai",
     "base_url": "https://api.xiaomimimo.com/v1", "model": "mimo-v2.6-pro",
     "key": os.environ.get("MIMO_API_KEY", ""),
     "max_tokens": 4096},

    # ---- 通义千问 DashScope（OpenAI 兼容模式）；官方要求长上下文可用 qwen-long ----
    {"id": "qwen-long", "name": "通义千问 · qwen-long（长上下文）", "backend": "openai",
     "base_url": os.environ.get("DASHSCOPE_BASE_URL", "https://dashscope.aliyuncs.com/compatible-mode/v1"),
     "model": "qwen-long", "key": os.environ.get("DASHSCOPE_API_KEY", ""),
     "max_tokens": 4096, "remark": "需填 DashScope 密钥，擅长超长合同"},

    # ---- 本机 Ollama（可选，数据不出内网）----
    {"id": "local-ollama", "name": "本机 Ollama · qwen2.5:7b（离线）", "backend": "ollama",
     "base_url": os.environ.get("OLLAMA_BASE_URL", "http://localhost:11434"),
     "model": os.environ.get("OLLAMA_CHAT_MODEL", "qwen2.5:7b"), "max_tokens": 4096},

    # ---- 其他 OpenAI 兼容服务：复制出来改三处即可 ----
    # {"id": "kimi", "name": "Kimi · moonshot-v1", "backend": "openai",
    #  "base_url": "https://api.moonshot.cn/v1", "model": "moonshot-v1-128k", "key": "sk-xxx"},
    # {"id": "vllm-local", "name": "内网 vLLM · Qwen2.5-32B", "backend": "openai",
    #  "base_url": "http://10.0.0.8:8000/v1", "model": "Qwen2.5-32B-Instruct", "key": "EMPTY"},
]

# 默认使用的模型（填上面某个条目的 id）
DEFAULT_MODEL = os.environ.get("DEFAULT_MODEL", "deepseek-flash")

# ============================================================
# 2) Embedding（向量化）方案 —— 【要加向量模型也在这里】
# ============================================================
# backend: local = 内置词法向量（零依赖、离线可用、无需密钥）
#          openai = OpenAI 兼容 /embeddings 接口
EMBED_MODELS = [
    {"id": "local-bow", "name": "本地词法向量（离线 fallback）", "backend": "local",
     "model": "jieba-boew-1024", "dim": 1024},

    {"id": "bge-m3-siliconflow", "name": "BAAI/bge-m3（SiliconFlow）", "backend": "openai",
     "base_url": "https://api.siliconflow.cn/v1", "model": "BAAI/bge-m3",
     "key": os.environ.get("SILICONFLOW_API_KEY", ""), "dim": 1024},

    {"id": "qwen-text-embedding-v3", "name": "通义 text-embedding-v3", "backend": "openai",
     "base_url": os.environ.get("DASHSCOPE_BASE_URL", "https://dashscope.aliyuncs.com/compatible-mode/v1"),
     "model": "text-embedding-v3", "key": os.environ.get("DASHSCOPE_API_KEY", ""), "dim": 1024},

    {"id": "openai-3-small", "name": "OpenAI text-embedding-3-small", "backend": "openai",
     "base_url": os.environ.get("OPENAI_BASE_URL", "https://api.openai.com/v1"),
     "model": "text-embedding-3-small", "key": os.environ.get("OPENAI_API_KEY", ""), "dim": 1536},
]

DEFAULT_EMBED_MODEL = os.environ.get("DEFAULT_EMBED_MODEL", "local-bow")
# 在线向量模型不可用时是否自动降级为本地词法向量（强烈建议 True，保证流程不中断）
EMBED_AUTO_FALLBACK = _env_bool("EMBED_AUTO_FALLBACK", True)

# ============================================================
# 3) 大模型调用通用参数
# ============================================================
LLM_TEMPERATURE = float(os.environ.get("LLM_TEMPERATURE", "0.1"))   # 审查场景要稳定，别太发散
LLM_TIMEOUT_SEC = int(os.environ.get("LLM_TIMEOUT_SEC", "120"))     # 单次请求超时
LLM_MAX_RETRY = int(os.environ.get("LLM_MAX_RETRY", "2"))           # 失败重试次数
LLM_RETRY_BASE_WAIT = float(os.environ.get("LLM_RETRY_BASE_WAIT", "1.5"))
# 是否尝试强制 JSON 输出（不支持的厂商会自动降级为普通输出 + 正则抽取）
LLM_JSON_MODE = _env_bool("LLM_JSON_MODE", True)
EXTERNAL_API_ENABLED = _env_bool("EXTERNAL_API_ENABLED", True)

# ============================================================
# 4) 文档解析
# ============================================================
# 解析引擎优先级：unstructured -> pdfplumber -> pypdf
DOC_PARSER = os.environ.get("DOC_PARSER", "auto")     # auto / unstructured / pdfplumber / pypdf
MAX_UPLOAD_MB = int(os.environ.get("MAX_UPLOAD_MB", "80"))
SUPPORTED_EXT = (".pdf", ".txt", ".md", ".docx")

# ============================================================
# 5) 文本切分（Agent 循环的基本单位 = 一个「大片段」）
# ============================================================
CHUNK_SIZE = int(os.environ.get("CHUNK_SIZE", "3500"))         # 每个大片段字符数
CHUNK_OVERLAP = int(os.environ.get("CHUNK_OVERLAP", "500"))    # 重叠字符数（防跨片段漏检）
SUMMARY_CHARS = int(os.environ.get("SUMMARY_CHARS", "220"))    # 片段尾部摘要（带入下一片段）
# 单片最大字符（硬截断保护，防止超大模型窗口）
CHUNK_HARD_MAX = int(os.environ.get("CHUNK_HARD_MAX", "6000"))
# 是否携带上一片段摘要（对策：解决超长文档上下文割裂）
CARRY_PREV_SUMMARY = _env_bool("CARRY_PREV_SUMMARY", True)

# ============================================================
# 6) Agent 运行参数（性能 / 时间预算）
# ============================================================
BATCH_SIZE = int(os.environ.get("BATCH_SIZE", "16"))           # 图每轮循环处理的大片段数
MAX_WORKERS = int(os.environ.get("MAX_WORKERS", "16"))         # 每轮并发调用大模型的线程数
TIME_BUDGET_SEC = int(os.environ.get("TIME_BUDGET_SEC", "110"))  # 分析阶段总预算（秒，满足 200 页 ≤120s）
MAX_LOOPS = int(os.environ.get("MAX_LOOPS", "200"))            # while 循环安全上限，防死循环
MAX_RECHECK = int(os.environ.get("MAX_RECHECK", "1"))          # 汇总后「缺法条/低置信」复核循环轮数
ENABLE_RECHECK = _env_bool("ENABLE_RECHECK", True)
MAX_RISK_PER_CHUNK = int(os.environ.get("MAX_RISK_PER_CHUNK", "8"))
MAX_RISK_IN_REPORT = int(os.environ.get("MAX_RISK_IN_REPORT", "80"))  # 报告保留的风险上限（按等级排序）

# ============================================================
# 7) 法律条文 RAG 知识库
# ============================================================
KNOWLEDGE_DIR = os.path.join(BASE_DIR, "knowledge_docs")
RAG_TOP_K = int(os.environ.get("RAG_TOP_K", "3"))              # 每个风险点检索法条条数
RAG_MIN_SCORE = float(os.environ.get("RAG_MIN_SCORE", "0.18"))  # 低于该分视为无相关法条
RAG_KEYWORD_BOOST = float(os.environ.get("RAG_KEYWORD_BOOST", "0.12"))  # 关键词命中加权

# ============================================================
# 8) 向量库：Milvus（正式） / 本地文件库（默认，开箱即用）
# ============================================================
VECTOR_BACKEND = os.environ.get("VECTOR_BACKEND", "auto")      # auto / milvus / local
MILVUS_URI = os.environ.get("MILVUS_URI", os.path.join(BASE_DIR, "data", "vector_store", "law_kb.db"))
MILVUS_TOKEN = os.environ.get("MILVUS_TOKEN", "")
MILVUS_COLLECTION = os.environ.get("MILVUS_COLLECTION", "law_kb")
MILVUS_DIM = int(os.environ.get("MILVUS_DIM", "1024"))
LOCAL_VECTOR_DIR = os.path.join(BASE_DIR, "data", "vector_store")

# ============================================================
# 9) 文件存储：MinIO（正式） / 本地磁盘（默认）
# ============================================================
STORAGE_BACKEND = os.environ.get("STORAGE_BACKEND", "auto")    # auto / minio / local
MINIO_ENDPOINT = os.environ.get("MINIO_ENDPOINT", "localhost:9000")
MINIO_ACCESS_KEY = os.environ.get("MINIO_ACCESS_KEY", "minioadmin")
MINIO_SECRET_KEY = os.environ.get("MINIO_SECRET_KEY", "minioadmin")
MINIO_SECURE = _env_bool("MINIO_SECURE", False)
MINIO_BUCKET_CONTRACT = os.environ.get("MINIO_BUCKET_CONTRACT", "contracts")
MINIO_BUCKET_REPORT = os.environ.get("MINIO_BUCKET_REPORT", "reports")
MINIO_REGION = os.environ.get("MINIO_REGION", "us-east-1")
LOCAL_STORAGE_DIR = os.path.join(BASE_DIR, "data", "storage")

# ============================================================
# 10) 路径 / 日志 / Web 服务
# ============================================================
DATA_DIR = os.path.join(BASE_DIR, "data")
LOG_DIR = os.path.join(DATA_DIR, "logs")
LOG_FILE = os.path.join(LOG_DIR, "app.log")
LOG_LEVEL = os.environ.get("LOG_LEVEL", "INFO")
LOG_MAX_BYTES = 2 * 1024 * 1024
LOG_BACKUP_COUNT = 3

TASK_DB_DIR = os.path.join(DATA_DIR, "tasks")
EXPORT_DIR = os.path.join(DATA_DIR, "exports")

WEB_HOST = os.environ.get("WEB_HOST", "0.0.0.0")
WEB_PORT = int(os.environ.get("WEB_PORT", "8000"))
STATIC_DIR = os.path.join(BASE_DIR, "static")

# ============================================================
# 11) Word 导出
# ============================================================
DOCX_TITLE = "合同风险审查报告"
DOCX_COMPANY = "长文档深度分析 Agent · 智能审查系统"
RISK_LEVELS = ["高", "中", "低"]        # 风险等级枚举
RISK_LEVEL_ORDER = {"高": 0, "中": 1, "低": 2}
