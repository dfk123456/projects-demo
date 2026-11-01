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
# config.py —— 全局配置
# 功能：集中管理整个系统全部可调参数：
#   * Ollama 服务地址 / 大模型名 / Embedding 模型名
#   * 两个独立 Chroma 向量库的持久化路径与集合名
#   * 检索 top_k、本地资料充分性判定阈值
#   * 文本切分参数、联网开关默认值、对话记忆轮数等
# 约定：本项目"离线优先"，默认不联网；代码只做接口预留，
#       不会在此处启动或内置运行 Ollama。
# ============================================================
import os

# 项目根目录（本文件所在目录）
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# ------------------------------------------------------------
# 1) Ollama 服务配置（默认 localhost:11434，后续部署无需改代码）
# ------------------------------------------------------------
OLLAMA_BASE_URL = os.environ.get("OLLAMA_BASE_URL", "http://localhost:11434")
# 对话 / 意图判别 / 汇总 统一使用的大模型（离线部署时 ollama pull qwen:7b）
CHAT_MODEL = os.environ.get("OLLAMA_CHAT_MODEL", "qwen:7b")
# 文本向量化模型（ollama pull bge-m3）
# 说明：原配置的 bge-small 在 Ollama 官方库中不存在，已改为 bge-m3
#      （多语言 Embedding，中文效果最好，约 1.2GB）
EMBEDDING_MODEL = os.environ.get("OLLAMA_EMBED_MODEL", "bge-m3")
# 模型上下文窗口（token）：需容纳「系统提示 + 检索片段 + 多轮历史 + 问题」。
# 注意：窗口越大，KV cache 越占显存，模型层越容易被挤下 GPU，速度反而更慢。
#       本机 6GB 显存 + qwen:7b 实测（单轮真实输入约 1738 tokens）：
#         num_ctx=8192 -> 运行时 9.0GB，仅 46% 层在 GPU，端到端 ~104s
#         num_ctx=4096 -> 运行时 6.4GB，100% GPU，端到端 ~51s
#         num_ctx=3072 -> 运行时 5.9GB，100% GPU，端到端 ~36s
#         num_ctx=2048 -> 运行时 5.3GB，100% GPU，端到端 ~23s
#       取 4096 的原因：满 4 轮历史（HISTORY_MAX_TURNS）时总需求约 3000~3100
#       tokens，4096 留有安全余量；2048/3072 在多轮后会截断（丢最早内容）。
#       若更看重速度：设 OLLAMA_NUM_CTX=2048 并把 HISTORY_MAX_TURNS 调小（如 1~2）。
LLM_NUM_CTX = int(os.environ.get("OLLAMA_NUM_CTX", "4096"))
# 加载到 GPU 的层数：0=纯 CPU 推理，99=尽量全部放 GPU（显存不足时自动部分卸载）
# 说明：驱动已升级至 581.57 / CUDA 13.0，Ollama 可正常使用 GPU，故默认启用。
#      若在无 GPU 环境运行，把环境变量 OLLAMA_NUM_GPU 设为 0 即可回退纯 CPU。
LLM_NUM_GPU = int(os.environ.get("OLLAMA_NUM_GPU", "99"))
# 生成温度：判别/汇总统一用 0，保证结果稳定
LLM_TEMPERATURE = 0.0

# ------------------------------------------------------------
# 2) 两个互相独立的 Chroma 向量库（均独立持久化，重启无需重建）
# ------------------------------------------------------------
VECTOR_STORE_ROOT = os.path.join(BASE_DIR, "vector_store")
# 知识库1：大模型技术（LangChain/LangGraph/RAG/Embedding/Agent/Ollama...）
TECH_DB_PERSIST_DIR = os.path.join(VECTOR_STORE_ROOT, "tech_db")
TECH_COLLECTION_NAME = "tech_kb"
# 知识库2：深度学习数学（卷积公式/向量/矩阵/概率/损失函数...）
MATH_DB_PERSIST_DIR = os.path.join(VECTOR_STORE_ROOT, "math_db")
MATH_COLLECTION_NAME = "math_kb"
# 增量入库清单（记录每个文件指纹，实现"追加不重建"）存放目录
MANIFEST_DIR = os.path.join(VECTOR_STORE_ROOT, "manifests")
TECH_MANIFEST_FILE = os.path.join(MANIFEST_DIR, "tech_manifest.json")
MATH_MANIFEST_FILE = os.path.join(MANIFEST_DIR, "math_manifest.json")

# ------------------------------------------------------------
# 3) 检索与"本地资料是否充足"判定参数
# ------------------------------------------------------------
RETRIEVAL_TOP_K = 4                 # 每个库检索返回的文档块数量
SUFFICIENT_TOP_SIM = 0.60           # 最高余弦相似度 >= 该值视为有直接相关片段
SUFFICIENT_MIN_CHARS = 120          # 相关片段合计字符数下限（太少视为不足）
# 实体命中时的相似度下限：问题点名的专有实体（LangGraph / 卷积 …）若出现在片段中，
# 即便向量相似度未达 SUFFICIENT_TOP_SIM，也认为资料可用。
# 原因：向量模型对形近专有名词（LangGraph / LangChain）区分度不足，实测出现过
#      「已检索到正确章节，却因相似度 0.58 < 0.60 被判资料不足」的情况。
SUFFICIENT_ENTITY_SIM = float(os.environ.get("SUFFICIENT_ENTITY_SIM", "0.45"))
# 阈值语义：sim>=top 且 相关片段合计字符>=下限 => 充足；
#          或 问题点名的实体命中片段 且 sim>=entity 下限 => 亦视为充足
DOC_SNIPPET_CHARS = 260             # 日志中单条检索片段展示的字符上限

# ------------------------------------------------------------
# 4) 文档切分参数（入库时使用）
# ------------------------------------------------------------
CHUNK_SIZE = 500
CHUNK_OVERLAP = 80

# ------------------------------------------------------------
# 5) 联网开关默认值（页面复选框初始状态，offline-first）
# ------------------------------------------------------------
DEFAULT_ENABLE_ONLINE_SEARCH = False

# ------------------------------------------------------------
# 6) 多轮对话记忆
# ------------------------------------------------------------
# 携带进上下文的历史轮数（仅最近 N 轮），长期记忆不落库
HISTORY_MAX_TURNS = 4

# ------------------------------------------------------------
# 7) 网络 / 服务超时
# ------------------------------------------------------------
NETWORK_TIMEOUT = 12                # 联网检索 HTTP 超时（秒）
# 本地模型调用的底层 HTTP 超时（秒）：Ollama 无响应时中断等待，
# 避免推理卡死时工作线程被无限挂住（外层另有 CHAT_TIMEOUT_SEC 兜底）
LLM_TIMEOUT_SEC = int(os.environ.get("OLLAMA_TIMEOUT", "180"))

# ------------------------------------------------------------
# 8) Web 服务
# ------------------------------------------------------------
WEB_HOST = "0.0.0.0"
WEB_PORT = 8000
STATIC_DIR = os.path.join(BASE_DIR, "static")
MAX_UPLOAD_MB = 50                  # 单次上传文件大小上限

# ------------------------------------------------------------
# 9) 会话隔离 / 并发保护 / 超时（Web 服务）
# ------------------------------------------------------------
# 说明：本地 CPU 推理单轮问答约 30~70 秒，必须限制并发，
#       否则多个请求会互相拖垮、并打满 Web 线程池。
WEB_MAX_CONCURRENCY = 4             # 同时处理的最大问答数
CHAT_TIMEOUT_SEC = 180              # 单轮问答超时（秒），超时返回友好提示
SESSION_TTL_SEC = 7200              # 会话空闲超过该秒数后清理（2 小时）
SESSION_MAX = 200                   # 进程内最多保留的会话数，超出淘汰最久未用的
STREAM_ENABLED = True               # 是否启用流式输出（SSE）；关闭则回退一次性返回

# ------------------------------------------------------------
# 10) 日志（落盘，便于排障）
# ------------------------------------------------------------
LOG_DIR = os.path.join(BASE_DIR, "logs")
LOG_FILE = os.path.join(LOG_DIR, "app.log")
LOG_LEVEL = os.environ.get("LOG_LEVEL", "INFO")   # DEBUG / INFO / WARNING / ERROR
LOG_MAX_BYTES = 2 * 1024 * 1024     # 单个日志文件上限，超过后自动轮转
LOG_BACKUP_COUNT = 3                # 保留的历史日志个数

# ------------------------------------------------------------
# 11) 用户与登录
# ------------------------------------------------------------
USERS_DB_FILE = os.path.join(BASE_DIR, "users.db")   # 用户库（含密码散列，勿外传）
# 是否要求登录后才能使用；设环境变量 AUTH_REQUIRED=0 可关闭（仅供内网调试）
AUTH_REQUIRED = os.environ.get("AUTH_REQUIRED", "1") not in ("0", "false", "False")
TOKEN_TTL_SEC = 7 * 24 * 3600       # 登录凭证有效期（7 天）

# ------------------------------------------------------------
# 12) 外部大模型 API（可选，OpenAI 兼容协议）
# ------------------------------------------------------------
# ============================================================
# 【要加模型就改这里】可选模型清单
# ============================================================
# 页面上「模型来源」下拉 = 下面这个列表，自己增删即可，支持任意模型。
#
# 每个条目的字段：
#   id        唯一标识（英文，自己起，别重复）
#   name      下拉里显示的名字（随便写，看得懂就行）
#   backend   ollama = 本机 Ollama（数据不出内网）
#             openai = 任意 OpenAI 兼容接口（DeepSeek / Kimi / 小米 MiMo / 通义 / vLLM …）
#   base_url  服务地址
#   model     模型名（必填，决定实际用哪个模型）
#   key       密钥（只有 backend=openai 才需要）
#
# 想加几个模型就加几条；同一家想用不同模型，复制一条改 model 即可。
MODELS = [
    # ---- 本机模型（默认）----
    {"id": "local-qwen7b", "name": "本地 Ollama · qwen:7b", "backend": "ollama",
     "base_url": "http://localhost:11434", "model": "qwen:7b"},

    # ---- 下面是三家常用服务的示例：复制出来、去掉注释、把 key 换成你自己的即可 ----
    #
    {"id": "deepseek-v4-flash", "name": "DeepSeek-v4-flash", "backend": "openai",
     "base_url": "https://api.deepseek.com/v1", "model": "deepseek-v4-flash",
     "key": os.environ.get("DEEPSEEK_API_KEY", "")},
    #
    {"id": "mimo v2.5 pro", "name": "MiMo V2.5 Pro", "backend": "openai",
     "base_url": "https://api.xiaomimimo.com/v1", "model": "mimo-v2.5-pro",
     "key": os.environ.get("MIMO_API_KEY", "")},
    #
    # 小米若用 Token Plan 订阅，base_url 换成控制台给出的专属地址即可。
]

# 默认使用哪个模型（填上面某个条目的 id）
DEFAULT_MODEL = os.environ.get("DEFAULT_MODEL", "local-qwen7b")

# 外部模型总开关：设 0 可一次性禁用所有 backend=openai 的模型
EXTERNAL_API_ENABLED = os.environ.get("EXTERNAL_API_ENABLED", "1") in ("1", "true", "True")
EXTERNAL_API_TIMEOUT = int(os.environ.get("EXTERNAL_API_TIMEOUT", "60"))
# 使用外部模型时，是否把本地课件片段一起发送（False = 只发问题，课件不外传）
EXTERNAL_SEND_CONTEXT = os.environ.get("EXTERNAL_SEND_CONTEXT", "1") in ("1", "true", "True")
