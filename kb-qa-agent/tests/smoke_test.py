# ============================================================
# tests/smoke_test.py —— 冒烟自检脚本
#
# 用法（在项目根目录 0903_demo 下执行）：
#   python tests/smoke_test.py          快速自检（约 30 秒，不跑完整问答）
#   python tests/smoke_test.py --full   追加 1 轮真实问答（CPU 推理约 1~2 分钟）
#
# 说明：只用标准库实现，不依赖 pytest；检查项覆盖依赖、Ollama、图、
#       向量库、检索、分类校正、拒绝纠正、超范围拦截、离线工具、入库幂等，失败返回非 0。
# ============================================================
import argparse
import os
import sys
import time
import urllib.request

# Windows 控制台默认 GBK，中文输出需切到 UTF-8，避免打印时崩溃
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, BASE_DIR)

import config

_RESULTS = []


def check(name: str, ok: bool, detail: str = ""):
    """记录并打印一项检查结果。"""
    _RESULTS.append((name, bool(ok), detail))
    print(f"[{'PASS' if ok else 'FAIL'}] {name}" + (f" —— {detail}" if detail else ""))


def run_safely(fn):
    """单个检查项抛异常时不中断整体流程。"""
    try:
        fn()
    except Exception as exc:
        check(fn.__name__, False, f"执行异常：{str(exc)[:200]}")


# ------------------------------------------------------------
# 1) 依赖是否安装
# ------------------------------------------------------------
def check_imports():
    import importlib.util
    mods = ["fastapi", "uvicorn", "requests", "pypdf", "langchain_core",
            "langchain_community", "langchain_text_splitters",
            "langchain_chroma", "langchain_ollama", "langgraph", "chromadb"]
    missing = [m for m in mods if importlib.util.find_spec(m) is None]
    check("依赖安装完整性", not missing,
          "全部就绪" if not missing else f"缺少：{missing}")


# ------------------------------------------------------------
# 2) Ollama 服务与模型
# ------------------------------------------------------------
def check_ollama():
    url = config.OLLAMA_BASE_URL + "/api/tags"
    try:
        with urllib.request.urlopen(url, timeout=5) as resp:
            data = __import__("json").loads(resp.read().decode("utf-8"))
    except Exception as exc:
        check("Ollama 服务连通性", False,
              f"无法访问 {url}（{str(exc)[:80]}），请先执行 ollama serve")
        return
    check("Ollama 服务连通性", True, url)

    names = {m.get("name", "") for m in data.get("models", [])}
    short = {n.split(":")[0] for n in names}

    def has(model: str) -> bool:
        return model in names or f"{model}:latest" in names or model.split(":")[0] in short

    check(f"对话模型 {config.CHAT_MODEL} 已拉取", has(config.CHAT_MODEL),
          f"ollama pull {config.CHAT_MODEL}" if not has(config.CHAT_MODEL) else "已就绪")
    check(f"向量模型 {config.EMBEDDING_MODEL} 已拉取", has(config.EMBEDDING_MODEL),
          f"ollama pull {config.EMBEDDING_MODEL}" if not has(config.EMBEDDING_MODEL) else "已就绪")


# ------------------------------------------------------------
# 3) LangGraph 图构建
# ------------------------------------------------------------
def check_graph():
    import graph_builder
    app = graph_builder.build_graph()
    nodes = set(app.get_graph().nodes.keys())
    required = {"dispatcher", "tech_rag", "math_rag", "assess",
                "calc_exec", "gate_online", "online_search", "summary"}
    missing = required - nodes
    check("LangGraph 图构建", not missing,
          f"节点 {len(nodes)} 个" if not missing else f"缺少节点：{missing}")


# ------------------------------------------------------------
# 4) 向量库是否已有内容
# ------------------------------------------------------------
def check_kb():
    from vector_store import math_kb, tech_kb
    t, m = tech_kb.count(), math_kb.count()
    check("知识库1（大模型技术）非空", t > 0,
          f"{t} 个文档块" if t else "请先上传课件到 tech 库")
    check("知识库2（深度学习数学）非空", m > 0,
          f"{m} 个文档块" if m else "请先上传课件到 math 库")


# ------------------------------------------------------------
# 5) 检索命中质量
# ------------------------------------------------------------
def _top_similarity(module, question: str, tries: int = 2) -> float:
    """取检索最高相似度；未达标时重试一次。

    说明：向量服务首次被调用时偶有抖动，重试可避免自检出现偶发误报。
    """
    best = 0.0
    for _ in range(tries):
        result = module.search(question, top_k=config.RETRIEVAL_TOP_K)
        docs = result.get("docs") or []
        best = max(best, max([d["similarity"] for d in docs], default=0.0))
        if best >= config.SUFFICIENT_TOP_SIM:
            break
    return best


def check_retrieval():
    from vector_store import math_kb, tech_kb
    cases = [("tech", tech_kb, "什么是 RAG 检索增强生成"),
             ("tech", tech_kb, "Ollama 怎么本地部署"),
             ("math", math_kb, "卷积输出尺寸怎么计算"),
             ("math", math_kb, "二项分布概率怎么算")]
    for name, module, q in cases:
        top = _top_similarity(module, q)
        check(f"检索命中[{name}] {q}", top >= config.SUFFICIENT_TOP_SIM,
              f"最高相似度 {top}（阈值 {config.SUFFICIENT_TOP_SIM}）")


# ------------------------------------------------------------
# 6) 分类交叉校正（纯规则，不调用模型）
# ------------------------------------------------------------
def check_category_guard():
    from agents.dispatcher_agent import DispatcherAgent
    cases = [
        ("输入 32x32，卷积核 3，填充 1，步长 1，输出尺寸是多少", "tech", "math"),
        ("LangGraph 怎么编排多智能体", "math", "tech"),
        ("什么是 RAG 检索增强生成", "tech", "tech"),
        ("矩阵乘法 A 是 2x3，B 是 3x4，结果维度是多少", "tech", "math"),
    ]
    for question, given, expect in cases:
        got, _reason = DispatcherAgent._guard_category(question, given, "")
        check(f"分类校正 {given}→{expect}：{question[:16]}…", got == expect,
              f"实际判定 {got}")


# ------------------------------------------------------------
# 6.5) 超范围话题兜底拦截（纯规则，不调用模型）
# ------------------------------------------------------------
def check_off_topic_guard():
    """闲聊 / 生活类问题即使被模型误判，也应被确定性兜底拦回 reject。"""
    from agents.dispatcher_agent import DispatcherAgent
    cases = [
        # 实测缺陷：问天气被小模型判成 math，进而去数学库检索并编造答案
        ("今天天气怎么样", "math", "reject"),
        ("给我讲个笑话", "tech", "reject"),
        ("帮我写一首情诗", "math", "reject"),
        # 反向保护：同时含专业特征词时不拦截，避免误伤正常问题
        ("深度学习模型可以用来预测天气吗", "tech", "tech"),
        ("用卷积神经网络预测股票走势", "math", "math"),
    ]
    for question, given, expect in cases:
        got, _reason = DispatcherAgent._guard_off_topic(question, given, "")
        check(f"超范围拦截 {given}→{expect}：{question[:14]}…", got == expect,
              f"实际判定 {got}")


# ------------------------------------------------------------
# 6.6) 拒绝误判纠正（模型把专业问题误判成 reject 时应拉回）
# ------------------------------------------------------------
def check_reject_guard():
    """实测缺陷：卷积计算题曾被模型误判为 reject 而遭拦截，应能被纠正回 math/tech。"""
    from agents.dispatcher_agent import DispatcherAgent
    cases = [
        ("输入 32x32，卷积核 3，填充 1，步长 1，输出尺寸是多少？", "math"),
        ("什么是 RAG 检索增强生成", "tech"),
        ("LangGraph 怎么编排多智能体", "tech"),
        # 反向保护：真正的闲聊不含专业特征词，仍应保持 reject
        ("你好", "reject"),
        ("今天天气怎么样", "reject"),
    ]
    for question, expect in cases:
        got, _reason = DispatcherAgent._guard_reject(question, "reject", "")
        check(f"拒绝误判纠正 reject→{expect}：{question[:14]}…", got == expect,
              f"实际判定 {got}")


# ------------------------------------------------------------
# 6.7) 实体重排 + 「实体命中」充分性通路
# ------------------------------------------------------------
def check_entity_handling():
    """实测缺陷：问 LangGraph 时 LangChain 章节排在首位，且正确章节因相似度
    0.58 < 0.60 被判「资料不足」。这里验证重排与实体通路是否生效。"""
    import graph_builder
    from vector_store import kb_filter, tech_kb

    q = "LangGraph 有哪些核心概念？"
    docs = tech_kb.search(q, top_k=config.RETRIEVAL_TOP_K).get("docs") or []
    reranked = kb_filter.rerank_by_question_entities(q, docs, kb_filter.TECH_ENTITIES)
    top_section = (reranked[0].get("section") or "") if reranked else ""
    check("实体重排：LangGraph 章节排首位", "LangGraph" in top_section,
          f"实际首位：{top_section or '（空）'}")

    state = {"question": q, "kb_docs": reranked, "kb_used": "tech",
             "enable_online_search": False, "log": []}
    out = graph_builder.assess_node(state)
    check("实体通路：资料判定为充足", out.get("sufficient") is True,
          f"sufficient={out.get('sufficient')}")


# ------------------------------------------------------------
# 6.8) 时效性缺口（问最新动态，而课件只有静态定义）
# ------------------------------------------------------------
def check_time_gap():
    """实测缺陷：问「MCP 协议最近有什么新进展」，课件只有基础定义，
    却被模型包装成「最新进展」。此处验证缺口识别规则。"""
    from agents.summary_agent import SummaryAgent

    mcp_doc = [{
        "content": "MCP（Model Context Protocol）：标准化地把外部工具、数据源、"
                   "资源接入大模型应用，降低工具集成成本。",
        "section": "八、进阶方向：GraphRAG 与 MCP",
    }]
    check("时效缺口：识别「最新动态」类提问",
          SummaryAgent._time_sensitive_gap("MCP 协议最近有什么新进展？", mcp_doc, False) != [],
          "应识别为时效缺口")
    check("时效缺口：普通问法不误判",
          SummaryAgent._time_sensitive_gap("MCP 是什么？", mcp_doc, False) == [],
          "不应误判为缺口")
    check("时效缺口：已有联网结果时不算缺口",
          SummaryAgent._time_sensitive_gap("MCP 协议最近有什么新进展？", mcp_doc, True) == [],
          "有联网结果时不应判缺口")
    check("时效缺口：资料确实含时效内容时不算缺口",
          SummaryAgent._time_sensitive_gap(
              "MCP 最近有什么新进展？",
              [{"content": "MCP 最新进展：2026 年新增了 XX 能力。", "section": "八"}],
              False) == [],
          "资料覆盖时不应判缺口")


# ------------------------------------------------------------
# 7) 离线工具
# ------------------------------------------------------------
def check_tools():
    from tools.calc_tool import offline_formula_calc
    from tools.online_search_tool import check_domain

    r1 = offline_formula_calc("卷积尺寸：输入 32x32，卷积核 3，填充 1，步长 1")
    check("calc_tool 卷积尺寸", "32 × 32" in r1, r1.splitlines()[-2] if r1 else "")

    r2 = offline_formula_calc("矩阵乘法：A=[[1,2],[3,4]]，B=[[5,6],[7,8]]")
    check("calc_tool 矩阵乘法", "19" in r2 and "43" in r2,
          "结果应含 19 / 43")

    r3 = offline_formula_calc("概率：二项分布 n=10，k=6，p=0.5")
    check("calc_tool 二项分布", "0.205078" in r3, "应约等于 0.205078")

    allowed, _ = check_domain("LangGraph 多智能体最新进展")
    check("联网领域自检：放行技术话题", allowed)

    allowed, reason = check_domain("今天的娱乐新闻八卦")
    check("联网领域自检：拦截非技术话题", not allowed, reason)


# ------------------------------------------------------------
# 8) 入库幂等（同一文件重复入库应跳过）
# ------------------------------------------------------------
def check_ingest_idempotent():
    from vector_store import document_loader
    name = "tech_大模型技术课件.txt"
    path = os.path.join(BASE_DIR, "knowledge_docs", name)
    if not os.path.exists(path):
        check("入库幂等", False, f"找不到示例课件：{path}")
        return
    with open(path, "rb") as fh:
        raw = fh.read()
    before = _count("tech")
    result = document_loader.add_document("tech", name, raw)
    after = _count("tech")
    check("入库幂等（重复上传自动跳过）", result["status"] == "skipped" and before == after,
          f"status={result['status']}，文档块 {before} → {after}")


def _count(kb: str) -> int:
    from vector_store import math_kb, tech_kb
    return tech_kb.count() if kb == "tech" else math_kb.count()


# ------------------------------------------------------------
# 9) 完整问答（仅 --full 模式，耗时较长）
# ------------------------------------------------------------
def check_full_chat():
    import graph_builder
    started = time.time()
    result = graph_builder.run_chat(
        "输入 32x32，卷积核 3，填充 1，步长 1，卷积输出尺寸是多少？", False)
    log_text = result["log_text"]
    check("完整问答：走数学知识库", "知识库2" in log_text,
          "分类判定应归属深度学习数学库")
    check("完整问答：输出计算结果", "32" in result["answer"],
          result["answer"][:60].replace("\n", " "))
    check("完整问答：耗时可接受", time.time() - started < config.CHAT_TIMEOUT_SEC,
          f"{time.time() - started:.1f}s")


def main():
    parser = argparse.ArgumentParser(description="离线专业知识库问答系统 —— 冒烟自检")
    parser.add_argument("--full", action="store_true",
                        help="追加 1 轮真实问答（调用本地大模型，耗时较长）")
    args = parser.parse_args()

    print("=" * 58)
    print("  离线专业知识库问答系统 —— 冒烟自检")
    print(f"  Ollama：{config.OLLAMA_BASE_URL} | 模型：{config.CHAT_MODEL}")
    print("=" * 58)

    started = time.time()
    for fn in (check_imports, check_ollama, check_graph, check_kb,
               check_retrieval, check_category_guard, check_reject_guard,
               check_off_topic_guard, check_entity_handling, check_time_gap,
               check_tools, check_ingest_idempotent):
        run_safely(fn)
    if args.full:
        run_safely(check_full_chat)

    passed = sum(1 for _, ok, _ in _RESULTS if ok)
    total = len(_RESULTS)
    print("\n" + "=" * 58)
    print(f"  自检完成：{passed}/{total} 通过，用时 {time.time() - started:.1f}s")
    if passed < total:
        print("  失败项：")
        for name, ok, detail in _RESULTS:
            if not ok:
                print(f"    - {name}：{detail}")
    print("=" * 58)
    sys.exit(0 if passed == total else 1)


if __name__ == "__main__":
    main()
