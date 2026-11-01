# ============================================================
# agents/tech_rag_agent.py —— 大模型技术知识库 RAG 检索 Agent
# 功能：当 dispatcher 判定问题归属 tech 时，本 Agent 从
#       vector_store.tech_kb 做相似度检索，产出结构化片段：
#       * kb_used   = "tech"
#       * kb_docs   = [{content, source, similarity}, ...]
#       * kb_error  = 检索出错时的友好提示（Ollama 离线等）
# 检索本身为纯本地（Embedding + Chroma），不触发任何联网请求。
# ============================================================
import config
from vector_store import kb_filter, tech_kb


def _extend_log(state, kind: str, title: str, content: str) -> list:
    """把日志追加进列表并返回新列表。state 兼容 dict(图状态)或直接传日志列表。"""
    if isinstance(state, dict):
        log = list(state.get("log") or [])
    else:
        log = list(state or [])
    log.append({"kind": kind, "title": title, "content": content})
    return log


def _snippet(text: str, limit: int = config.DOC_SNIPPET_CHARS) -> str:
    """文档片段展示裁剪，避免日志刷屏。"""
    text = (text or "").strip().replace("\n", " ")
    return text if len(text) <= limit else text[:limit] + "…"


class TechRAGAgent:
    """知识库1（大模型技术）检索 Agent。"""

    def run(self, state: dict) -> dict:
        question = (state.get("question") or "").strip()
        # 检索用查询：追问场景下由 dispatcher 补全了上文语义（详见 kb_filter.resolve_followup_query）
        search_query = (state.get("search_query") or "").strip() or question
        updates = {"kb_used": "tech", "kb_docs": [], "kb_error": "",
                   "log": state.get("log") or []}

        # 0) 打开日志：记录正在调用哪个知识库
        updates["log"] = _extend_log(
            state, "知识库检索", "开始检索：知识库1（大模型技术）",
            f"使用本地 Chroma 向量库检索 top{config.RETRIEVAL_TOP_K}，"
            f"问题：{search_query[:80]}")

        # 1) 执行检索（本地，不联网）
        result = tech_kb.search(search_query, top_k=config.RETRIEVAL_TOP_K)

        # 2) 处理错误
        if result.get("error"):
            updates["kb_error"] = result["error"]
            updates["log"] = _extend_log(
                updates["log"], "错误", "知识库1检索失败",
                result["error"])
            return updates

        # 3) 库为空
        if result.get("empty"):
            updates["log"] = _extend_log(
                updates["log"], "知识库检索", "知识库1暂无内容",
                "当前【大模型技术】向量库尚未入库任何课件/笔记，请先上传 PDF/TXT 文档。")
            return updates

        # 4) 实体优先重排 + 实体一致性过滤
        #    重排：把命中问题主实体的片段前移（向量模型对形近专有名词区分度不足，
        #          实测问 LangGraph 时 LangChain 章节曾排到第一位）
        #    过滤：问题点名 A 时剔除讲 B 的片段（防跨概念混淆）
        docs = result.get("docs", [])
        docs = kb_filter.rerank_by_question_entities(
            question, docs, kb_filter.TECH_ENTITIES)
        docs, filter_note = kb_filter.filter_by_question_entities(
            question, docs, kb_filter.TECH_ENTITIES)
        if filter_note:
            updates["log"] = _extend_log(
                updates["log"], "知识库检索", "实体一致性过滤", filter_note)
        updates["kb_docs"] = docs
        snippet_lines = []
        for i, doc in enumerate(docs, start=1):
            section = doc.get("section") or ""
            source = doc["source"] + (f" · {section}" if section else "")
            snippet_lines.append(
                f"[{i}] 相似度 {doc['similarity']} | 来源：{source}\n"
                f"    片段：{_snippet(doc['content'])}")
        updates["log"] = _extend_log(
            updates["log"], "知识库检索", f"检索到 {len(docs)} 个相关文档块",
            "\n".join(snippet_lines))
        return updates
