# ============================================================
# agents/math_rag_agent.py —— 深度学习数学知识库 RAG 检索 Agent
# 功能：当 dispatcher 判定问题归属 math 时，本 Agent 从
#       vector_store.math_kb 做相似度检索，产出结构化片段：
#       * kb_used   = "math"
#       * kb_docs   = [{content, source, similarity}, ...]
#       * kb_error  = 检索出错时的友好提示
# 说明：知识库2只收录深度学习配套数学资料（卷积公式/向量/矩阵/
#       概率/损失函数），检索为纯本地，不触发任何联网请求。
# ============================================================
import config
from vector_store import kb_filter, math_kb


def _extend_log(state, kind: str, title: str, content: str) -> list:
    """把日志追加进列表并返回新列表。state 兼容 dict(图状态)或直接传日志列表。"""
    if isinstance(state, dict):
        log = list(state.get("log") or [])
    else:
        log = list(state or [])
    log.append({"kind": kind, "title": title, "content": content})
    return log


def _snippet(text: str, limit: int = config.DOC_SNIPPET_CHARS) -> str:
    text = (text or "").strip().replace("\n", " ")
    return text if len(text) <= limit else text[:limit] + "…"


class MathRAGAgent:
    """知识库2（深度学习数学）检索 Agent。"""

    def run(self, state: dict) -> dict:
        question = (state.get("question") or "").strip()
        # 检索用查询：追问场景下由 dispatcher 补全了上文语义
        search_query = (state.get("search_query") or "").strip() or question
        updates = {"kb_used": "math", "kb_docs": [], "kb_error": "",
                   "log": state.get("log") or []}

        updates["log"] = _extend_log(
            state, "知识库检索", "开始检索：知识库2（深度学习数学）",
            f"使用本地 Chroma 向量库检索 top{config.RETRIEVAL_TOP_K}，"
            f"问题：{search_query[:80]}")

        result = math_kb.search(search_query, top_k=config.RETRIEVAL_TOP_K)

        if result.get("error"):
            updates["kb_error"] = result["error"]
            updates["log"] = _extend_log(
                updates["log"], "错误", "知识库2检索失败",
                result["error"])
            return updates

        if result.get("empty"):
            updates["log"] = _extend_log(
                updates["log"], "知识库检索", "知识库2暂无内容",
                "当前【深度学习数学】向量库尚未入库任何课件/笔记，请先上传 PDF/TXT 文档。")
            return updates

        # 实体优先重排 + 实体一致性过滤（说明见 tech_rag_agent 同名步骤）
        docs = result.get("docs", [])
        docs = kb_filter.rerank_by_question_entities(
            question, docs, kb_filter.MATH_ENTITIES)
        docs, filter_note = kb_filter.filter_by_question_entities(
            question, docs, kb_filter.MATH_ENTITIES)
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
