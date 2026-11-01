# ============================================================
# graph_builder.py —— LangGraph 状态图：串联全部智能体与工具
#
# 节点总览（对应文件 modules）：
#   dispatcher      -> agents/dispatcher_agent.py   中控调度
#   tech_rag        -> agents/tech_rag_agent.py     大模型技术库 RAG
#   math_rag        -> agents/math_rag_agent.py     深度学习数学库 RAG
#   assess          -> 本地资料充分性判定（rule-based，图内实现）
#   calc_exec       -> tools/calc_tool.py           离线公式演算（可选）
#   gate_online     -> dispatcher.judge_online      前沿联网许可判定（可选）
#   online_search   -> tools/online_search_tool.py  联网检索（可选）
#   summary         -> agents/summary_agent.py      汇总&幻觉校验
#
# 强制业务逻辑（离线优先）：
#   1) dispatcher 拦截闲聊/娱乐/生活 → 直接 END，不调用任何工具
#   2) tech/math 二选一进入对应本地向量库检索
#   3) assess 判定本地资料充分 → summary 离线作答（禁止联网）
#   4) 资料不足 & 联网关闭      → summary 如实告知不足，不联网
#   5) 资料不足 & 联网开启      → gate_online：仅前沿技术放行；
#      基础知识点即使缺资料也禁止联网；普通话题同样禁止
#   6) 联网工具仅检索 AI/深度学习技术资料，只提交精简关键词
#   7) summary 严格依据给定资料作答并标注来源，杜绝编造
# ============================================================
import queue
import re
import threading
from typing import TypedDict

import config
from agents.dispatcher_agent import DispatcherAgent
from agents.math_rag_agent import MathRAGAgent
from agents.summary_agent import SummaryAgent
from agents.tech_rag_agent import TechRAGAgent
from tools.calc_tool import offline_formula_calc
from tools.online_search_tool import online_tech_search

# ------------------------------------------------------------
# 状态定义
# ------------------------------------------------------------
class GraphState(TypedDict, total=False):
    question: str                # 用户问题
    search_query: str            # 检索用查询（追问场景下已补全上文语义）
    enable_online_search: bool   # 联网开关（本次请求）
    chat_history: list           # 多轮对话记忆
    log: list                    # 结构化推理日志 [{kind,title,content}]
    log_text: str                # 最终格式化日志（纯文本，给前端）
    category: str                # tech / math / reject
    reason: str                  # 分类理由
    use_calc: bool               # 是否需要离线公式演算
    search_keywords: str         # 精简联网关键词
    kb_used: str                 # tech / math
    kb_docs: list                # 检索到的本地片段
    kb_error: str                # 检索异常提示
    sufficient: bool             # 本地资料是否充足
    calc_result: str             # 公式演算输出
    online_allowed: bool         # 联网许可（gate 判定）
    online_reason: str
    online_results: list         # 联网检索结果
    answer: str                  # 最终回答
    token_sink: object           # 可选：流式输出通道（queue.Queue），用于逐 token 推送
    provider: str                # 模型来源：local（本地 Ollama）/ external（外部 API）


# ------------------------------------------------------------
# 日志小工具
# ------------------------------------------------------------
def _extend_log(state, kind: str, title: str, content: str) -> list:
    """把日志追加进列表并返回新列表。state 兼容 dict(图状态)或直接传日志列表。"""
    if isinstance(state, dict):
        log = list(state.get("log") or [])
    else:
        log = list(state or [])
    log.append({"kind": kind, "title": title, "content": content})
    return log


# ------------------------------------------------------------
# Agent 实例（懒加载依赖，构建期轻量）
# ------------------------------------------------------------
_dispatcher = DispatcherAgent()
_tech_rag = TechRAGAgent()
_math_rag = MathRAGAgent()
_summary = SummaryAgent()


# ------------------------------------------------------------
# 图节点函数
# ------------------------------------------------------------
def dispatch_node(state: dict) -> dict:
    return _dispatcher.dispatch(state)


def tech_rag_node(state: dict) -> dict:
    return _tech_rag.run(state)


def math_rag_node(state: dict) -> dict:
    return _math_rag.run(state)


def _matched_entity(state: dict) -> str:
    """判断问题点名的专有实体是否出现在检索片段中（充分性判定的实体通路）。

    返回命中的实体名；未命中或未点名实体时返回空串。
    """
    from vector_store import kb_filter
    entities = (kb_filter.MATH_ENTITIES if state.get("kb_used") == "math"
                else kb_filter.TECH_ENTITIES)
    return kb_filter.matched_entity_name(
        state.get("search_query") or state.get("question", ""),
        state.get("kb_docs") or [], entities)


def assess_node(state: dict) -> dict:
    """本地资料充分性判定（规则化、可离线、可解释）。"""
    docs = state.get("kb_docs") or []
    kb_error = state.get("kb_error", "")
    question = state.get("question", "")
    updates = {"sufficient": False, "log": state.get("log") or []}

    # 检索异常 → 资料视为不足
    if kb_error:
        updates["sufficient"] = False
        updates["log"] = _extend_log(
            state, "意图判断", "充分性判定：资料不足（检索异常）",
            f"本地检索异常：{kb_error}\n将如实告知用户，不编造内容。")
        return updates

    # 未检索到内容 → 资料不足
    if not docs:
        updates["sufficient"] = False
        updates["log"] = _extend_log(
            state, "意图判断", "充分性判定：资料不足（无相关片段）",
            f"本地向量库未返回任何片段，问题「{question[:60]}」暂无课件支撑。")
        return updates

    # 规则判定：最高相似度 + 相关片段合计长度
    top_sim = max((d.get("similarity", 0.0) or 0.0) for d in docs)
    total_chars = sum(len(d.get("content", "") or "") for d in docs)
    good_docs = sum(1 for d in docs if (d.get("similarity", 0.0) or 0.0) >= config.SUFFICIENT_TOP_SIM)
    enough = (top_sim >= config.SUFFICIENT_TOP_SIM and total_chars >= config.SUFFICIENT_MIN_CHARS)

    # 第二条通路：问题点名的专有实体确实命中片段时，放宽相似度要求。
    # 原因：向量模型对形近专有名词（LangGraph / LangChain）区分度不足，实测出现过
    #      「已检索到正确章节，却因相似度 0.58 < 0.60 被判资料不足」的情况。
    entity_name = ""
    if not enough and total_chars >= config.SUFFICIENT_MIN_CHARS:
        entity_name = _matched_entity(state)
        if entity_name and top_sim >= config.SUFFICIENT_ENTITY_SIM:
            enough = True

    # 反向通路：时效性缺口 → 判为不足。
    # 问题问「最新/最近」等动态信息，而课件是静态资料且未包含对应内容时，
    # 即便检索命中了该实体的基础定义也不足以回答。判为不足之后：
    #   ① 开启联网时可进入联网闸门，尝试取回最新信息；
    #   ② 未联网时由汇总阶段如实告知，而不是把定义包装成「新进展」。
    time_hits = []
    if enough:
        from vector_store import kb_filter
        time_hits = kb_filter.time_sensitive_gap(question, docs, has_online=False)
        if time_hits:
            enough = False

    updates["sufficient"] = bool(enough)
    entity_note = (f"实体命中「{entity_name}」，按实体通路判定为充足"
                   if (entity_name and enough) else "")
    time_note = (f"时效性缺口：问题命中时效词 {time_hits}，本地课件未包含该内容，判为不足"
                 if time_hits else "")
    updates["log"] = _extend_log(
        state, "意图判断", f"充分性判定：{'充足' if enough else '不足'}",
        f"最高相似度 {top_sim}（阈值 {config.SUFFICIENT_TOP_SIM}）；"
        f"高相关片段 {good_docs}/{len(docs)} 个；片段合计 {total_chars} 字符"
        f"（下限 {config.SUFFICIENT_MIN_CHARS}）。"
        + (f"\n{entity_note}（实体通路下限 {config.SUFFICIENT_ENTITY_SIM}）。"
           if entity_note else "")
        + (f"\n{time_note}。" if time_note else "")
        + "\n"
        + ("本地资料足以支撑回答 → 离线直接作答，禁止联网。"
           if enough else
           ("本地资料不足且联网开关开启 → 下一步进行前沿技术联网许可判定。"
            if state.get("enable_online_search")
            else "本地资料不足且联网开关关闭 → 不再联网，如实告知资料不足。")))
    return updates


def calc_exec_node(state: dict) -> dict:
    """可选节点：需要公式演算时调用离线 calc_tool。"""
    updates = {"calc_result": "", "log": state.get("log") or []}
    if not state.get("use_calc"):
        return updates
    question = state.get("question", "")
    updates["log"] = _extend_log(
        state, "工具调用", "调用工具：offline_formula_calc（本地离线计算）",
        f"输入：{question[:120]}")
    try:
        result = offline_formula_calc(question)
    except Exception as exc:
        result = f"公式演算工具执行异常：{str(exc)[:200]}"

    # 关键：工具返回的「无法识别 / 执行异常」属于失败说明，不是可引用的计算结果，
    # 绝不能写入 calc_result——否则汇总阶段会误以为「已有计算结果」，
    # 从而绕过零证据保护、自由生成答案（实测因此编造出课件中并不存在的内容）。
    if result.startswith(("无法识别", "公式演算工具执行异常")):
        updates["calc_result"] = ""
        updates["log"] = _extend_log(
            updates["log"], "工具调用", "公式演算未产出结果（不作为计算依据）",
            f"{result[:300]}\n（说明：该提示仅作日志记录，不进入答案依据。）")
    else:
        updates["calc_result"] = result
        updates["log"] = _extend_log(
            updates["log"], "工具调用", "公式演算结果",
            result[:600])
    return updates


def gate_online_node(state: dict) -> dict:
    """可选节点：本地资料不足时，判断该问题是否属于前沿新技术而值得联网。"""
    return _dispatcher.judge_online(state)


def online_search_node(state: dict) -> dict:
    """可选节点：联网检索 AI/深度学习前沿技术资料（仅提交精简关键词）。"""
    keywords = state.get("search_keywords") or state.get("question", "")
    updates = {"online_results": [], "log": state.get("log") or []}
    updates["log"] = _extend_log(
        state, "联网检索", "调用工具：online_tech_search",
        f"仅提交精简专业关键词（不携带本地文档原文）：{keywords[:80]}")

    try:
        raw = online_tech_search(keywords)
    except Exception as exc:
        raw = f"[联网检索失败] 检索工具执行异常：{str(exc)[:160]}"

    # 解析成结构化条目（多结果时逐条拆分）
    items = []
    pattern = re.compile(
        r"(\d+)\.\s*标题：(.+?)\n\s*摘要：(.+?)\n\s*来源：(https?://\S+)", re.S)
    for m in pattern.finditer(raw):
        items.append({"source": m.group(4), "content": f"标题：{m.group(2)}\n摘要：{m.group(3)}"})

    # 关键：失败 / 被拒绝时绝不写入 online_results。
    # 失败信息只是错误说明，不是可引用的外部资料；一旦写入，汇总阶段会误以为
    # "已有外部证据"，进而据一段错误文本编造答案并错标来源（实测出现过把失败
    # 信息当成【外部补充资料】写进答案的情况）。
    is_failure = raw.startswith("[")
    if is_failure:
        items = []
    elif not items:
        items = [{"source": "外部网络检索", "content": raw}]

    updates["online_results"] = items
    if is_failure:
        updates["log"] = _extend_log(
            updates["log"], "联网检索", "联网结果（失败/拒绝）",
            f"{raw}\n（说明：失败信息不作为外部资料进入答案，避免被误当成引用依据。）")
    else:
        updates["log"] = _extend_log(
            updates["log"], "联网检索", f"联网检索成功，获得 {len(items)} 条外部资料",
            "\n".join(f"- {it['source']}: {it['content'][:150]}" for it in items))
    return updates


def summary_node(state: dict) -> dict:
    return _summary.run(state)


# ------------------------------------------------------------
# 条件路由
# ------------------------------------------------------------
def route_after_dispatch(state: dict) -> str:
    """reject -> refusal（话题拦截终点）；tech/math -> 对应 RAG 节点。"""
    category = state.get("category", "tech")
    if category == "reject":
        return "refusal"
    return "tech_rag" if category == "tech" else "math_rag"


def route_after_calc(state: dict) -> str:
    """本地资料不足且联网开启 -> gate_online；否则直接 summary。"""
    if not state.get("sufficient") and state.get("enable_online_search"):
        return "gate_online"
    return "summary"


def route_after_gate(state: dict) -> str:
    """联网许可通过 -> online_search；否则 -> summary。"""
    return "online_search" if state.get("online_allowed") else "summary"


# ------------------------------------------------------------
# 构图
# ------------------------------------------------------------
def build_graph():
    from langgraph.graph import END, START, StateGraph

    builder = StateGraph(GraphState)

    builder.add_node("dispatcher", dispatch_node)
    builder.add_node("refusal", lambda s: {"log": s.get("log")})  # 话题拦截终点（空操作）
    builder.add_node("tech_rag", tech_rag_node)
    builder.add_node("math_rag", math_rag_node)
    builder.add_node("assess", assess_node)
    builder.add_node("calc_exec", calc_exec_node)
    builder.add_node("gate_online", gate_online_node)
    builder.add_node("online_search", online_search_node)
    builder.add_node("summary", summary_node)

    # 入口 -> 中控调度
    builder.add_edge(START, "dispatcher")
    # 调度 -> (拦截: refusal | 对应知识库 RAG)
    builder.add_conditional_edges("dispatcher", route_after_dispatch)
    builder.add_edge("refusal", END)
    # 两个 RAG 节点汇合到"充分性判定"
    builder.add_edge("tech_rag", "assess")
    builder.add_edge("math_rag", "assess")
    # 判定 -> 公式演算（始终经过，无需求时为空操作，保证日志顺序）
    builder.add_edge("assess", "calc_exec")
    # 演算 -> (资料不足且联网开启: gate_online | 否则: summary)
    builder.add_conditional_edges("calc_exec", route_after_calc)
    # 联网许可 -> (放行: online_search | 拒绝: summary)
    builder.add_conditional_edges("gate_online", route_after_gate)
    builder.add_edge("online_search", "summary")
    builder.add_edge("summary", END)

    return builder.compile()


# ------------------------------------------------------------
# 便捷运行入口（控制台 / Web 共用）
# ------------------------------------------------------------
_graph_app = None
_graph_lock = threading.Lock()


def _get_app():
    """返回已编译的图（进程内单例，线程安全懒加载）。"""
    global _graph_app
    if _graph_app is None:
        with _graph_lock:
            if _graph_app is None:
                _graph_app = build_graph()
    return _graph_app


def _initial_state(question: str, enable_online_search: bool,
                   chat_history: list = None, token_sink=None,
                   provider: str = "") -> dict:
    """构造一次对话的初始状态（一次性返回与流式输出共用）。"""
    state = {
        "question": (question or "").strip(),
        # 检索用查询默认同问题；追问场景由 dispatcher 用上一轮问题补全语义
        "search_query": (question or "").strip(),
        "enable_online_search": bool(enable_online_search),
        "chat_history": list(chat_history or []),
        "provider": (provider or config.DEFAULT_MODEL or "").strip(),
        "log": [],
        "log_text": "",
        "category": "tech",
        "reason": "",
        "use_calc": False,
        "search_keywords": "",
        "kb_used": "",
        "kb_docs": [],
        "kb_error": "",
        "sufficient": False,
        "calc_result": "",
        "online_allowed": False,
        "online_reason": "",
        "online_results": [],
        "answer": "",
    }
    if token_sink is not None:
        state["token_sink"] = token_sink
    return state


def run_chat(question: str, enable_online_search: bool,
             chat_history: list = None, _app=None, provider: str = "") -> dict:
    """执行一轮完整对话，返回 {answer, log_steps, log_text}（一次性返回）。

    provider：模型来源（local / external），None 表示用配置默认值。
    异常时返回友好错误消息，保证调用方（Web/控制台）不崩溃。
    """
    app = _app or _get_app()
    state = _initial_state(question, enable_online_search, chat_history,
                           provider=provider)
    try:
        final = app.invoke(state)
        answer = final.get("answer") or "抱歉，本次未能生成有效回答。"
        log_text = final.get("log_text") or ""
        if not log_text:  # 兜底：即使没走到 summary 也要输出日志
            from agents.summary_agent import SummaryAgent
            log_text = SummaryAgent._format_log_text(final.get("log") or [])
        return {"answer": answer, "log_steps": final.get("log") or [],
                "log_text": log_text}
    except Exception as exc:
        return {
            "answer": f"系统处理本轮问题出现异常：{str(exc)[:200]}。请稍后重试。",
            "log_steps": [{"kind": "错误", "title": "图执行异常",
                           "content": str(exc)[:500]}],
            "log_text": f"[1] (错误) 图执行异常\n     {str(exc)[:500]}",
        }


# ------------------------------------------------------------
# 流式输出：按发生顺序产出事件，供 SSE 接口推送给前端（打字机效果）
# ------------------------------------------------------------
def iter_chat_events(question: str, enable_online_search: bool,
                     chat_history: list = None, provider: str = ""):
    """同步生成器：逐步产出 (kind, payload)。

    kind 说明：
      "log"    -> 新增的一条结构化日志 {"kind", "title", "content"}
      "status" -> 阶段状态提示（字符串），如"汇总 Agent 正在生成答案…"
      "token"  -> 回答的增量文本（字符串），前端逐块追加即可
      "done"   -> 完成，payload = {"answer", "log_text"}
      "error"  -> 图执行异常信息（字符串）

    实现要点：图在后台线程执行（同步阻塞），通过队列把事件传回本生成器，
    因此消费方可以边读边推，不会阻塞图执行。
    """
    sink = queue.Queue()
    app = _get_app()
    state = _initial_state(question, enable_online_search, chat_history, sink,
                           provider=provider)

    def worker():
        try:
            for value in app.stream(state, stream_mode="values"):
                if isinstance(value, dict):
                    sink.put(("state", value))
        except Exception as exc:
            sink.put(("error", exc))
            sink.put(("__end__", None))
            return
        sink.put(("__end__", None))

    threading.Thread(target=worker, daemon=True).start()

    emitted = 0      # 已推送的日志条数，避免重复推送同一条
    final = {}
    while True:
        kind, payload = sink.get()          # 阻塞直到下一个事件
        if kind == "__end__":
            break
        if kind == "error":
            yield ("error", str(payload))
            return
        if kind in ("token", "status"):
            yield (kind, payload)
            continue
        if kind == "state":
            final = payload
            logs = payload.get("log") or []
            while emitted < len(logs):      # 只推送本次新增的日志
                yield ("log", logs[emitted])
                emitted += 1

    answer = (final.get("answer") or "").strip() or "抱歉，本次未能生成有效回答。"
    log_text = final.get("log_text") or SummaryAgent._format_log_text(final.get("log") or [])
    yield ("done", {"answer": answer, "log_text": log_text})


if __name__ == "__main__":  # 允许独立调试构图
    graph = build_graph()
    print("LangGraph 图构建成功，节点：",
          list(graph.get_graph().nodes.keys()) if hasattr(graph, "get_graph") else "OK")
