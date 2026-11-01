# ============================================================
# agents/dispatcher_agent.py —— 中控调度 Agent
# 功能（严格按离线优先业务规则实现）：
#   1) 话题过滤：闲聊/娱乐/生活/政治类问题直接拦截，不调用任何工具
#   2) 多轮对话记忆：携带最近若干轮历史，识别指代（"它/上面/该框架"等）
#   3) 问题分类：区分【大模型技术】tech 与【深度学习数学】math
#   4) 工具调用决策：判断是否需要离线公式演算（use_calc）、是否计算关键词
#   5) 联网许可判断：仅当本地资料不足且联网开关开启时才调用 judge_online，
#      且只有"前沿新技术/框架/算法"才允许放行（由 graph_builder 控制调用时机）
# 兜底策略：本地模型不可用时自动降级为关键词启发式分类，流程绝不中断。
# ============================================================
import re

import config
from llm_client import MissingDependencyError, OllamaError, get_llm_client, is_external
from vector_store import kb_filter

# 闲聊/问候拦截词（启发式兜底，主判别仍由本地大模型完成）
_GREETING_WORDS = ["你好", "您好", "嗨", "hi", "hello", "hey", "在吗",
                   "早上好", "下午好", "晚上好", "谢谢", "感谢", "再见", "拜拜"]
_MATH_WORDS = ["卷积", "池化", "矩阵", "向量", "概率", "二项", "损失函数",
               "导数", "梯度", "期望", "方差", "交叉熵", "线性代数", "公式",
               "推导", "下采样", "上采样", "stride", "padding"]
_TECH_WORDS = ["langchain", "langgraph", "rag", "embedding", "agent", "ollama",
               "大模型", "智能体", "检索增强", "向量库", "提示词", "多智能体",
               "mcp", "工具调用", "function calling", "向量检索", "模型微调",
               "知识库", "工作流", "graphrag", "token", "模型部署"]
_CALC_WORDS = ["卷积", "池化", "核大小", "卷积核", "步长", "填充", "stride",
               "padding", "矩阵", "向量", "点积", "概率", "二项", "组合数",
               "行列式", "相乘", "维度", "尺寸", "reshape"]
# 分类交叉校正词表：本地小模型常把"纯数学计算题"误判为 tech，
# 这里用关键词对模型判定结果做双向校正（模型判定优先，仅在明显偏斜时纠正）
_STRONG_MATH_CALC = ["卷积", "池化", "卷积核", "核大小", "步长", "填充", "stride",
                     "padding", "矩阵", "向量", "点积", "概率", "二项", "组合数",
                     "行列式", "损失函数", "交叉熵", "梯度", "导数", "期望",
                     "方差", "输出尺寸", "感受野"]
_TECH_FRAMEWORK = ["langchain", "langgraph", "rag", "agent", "智能体", "ollama",
                   "chroma", "embedding", "提示词", "向量库", "多智能体", "mcp",
                   "graphrag", "微调", "知识库", "大模型", "检索增强", "部署",
                   # 以下为「保护词」：出现即说明确属 AI 领域，避免被超范围拦截误伤
                   "深度学习", "神经网络", "机器学习", "模型训练", "训练模型"]
# 超范围话题兜底词表（闲聊 / 生活 / 娱乐类）。
# 背景：本地小模型常把这类问题误判成技术或数学（实测「今天天气怎么样」被判成 math，
#       进而去数学库检索并编造答案），故在此加一层确定性拦截：
#       命中本表、且不含任何专业特征词 → 一律拦截，不进入检索与生成。
_OFF_TOPIC_WORDS = ["天气", "气温", "降雨", "下雨", "下雪", "空气质量", "股票", "基金",
                    "彩票", "油价", "房价", "新闻", "明星", "八卦", "娱乐", "电影",
                    "电视剧", "综艺", "音乐", "歌曲", "游戏", "漫画", "小说", "菜谱",
                    "食谱", "做饭", "美食", "旅游", "景点", "酒店", "机票", "航班",
                    "恋爱", "失恋", "情感", "情诗", "写诗", "作文", "星座", "运势",
                    "算命", "养生", "减肥", "法律", "税务", "简历", "讲故事",
                    "讲个笑话", "无聊", "随便聊聊"]
# 允许联网的前沿关键词（启发式兜底）
_FRONTIER_WORDS = ["前沿", "最新", "新发布", "新框架", "新架构", "新算法",
                   "graphrag", "mcp", "多智能体", "新版本", "论文", "arxiv",
                   "2024", "2025", "2026", "最新进展", "新工具", "新特性"]


def _extend_log(state, kind: str, title: str, content: str) -> list:
    """把一条日志追加进日志列表并返回新列表。

    state 兼容两种传法：整个图状态 dict（内部取 state['log']），
    或直接传入当前的日志列表。返回合并后的新列表。
    """
    if isinstance(state, dict):
        log = list(state.get("log") or [])
    else:
        log = list(state or [])
    log.append({"kind": kind, "title": title, "content": content})
    return log


class DispatcherAgent:
    """中控调度 Agent：负责每轮对话的入口判别。"""

    def __init__(self, llm_client=None):
        self.llm = llm_client or get_llm_client()

    # --------------------------------------------------------
    # 内部小工具
    # --------------------------------------------------------
    def _history_text(self, history) -> str:
        """把最近几轮历史压缩成文本，供模型理解指代。"""
        turns = (history or [])[-config.HISTORY_MAX_TURNS * 2:]
        if not turns:
            return "（无历史，本问题为会话首轮）"
        lines = []
        for item in turns:
            role = "用户" if item.get("role") == "user" else "助手"
            content = str(item.get("content", ""))[:180]
            lines.append(f"{role}：{content}")
        return "\n".join(lines)

    # --------------------------------------------------------
    # 主入口：dispatch
    # --------------------------------------------------------
    def dispatch(self, state: dict) -> dict:
        """执行话题过滤 + 问题分类 + 工具决策。

        返回需要更新到图状态中的字段（部分更新，图自动合并）。
        """
        question = (state.get("question") or "").strip()
        history = state.get("chat_history") or []
        provider = (state.get("provider") or "").strip()
        updates = {"category": "reject", "reason": "", "use_calc": False,
                   "search_keywords": "", "answer": "", "log": state.get("log") or []}

        # 0) 空输入直接拦截
        if not question:
            updates["log"] = _extend_log(
                state, "错误", "输入为空", "用户提交了空问题，直接拒绝进入检索流程。")
            updates["answer"] = "未收到有效问题，请输入你想了解的大模型技术或深度学习数学问题。"
            return updates

        # 1) 记录本轮问题与历史
        updates["log"] = _extend_log(
            state, "意图判断", "本轮输入", f"用户问题：{question}")

        # 2) 调用本地大模型做统一判别（离线）
        sys_prompt = (
            "你是一个离线专业知识库问答系统的【中控调度员】。系统只提供两类服务：\n"
            "A. 大模型技术问题（LangChain / LangGraph / RAG / Embedding / Agent / Ollama / 提示词 / 向量检索等）；\n"
            "B. 深度学习配套数学问题（卷积尺寸公式 / 矩阵与向量运算 / 概率计算 / 损失函数 / 梯度推导等）。\n"
            "硬性规则：\n"
            "1) 闲聊、问候、娱乐、生活、政治、情感、新闻等与 A/B 无关的问题 → action 必须为 reject；\n"
            "2) 属于 A → action=tech；属于 B → action=math；两边都沾边时优先更主要的那个；\n"
            "3) 问题包含需要数值/公式演算（卷积尺寸、矩阵乘法、向量点积、二项概率等）时 use_calc=true；\n"
            "4) 只要属于 A 或 B 领域，就必须归入 tech 或 math——哪怕问题表述简短、"
            "像一道填空题（如「输入 32x32，卷积核 3，填充 1，步长 1，输出尺寸是多少」），"
            "也绝不可判为 reject；只有寒暄、闲聊、生活娱乐等才 reject；\n"
            "5) 只输出 JSON，不要任何解释。\n"
            'JSON 格式：{"action": "reject"|"tech"|"math", "reason": "一句话理由", "use_calc": true|false}'
        )
        user_prompt = (
            f"【历史对话】\n{self._history_text(history)}\n\n"
            f"【本轮问题】{question}\n\n请按规则输出 JSON。"
        )
        try:
            # 本地模式先探测 Ollama 连通性，避免 ask_json 吞掉连接错误后误判为 reject；
            # 外部模式直接请求 API，不需要探测本地服务
            if not is_external(provider) and not self.llm.ping():
                raise OllamaError("Ollama 服务未就绪（ping 失败）。")
            prompt = [("system", sys_prompt), ("user", user_prompt)]
            parsed = self.llm.ask_json(prompt, fallback={
                "action": "reject", "reason": "模型未返回可解析的 JSON",
                "use_calc": False}, provider=provider)
            action = str(parsed.get("action", "")).lower().strip()
            reason = str(parsed.get("reason", "")).strip()
            use_calc = bool(parsed.get("use_calc", False))
            # 规则校验：模型偶尔会给无关问题打上 use_calc=true（实测「什么是傅里叶变换」
            # 也被判成需要计算），这里要求问题确实含计算特征词才真正调用工具，
            # 避免无效调用产生的「无法识别计算类型」提示污染后续充分性判定。
            if use_calc and not any(w in question.lower() for w in _CALC_WORDS):
                use_calc = False
                updates["log"] = _extend_log(
                    updates["log"], "工具调用", "取消公式演算（无计算特征词）",
                    "模型判定需要演算，但问题中未出现卷积/矩阵/概率等计算特征词，"
                    "已取消调用，避免无效工具调用。")
            if action not in ("tech", "math", "reject"):
                action = "reject"
        except (OllamaError, MissingDependencyError) as exc:
            # 本地模型不可用：降级为关键词启发式，保证流程可继续
            action, reason = self._heuristic_classify(question)
            use_calc = any(w in question.lower() for w in _CALC_WORDS)
            lg = list(updates["log"])
            lg.append({"kind": "错误", "title": "本地模型调用失败，已降级启发式分类",
                       "content": f"{exc}（关键词兜底：action={action}）"})
            updates["log"] = lg

        # 3) 分类交叉校正：修正小模型把纯计算题误判方向的情况
        before_action = action
        action, reason = self._guard_category(question, action, reason)
        if action != before_action:
            updates["log"] = _extend_log(
                updates["log"], "意图判断", "分类交叉校正",
                f"模型/启发式判定 {before_action} → 校正为 {action}。\n理由：{reason}")

        # 3.4) 拒绝误判纠正：模型偶尔把明确的技术/数学题判成 reject（详见 _guard_reject）
        before_reject = action
        action, reason = self._guard_reject(question, action, reason)
        if action != before_reject:
            updates["log"] = _extend_log(
                updates["log"], "意图判断", "拒绝误判纠正",
                f"模型原判 reject → 校正为 {action}。\n理由：{reason}")

        # 3.5) 超范围话题兜底拦截（规则详见 _guard_off_topic 的说明）
        before_off = action
        action, reason = self._guard_off_topic(question, action, reason)
        if action != before_off:
            updates["log"] = _extend_log(
                updates["log"], "意图判断", "超范围话题兜底拦截",
                f"原判定为 {before_off}，但问题属生活/闲聊类且无专业特征词 → 拦截。\n"
                f"理由：{reason}")

        # 4) 话题过滤：reject 直接终止，不调用任何工具
        if action == "reject":
            updates["log"] = _extend_log(
                updates["log"], "意图判断", "话题过滤：拦截（不调用任何工具）",
                f"分类结果：闲聊/无关话题。理由：{reason or '不在本系统服务范围内'}。")
            updates["answer"] = (
                "【话题拦截】本系统是离线专业知识库问答助手，只回答"
                "「大模型技术（LangChain/LangGraph/RAG/Agent…）」与"
                "「深度学习数学（卷积/矩阵/向量/概率/损失函数）」两类问题；"
                "闲聊、娱乐、生活类提问不在服务范围内，感谢理解。")
            return updates

        # 5) 正常路由
        kb_name = {"tech": "大模型技术知识库", "math": "深度学习数学知识库"}[action]
        updates["category"] = action
        updates["reason"] = f"分类=「{'tech' if action == 'tech' else 'math'}」，理由：{reason}"
        updates["log"] = _extend_log(
            updates["log"], "意图判断", "问题分类完成",
            f"判定归属：{kb_name}\n理由：{reason or '（模型未给出明确理由）'}")

        # 5.1) 追问改写：把上一轮问题拼进来，补全「第二步具体怎么做」这类追问的语义。
        #      否则短追问直接检索会一无所获，被误判成「资料不足」。
        search_query = kb_filter.resolve_followup_query(question, history)
        updates["search_query"] = search_query
        if search_query != question:
            updates["log"] = _extend_log(
                updates["log"], "意图判断", "追问改写（补全检索语义）",
                f"原问题：{question}\n用于检索的查询：{search_query}")

        # 6) 工具调用决策：是否需要离线公式演算
        if use_calc:
            updates["log"] = _extend_log(
                updates["log"], "工具调用", "决策：调用离线公式演算工具",
                f"命中数值/公式演算需求（use_calc=true）。原问题：{question}")
        else:
            updates["log"] = _extend_log(
                updates["log"], "工具调用", "决策：无需公式演算",
                "本问题不涉及卷积/矩阵/概率等数值计算，跳过 calc_tool。")
        updates["use_calc"] = use_calc

        # 7) 联网关键词预处理（只保留精简专业词，绝不带文档原文）
        keywords = self.build_search_keywords(question)
        updates["search_keywords"] = keywords
        updates["log"] = _extend_log(
            updates["log"], "意图判断", "联网关键词准备（备用，是否联网见后续判定）",
            f"精简专业检索词：{keywords}")
        return updates

    # --------------------------------------------------------
    # 启发式降级分类
    # --------------------------------------------------------
    def _heuristic_classify(self, question: str):
        """本地模型不可用时的关键词兜底分类。"""
        q = question.lower()
        if any(w in q for w in _GREETING_WORDS):
            return "reject", "命中问候/闲聊关键词，按规则拦截。"
        math_hit = [w for w in _MATH_WORDS if w in q]
        tech_hit = [w for w in _TECH_WORDS if w in q]
        if math_hit and not tech_hit:
            return "math", f"启发式命中数学关键词：{math_hit}"
        if tech_hit and not math_hit:
            return "tech", f"启发式命中技术关键词：{tech_hit}"
        if math_hit and tech_hit:
            # 两边都命中：卷积/概率类数学词优先归 math，其余归 tech
            strong_math = any(w in q for w in ("卷积", "矩阵", "向量", "概率", "损失函数"))
            if strong_math:
                return "math", f"启发式双命中但数学意图更强：{math_hit}"
            return "tech", f"启发式双命中：tech={tech_hit}, math={math_hit}"
        return "tech", "未命中明确拦截词，按默认大模型技术问题路由尝试检索。"

    # --------------------------------------------------------
    # 超范围话题兜底拦截
    # --------------------------------------------------------
    @staticmethod
    def _guard_off_topic(question: str, action: str, reason: str):
        """命中生活/闲聊词、且无任何专业特征词时，强制拦截。

        背景：本地小模型会把生活类问题误判成 tech/math（实测「今天天气怎么样」
        被判成 math，进而去数学库检索并编造答案）。这里做确定性兜底，不依赖模型。
        返回 (校正后的 action, 补充后的 reason)。
        """
        if action not in ("tech", "math"):
            return action, reason
        q = (question or "").lower()
        off_hits = [w for w in _OFF_TOPIC_WORDS if w in q]
        if not off_hits:
            return action, reason
        # 含任一专业特征词（含「深度学习/神经网络」等保护词）则放行，避免误伤
        prof_hits = ([w for w in _STRONG_MATH_CALC if w in q]
                     + [w for w in _TECH_FRAMEWORK if w in q])
        if prof_hits:
            return action, reason
        return "reject", (f"命中超范围话题词 {off_hits}，且无任何专业特征词，"
                          f"判定为闲聊 / 生活类问题。")

    # --------------------------------------------------------
    # 拒绝误判纠正（reject -> tech/math）
    # --------------------------------------------------------
    @staticmethod
    def _guard_reject(question: str, action: str, reason: str):
        """模型误判为 reject 时的反向纠正。

        背景：本地小模型偶尔会把明确的技术/数学问题判成 reject，实测出现过
        「输入 32x32，卷积核 3，填充 1，步长 1，输出尺寸是多少」被拦截的情况，
        表现为正常题目被拒之门外。若问题含强专业特征词，则纠正回专业分类。
        注意：闲聊/生活类问题不含这些特征词，因此不会被误放行。
        返回 (校正后的 action, 补充后的 reason)。
        """
        if action != "reject":
            return action, reason
        q = (question or "").lower()
        math_hit = [w for w in _STRONG_MATH_CALC if w in q]
        tech_hit = [w for w in _TECH_FRAMEWORK if w in q]
        if math_hit and not tech_hit:
            return "math", f"模型误判为拦截，但命中数学特征词 {math_hit}，纠正为 math。"
        if tech_hit and not math_hit:
            return "tech", f"模型误判为拦截，但命中技术框架词 {tech_hit}，纠正为 tech。"
        if math_hit and tech_hit:
            return "math", (f"模型误判为拦截，但数学与技术特征词均命中"
                            f"（{math_hit} / {tech_hit}），按数学优先纠正为 math。")
        return action, reason

    # --------------------------------------------------------
    # 分类交叉校正
    # --------------------------------------------------------
    @staticmethod
    def _guard_category(question: str, action: str, reason: str):
        """对模型（或启发式）给出的分类做关键词交叉校正。

        本地小模型容易把纯数学计算题（如"输入32x32，卷积核3，输出尺寸"）
        判成 tech，导致去技术库检索而落空。这里只在"一侧命中、另一侧完全
        没命中"的明显偏斜情况下纠正，双向生效，reject 不参与校正。
        返回 (校正后的 action, 补充后的 reason)。
        """
        q = (question or "").lower()
        math_hit = [w for w in _STRONG_MATH_CALC if w in q]
        tech_hit = [w for w in _TECH_FRAMEWORK if w in q]

        if action == "tech" and math_hit and not tech_hit:
            return "math", (f"{reason}（关键词校正：命中数学计算特征 {math_hit}，"
                            f"且无任何技术框架词，改判 math）")
        if action == "math" and tech_hit and not math_hit:
            return "tech", (f"{reason}（关键词校正：命中技术框架词 {tech_hit}，"
                            f"且无数学计算特征，改判 tech）")
        return action, reason

    # --------------------------------------------------------
    # 联网许可判定（只在本地资料不足 & 联网开关开启时被调用）
    # --------------------------------------------------------
    def judge_online(self, state: dict) -> dict:
        """判断：即使本地资料不足且联网开启，也仅有"前沿新技术/框架/算法"可放行。"""
        question = (state.get("question") or "").strip()
        provider = (state.get("provider") or "").strip()
        updates = {"online_allowed": False, "online_reason": "", "log": state.get("log") or []}

        sys_prompt = (
            "你是离线问答系统的【联网许可审核员】。系统严格离线优先，只有满足全部条件才能允许联网：\n"
            "1) 提问确实属于 AI/深度学习/大模型领域的【前沿新技术、新框架、新算法】\n"
            "   （例如新发布的模型/论文/框架版本、GraphRAG、MCP、新型多智能体范式等）；\n"
            "2) 普通基础知识点（RAG是什么、卷积公式等经典知识）即使本地资料缺失也【禁止】联网；\n"
            "3) 新闻、生活、娱乐类内容永远禁止联网。\n"
            "只输出 JSON：{\"allow\": true|false, \"reason\": \"一句话理由\"}"
        )
        user_prompt = f"待审核问题：{question}\n请输出 JSON。"
        try:
            if not is_external(provider) and not self.llm.ping():
                raise OllamaError("Ollama 服务未就绪（ping 失败）。")
            parsed = self.llm.ask_json(
                [("system", sys_prompt), ("user", user_prompt)],
                fallback={"allow": False, "reason": "模型未返回有效判定"},
                provider=provider)
            allow = bool(parsed.get("allow", False))
            reason = str(parsed.get("reason", "")).strip() or "（未给出理由）"
        except (OllamaError, MissingDependencyError) as exc:
            # 模型不可用 -> 保守策略：默认拒绝联网（离线优先）
            allow = False
            reason = f"本地模型不可用，按保守策略默认不联网（{exc}）"
            updates["log"] = _extend_log(
                updates["log"], "错误", "联网判定降级", reason)

        # 双保险：即使模型放行，也做关键词交叉验证
        if allow and not any(w in question.lower() for w in _FRONTIER_WORDS):
            allow = False
            reason = (f"模型倾向放行但问题缺乏「前沿/最新/新框架/论文」等新内容特征"
                      f"（{reason}），交叉验证不通过，拒绝联网。")

        if allow:
            updates["online_allowed"] = True
            updates["log"] = _extend_log(
                updates["log"], "意图判断", "联网许可：放行",
                f"判定为 AI 前沿新技术问题，允许联网检索。理由：{reason}")
        else:
            updates["online_allowed"] = False
            updates["log"] = _extend_log(
                updates["log"], "意图判断", "联网许可：拒绝",
                f"判定为非前沿新技术（普通知识点/无关话题），即使本地资料不足也不联网。理由：{reason}")
        updates["online_reason"] = reason
        return updates

    # --------------------------------------------------------
    # 关键词构造
    # --------------------------------------------------------
    @staticmethod
    def build_search_keywords(question: str) -> str:
        """从问题中提炼精简专业检索词（清理疑问句壳，最长约 60 字符）。"""
        q = question.strip()
        q = re.sub(r"[?？。！!，,、：:;；\"'”“‘’]", " ", q)
        for phrase in ("请帮我", "帮我", "请问", "如何实现", "怎么实现", "介绍一下",
                       "是什么", "怎么做", "如何", "如何做", "解释一下", "说明一下",
                       "我想知道", "请解释", "请介绍"):
            q = q.replace(phrase, " ")
        words = [w for w in re.split(r"\s+", q) if w and len(w) <= 30]
        cleaned = " ".join(words).strip()
        return cleaned[:60] or question[:60]
