# ============================================================
# tools/online_search_tool.py —— 扩展联网检索工具（可开关，@tool 注册）
# 功能：
#   1) 仅检索 AI / 深度学习 / 大模型前沿专业技术资料，
#      命中新闻、娱乐、生活等话题时直接拒绝，不发起请求
#   2) 只提交精简专业关键词（query），绝不附带本地文档原文
#   3) 检索实现：Bing 中文搜索页轻量抓取，失败自动回退 DuckDuckGo，
#      不引入第三方搜索 SDK；网络异常/被拦截时给出友好失败信息，不影响系统可用性
#      说明：早期只接 DuckDuckGo，但部分网络环境访问不到该站点（实测连接超时），
#            故改为「Bing 优先 + DuckDuckGo 回退」的多源方案。
# 注意：本工具需在 dispatcher 的"前沿技术许可"判定通过后才被图调用；
#       本文件内再次做领域自检，形成双重防线。
# ============================================================
import html as html_mod
import re

from langchain_core.tools import tool

import config

ALLOWED_TOPICS = (
    "大模型", "llm", "gpt", "langchain", "langgraph", "rag", "graphrag",
    "agent", "multi-agent", "智能体", "embedding", "向量", "ollama",
    "检索增强", "提示词", "prompt", "token", "深度学习", "神经网络",
    "transformer", "扩散模型", "多模态", "模型微调", "微调", "finetune",
    "算法", "mcp", "function calling", "工具调用", "模型部署", "训练",
    "推理", "量化", "蒸馏", "论文", "arxiv", "ai",
)
BLOCKED_TOPICS = (
    "新闻", "时事", "娱乐", "明星", "八卦", "电影", "综艺", "体育",
    "股票", "基金", "财经", "楼市", "美食", "旅游", "游戏攻略", "天气",
    "政治", "军事", "招聘", "购物", "明星绯闻", "八卦新闻", "社会热点",
)

_UA = ("Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
       "(KHTML, like Gecko) Chrome/124.0 Safari/537.36")


# ------------------------------------------------------------
# 领域自检
# ------------------------------------------------------------
def check_domain(query: str) -> tuple:
    """领域自检：返回 (是否允许检索, 理由)。

    规则：
      * 命中 BLOCKED_TOPICS         -> 直接拒绝（新闻/生活等一律不许）
      * 命中至少一个 ALLOWED_TOPICS -> 允许
      * 两者都未命中               -> 无法确认是 AI 技术话题，保守拒绝
    """
    q = query.lower()
    for word in BLOCKED_TOPICS:
        if word.lower() in q:
            return False, f"查询命中禁止话题「{word}」，联网工具只服务 AI/深度学习前沿技术。"
    for word in ALLOWED_TOPICS:
        if word.lower() in q:
            return True, ""
    return False, "查询不属于可识别的 AI/深度学习技术话题，拒绝联网检索。"


def _strip_html_tags(raw: str) -> str:
    return re.sub(r"\s+", " ", re.sub(r"<[^>]+>", "", raw)).strip()


# ------------------------------------------------------------
# 检索实现
# ------------------------------------------------------------
def _duckduckgo_search(keywords: str) -> str:
    """抓取 DuckDuckGo HTML 结果页并整理为文本（最多 4 条）。"""
    import requests
    resp = requests.get(
        "https://html.duckduckgo.com/html/",
        params={"q": keywords, "kl": "cn-zh"},
        headers={"User-Agent": _UA, "Accept-Language": "zh-CN,zh;q=0.9,en;q=0.5"},
        timeout=config.NETWORK_TIMEOUT,
    )
    if resp.status_code != 200:
        raise RuntimeError(f"搜索引擎返回异常状态码 {resp.status_code}")
    page = resp.text

    # 结果块级提取：result__a(标题/链接) + result__snippet(摘要)
    titles = re.findall(
        r'<a[^>]*class="[^"]*result__a[^"]*"[^>]*href="([^"]+)"[^>]*>(.*?)</a>',
        page, flags=re.S)
    snippets = re.findall(
        r'<a[^>]*class="[^"]*result__snippet[^"]*"[^>]*>(.*?)</a>',
        page, flags=re.S)

    if not titles:
        raise RuntimeError("未解析到搜索结果（可能网络受限或页面结构变化）")

    lines = []
    for i, (url, title) in enumerate(titles[:4], start=1):
        title_txt = html_mod.unescape(_strip_html_tags(title))[:120]
        snippet_txt = ""
        if i - 1 < len(snippets):
            snippet_txt = html_mod.unescape(_strip_html_tags(snippets[i - 1]))[:300]
        lines.append(
            f"{i}. 标题：{title_txt}\n   摘要：{snippet_txt}\n   来源：{url[:200]}")
    return "\n\n".join(lines)


def _bing_search(keywords: str) -> str:
    """抓取 Bing 中文搜索页并整理为文本（最多 4 条）。

    输出格式与 _duckduckgo_search 保持一致，便于上层统一解析。
    """
    import requests
    resp = requests.get(
        "https://cn.bing.com/search",
        params={"q": keywords, "setlang": "zh-CN"},
        headers={"User-Agent": _UA, "Accept-Language": "zh-CN,zh;q=0.9,en;q=0.5"},
        timeout=config.NETWORK_TIMEOUT,
    )
    if resp.status_code != 200:
        raise RuntimeError(f"搜索引擎返回异常状态码 {resp.status_code}")
    page = resp.text

    titles = re.findall(
        r'<h2[^>]*>\s*<a[^>]*href="([^"]+)"[^>]*>(.*?)</a>', page, flags=re.S)
    snippets = re.findall(
        r'<p[^>]*class="[^"]*b_lineclamp[^"]*"[^>]*>(.*?)</p>', page, flags=re.S)

    if not titles:
        raise RuntimeError("未解析到搜索结果（可能网络受限或页面结构变化）")

    lines = []
    for i, (url, title) in enumerate(titles[:4], start=1):
        title_txt = html_mod.unescape(_strip_html_tags(title))[:120]
        snippet_txt = ""
        if i - 1 < len(snippets):
            snippet_txt = html_mod.unescape(_strip_html_tags(snippets[i - 1]))[:300]
        lines.append(
            f"{i}. 标题：{title_txt}\n   摘要：{snippet_txt}\n   来源：{url[:200]}")
    return "\n\n".join(lines)


# 检索源优先级：按顺序逐个尝试，任一成功即返回（多源回退，避免单点不可达）
_SEARCH_PROVIDERS = (
    ("Bing 中文", _bing_search),
    ("DuckDuckGo", _duckduckgo_search),
)


# ------------------------------------------------------------
# 对外主函数（@tool 注册）
# ------------------------------------------------------------
def online_tech_search(keywords: str) -> str:
    """联网检索 AI / 深度学习 / 大模型前沿技术资料（仅限技术话题）。

    keywords：精简的专业检索词（如 "GraphRAG 最新架构 2025"）。
    返回检索片段列表，每条标注来源 URL；被拦截或全部搜索源不可用时返回提示文本。
    """
    keywords = (keywords or "").strip()
    if not keywords:
        return "检索关键词为空，请提供精简的专业技术关键词。"

    # 第一道防线：领域自检
    allowed, reason = check_domain(keywords)
    if not allowed:
        return f"[联网检索被拒绝] {reason}"

    # 第二道防线：多源请求（异常一律友好降级）
    errors = []
    for name, fetch in _SEARCH_PROVIDERS:
        try:
            result = fetch(keywords)
            return f"【外部补充资料】关键词「{keywords}」联网检索结果：\n{result}"
        except Exception as exc:
            errors.append(f"{name}（{str(exc)[:70]}）")
            continue
    return ("[联网检索失败] 所有搜索源均不可用：" + "；".join(errors) +
            "。系统将如实告知资料不足，不会编造内容。")


# 注册为 LangChain Tool 对象（同时保留原函数便于直接调用）
online_search_tool_obj = tool(online_tech_search)

if __name__ == "__main__":  # 允许单独调试
    print(online_tech_search("LangGraph 最新版本 多智能体"))
