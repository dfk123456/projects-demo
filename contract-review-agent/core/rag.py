# ============================================================
# rag.py —— 法律法规条文 RAG 知识库
#
# 知识源：knowledge_docs/ 下的 .md / .txt，格式：
#   # 《中华人民共和国民法典》第五百八十五条
#   分类：违约金
#   关键词：违约金,损失赔偿,过高调整
#   正文：当事人可以约定一方违约时应当根据违约情况向对方支付一定数额的违约金……
#
# 检索：向量召回 + 关键词命中加权，返回可直接写进报告的「参考法条」。
# ============================================================
import hashlib
import os
import re
from typing import Any, Dict, List

import config
from core.embeddings import tokenize
from core.llm_client import get_llm_client
from core.logging_setup import setup_logger
from core.vectorstore import VectorDoc, get_vector_store

logger = setup_logger()

ARTICLE_SPLIT = re.compile(r"^#\s+(.+)$", re.MULTILINE)
FIELD_RE = {
    "category": re.compile(r"^分类[：:]\s*(.+)$", re.MULTILINE),
    "keywords": re.compile(r"^关键词[：:]\s*(.+)$", re.MULTILINE),
    "body": re.compile(r"^正文[：:]\s*([\s\S]+?)(?=\n#\s|\Z)", re.MULTILINE),
}


class LawArticle(dict):
    """{'id','title','law','article','category','keywords','text'}"""


def parse_law_file(path: str) -> List[LawArticle]:
    """把一个知识库文件解析为若干法条条目。"""
    with open(path, "r", encoding="utf-8", errors="ignore") as fh:
        raw = fh.read()
    blocks = ARTICLE_SPLIT.split(raw)
    # split 结果：[前缀, 标题1, 内容1, 标题2, 内容2, ...]
    articles: List[LawArticle] = []
    for i in range(1, len(blocks), 2):
        title = blocks[i].strip()
        body_text = blocks[i + 1] if i + 1 < len(blocks) else ""
        cat = FIELD_RE["category"].search(body_text)
        kw = FIELD_RE["keywords"].search(body_text)
        body = FIELD_RE["body"].search(body_text)
        text = (body.group(1) if body else body_text).strip()
        text = re.sub(r"^\s*(分类|关键词)[：:].*$", "", text, flags=re.MULTILINE).strip()
        if not text:
            continue
        if "《" not in title:
            continue                                   # 跳过说明性段落（须为『《xx法》第x条』格式）
        law_name, article_no = split_law_ref(title)
        articles.append(LawArticle(
            id=hashlib.md5(title.encode("utf-8")).hexdigest()[:16],
            title=title, law=law_name, article=article_no,
            category=(cat.group(1).strip() if cat else ""),
            keywords=(kw.group(1).strip() if kw else ""),
            text=text[:1500]))
    return articles


def split_law_ref(title: str):
    """从『《民法典》第五百八十五条』中拆出法律名与条款号。"""
    m = re.search(r"《(.+?)》\s*(.*)$", title)
    if not m:
        return title.strip(), ""
    law = m.group(1).strip()
    no = m.group(2).strip()
    no_match = re.search(r"第[零一二三四五六七八九十百千0-9]+条", no)
    return law, (no_match.group(0) if no_match else no)


def reference_of(article: Dict[str, Any]) -> str:
    """生成规范的法条引用：『《中华人民共和国民法典》第五百八十五条』。"""
    if article.get("law") and article.get("article"):
        return f"《{article['law']}》{article['article']}"
    return article.get("title", "")


class LawKnowledgeBase:
    """法律法规知识库：建库（入库） + 检索。"""

    COLLECTION = config.MILVUS_COLLECTION

    def __init__(self, embed_id: str = ""):
        self.embed_id = embed_id
        self.store = get_vector_store(self.COLLECTION)

    # ---------------- 建库 ----------------
    def build(self, knowledge_dir: str = None, rebuild: bool = False) -> Dict[str, Any]:
        knowledge_dir = knowledge_dir or config.KNOWLEDGE_DIR
        if rebuild:
            self.store.clear()
        files = []
        if os.path.isdir(knowledge_dir):
            for name in sorted(os.listdir(knowledge_dir)):
                if name.lower().endswith((".md", ".txt")):
                    files.append(os.path.join(knowledge_dir, name))
        if not files:
            logger.warning("知识库目录为空：%s", knowledge_dir)
            return {"files": 0, "added": 0, "total": self.store.count()}

        articles: List[LawArticle] = []
        for path in files:
            articles.extend(parse_law_file(path))
        logger.info("知识库解析：%s 个文件 → %s 条法条", len(files), len(articles))

        # vector store 已按 id 做 upsert，重复入库不会产生重复条目（幂等）
        pending = articles
        added = 0
        batch_size = 16
        llm = get_llm_client()
        for i in range(0, len(pending), batch_size):
            batch = pending[i:i + batch_size]
            texts = [f"{a['title']}\n{a['keywords']}\n{a['text']}" for a in batch]
            vectors = llm.embed_texts(texts, embed_id=self.embed_id)
            docs = [VectorDoc(id=a["id"], text=a["text"],
                              metadata={"title": a["title"], "law": a["law"],
                                        "article": a["article"], "category": a["category"],
                                        "keywords": a["keywords"], "reference": reference_of(a)},
                              vector=vectors[j])
                    for j, a in enumerate(batch)
                    if j < len(vectors) and len(vectors[j]) == len(vectors[0])]
            if docs:
                self.store.upsert(docs)
                added += len(docs)
        result = {"files": len(files), "added": added, "total": self.store.count()}
        logger.info("知识库入库完成：新增 %s 条，总计 %s 条", added, result["total"])
        return result

    # ---------------- 检索 ----------------
    def retrieve(self, query: str, top_k: int = None, trace=None) -> List[Dict[str, Any]]:
        """检索与 query 最相关的法条，返回 [{reference, excerpt, score, category}]。"""
        top_k = int(top_k or config.RAG_TOP_K)
        try:
            vector = get_llm_client().embed_query(query, embed_id=self.embed_id)
            hits = self.store.search(vector, top_k=max(top_k * 3, 10))
        except Exception as exc:
            if trace and hasattr(trace, "append"):
                trace.append(f"法条检索失败：{str(exc)[:120]}")
            return []
        q_tokens = set(tokenize(query))
        out = []
        for hit in hits:
            meta = hit.get("metadata", {}) or {}
            kw = {t for t in re.split(r"[,，、\s]+", meta.get("keywords", "")) if t}
            hit_tokens = set(tokenize((meta.get("title", "") + " " + meta.get("keywords", ""))))
            overlap = len(q_tokens & (hit_tokens | kw))
            score = float(hit.get("score", 0.0)) + config.RAG_KEYWORD_BOOST * min(overlap, 5)
            out.append({"reference": meta.get("reference") or meta.get("title", ""),
                        "excerpt": (hit.get("text", "") or "")[:220],
                        "category": meta.get("category", ""),
                        "score": round(score, 4)})
        out.sort(key=lambda x: x["score"], reverse=True)
        return out[:top_k]

    # ---------------- 给风险点补充法条 ----------------
    def enrich(self, risk: Dict[str, Any], top_k: int = None) -> List[Dict[str, Any]]:
        """为单个风险点补充参考法条（过滤低分）。"""
        query = " ".join(str(risk.get(k, "")) for k in
                         ("risk_type", "description", "quote"))[:500]
        refs = self.retrieve(query, top_k=top_k)
        return [r for r in refs if r["score"] >= config.RAG_MIN_SCORE]

    def count(self) -> int:
        return self.store.count()
