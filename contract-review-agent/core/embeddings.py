# ============================================================
# embeddings.py —— 文本向量化（在线向量模型 + 本地词法向量兜底）
#
# 对外只暴露 LocalLexicalEmbedder（离线、零依赖、无需密钥）。
# 在线 Embedding 的调用统一放在 llm_client.py 里，共用同一份模型清单式配置。
# ============================================================
import hashlib
import math
import re
from typing import List

try:
    import jieba
except ImportError:  # pragma: no cover
    jieba = None

import numpy as np


def tokenize(text: str) -> List[str]:
    """中文分词 + 英文/数字切词，返回 token 列表。"""
    text = (text or "").lower()
    if jieba is not None:
        try:
            tokens = [t for t in jieba.lcut_for_search(text) if t.strip()]
        except Exception:
            tokens = re.findall(r"[\u4e00-\u9fa5]+|[a-z0-9]+", text)
    else:
        tokens = re.findall(r"[\u4e00-\u9fa5]+|[a-z0-9]+", text)
    out = []
    for tk in tokens:
        out.append(tk)
        # 中文二元组：提升短 query 与新词召回（“违约金”/“免赔”等）
        if re.fullmatch(r"[\u4e00-\u9fa5]{2,}", tk):
            for i in range(len(tk) - 1):
                out.append(tk[i:i + 2])
    return out


class LocalLexicalEmbedder:
    """本地词法向量（hashing + sublinear TF + 长度归一化）。

    特点：
      * 不需要任何外部服务 / 密钥，离线可用；
      * 向量维度固定 → 内积即加权词共现相似度，配合 jieba 二元组，
        对「违约金 / 免责 / 期限」类法条检索效果可用；
      * 用于在线 Embedding 不可用时的兜底，保证 RAG 环节不中断。
    """

    def __init__(self, dim: int = 1024):
        self.dim = int(dim)

    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        return [self._embed_one(t) for t in texts]

    def embed_query(self, text: str) -> List[float]:
        return self._embed_one(text)

    def _embed_one(self, text: str) -> List[float]:
        vec = np.zeros(self.dim, dtype=np.float32)
        tokens = tokenize(text)
        if not tokens:
            return vec.tolist()
        tf: dict = {}
        for tk in tokens:
            key = int(hashlib.md5(tk.encode("utf-8")).hexdigest()[:8], 16) % self.dim
            # 短 token（单字）降权，避免噪声
            tf[key] = tf.get(key, 0.0) + (1.0 if len(tk) > 1 else 0.35)
        for idx, freq in tf.items():
            vec[idx] = 1.0 + math.log(freq)          # sublinear TF
        norm = float(np.linalg.norm(vec))
        if norm > 0:
            vec = vec / norm
        return vec.tolist()


def cosine(a: List[float], b: List[float]) -> float:
    """余弦相似度（向量均已归一化时等价于内积）。"""
    va, vb = np.asarray(a, dtype=np.float32), np.asarray(b, dtype=np.float32)
    den = float(np.linalg.norm(va) * np.linalg.norm(vb))
    if den <= 0:
        return 0.0
    return float(np.dot(va, vb) / den)
