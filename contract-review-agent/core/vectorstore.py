# ============================================================
# vectorstore.py —— 向量库抽象（Milvus 正式 + 本地文件库开箱即用）
#
# 统一接口：upsert / search / count / clear / persist
# 切换方式：config.VECTOR_BACKEND = auto | milvus | local
# 说明：本机无 Docker / Milvus 服务时自动使用本地文件向量库（numpy 余弦检索），
#       功能等价、重启不丢数据，部署时把 VECTOR_BACKEND 改成 milvus 即可。
# ============================================================
import json
import os
import threading
from typing import Any, Dict, List, Optional

import numpy as np

import config
from core.logging_setup import setup_logger

logger = setup_logger()


class VectorDoc(dict):
    """{'id': str, 'text': str, 'vector': list[float], 'metadata': dict}"""


class BaseVectorStore:
    def upsert(self, docs: List[VectorDoc]):
        raise NotImplementedError

    def search(self, vector: List[float], top_k: int = 5) -> List[Dict[str, Any]]:
        raise NotImplementedError

    def count(self) -> int:
        raise NotImplementedError

    def clear(self):
        raise NotImplementedError

    def persist(self):
        pass


# ------------------------------------------------------------
# 本地文件向量库
# ------------------------------------------------------------
class LocalVectorStore(BaseVectorStore):
    """numpy 暴力余弦检索 + JSON 落盘，零外部依赖。"""

    def __init__(self, name: str, dir_path: str, dim: int = None):
        self.name = name
        self.dir_path = dir_path
        self.dim = dim or config.MILVUS_DIM
        os.makedirs(dir_path, exist_ok=True)
        self.index_file = os.path.join(dir_path, f"{name}.json")
        self.vector_file = os.path.join(dir_path, f"{name}.npy")
        self.docs: List[Dict[str, Any]] = []
        self.matrix: Optional[np.ndarray] = None
        self._lock = threading.RLock()
        self._load()

    def _load(self):
        if os.path.exists(self.index_file) and os.path.exists(self.vector_file):
            try:
                with open(self.index_file, "r", encoding="utf-8") as fh:
                    self.docs = json.load(fh)
                self.matrix = np.load(self.vector_file)
                logger.info("已加载本地向量库 %s：%s 条", self.name, len(self.docs))
            except Exception as exc:
                logger.warning("本地向量库加载失败（将重建）：%s", str(exc)[:120])
                self.docs, self.matrix = [], None

    def persist(self):
        with self._lock:
            if not self.docs:
                np.save(self.vector_file, np.zeros((0, self.dim), dtype=np.float32))
            else:
                np.save(self.vector_file, np.asarray(self.matrix, dtype=np.float32))
            with open(self.index_file, "w", encoding="utf-8") as fh:
                json.dump(self.docs, fh, ensure_ascii=False)

    def upsert(self, docs: List[VectorDoc]):
        with self._lock:
            existing = {d["id"]: i for i, d in enumerate(self.docs)}
            vectors = list(self.matrix) if self.matrix is not None else []
            for doc in docs:
                vec = np.asarray(doc["vector"], dtype=np.float32)
                norm = np.linalg.norm(vec)
                if norm > 0:
                    vec = vec / norm
                record = {"id": doc["id"], "text": doc["text"],
                          "metadata": doc.get("metadata", {})}
                if doc["id"] in existing:
                    self.docs[existing[doc["id"]]] = record
                    vectors[existing[doc["id"]]] = vec
                else:
                    existing[doc["id"]] = len(self.docs)
                    self.docs.append(record)
                    vectors.append(vec)
            self.matrix = (np.vstack(vectors) if vectors
                           else np.zeros((0, self.dim), dtype=np.float32))
            self.persist()

    def search(self, vector: List[float], top_k: int = 5) -> List[Dict[str, Any]]:
        with self._lock:
            if not self.docs or self.matrix is None or len(self.matrix) == 0:
                return []
            q = np.asarray(vector, dtype=np.float32)
            norm = np.linalg.norm(q)
            if norm > 0:
                q = q / norm
            scores = self.matrix @ q
            k = min(int(top_k), len(scores))
            top = np.argpartition(-scores, k - 1)[:k]
            top = top[np.argsort(-scores[top])]
            return [{"id": self.docs[i]["id"], "text": self.docs[i]["text"],
                     "score": float(scores[i]), "metadata": self.docs[i].get("metadata", {})}
                    for i in top]

    def count(self) -> int:
        return len(self.docs)

    def clear(self):
        with self._lock:
            self.docs, self.matrix = [], None
            self.persist()


# ------------------------------------------------------------
# Milvus（pymilvus，支持 Milvus Lite 本地文件与服务模式）
# ------------------------------------------------------------
class MilvusVectorStore(BaseVectorStore):
    def __init__(self, name: str, uri: str = None, dim: int = None, token: str = ""):
        from pymilvus import MilvusClient  # 延迟导入，未安装时不影响本地模式

        self.name = name
        self.dim = int(dim or config.MILVUS_DIM)
        self.uri = uri or config.MILVUS_URI
        os.makedirs(os.path.dirname(self.uri) if os.path.splitext(self.uri)[1] else ".",
                    exist_ok=True)
        self.client = MilvusClient(uri=self.uri, token=token or config.MILVUS_TOKEN or None)
        self._ensure_collection()

    def _ensure_collection(self):
        if self.client.has_collection(self.name):
            return
        self.client.create_collection(collection_name=self.name, dimension=self.dim,
                                      metric_type="COSINE", auto_id=False,
                                      consistency_level="Strong")

    def upsert(self, docs: List[VectorDoc]):
        if not docs:
            return
        rows = [{"id": d["id"], "vector": list(d["vector"]),
                 "text": d["text"][:2000], **{"metadata": json.dumps(d.get("metadata", {}),
                                                                    ensure_ascii=False)}}
                for d in docs]
        self.client.upsert(collection_name=self.name, data=rows)

    def search(self, vector: List[float], top_k: int = 5) -> List[Dict[str, Any]]:
        res = self.client.search(collection_name=self.name, data=[list(vector)],
                                 limit=int(top_k),
                                 output_fields=["text", "metadata"])
        out = []
        for hits in res or []:
            for hit in hits:
                entity = hit.get("entity", {})
                meta = entity.get("metadata") or "{}"
                try:
                    meta = json.loads(meta)
                except Exception:
                    meta = {}
                out.append({"id": hit.get("id"), "text": entity.get("text", ""),
                            "score": float(hit.get("distance", 0.0)), "metadata": meta})
        return out

    def count(self) -> int:
        try:
            stat = self.client.get_collection_stats(self.name)
            return int(stat.get("row_count", 0))
        except Exception:
            return -1

    def clear(self):
        if self.client.has_collection(self.name):
            self.client.drop_collection(self.name)
        self._ensure_collection()


# ------------------------------------------------------------
# 工厂
# ------------------------------------------------------------
_STORE_CACHE: Dict[str, BaseVectorStore] = {}
_STORE_LOCK = threading.Lock()


def get_vector_store(name: str = None, backend: str = None) -> BaseVectorStore:
    """获取（带缓存的）向量库实例。"""
    name = name or config.MILVUS_COLLECTION
    backend = (backend or config.VECTOR_BACKEND or "auto").lower()
    key = f"{backend}::{name}"
    if key in _STORE_CACHE:
        return _STORE_CACHE[key]

    with _STORE_LOCK:
        if key in _STORE_CACHE:
            return _STORE_CACHE[key]
        store = None
        if backend in ("auto", "milvus"):
            try:
                store = MilvusVectorStore(name)
                logger.info("向量库后端：Milvus（%s）", config.MILVUS_URI)
            except Exception as exc:
                if backend == "milvus":
                    raise
                logger.warning("Milvus 不可用（%s），回退本地向量库", str(exc)[:120])
        if store is None:
            store = LocalVectorStore(name, config.LOCAL_VECTOR_DIR)
            logger.info("向量库后端：本地文件库（%s）", config.LOCAL_VECTOR_DIR)
        _STORE_CACHE[key] = store
        return store
