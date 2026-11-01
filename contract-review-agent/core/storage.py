# ============================================================
# storage.py —— 文件存储抽象（MinIO 正式 + 本地磁盘默认）
#
# 统一接口：save / read / exists / delete / url
# 切换方式：config.STORAGE_BACKEND = auto | minio | local
# 说明：正式环境用 MinIO（满足非功能性需求“文件存储 MinIO”）；
#       本机无 MinIO 服务时自动落本地磁盘 data/storage，功能完全等价。
# ============================================================
import os
from typing import Optional

import config
from core.logging_setup import setup_logger

logger = setup_logger()


class BaseStorage:
    def save(self, key: str, data: bytes, content_type: str = "application/octet-stream") -> str:
        raise NotImplementedError

    def read(self, key: str) -> Optional[bytes]:
        raise NotImplementedError

    def exists(self, key: str) -> bool:
        raise NotImplementedError

    def delete(self, key: str) -> bool:
        raise NotImplementedError

    def url(self, key: str) -> str:
        raise NotImplementedError


class LocalStorage(BaseStorage):
    """本地磁盘存储（按 bucket 分子目录）。"""

    def __init__(self, root: str = None):
        self.root = root or config.LOCAL_STORAGE_DIR
        os.makedirs(self.root, exist_ok=True)

    def _path(self, key: str) -> str:
        safe = key.replace("\\", "/").lstrip("/")
        return os.path.join(self.root, *safe.split("/"))

    def save(self, key: str, data: bytes, content_type: str = "application/octet-stream") -> str:
        path = self._path(key)
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, "wb") as fh:
            fh.write(data)
        return path

    def read(self, key: str) -> Optional[bytes]:
        path = self._path(key)
        if not os.path.exists(path):
            return None
        with open(path, "rb") as fh:
            return fh.read()

    def exists(self, key: str) -> bool:
        return os.path.exists(self._path(key))

    def delete(self, key: str) -> bool:
        path = self._path(key)
        if os.path.exists(path):
            os.remove(path)
            return True
        return False

    def url(self, key: str) -> str:
        return "/files/" + key.replace("\\", "/").lstrip("/")


class MinioStorage(BaseStorage):
    """MinIO 对象存储（S3 协议）。"""

    def __init__(self):
        from minio import Minio

        self.client = Minio(config.MINIO_ENDPOINT, access_key=config.MINIO_ACCESS_KEY,
                            secret_key=config.MINIO_SECRET_KEY, secure=config.MINIO_SECURE,
                            region=config.MINIO_REGION)
        self.buckets = {config.MINIO_BUCKET_CONTRACT, config.MINIO_BUCKET_REPORT}
        for bucket in self.buckets:
            if not self.client.bucket_exists(bucket):
                self.client.make_bucket(bucket)

    @staticmethod
    def _split(key: str):
        bucket, _, rest = key.partition("/")
        return bucket, rest or key

    def save(self, key: str, data: bytes, content_type: str = "application/octet-stream") -> str:
        import io
        bucket, obj = self._split(key)
        self.client.put_object(bucket, obj, io.BytesIO(data), length=len(data),
                               content_type=content_type)
        return key

    def read(self, key: str) -> Optional[bytes]:
        bucket, obj = self._split(key)
        try:
            return self.client.get_object(bucket, obj).read()
        except Exception:
            return None

    def exists(self, key: str) -> bool:
        bucket, obj = self._split(key)
        try:
            self.client.stat_object(bucket, obj)
            return True
        except Exception:
            return False

    def delete(self, key: str) -> bool:
        bucket, obj = self._split(key)
        try:
            self.client.remove_object(bucket, obj)
            return True
        except Exception:
            return False

    def url(self, key: str) -> str:
        bucket, obj = self._split(key)
        try:
            return self.client.presigned_get_object(bucket, obj)
        except Exception:
            return key


_storage = None
_storage_lock = None


def get_storage(backend: str = None) -> BaseStorage:
    """获取存储实例（进程内单例，auto 模式自动探测 MinIO）。"""
    global _storage, _storage_lock
    import threading

    if _storage_lock is None:
        _storage_lock = threading.Lock()
    if _storage is not None:
        return _storage
    with _storage_lock:
        if _storage is not None:
            return _storage
        want = (backend or config.STORAGE_BACKEND or "auto").lower()
        if want in ("auto", "minio"):
            try:
                _storage = MinioStorage()
                logger.info("文件存储后端：MinIO（%s）", config.MINIO_ENDPOINT)
                return _storage
            except Exception as exc:
                if want == "minio":
                    raise
                logger.warning("MinIO 不可用（%s），回退本地磁盘存储", str(exc)[:120])
        _storage = LocalStorage()
        logger.info("文件存储后端：本地磁盘（%s）", config.LOCAL_STORAGE_DIR)
        return _storage
