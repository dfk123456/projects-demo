# ============================================================
# auth.py —— 用户注册 / 登录 / 会话凭证
#
# 设计原则：
#   1) 零第三方依赖：SQLite(sqlite3) 存用户，PBKDF2-HMAC 做密码散列；
#   2) 不存明文密码：每个用户独立随机盐 + 10 万次迭代；
#   3) 登录凭证为随机 token，服务端记录并用过期时间清理；
#   4) 仅面向内网/本机使用，不处理找回密码、邮箱验证等流程。
# ============================================================
import hashlib
import hmac
import os
import re
import secrets
import sqlite3
import threading
import time

import config

_LOCK = threading.Lock()
_USERNAME_RE = re.compile(r"^[\w一-龥]{3,20}$")   # 字母数字下划线或中文，3~20 位
_ITERATIONS = 100_000


# ------------------------------------------------------------
# 数据库连接
# ------------------------------------------------------------
def _connect():
    con = sqlite3.connect(config.USERS_DB_FILE, timeout=10)
    con.row_factory = sqlite3.Row
    return con


def init_db():
    """建表（幂等）；服务启动时调用一次。"""
    with _LOCK, _connect() as con:
        con.execute(
            "CREATE TABLE IF NOT EXISTS users ("
            "  id INTEGER PRIMARY KEY AUTOINCREMENT,"
            "  username TEXT UNIQUE NOT NULL,"
            "  password_hash TEXT NOT NULL,"
            "  salt TEXT NOT NULL,"
            "  nickname TEXT DEFAULT '',"
            "  created_at TEXT NOT NULL)")
        con.execute(
            "CREATE TABLE IF NOT EXISTS tokens ("
            "  token TEXT PRIMARY KEY,"
            "  username TEXT NOT NULL,"
            "  created_at REAL NOT NULL)")
        con.execute("CREATE INDEX IF NOT EXISTS idx_tokens_user ON tokens(username)")


# ------------------------------------------------------------
# 密码散列
# ------------------------------------------------------------
def _hash_password(password: str, salt: str) -> str:
    return hashlib.pbkdf2_hmac(
        "sha256", password.encode("utf-8"), salt.encode("utf-8"), _ITERATIONS).hex()


def _safe_equal(a: str, b: str) -> bool:
    return hmac.compare_digest(a, b)


# ------------------------------------------------------------
# 注册 / 登录 / 校验
# ------------------------------------------------------------
def register(username: str, password: str, nickname: str = "") -> dict:
    """注册新用户。失败时抛出 ValueError（错误信息可直接展示给用户）。"""
    username = (username or "").strip()
    password = password or ""
    nickname = (nickname or "").strip() or username

    if not _USERNAME_RE.match(username):
        raise ValueError("用户名需为 3~20 位字母、数字、下划线或中文。")
    if len(password) < 6:
        raise ValueError("密码至少 6 位。")
    if len(password) > 64:
        raise ValueError("密码过长（最多 64 位）。")

    salt = secrets.token_hex(16)
    pwd_hash = _hash_password(password, salt)
    created = time.strftime("%Y-%m-%d %H:%M:%S")

    try:
        with _LOCK, _connect() as con:
            con.execute(
                "INSERT INTO users(username, password_hash, salt, nickname, created_at)"
                " VALUES(?,?,?,?,?)",
                (username, pwd_hash, salt, nickname, created))
    except sqlite3.IntegrityError:
        raise ValueError("该用户名已被注册，请换一个。")

    return {"username": username, "nickname": nickname}


def login(username: str, password: str) -> dict:
    """校验账号密码，成功则返回登录凭证 token。失败抛出 ValueError。"""
    username = (username or "").strip()
    with _LOCK, _connect() as con:
        row = con.execute(
            "SELECT username, password_hash, salt, nickname FROM users WHERE username=?",
            (username,)).fetchone()

    if row is None or not _safe_equal(row["password_hash"],
                                      _hash_password(password or "", row["salt"])):
        raise ValueError("用户名或密码不正确。")

    token = secrets.token_urlsafe(32)
    with _LOCK, _connect() as con:
        con.execute("INSERT INTO tokens(token, username, created_at) VALUES(?,?,?)",
                    (token, username, time.time()))
        _purge_expired(con)
    return {"token": token, "username": row["username"],
            "nickname": row["nickname"] or username}


def verify_token(token: str):
    """校验登录凭证：有效返回 username，无效返回 None。"""
    if not token:
        return None
    with _LOCK, _connect() as con:
        row = con.execute(
            "SELECT username, created_at FROM tokens WHERE token=?", (token,)).fetchone()
        if row is None:
            return None
        if time.time() - row["created_at"] > config.TOKEN_TTL_SEC:
            con.execute("DELETE FROM tokens WHERE token=?", (token,))
            return None
    return row["username"]


def logout(token: str) -> bool:
    """注销登录凭证。"""
    if not token:
        return False
    with _LOCK, _connect() as con:
        cur = con.execute("DELETE FROM tokens WHERE token=?", (token,))
        return cur.rowcount > 0


def user_count() -> int:
    """已注册用户数（用于判断是否需要引导注册）。"""
    try:
        with _LOCK, _connect() as con:
            return con.execute("SELECT COUNT(*) AS c FROM users").fetchone()["c"]
    except Exception:
        return 0


def _purge_expired(con):
    """清理过期凭证（内部调用，需在锁内）。"""
    deadline = time.time() - config.TOKEN_TTL_SEC
    con.execute("DELETE FROM tokens WHERE created_at < ?", (deadline,))
