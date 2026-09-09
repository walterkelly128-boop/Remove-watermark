from __future__ import annotations

import hashlib
import hmac
import os
import secrets
import sqlite3
from datetime import datetime, timezone
from pathlib import Path

DB_PATH = Path(os.getenv("ACCOUNT_DB_PATH", "/app/data/accounts.db"))
FREE_CREDITS = int(os.getenv("FREE_CREDITS", "5"))
ADMIN_USERNAME = os.getenv("ADMIN_USERNAME", "admin")
ADMIN_PASSWORD = os.getenv("ADMIN_PASSWORD", "change-me")


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _connect():
    DB_PATH.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(DB_PATH, timeout=30)
    conn.row_factory = sqlite3.Row
    return conn


def init_db():
    with _connect() as db:
        db.executescript(
            """
            CREATE TABLE IF NOT EXISTS users (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                username TEXT UNIQUE NOT NULL,
                password_hash TEXT NOT NULL,
                credits INTEGER NOT NULL DEFAULT 0,
                is_admin INTEGER NOT NULL DEFAULT 0,
                created_at TEXT NOT NULL,
                last_login TEXT
            );
            CREATE TABLE IF NOT EXISTS usage_logs (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                user_id INTEGER NOT NULL,
                username TEXT NOT NULL,
                provider TEXT NOT NULL,
                credits INTEGER NOT NULL,
                success INTEGER NOT NULL,
                detail TEXT,
                created_at TEXT NOT NULL,
                FOREIGN KEY(user_id) REFERENCES users(id)
            );
            """
        )
        row = db.execute("SELECT id FROM users WHERE username=?", (ADMIN_USERNAME,)).fetchone()
        if not row:
            db.execute(
                "INSERT INTO users(username,password_hash,credits,is_admin,created_at) VALUES(?,?,?,?,?)",
                (ADMIN_USERNAME, _hash_password(ADMIN_PASSWORD), 0, 1, _now()),
            )


def _hash_password(password: str) -> str:
    salt = secrets.token_bytes(16)
    digest = hashlib.pbkdf2_hmac("sha256", password.encode(), salt, 240000)
    return f"pbkdf2$240000${salt.hex()}${digest.hex()}"


def _verify_password(password: str, stored: str) -> bool:
    try:
        _, rounds, salt_hex, digest_hex = stored.split("$", 3)
        digest = hashlib.pbkdf2_hmac("sha256", password.encode(), bytes.fromhex(salt_hex), int(rounds))
        return hmac.compare_digest(digest.hex(), digest_hex)
    except Exception:
        return False


def register(username: str, password: str):
    username = (username or "").strip()
    if len(username) < 3 or len(username) > 32:
        return False, "用户名需要 3-32 个字符。"
    if len(password or "") < 6:
        return False, "密码至少 6 位。"
    try:
        with _connect() as db:
            db.execute(
                "INSERT INTO users(username,password_hash,credits,is_admin,created_at) VALUES(?,?,?,?,?)",
                (username, _hash_password(password), FREE_CREDITS, 0, _now()),
            )
        return True, f"注册成功，已赠送 {FREE_CREDITS} 次额度。"
    except sqlite3.IntegrityError:
        return False, "用户名已存在。"


def login(username: str, password: str):
    with _connect() as db:
        row = db.execute("SELECT * FROM users WHERE username=?", ((username or "").strip(),)).fetchone()
        if not row or not _verify_password(password or "", row["password_hash"]):
            return None, "用户名或密码错误。"
        db.execute("UPDATE users SET last_login=? WHERE id=?", (_now(), row["id"]))
        return dict(row), "登录成功。"


def get_user(user_id: int):
    with _connect() as db:
        row = db.execute("SELECT * FROM users WHERE id=?", (user_id,)).fetchone()
        return dict(row) if row else None


def consume_credit(user_id: int, provider: str, cost: int = 1):
    with _connect() as db:
        db.execute("BEGIN IMMEDIATE")
        row = db.execute("SELECT * FROM users WHERE id=?", (user_id,)).fetchone()
        if not row:
            return False, "账户不存在。"
        if row["is_admin"]:
            return True, "管理员账户不扣额度。"
        if row["credits"] < cost:
            return False, "额度不足，请联系管理员充值。"
        db.execute("UPDATE users SET credits=credits-? WHERE id=?", (cost, user_id))
        return True, "额度已预扣。"


def refund_credit(user_id: int, provider: str, cost: int = 1):
    with _connect() as db:
        db.execute("UPDATE users SET credits=credits+? WHERE id=?", (cost, user_id))


def log_usage(user_id: int, provider: str, cost: int, success: bool, detail: str = ""):
    user = get_user(user_id)
    if not user:
        return
    with _connect() as db:
        db.execute(
            "INSERT INTO usage_logs(user_id,username,provider,credits,success,detail,created_at) VALUES(?,?,?,?,?,?,?)",
            (user_id, user["username"], provider, cost, int(success), detail[:500], _now()),
        )


def usage_for_user(user_id: int, limit: int = 100):
    with _connect() as db:
        return [dict(r) for r in db.execute("SELECT * FROM usage_logs WHERE user_id=? ORDER BY id DESC LIMIT ?", (user_id, limit)).fetchall()]


def admin_users():
    with _connect() as db:
        return [dict(r) for r in db.execute("SELECT id,username,credits,is_admin,created_at,last_login FROM users ORDER BY id DESC").fetchall()]


def admin_usage(limit: int = 500):
    with _connect() as db:
        return [dict(r) for r in db.execute("SELECT * FROM usage_logs ORDER BY id DESC LIMIT ?", (limit,)).fetchall()]


def add_credits(user_id: int, amount: int):
    with _connect() as db:
        db.execute("UPDATE users SET credits=credits+? WHERE id=?", (int(amount), user_id))


def set_credits(user_id: int, amount: int):
    with _connect() as db:
        db.execute("UPDATE users SET credits=? WHERE id=?", (max(0, int(amount)), user_id))


init_db()
