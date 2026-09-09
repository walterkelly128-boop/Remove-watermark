from __future__ import annotations

import hashlib
import hmac
import os
import secrets
import sqlite3
from datetime import datetime, timezone
from pathlib import Path

DB_PATH = Path(os.getenv("ACCOUNT_DB_PATH", "/app/data/accounts.db"))
ADMIN_USERNAME = os.getenv("ADMIN_USERNAME", "admin")
ADMIN_PASSWORD = os.getenv("ADMIN_PASSWORD", "")
FREE_CREDITS = int(os.getenv("FREE_CREDITS", "5"))


def now():
    return datetime.now(timezone.utc).isoformat()


def _conn():
    DB_PATH.parent.mkdir(parents=True, exist_ok=True)
    c = sqlite3.connect(DB_PATH)
    c.row_factory = sqlite3.Row
    c.execute("PRAGMA foreign_keys=ON")
    return c


def _hash(password, salt=None):
    salt = salt or secrets.token_hex(16)
    digest = hashlib.pbkdf2_hmac("sha256", password.encode(), salt.encode(), 240000).hex()
    return f"pbkdf2$240000${salt}${digest}"


def _verify(password, encoded):
    try:
        _, rounds, salt, digest = encoded.split("$", 3)
        test = hashlib.pbkdf2_hmac("sha256", password.encode(), salt.encode(), int(rounds)).hex()
        return hmac.compare_digest(test, digest)
    except Exception:
        return False


def init_db():
    with _conn() as c:
        c.executescript("""
        CREATE TABLE IF NOT EXISTS users (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            username TEXT UNIQUE NOT NULL,
            password_hash TEXT NOT NULL,
            credits INTEGER NOT NULL DEFAULT 0,
            disabled INTEGER NOT NULL DEFAULT 0,
            is_admin INTEGER NOT NULL DEFAULT 0,
            created_at TEXT NOT NULL,
            last_login TEXT
        );
        CREATE TABLE IF NOT EXISTS usage_logs (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            user_id INTEGER NOT NULL,
            provider TEXT NOT NULL,
            credits INTEGER NOT NULL,
            success INTEGER NOT NULL,
            detail TEXT,
            created_at TEXT NOT NULL,
            FOREIGN KEY(user_id) REFERENCES users(id) ON DELETE CASCADE
        );
        CREATE INDEX IF NOT EXISTS idx_users_username ON users(username);
        CREATE INDEX IF NOT EXISTS idx_usage_user ON usage_logs(user_id, created_at DESC);
        CREATE INDEX IF NOT EXISTS idx_usage_provider ON usage_logs(provider);
        """)
        if ADMIN_PASSWORD:
            row = c.execute("SELECT id FROM users WHERE username=?", (ADMIN_USERNAME,)).fetchone()
            if not row:
                c.execute("INSERT INTO users(username,password_hash,credits,disabled,is_admin,created_at) VALUES(?,?,?,?,?,?)",
                          (ADMIN_USERNAME, _hash(ADMIN_PASSWORD), 0, 0, 1, now()))


def register(username, password):
    username = (username or "").strip()
    if len(username) < 3 or len(username) > 40 or len(password or "") < 6:
        return None, "用户名至少 3 位，密码至少 6 位"
    try:
        with _conn() as c:
            cur = c.execute("INSERT INTO users(username,password_hash,credits,created_at) VALUES(?,?,?,?)",
                            (username, _hash(password), FREE_CREDITS, now()))
            return cur.lastrowid, "注册成功"
    except sqlite3.IntegrityError:
        return None, "用户名已存在"


def login(username, password):
    with _conn() as c:
        row = c.execute("SELECT * FROM users WHERE username=?", ((username or "").strip(),)).fetchone()
        if not row or row["disabled"] or not _verify(password or "", row["password_hash"]):
            return None, "用户名或密码错误，或账号已被禁用"
        c.execute("UPDATE users SET last_login=? WHERE id=?", (now(), row["id"]))
        return dict(row), "登录成功"


def get_user(user_id):
    with _conn() as c:
        row = c.execute("SELECT id,username,credits,disabled,is_admin,created_at,last_login FROM users WHERE id=?", (user_id,)).fetchone()
        return dict(row) if row else None


def consume(user_id, provider, cost, success=True, detail=""):
    cost = int(cost)
    with _conn() as c:
        c.execute("BEGIN IMMEDIATE")
        row = c.execute("SELECT credits,disabled FROM users WHERE id=?", (user_id,)).fetchone()
        if not row or row["disabled"]:
            raise ValueError("账号不可用")
        if cost > 0 and row["credits"] < cost:
            raise ValueError("额度不足")
        if success and cost > 0:
            c.execute("UPDATE users SET credits=credits-? WHERE id=?", (cost, user_id))
        c.execute("INSERT INTO usage_logs(user_id,provider,credits,success,detail,created_at) VALUES(?,?,?,?,?,?)",
                  (user_id, provider, cost if success else 0, int(success), detail, now()))
        return c.execute("SELECT credits FROM users WHERE id=?", (user_id,)).fetchone()[0]


def add_credits(user_id, amount, detail="admin adjustment"):
    amount = int(amount)
    with _conn() as c:
        c.execute("UPDATE users SET credits=MAX(0,credits+?) WHERE id=?", (amount, user_id))
        row = c.execute("SELECT credits FROM users WHERE id=?", (user_id,)).fetchone()
        return row[0] if row else None


def set_disabled(user_id, disabled):
    with _conn() as c:
        c.execute("UPDATE users SET disabled=? WHERE id=?", (int(bool(disabled)), user_id))


def search_users(keyword=""):
    keyword = f"%{(keyword or '').strip()}%"
    with _conn() as c:
        return [dict(r) for r in c.execute("SELECT id,username,credits,disabled,is_admin,created_at,last_login FROM users WHERE username LIKE ? ORDER BY id DESC", (keyword,)).fetchall()]


def user_usage(user_id, limit=200):
    with _conn() as c:
        return [dict(r) for r in c.execute("SELECT id,provider,credits,success,detail,created_at FROM usage_logs WHERE user_id=? ORDER BY id DESC LIMIT ?", (user_id, limit)).fetchall()]


def stats():
    with _conn() as c:
        users = c.execute("SELECT COUNT(*) FROM users").fetchone()[0]
        active = c.execute("SELECT COUNT(*) FROM users WHERE disabled=0").fetchone()[0]
        credits = c.execute("SELECT COALESCE(SUM(credits),0) FROM users").fetchone()[0]
        total = c.execute("SELECT COUNT(*) FROM usage_logs").fetchone()[0]
        success = c.execute("SELECT COUNT(*) FROM usage_logs WHERE success=1").fetchone()[0]
        gemini = c.execute("SELECT COALESCE(SUM(credits),0) FROM usage_logs WHERE provider='gemini' AND success=1").fetchone()[0]
        openai = c.execute("SELECT COALESCE(SUM(credits),0) FROM usage_logs WHERE provider='openai' AND success=1").fetchone()[0]
        local = c.execute("SELECT COUNT(*) FROM usage_logs WHERE provider='local' AND success=1").fetchone()[0]
        return {"users": users, "active": active, "credits": credits, "total": total, "success": success, "gemini": gemini, "openai": openai, "local": local}


init_db()
