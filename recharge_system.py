from __future__ import annotations
import os
import secrets
from datetime import datetime, timezone
import account_system as accounts

# Format: credits:price, e.g. 10:5,50:20,100:35
DEFAULT_PACKAGES = [(10, 5), (50, 20), (100, 35)]

def _now():
    return datetime.now(timezone.utc).isoformat()

def _packages():
    raw = os.getenv("RECHARGE_PACKAGES", "")
    if not raw.strip():
        return DEFAULT_PACKAGES
    result = []
    for item in raw.split(","):
        try:
            credits, price = item.strip().split(":", 1)
            credits, price = int(credits), float(price)
            if credits > 0 and price >= 0:
                result.append((credits, price))
        except Exception:
            continue
    return result or DEFAULT_PACKAGES

def package_choices():
    return [f"{c} 次 — ¥{p:g}" for c, p in _packages()]

def package_from_choice(choice):
    choices = package_choices()
    try:
        idx = choices.index(choice)
        return _packages()[idx]
    except Exception:
        return None, None

def payment_instructions():
    return os.getenv("RECHARGE_PAYMENT_INSTRUCTIONS", "请先提交充值申请，管理员确认收款后会手动审核并增加额度。")

def init_db():
    with accounts._connect() as db:
        db.executescript('''
        CREATE TABLE IF NOT EXISTS recharge_orders(
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            order_no TEXT UNIQUE NOT NULL,
            user_id INTEGER NOT NULL,
            username TEXT NOT NULL,
            credits INTEGER NOT NULL,
            amount REAL NOT NULL,
            payment_note TEXT,
            status TEXT NOT NULL DEFAULT 'pending',
            admin_user_id INTEGER,
            admin_username TEXT,
            admin_note TEXT,
            created_at TEXT NOT NULL,
            reviewed_at TEXT
        );
        CREATE TABLE IF NOT EXISTS credit_transactions(
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            user_id INTEGER NOT NULL,
            order_id INTEGER,
            type TEXT NOT NULL,
            amount INTEGER NOT NULL,
            balance_after INTEGER NOT NULL,
            detail TEXT,
            created_at TEXT NOT NULL
        );
        CREATE INDEX IF NOT EXISTS idx_recharge_user ON recharge_orders(user_id, id DESC);
        CREATE INDEX IF NOT EXISTS idx_recharge_status ON recharge_orders(status, id DESC);
        CREATE INDEX IF NOT EXISTS idx_credit_tx_user ON credit_transactions(user_id, id DESC);
        ''')

def create_order(user_id, choice, payment_note=""):
    user = accounts.get_user(user_id)
    if not user or user.get("disabled"):
        return False, "账户不存在或已被禁用。", None
    credits, amount = package_from_choice(choice)
    if not credits:
        return False, "请选择有效的充值套餐。", None
    order_no = "R" + datetime.now(timezone.utc).strftime("%Y%m%d%H%M%S") + secrets.token_hex(3).upper()
    with accounts._connect() as db:
        db.execute("INSERT INTO recharge_orders(order_no,user_id,username,credits,amount,payment_note,status,created_at) VALUES(?,?,?,?,?,?,?,?)", (order_no, user_id, user["username"], credits, amount, (payment_note or "").strip()[:300], "pending", _now()))
    return True, f"充值申请已提交：{order_no}，等待管理员审核。", order_no

def user_orders(user_id, limit=50):
    with accounts._connect() as db:
        rows = db.execute("SELECT * FROM recharge_orders WHERE user_id=? ORDER BY id DESC LIMIT ?", (user_id, limit)).fetchall()
        return [dict(r) for r in rows]

def credit_history(user_id, limit=100):
    with accounts._connect() as db:
        rows = db.execute("SELECT * FROM credit_transactions WHERE user_id=? ORDER BY id DESC LIMIT ?", (user_id, limit)).fetchall()
        return [dict(r) for r in rows]

def admin_orders(status="all", limit=500):
    with accounts._connect() as db:
        if status == "pending":
            rows = db.execute("SELECT * FROM recharge_orders WHERE status='pending' ORDER BY id DESC LIMIT ?", (limit,)).fetchall()
        elif status in ("approved", "rejected"):
            rows = db.execute("SELECT * FROM recharge_orders WHERE status=? ORDER BY id DESC LIMIT ?", (status, limit)).fetchall()
        else:
            rows = db.execute("SELECT * FROM recharge_orders ORDER BY id DESC LIMIT ?", (limit,)).fetchall()
        return [dict(r) for r in rows]

def _audit_same_db(db, admin, action, target_user_id, detail):
    target = db.execute("SELECT username FROM users WHERE id=?", (target_user_id,)).fetchone()
    db.execute("INSERT INTO admin_audit(admin_user_id,admin_username,action,target_user_id,target_username,detail,created_at) VALUES(?,?,?,?,?,?,?)", (admin["id"], admin["username"], action, target_user_id, target["username"] if target else "", detail[:500], _now()))

def review_order(admin_user_id, order_id, approve, admin_note=""):
    admin = accounts.get_user(admin_user_id)
    if not admin or not admin.get("is_admin"):
        return False, "无管理员权限。"
    order_id = int(order_id)
    with accounts._connect() as db:
        db.execute("BEGIN IMMEDIATE")
        order = db.execute("SELECT * FROM recharge_orders WHERE id=?", (order_id,)).fetchone()
        if not order:
            return False, "充值订单不存在。"
        if order["status"] != "pending":
            return False, f"该订单已经处理，当前状态：{order['status']}。"
        now = _now(); note = (admin_note or "").strip()[:300]
        if not approve:
            db.execute("UPDATE recharge_orders SET status='rejected',admin_user_id=?,admin_username=?,admin_note=?,reviewed_at=? WHERE id=? AND status='pending'", (admin_user_id, admin["username"], note, now, order_id))
            _audit_same_db(db, admin, "拒绝充值", order["user_id"], f"订单 {order['order_no']}，{note or '未填写原因'}")
            return True, f"已拒绝订单 {order['order_no']}。"
        user = db.execute("SELECT * FROM users WHERE id=?", (order["user_id"],)).fetchone()
        if not user:
            return False, "充值用户不存在。"
        new_balance = int(user["credits"]) + int(order["credits"])
        updated = db.execute("UPDATE users SET credits=? WHERE id=?", (new_balance, order["user_id"]))
        if updated.rowcount != 1:
            return False, "增加额度失败。"
        db.execute("UPDATE recharge_orders SET status='approved',admin_user_id=?,admin_username=?,admin_note=?,reviewed_at=? WHERE id=? AND status='pending'", (admin_user_id, admin["username"], note, now, order_id))
        db.execute("INSERT INTO credit_transactions(user_id,order_id,type,amount,balance_after,detail,created_at) VALUES(?,?,?,?,?,?,?)", (order["user_id"], order_id, "recharge", int(order["credits"]), new_balance, f"充值订单 {order['order_no']}", now))
        _audit_same_db(db, admin, "通过充值", order["user_id"], f"订单 {order['order_no']}，增加 {order['credits']} 次，余额 {new_balance} 次")
        return True, f"已通过订单 {order['order_no']}，用户增加 {order['credits']} 次，当前余额 {new_balance} 次。"

def pending_count():
    with accounts._connect() as db:
        return int(db.execute("SELECT COUNT(*) FROM recharge_orders WHERE status='pending'").fetchone()[0])

init_db()
