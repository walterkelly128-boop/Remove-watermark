from __future__ import annotations
import hashlib, hmac, os, secrets, sqlite3
from datetime import datetime, timezone
from pathlib import Path
DB_PATH=Path(os.getenv("ACCOUNT_DB_PATH","/app/data/accounts.db")); FREE_CREDITS=int(os.getenv("FREE_CREDITS","5")); ADMIN_USERNAME=os.getenv("ADMIN_USERNAME","admin"); ADMIN_PASSWORD=os.getenv("ADMIN_PASSWORD","")
def _now(): return datetime.now(timezone.utc).isoformat()
def _connect():
 DB_PATH.parent.mkdir(parents=True,exist_ok=True); c=sqlite3.connect(DB_PATH,timeout=30); c.row_factory=sqlite3.Row; return c
def _hash_password(password):
 salt=secrets.token_bytes(16); digest=hashlib.pbkdf2_hmac("sha256",password.encode(),salt,240000); return f"pbkdf2$240000${salt.hex()}${digest.hex()}"
def _verify_password(password,stored):
 try:
  _,rounds,salt,digest=stored.split("$",3); test=hashlib.pbkdf2_hmac("sha256",password.encode(),bytes.fromhex(salt),int(rounds)); return hmac.compare_digest(test.hex(),digest)
 except Exception:return False
def init_db():
 with _connect() as db:
  db.executescript('''CREATE TABLE IF NOT EXISTS users(id INTEGER PRIMARY KEY AUTOINCREMENT,username TEXT UNIQUE NOT NULL,password_hash TEXT NOT NULL,credits INTEGER NOT NULL DEFAULT 0,disabled INTEGER NOT NULL DEFAULT 0,is_admin INTEGER NOT NULL DEFAULT 0,created_at TEXT NOT NULL,last_login TEXT); CREATE TABLE IF NOT EXISTS usage_logs(id INTEGER PRIMARY KEY AUTOINCREMENT,user_id INTEGER NOT NULL,username TEXT NOT NULL,provider TEXT NOT NULL,credits INTEGER NOT NULL,success INTEGER NOT NULL,detail TEXT,created_at TEXT NOT NULL,FOREIGN KEY(user_id) REFERENCES users(id)); CREATE INDEX IF NOT EXISTS idx_users_username ON users(username); CREATE INDEX IF NOT EXISTS idx_usage_user ON usage_logs(user_id,created_at DESC);''')
  if ADMIN_PASSWORD and not db.execute("SELECT id FROM users WHERE username=?",(ADMIN_USERNAME,)).fetchone(): db.execute("INSERT INTO users(username,password_hash,is_admin,created_at) VALUES(?,?,1,?)",(ADMIN_USERNAME,_hash_password(ADMIN_PASSWORD),_now()))
def register(username,password):
 username=(username or '').strip()
 if len(username)<3 or len(username)>32:return False,"用户名需要 3-32 个字符。"
 if len(password or '')<6:return False,"密码至少 6 位。"
 try:
  with _connect() as db: db.execute("INSERT INTO users(username,password_hash,credits,created_at) VALUES(?,?,?,?)",(username,_hash_password(password),FREE_CREDITS,_now()))
  return True,f"注册成功，已赠送 {FREE_CREDITS} 次额度。"
 except sqlite3.IntegrityError:return False,"用户名已存在。"
def login(username,password):
 with _connect() as db:
  row=db.execute("SELECT * FROM users WHERE username=?",((username or '').strip(),)).fetchone()
  if not row or row['disabled'] or not _verify_password(password or '',row['password_hash']):return None,"用户名或密码错误，或账号已被禁用。"
  db.execute("UPDATE users SET last_login=? WHERE id=?",(_now(),row['id'])); return dict(row),"登录成功。"
def get_user(user_id):
 with _connect() as db:
  r=db.execute("SELECT * FROM users WHERE id=?",(user_id,)).fetchone(); return dict(r) if r else None
def consume_credit(user_id,provider,cost=1):
 with _connect() as db:
  db.execute("BEGIN IMMEDIATE"); r=db.execute("SELECT * FROM users WHERE id=?",(user_id,)).fetchone()
  if not r:return False,"账户不存在。"
  if r['disabled']:return False,"账号已被禁用。"
  if r['is_admin']:return True,"管理员账户不扣额度。"
  if r['credits']<cost:return False,"额度不足，请联系管理员。"
  db.execute("UPDATE users SET credits=credits-? WHERE id=?",(cost,user_id)); return True,"额度已预扣。"
def refund_credit(user_id,provider,cost=1):
 with _connect() as db: db.execute("UPDATE users SET credits=credits+? WHERE id=?",(cost,user_id))
def log_usage(user_id,provider,cost,success,detail=''):
 user=get_user(user_id)
 if not user:return
 with _connect() as db: db.execute("INSERT INTO usage_logs(user_id,username,provider,credits,success,detail,created_at) VALUES(?,?,?,?,?,?,?)",(user_id,user['username'],provider,cost,int(success),detail[:500],_now()))
def usage_for_user(user_id,limit=200):
 with _connect() as db:return [dict(r) for r in db.execute("SELECT * FROM usage_logs WHERE user_id=? ORDER BY id DESC LIMIT ?",(user_id,limit)).fetchall()]
def admin_users(keyword=''):
 with _connect() as db:
  q=(keyword or '').strip(); rows=db.execute("SELECT id,username,credits,disabled,is_admin,created_at,last_login FROM users WHERE username LIKE ? ORDER BY id DESC",(f'%{q}%',)).fetchall(); return [dict(r) for r in rows]
def admin_usage(limit=500):
 with _connect() as db:return [dict(r) for r in db.execute("SELECT * FROM usage_logs ORDER BY id DESC LIMIT ?",(limit,)).fetchall()]
def add_credits(user_id,amount):
 with _connect() as db: db.execute("UPDATE users SET credits=MAX(0,credits+?) WHERE id=?",(int(amount),user_id))
def set_credits(user_id,amount):
 with _connect() as db: db.execute("UPDATE users SET credits=MAX(0,?) WHERE id=?",(int(amount),user_id))
def set_disabled(user_id,disabled):
 with _connect() as db: db.execute("UPDATE users SET disabled=? WHERE id=? AND is_admin=0",(int(bool(disabled)),user_id))
def stats():
 with _connect() as db:
  return {'users':db.execute("SELECT COUNT(*) FROM users WHERE is_admin=0").fetchone()[0],'active':db.execute("SELECT COUNT(*) FROM users WHERE is_admin=0 AND disabled=0").fetchone()[0],'credits':db.execute("SELECT COALESCE(SUM(credits),0) FROM users WHERE is_admin=0").fetchone()[0],'total_usage':db.execute("SELECT COUNT(*) FROM usage_logs").fetchone()[0],'success':db.execute("SELECT COUNT(*) FROM usage_logs WHERE success=1").fetchone()[0],'gemini':db.execute("SELECT COALESCE(SUM(credits),0) FROM usage_logs WHERE provider='gemini' AND success=1").fetchone()[0],'openai':db.execute("SELECT COALESCE(SUM(credits),0) FROM usage_logs WHERE provider='openai' AND success=1").fetchone()[0],'local':db.execute("SELECT COUNT(*) FROM usage_logs WHERE provider='local' AND success=1").fetchone()[0]}
init_db()
