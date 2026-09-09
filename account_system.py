from __future__ import annotations
import hashlib, hmac, os, secrets, sqlite3
from datetime import datetime, timezone, timedelta
from pathlib import Path
DB_PATH=Path(os.getenv("ACCOUNT_DB_PATH","/app/data/accounts.db")); FREE_CREDITS=int(os.getenv("FREE_CREDITS","5")); GUEST_CREDITS=int(os.getenv("GUEST_CREDITS","5")); ADMIN_USERNAME=os.getenv("ADMIN_USERNAME","admin"); ADMIN_PASSWORD=os.getenv("ADMIN_PASSWORD",""); GUEST_IP_SALT=os.getenv("GUEST_IP_SALT","") or secrets.token_hex(32)
PBKDF2_ROUNDS=600000; LOGIN_MAX_FAILURES=5; LOGIN_LOCK_MINUTES=15

def _now(): return datetime.now(timezone.utc).isoformat()
def _connect():
 DB_PATH.parent.mkdir(parents=True,exist_ok=True); c=sqlite3.connect(DB_PATH,timeout=30); c.row_factory=sqlite3.Row; return c
def _hash_password(password):
 salt=secrets.token_bytes(16); digest=hashlib.pbkdf2_hmac("sha256",password.encode("utf-8"),salt,PBKDF2_ROUNDS); return f"pbkdf2${PBKDF2_ROUNDS}${salt.hex()}${digest.hex()}"
def _verify_password(password,stored):
 try:
  _,rounds,salt,digest=stored.split("$",3); test=hashlib.pbkdf2_hmac("sha256",password.encode("utf-8"),bytes.fromhex(salt),int(rounds)); return hmac.compare_digest(test.hex(),digest)
 except Exception:return False
def _guest_key(ip): return hmac.new(GUEST_IP_SALT.encode("utf-8"),ip.encode("utf-8"),hashlib.sha256).hexdigest()
def _migrate_columns(db):
 cols={r[1] for r in db.execute("PRAGMA table_info(users)").fetchall()}
 for name,definition in (("disabled","INTEGER NOT NULL DEFAULT 0"),("is_admin","INTEGER NOT NULL DEFAULT 0"),("failed_logins","INTEGER NOT NULL DEFAULT 0"),("locked_until","TEXT")):
  if name not in cols: db.execute(f"ALTER TABLE users ADD COLUMN {name} {definition}")
def init_db():
 with _connect() as db:
  db.executescript('''CREATE TABLE IF NOT EXISTS users(id INTEGER PRIMARY KEY AUTOINCREMENT,username TEXT UNIQUE NOT NULL,password_hash TEXT NOT NULL,credits INTEGER NOT NULL DEFAULT 0,disabled INTEGER NOT NULL DEFAULT 0,is_admin INTEGER NOT NULL DEFAULT 0,created_at TEXT NOT NULL,last_login TEXT,failed_logins INTEGER NOT NULL DEFAULT 0,locked_until TEXT); CREATE TABLE IF NOT EXISTS usage_logs(id INTEGER PRIMARY KEY AUTOINCREMENT,user_id INTEGER NOT NULL,username TEXT NOT NULL,provider TEXT NOT NULL,credits INTEGER NOT NULL,success INTEGER NOT NULL,detail TEXT,created_at TEXT NOT NULL,FOREIGN KEY(user_id) REFERENCES users(id)); CREATE TABLE IF NOT EXISTS admin_audit(id INTEGER PRIMARY KEY AUTOINCREMENT,admin_user_id INTEGER NOT NULL,admin_username TEXT NOT NULL,action TEXT NOT NULL,target_user_id INTEGER,target_username TEXT,detail TEXT,created_at TEXT NOT NULL); CREATE TABLE IF NOT EXISTS guest_usage(id INTEGER PRIMARY KEY AUTOINCREMENT,guest_key TEXT UNIQUE NOT NULL,credits INTEGER NOT NULL,created_at TEXT NOT NULL,last_used_at TEXT NOT NULL,total_used INTEGER NOT NULL DEFAULT 0); CREATE INDEX IF NOT EXISTS idx_users_username ON users(username); CREATE INDEX IF NOT EXISTS idx_usage_user ON usage_logs(user_id,created_at DESC); CREATE INDEX IF NOT EXISTS idx_audit_created ON admin_audit(created_at DESC); CREATE INDEX IF NOT EXISTS idx_guest_last_used ON guest_usage(last_used_at DESC);''')
  _migrate_columns(db)
  if ADMIN_PASSWORD and not db.execute("SELECT id FROM users WHERE username=?",(ADMIN_USERNAME,)).fetchone(): db.execute("INSERT INTO users(username,password_hash,is_admin,created_at) VALUES(?,?,1,?)",(ADMIN_USERNAME,_hash_password(ADMIN_PASSWORD)))
def register(username,password):
 username=(username or '').strip()
 if len(username)<3 or len(username)>32:return False,"用户名需要 3-32 个字符。"
 if len(password or '')<8:return False,"密码至少 8 位。"
 try:
  with _connect() as db: db.execute("INSERT INTO users(username,password_hash,credits,created_at) VALUES(?,?,?,?)",(username,_hash_password(password),FREE_CREDITS,_now()))
  return True,f"注册成功，已赠送 {FREE_CREDITS} 次额度。"
 except sqlite3.IntegrityError:return False,"用户名已存在。"
def login(username,password):
 username=(username or '').strip()
 with _connect() as db:
  row=db.execute("SELECT * FROM users WHERE username=?",(username,)).fetchone()
  if not row:return None,"用户名或密码错误。"
  if row['disabled']:return None,"账号已被禁用。"
  locked=row['locked_until']
  if locked:
   try:
    if datetime.fromisoformat(locked)>datetime.now(timezone.utc): return None,"登录失败次数过多，请稍后再试。"
   except Exception: pass
   db.execute("UPDATE users SET failed_logins=0,locked_until=NULL WHERE id=?",(row['id'],)); row=db.execute("SELECT * FROM users WHERE id=?",(row['id'],)).fetchone()
  if not _verify_password(password or '',row['password_hash']):
   failures=int(row['failed_logins'] or 0)+1
   if failures>=LOGIN_MAX_FAILURES:
    until=(datetime.now(timezone.utc)+timedelta(minutes=LOGIN_LOCK_MINUTES)).isoformat(); db.execute("UPDATE users SET failed_logins=?,locked_until=? WHERE id=?",(failures,until,row['id']))
    return None,"登录失败次数过多，账号已临时锁定 15 分钟。"
   db.execute("UPDATE users SET failed_logins=? WHERE id=?",(failures,row['id'])); return None,"用户名或密码错误。"
  db.execute("UPDATE users SET last_login=?,failed_logins=0,locked_until=NULL WHERE id=?",(_now(),row['id'])); row=dict(row); row['failed_logins']=0; row['locked_until']=None; return row,"登录成功。"
def get_user(user_id):
 with _connect() as db:
  r=db.execute("SELECT * FROM users WHERE id=?",(user_id,)).fetchone(); return dict(r) if r else None
def change_password(user_id,current_password,new_password):
 if len(new_password or '')<8:return False,"新密码至少 8 位。"
 with _connect() as db:
  r=db.execute("SELECT * FROM users WHERE id=?",(user_id,)).fetchone()
  if not r:return False,"账户不存在。"
  if not _verify_password(current_password or '',r['password_hash']):return False,"当前密码错误。"
  if hmac.compare_digest(current_password or '',new_password or ''):return False,"新密码不能与旧密码相同。"
  db.execute("UPDATE users SET password_hash=?,failed_logins=0,locked_until=NULL WHERE id=?",(_hash_password(new_password),user_id)); return True,"密码修改成功，请重新登录。"
def consume_credit(user_id,provider,cost=1):
 with _connect() as db:
  db.execute("BEGIN IMMEDIATE"); r=db.execute("SELECT * FROM users WHERE id=?",(user_id,)).fetchone()
  if not r:return False,"账户不存在。"
  if r['disabled']:return False,"账号已被禁用。"
  if r['is_admin']:return True,"管理员账户不扣额度。"
  if r['credits']<cost:return False,"额度不足，请充值后继续使用。"
  db.execute("UPDATE users SET credits=credits-? WHERE id=?",(cost,user_id)); return True,"额度已预扣。"
def consume_guest(ip,provider,cost=1):
 if not ip:return False,"无法识别访客网络地址，请登录后继续。"
 if cost<=0:return True,"免费操作。"
 key=_guest_key(ip); now=_now()
 with _connect() as db:
  db.execute("BEGIN IMMEDIATE")
  row=db.execute("SELECT * FROM guest_usage WHERE guest_key=?",(key,)).fetchone()
  if not row:
   remaining=GUEST_CREDITS-cost
   if remaining<0:return False,f"游客免费额度为 {GUEST_CREDITS} 次，请注册或登录后继续。"
   db.execute("INSERT INTO guest_usage(guest_key,credits,created_at,last_used_at,total_used) VALUES(?,?,?,?,?)",(key,remaining,now,now,cost)); return True,"游客额度已预扣。"
  if row['credits']<cost:return False,f"游客免费额度已用完（{GUEST_CREDITS} 次），请注册或登录后充值继续。"
  db.execute("UPDATE guest_usage SET credits=credits-?,last_used_at=?,total_used=total_used+? WHERE guest_key=?",(cost,now,cost,key)); return True,"游客额度已预扣。"
def refund_guest(ip,cost=1):
 if ip and cost>0:
  with _connect() as db: db.execute("UPDATE guest_usage SET credits=MIN(?,credits+?) WHERE guest_key=?",(GUEST_CREDITS,cost,_guest_key(ip)))
def guest_remaining(ip):
 if not ip:return GUEST_CREDITS
 with _connect() as db:
  r=db.execute("SELECT credits FROM guest_usage WHERE guest_key=?",(_guest_key(ip),)).fetchone(); return int(r['credits']) if r else GUEST_CREDITS
def guest_stats():
 with _connect() as db:return {'guests':db.execute("SELECT COUNT(*) FROM guest_usage").fetchone()[0],'guest_used':db.execute("SELECT COALESCE(SUM(total_used),0) FROM guest_usage").fetchone()[0]}
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
def admin_audit_logs(limit=200):
 with _connect() as db:return [dict(r) for r in db.execute("SELECT * FROM admin_audit ORDER BY id DESC LIMIT ?",(limit,)).fetchall()]
def audit_admin(admin_user_id,action,target_user_id=None,detail=''):
 admin=get_user(admin_user_id)
 if not admin or not admin['is_admin']:return False
 target=get_user(target_user_id) if target_user_id else None
 with _connect() as db: db.execute("INSERT INTO admin_audit(admin_user_id,admin_username,action,target_user_id,target_username,detail,created_at) VALUES(?,?,?,?,?,?,?)",(admin_user_id,admin['username'],action,target_user_id,target['username'] if target else '',detail[:500],_now()))
 return True
def add_credits(user_id,amount):
 with _connect() as db: db.execute("UPDATE users SET credits=MAX(0,credits+?) WHERE id=?",(int(amount),user_id))
def set_credits(user_id,amount):
 with _connect() as db: db.execute("UPDATE users SET credits=MAX(0,?) WHERE id=?",(int(amount),user_id))
def set_disabled(user_id,disabled):
 with _connect() as db: db.execute("UPDATE users SET disabled=? WHERE id=? AND is_admin=0",(int(bool(disabled)),user_id))
def stats():
 with _connect() as db:
  return {'users':db.execute("SELECT COUNT(*) FROM users WHERE is_admin=0").fetchone()[0],'active':db.execute("SELECT COUNT(*) FROM users WHERE is_admin=0 AND disabled=0").fetchone()[0],'credits':db.execute("SELECT COALESCE(SUM(credits),0) FROM users WHERE is_admin=0").fetchone()[0],'total_usage':db.execute("SELECT COUNT(*) FROM usage_logs").fetchone()[0],'success':db.execute("SELECT COUNT(*) FROM usage_logs WHERE success=1").fetchone()[0],'gemini':db.execute("SELECT COALESCE(SUM(credits),0) FROM usage_logs WHERE provider='gemini' AND success=1").fetchone()[0],'openai':db.execute("SELECT COALESCE(SUM(credits),0) FROM usage_logs WHERE provider='openai' AND success=1").fetchone()[0],'local':db.execute("SELECT COUNT(*) FROM usage_logs WHERE provider='local' AND success=1").fetchone()[0],**guest_stats()}
init_db()
