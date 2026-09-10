from __future__ import annotations
import os, traceback
import numpy as np
import gradio as gr
from PIL import Image
from app import CSS, EDITOR_JS, _pil, auto_detect, decode_mask, reset_editor
from ai_provider_engine import RCImageEngine
from core.inpaint_engine import InpaintEngine
import account_system as accounts
import recharge_system as recharge

local_engine=InpaintEngine(); PROVIDER_COST={"local":0,"gemini":1,"openai":3}
def _engine_status(p):
 if p=="local": return "● 已就绪","本地 LaMa · CPU · 免费"
 e=RCImageEngine(p); return ("● 已配置" if e.api_key else "○ 未配置",f"{p.title()} · {e.model} · 每次 {PROVIDER_COST[p]} 积分")
def refresh_status(*_):
 return sum((_engine_status(p) for p in ("local","gemini","openai")),())
def account_text(u): return f"### 👤 {u['username']} · {'管理员' if u['is_admin'] else '普通用户'}\n**剩余积分：{u['credits']}**" if u else ""
def _client_ip(request):
 if request is None:return ""
 try:
  if os.getenv("TRUST_PROXY","0")=="1":
   xff=request.headers.get("x-forwarded-for","")
   if xff:return xff.split(",")[0].strip()
  return request.client.host if request.client else ""
 except Exception:return ""
def _user_agent(request):
 try:return (request.headers.get("user-agent","") if request else "")[:500]
 except Exception:return ""
def auth_login(username,password,totp_code,request: gr.Request=None):
 u,m=accounts.login(username,password,_client_ip(request),_user_agent(request),totp_code)
 if u:
  return (u["id"],u["session_token"],u["csrf_token"],gr.update(visible=False),gr.update(visible=True),gr.update(visible=True),f"✅ 欢迎回来，**{u['username']}**",account_text(u),gr.update(visible=bool(u['is_admin'])))
 return (None,None,None,gr.update(visible=True),gr.update(visible=True),gr.update(visible=False),f"❌ {m}","",gr.update(visible=False))
def auth_register(username,password,request: gr.Request=None):
 ok,m=accounts.register(username,password,_client_ip(request)); return f"{'✅' if ok else '❌'} {m}",gr.update(value=username if ok else None)
def logout(token):
 accounts.revoke_session(token)
 return None,None,None,gr.update(visible=True),gr.update(visible=True),gr.update(visible=False),"","",gr.update(visible=False)
def require_user(uid,token,csrf=None,admin=False):
 if not uid or not token:return None,"请先登录。"
 s,m=accounts.validate_session(token,csrf,admin)
 if not s:return None,m
 if int(s['user_id'])!=int(uid):return None,"会话用户不匹配，请重新登录。"
 return s,""
def change_my_password(uid,token,csrf,current,new):
 if not uid:return "❌ 请先登录。"
 ok,m=accounts.change_password(uid,current,new,token,csrf); return f"{'✅' if ok else '❌'} {m}"
def ai_restore(provider,image,mask_data,user_id,session_token,csrf_token,request: gr.Request=None):
 pil=_pil(image)
 if pil is None: raise gr.Error("请先上传图片。")
 mask=decode_mask(mask_data,pil.size)
 if int(np.count_nonzero(mask))==0: raise gr.Error("Mask 是空的：请先自动识别，或用画笔涂满需要修复的区域。")
 cost=PROVIDER_COST.get(provider,1); reserved=False; guest_ip=""
 try:
  if user_id:
   s,m=require_user(user_id,session_token,csrf_token)
   if not s: raise gr.Error(m)
   ok,m=accounts.consume_credit(user_id,provider,cost)
  elif cost>0:
   guest_ip=_client_ip(request); ok,m=accounts.consume_guest(guest_ip,provider,cost)
  else: ok,m=True,"免费操作。"
  if not ok: raise gr.Error(f"{m} 当前需要 {cost} 积分。")
  reserved=cost>0
  if provider=="local": result=local_engine.run(pil,Image.fromarray(mask.astype(np.uint8),"L"))
  else:
   e=RCImageEngine(provider)
   if not e.available: raise RuntimeError("管理员尚未配置第三方 API。")
   result=e.repair(pil,mask)
  if user_id: accounts.log_usage(user_id,provider,cost,True,"repair success")
  return result
 except gr.Error:
  if reserved:
   if user_id: accounts.refund_credit(user_id,provider,cost)
   elif guest_ip: accounts.refund_guest(guest_ip,cost)
  if user_id: accounts.log_usage(user_id,provider,cost,False,"credit refunded")
  raise
 except Exception as exc:
  if reserved:
   if user_id: accounts.refund_credit(user_id,provider,cost)
   elif guest_ip: accounts.refund_guest(guest_ip,cost)
  if user_id: accounts.log_usage(user_id,provider,cost,False,str(exc))
  traceback.print_exc(); raise gr.Error(f"修复失败：{type(exc).__name__}: {exc}") from exc

def admin_users_view(keyword=""):
 rows=accounts.admin_users(keyword); return [[r["id"],r["username"],r["credits"],"禁用" if r.get("disabled") else "正常","管理员" if r["is_admin"] else "用户",r["created_at"],r["last_login"] or "-"] for r in rows]
def admin_logs(uid=None):
 try:
  rows=accounts.usage_for_user(int(uid)) if uid else accounts.admin_usage()
  return [[r.get("id",""),r.get("username",""),r.get("provider",""),int(r.get("credits",0)),"成功" if r.get("success") else "失败",r.get("detail") or "",r.get("created_at","")] for r in rows]
 except Exception as exc:
  traceback.print_exc(); return [["","系统","","","错误",f"读取使用记录失败：{exc}",""]]
def admin_audits():
 rows=accounts.admin_audit_logs(); return [[r["id"],r["admin_username"],r["action"],r["target_username"] or "-",r["detail"] or "",r["created_at"]] for r in rows]
def admin_stats():
 s=accounts.stats(); return f"### 📊 管理员统计\n**用户：{s['users']}**　**活跃：{s['active']}**　**剩余积分：{s['credits']}**　**总记录：{s['total_usage']}**　**成功：{s['success']}**\n\n✨ Gemini：**{s['gemini']} 积分**　◉ OpenAI：**{s['openai']} 积分**　🖥️ LaMa：**{s['local']}**\n\n👥 游客 IP：**{s['guests']}**　🎁 游客已使用：**{s['guest_used']} 积分**　💰 待审核充值：**{recharge.pending_count()}**"
def _admin(uid,token,csrf): return require_user(uid,token,csrf,True)
def admin_adjust(uid,token,csrf,target,amount,mode):
 s,m=_admin(uid,token,csrf)
 if not s:return f"❌ {m}",[],[]
 try:
  target=int(target); amount=abs(int(amount)); u=accounts.get_user(target)
  if not u:return "❌ 用户不存在。",[],[]
  if u["is_admin"] or target==uid:return "❌ 不能修改管理员账户。",[],[]
  delta=amount if mode=="增加" else -amount; accounts.add_credits(target,delta); new=accounts.get_user(target)['credits']; accounts.audit_admin(uid,"调整积分",target,f"{mode} {amount} 积分，结果 {new} 积分")
  return f"✅ {u['username']} 当前积分：{new}",admin_users_view(),admin_logs()
 except Exception as e:return f"❌ 操作失败：{e}",admin_users_view(),admin_logs()
def admin_set(uid,token,csrf,target,amount):
 s,m=_admin(uid,token,csrf)
 if not s:return f"❌ {m}",[],[]
 try:
  target=int(target); u=accounts.get_user(target)
  if not u:return "❌ 用户不存在。",[],[]
  if u["is_admin"] or target==uid:return "❌ 不能修改管理员账户。",[],[]
  new=max(0,int(amount)); accounts.set_credits(target,new); accounts.audit_admin(uid,"设置积分",target,f"设置为 {new} 积分")
  return f"✅ 已设置 {u['username']} 为 {new} 积分。",admin_users_view(),admin_logs()
 except Exception as e:return f"❌ 操作失败：{e}",admin_users_view(),admin_logs()
def admin_disable(uid,token,csrf,target,disabled):
 s,m=_admin(uid,token,csrf)
 if not s:return f"❌ {m}",[]
 try:
  target=int(target); u=accounts.get_user(target)
  if not u:return "❌ 用户不存在。",[]
  if u["is_admin"] or target==uid:return "❌ 不能禁用管理员账户。",[]
  accounts.set_disabled(target,disabled); accounts.audit_admin(uid,"禁用用户" if disabled else "启用用户",target,"状态已修改"); return f"✅ {u['username']} 已{'禁用' if disabled else '启用'}。",admin_users_view()
 except Exception as e:return f"❌ 操作失败：{e}",admin_users_view()
def admin_refresh_view(uid,token,csrf):
 s,m=_admin(uid,token,csrf)
 if not s:return f"❌ {m}",[],[]
 return admin_stats(),admin_users_view(),admin_logs()
def admin_search_view(uid,token,csrf,keyword):
 s,m=_admin(uid,token,csrf); return admin_users_view(keyword) if s else []
def admin_usage_view(uid,token,csrf):
 s,m=_admin(uid,token,csrf); return admin_logs() if s else []
def admin_audit_view(uid,token,csrf):
 s,m=_admin(uid,token,csrf); return admin_audits() if s else []
def usage_view(uid,token,csrf):
 s,m=require_user(uid,token,csrf)
 if not s:return []
 return [[r["created_at"],r["provider"],r["credits"],"成功" if r["success"] else "失败",r["detail"]] for r in accounts.usage_for_user(uid)]

def recharge_submit(uid,token,csrf,choice,note):
 s,m=require_user(uid,token,csrf)
 if not s:return "❌ "+m, recharge_user_orders(uid) if uid else []
 ok,msg,_=recharge.create_order(uid,choice,note)
 return ("✅ "+msg if ok else "❌ "+msg), recharge_user_orders(uid)
def recharge_user_orders(uid):
 if not uid:return []
 rows=recharge.user_orders(uid)
 return [[r["order_no"],r["credits"],f"¥{r['amount']:g}",r["status"],r["payment_note"] or "",r["created_at"],r["reviewed_at"] or "-"] for r in rows]
def recharge_history_view(uid,token,csrf):
 s,m=require_user(uid,token,csrf); return recharge_user_orders(uid) if s else []
def admin_recharge_orders(uid,token,csrf,status="all"):
 s,m=_admin(uid,token,csrf)
 if not s:return []
 rows=recharge.admin_orders(status)
 return [[r["id"],r["order_no"],r["user_id"],r["username"],r["credits"],f"¥{r['amount']:g}",r["payment_note"] or "",r["status"],r["admin_username"] or "-",r["admin_note"] or "",r["created_at"],r["reviewed_at"] or "-"] for r in rows]
def admin_recharge_review(uid,token,csrf,order_id,approve,note):
 s,m=_admin(uid,token,csrf)
 if not s:return "❌ "+m,[]
 try: oid=int(order_id)
 except Exception:return "❌ 请输入有效的订单 ID。",admin_recharge_orders(uid,token,csrf)
 ok,msg=recharge.review_order(uid,oid,approve,note)
 return ("✅ "+msg if ok else "❌ "+msg),admin_recharge_orders(uid,token,csrf)

CARD_CSS=""".engine-card{border:1px solid var(--border-color-primary);border-radius:14px;padding:16px;min-height:145px}.engine-card:hover{border-color:var(--primary-500);transform:translateY(-2px)}"""
with gr.Blocks(title="AI 图片智能修复",theme=gr.themes.Soft(),css=CSS+CARD_CSS,head=EDITOR_JS) as demo:
 user_id=gr.State(None); session_token=gr.State(None); csrf_token=gr.State(None)
 with gr.Column(visible=True) as auth_panel:
  gr.Markdown("## 🔐 登录 / 注册（可选）\n游客可直接体验图片去水印；登录后可查看积分、充值和使用记录。")
  with gr.Tabs():
   with gr.Tab("登录"):
    login_user=gr.Textbox(label="用户名"); login_pass=gr.Textbox(label="密码",type="password"); login_totp=gr.Textbox(label="管理员二次验证码（6位，可选）",type="password"); login_btn=gr.Button("登录",variant="primary")
   with gr.Tab("注册"):
    reg_user=gr.Textbox(label="用户名（3-32位）"); reg_pass=gr.Textbox(label="密码（至少8位）",type="password"); reg_btn=gr.Button("注册",variant="primary")
  auth_message=gr.Markdown()
 with gr.Column(visible=True) as app_panel:
  with gr.Row():
   with gr.Column(scale=5): gr.Markdown("# AI 图片智能修复\n自动识别候选区域 + 手动画笔/橡皮擦 Mask。支持本地 CPU LaMa、Gemini 和 OpenAI。")
   with gr.Column(scale=2): account_info=gr.Markdown(); logout_btn=gr.Button("退出登录", visible=False)
  with gr.Accordion("🔐 账户安全",open=False):
   with gr.Row(): old_pass=gr.Textbox(label="当前密码",type="password"); new_pass=gr.Textbox(label="新密码（至少8位）",type="password"); change_pass_btn=gr.Button("修改密码")
   password_message=gr.Markdown()
  with gr.Tabs():
   with gr.Tab("🖼️ 图片修复"):
    with gr.Row():
     with gr.Column():
      source=gr.Image(label="原图",type="pil")
      with gr.Row(): auto_btn=gr.Button("✨ 自动识别候选区域",variant="primary"); clear_btn=gr.Button("清除 Mask")
      status=gr.Markdown("上传图片后开始。")
     with gr.Column(): preview=gr.Image(label="自动识别预览（红色=候选区域）",type="pil")
    gr.Markdown("## Mask 编辑器"); editor=gr.HTML(label="Mask 编辑器"); mask_data=gr.Textbox(label="",elem_id="mask-data",visible=True,container=False)
    gr.Markdown("## 选择 AI 修复引擎")
    with gr.Row():
     with gr.Column(elem_classes=["engine-card"]): gr.Markdown("### 🖥️ 本地 LaMa"); local_status=gr.Markdown("● 已就绪"); local_desc=gr.Markdown("本地 CPU · 免费"); local_btn=gr.Button("选择本地",variant="primary")
     with gr.Column(elem_classes=["engine-card"]): gr.Markdown("### ✨ Gemini"); gemini_status=gr.Markdown("○ 检测中…"); gemini_desc=gr.Markdown("每次 1 积分"); gemini_btn=gr.Button("选择 Gemini")
     with gr.Column(elem_classes=["engine-card"]): gr.Markdown("### ◉ OpenAI"); openai_status=gr.Markdown("○ 检测中…"); openai_desc=gr.Markdown("每次 3 积分"); openai_btn=gr.Button("选择 OpenAI")
    selected=gr.State("local"); selected_text=gr.Markdown("**当前引擎：本地 LaMa（免费）**"); restore_btn=gr.Button("🚀 开始修复",variant="primary"); result=gr.Image(label="修复结果",type="pil",format="png")
   with gr.Tab("💰 充值积分"):
    gr.Markdown("## 手动充值\n选择套餐后提交充值申请。管理员确认收款后，会手动审核并自动增加你的积分。**本版本不接入任何线上自动支付。**")
    gr.Markdown(recharge.payment_instructions())
    recharge_choice=gr.Radio(recharge.package_choices(),label="充值套餐",value=recharge.package_choices()[0] if recharge.package_choices() else None)
    recharge_note=gr.Textbox(label="付款说明 / 流水号（可选）",placeholder="例如：微信/支付宝转账后填写末四位或备注，方便管理员核对",max_lines=2)
    recharge_submit_btn=gr.Button("提交充值申请",variant="primary")
    recharge_message=gr.Markdown(); recharge_refresh_btn=gr.Button("刷新充值记录")
    recharge_table=gr.Dataframe(headers=["订单号","积分","金额","状态","付款说明","申请时间","审核时间"],interactive=False)
   with gr.Tab("📊 使用记录"):
    usage_table=gr.Dataframe(headers=["时间","引擎","积分","成功","详情"],interactive=False); refresh_usage_btn=gr.Button("刷新记录")
   with gr.Tab("⚙️ 管理后台",visible=False) as admin_tab:
    admin_stat=gr.Markdown(); admin_refresh=gr.Button("🔄 刷新统计")
    gr.Markdown("## 💰 充值订单管理")
    with gr.Row(): recharge_status_filter=gr.Radio(["all","pending","approved","rejected"],value="pending",label="订单状态"); admin_recharge_refresh=gr.Button("刷新充值订单")
    recharge_orders_table=gr.Dataframe(headers=["ID","订单号","用户ID","用户名","积分","金额","付款说明","状态","审核管理员","管理员备注","申请时间","审核时间"],interactive=False)
    with gr.Row(): recharge_order_id=gr.Number(label="订单 ID",precision=0); recharge_approve=gr.Button("✅ 通过并增加积分",variant="primary"); recharge_reject=gr.Button("❌ 拒绝申请")
    recharge_admin_note=gr.Textbox(label="管理员备注",placeholder="可填写收款核对结果或拒绝原因",max_lines=2); recharge_admin_message=gr.Markdown()
    gr.Markdown("## 👥 用户管理")
    with gr.Row(): admin_search=gr.Textbox(label="搜索用户名"); admin_search_btn=gr.Button("🔎 搜索"); admin_all_btn=gr.Button("显示全部")
    users_table=gr.Dataframe(headers=["ID","用户名","积分","状态","角色","注册时间","最后登录"],interactive=False)
    with gr.Row(): target_id=gr.Number(label="用户 ID",precision=0); amount=gr.Number(label="积分",value=10,precision=0); mode=gr.Radio(["增加","扣除"],value="增加",label="操作"); adjust_btn=gr.Button("执行")
    with gr.Row(): set_amount=gr.Number(label="设置为",precision=0); set_btn=gr.Button("设置积分"); disable_btn=gr.Button("禁用用户"); enable_btn=gr.Button("启用用户")
    admin_message=gr.Markdown(); gr.Markdown("## 📋 使用记录"); admin_usage_btn=gr.Button("查看全部记录"); logs_table=gr.Dataframe(headers=["ID","用户","引擎","积分","状态","详情","时间"],interactive=False)
    gr.Markdown("## 🔐 管理员操作审计"); admin_audit_btn=gr.Button("查看审计记录"); audit_table=gr.Dataframe(headers=["ID","管理员","操作","目标用户","详情","时间"],interactive=False)
 login_btn.click(auth_login,[login_user,login_pass,login_totp],[user_id,session_token,csrf_token,auth_panel,app_panel,logout_btn,auth_message,account_info,admin_tab]); reg_btn.click(auth_register,[reg_user,reg_pass],[auth_message,login_user]); logout_btn.click(logout,[session_token],[user_id,session_token,csrf_token,auth_panel,app_panel,auth_message,account_info,admin_tab]); change_pass_btn.click(change_my_password,[user_id,session_token,csrf_token,old_pass,new_pass],password_message)
 source.change(reset_editor,source,[editor,mask_data]); auto_btn.click(auto_detect,source,[preview,editor,status,mask_data],show_progress="minimal"); clear_btn.click(reset_editor,source,[editor,mask_data]); demo.load(refresh_status,[local_status,local_desc,gemini_status,gemini_desc,openai_status,openai_desc])
 local_btn.click(lambda:("local","**当前引擎：本地 LaMa（免费）**"),outputs=[selected,selected_text]); gemini_btn.click(lambda:("gemini","**当前引擎：Gemini（1 积分）**"),outputs=[selected,selected_text]); openai_btn.click(lambda:("openai","**当前引擎：OpenAI（3 积分）**"),outputs=[selected,selected_text]); restore_btn.click(ai_restore,[selected,source,mask_data,user_id,session_token,csrf_token],result)
 recharge_submit_btn.click(recharge_submit,[user_id,session_token,csrf_token,recharge_choice,recharge_note],[recharge_message,recharge_table]); recharge_refresh_btn.click(recharge_history_view,[user_id,session_token,csrf_token],recharge_table)
 refresh_usage_btn.click(usage_view,[user_id,session_token,csrf_token],usage_table)
 admin_refresh.click(admin_refresh_view,[user_id,session_token,csrf_token],[admin_stat,users_table,logs_table]); admin_search_btn.click(admin_search_view,[user_id,session_token,csrf_token,admin_search],users_table); admin_all_btn.click(admin_search_view,[user_id,session_token,csrf_token,admin_search],users_table); admin_usage_btn.click(admin_usage_view,[user_id,session_token,csrf_token],logs_table); admin_audit_btn.click(admin_audit_view,[user_id,session_token,csrf_token],audit_table)
 admin_recharge_refresh.click(admin_recharge_orders,[user_id,session_token,csrf_token,recharge_status_filter],recharge_orders_table); recharge_status_filter.change(admin_recharge_orders,[user_id,session_token,csrf_token,recharge_status_filter],recharge_orders_table)
 recharge_approve.click(lambda uid,tok,csrf,oid,note:admin_recharge_review(uid,tok,csrf,oid,True,note),[user_id,session_token,csrf_token,recharge_order_id,recharge_admin_note],[recharge_admin_message,recharge_orders_table]); recharge_reject.click(lambda uid,tok,csrf,oid,note:admin_recharge_review(uid,tok,csrf,oid,False,note),[user_id,session_token,csrf_token,recharge_order_id,recharge_admin_note],[recharge_admin_message,recharge_orders_table])
 adjust_btn.click(admin_adjust,[user_id,session_token,csrf_token,target_id,amount,mode],[admin_message,users_table,logs_table]); set_btn.click(admin_set,[user_id,session_token,csrf_token,target_id,set_amount],[admin_message,users_table,logs_table]); disable_btn.click(lambda uid,tok,csrf,t:admin_disable(uid,tok,csrf,t,True),[user_id,session_token,csrf_token,target_id],[admin_message,users_table]); enable_btn.click(lambda uid,tok,csrf,t:admin_disable(uid,tok,csrf,t,False),[user_id,session_token,csrf_token,target_id],[admin_message,users_table])
if __name__=="__main__": demo.launch(server_name="0.0.0.0",server_port=7860,show_error=True)
