from __future__ import annotations

import os
import gradio as gr

import gemini_app as base

BROWSER_STATE_SECRET = os.getenv("BROWSER_STATE_SECRET", "zolfox-browser-session-v1")
BROWSER_STATE_KEY = "zolfox_shared_session"

CSS = """
.gradio-container{max-width:1400px!important;width:100%!important;margin:auto!important}
.zf-admin{padding:24px 3vw 48px}
.zf-admin-head{display:flex;align-items:center;justify-content:space-between;gap:16px;margin-bottom:22px}
.zf-admin-title{font-size:30px;font-weight:850;letter-spacing:-1px}
.zf-admin-sub{color:#667085;margin-top:5px}
"""


def restore(browser_session):
    try:
        blob=(browser_session or "").strip()
        if not blob or "|" not in blob:
            raise ValueError("未登录")
        token,csrf=blob.split("|",1)
        session,msg=base.accounts.validate_session(token,csrf,True)
        if not session:
            raise ValueError(msg)
        uid=int(session["user_id"])
        return (
            uid,token,csrf,
            gr.update(visible=True),
            gr.update(value=f"### 🔐 管理员：{session['username']}\n管理员权限已启用。",visible=True),
            base.admin_stats(),base.admin_users_view(),base.admin_logs(),base.admin_audits(),
        )
    except Exception as exc:
        return (
            None,None,None,
            gr.update(visible=False),
            gr.update(value=f"⚠️ {exc}\n\n请先在 ZOLFOX Tools 登录管理员账号，然后刷新本页面。",visible=True),
            "",[],[],[],
        )


def admin_refresh(uid,token,csrf): return base.admin_refresh_view(uid,token,csrf)
def search(uid,token,csrf,keyword): return base.admin_search_view(uid,token,csrf,keyword)
def users_all(uid,token,csrf): return base.admin_search_view(uid,token,csrf,"")
def usage(uid,token,csrf): return base.admin_usage_view(uid,token,csrf)
def audits(uid,token,csrf):
    s,m=base._admin(uid,token,csrf)
    return base.admin_audits() if s else []
def recharge_orders(uid,token,csrf,status): return base.admin_recharge_orders(uid,token,csrf,status)
def recharge_review(uid,token,csrf,order_id,approve,note): return base.admin_recharge_review(uid,token,csrf,order_id,approve,note)
def adjust(uid,token,csrf,target,amount,mode): return base.admin_adjust(uid,token,csrf,target,amount,mode)
def set_credits(uid,token,csrf,target,amount): return base.admin_set(uid,token,csrf,target,amount)
def disable(uid,token,csrf,target,disabled): return base.admin_disable(uid,token,csrf,target,disabled)

with gr.Blocks(title="ZOLFOX 管理后台",theme=gr.themes.Soft(),css=CSS) as admin_app:
    browser_session=gr.BrowserState("",storage_key=BROWSER_STATE_KEY,secret=BROWSER_STATE_SECRET)
    user_id=gr.State(None); session_token=gr.State(None); csrf_token=gr.State(None)
    with gr.Column(elem_classes=["zf-admin"]):
        with gr.Row(elem_classes=["zf-admin-head"]):
            with gr.Column():
                gr.Markdown("# ZOLFOX 管理后台",elem_classes=["zf-admin-title"])
                admin_message=gr.Markdown("正在验证管理员权限…",elem_classes=["zf-admin-sub"])
            gr.HTML('<a href="/"><button>← 返回 ZOLFOX Tools</button></a>')
        with gr.Column(visible=False) as panel:
            gr.Markdown("## 📊 数据概览")
            admin_stat=gr.Markdown()
            refresh_btn=gr.Button("🔄 刷新统计",variant="primary")
            with gr.Tabs():
                with gr.Tab("👥 用户管理"):
                    with gr.Row():
                        keyword=gr.Textbox(label="搜索用户名"); search_btn=gr.Button("🔎 搜索"); all_btn=gr.Button("显示全部")
                    users_table=gr.Dataframe(headers=["ID","用户名","积分","状态","角色","注册时间","最后登录"],interactive=False)
                    with gr.Row():
                        target_id=gr.Number(label="用户 ID",precision=0); amount=gr.Number(label="积分",value=10,precision=0); mode=gr.Radio(["增加","扣除"],value="增加",label="操作"); adjust_btn=gr.Button("执行")
                    with gr.Row():
                        set_amount=gr.Number(label="设置为",precision=0); set_btn=gr.Button("设置积分"); disable_btn=gr.Button("禁用用户"); enable_btn=gr.Button("启用用户")
                    user_message=gr.Markdown()
                with gr.Tab("💰 充值审核"):
                    with gr.Row():
                        recharge_filter=gr.Radio(["all","pending","approved","rejected"],value="pending",label="订单状态"); recharge_refresh=gr.Button("刷新充值订单")
                    recharge_table=gr.Dataframe(headers=["ID","订单号","用户ID","用户名","积分","金额","付款说明","状态","审核管理员","管理员备注","申请时间","审核时间"],interactive=False)
                    with gr.Row():
                        order_id=gr.Number(label="订单 ID",precision=0); approve_btn=gr.Button("✅ 通过并增加积分",variant="primary"); reject_btn=gr.Button("❌ 拒绝申请")
                    admin_note=gr.Textbox(label="管理员备注",max_lines=2); recharge_message=gr.Markdown()
                with gr.Tab("📋 使用记录"):
                    usage_btn=gr.Button("查看全部记录"); logs_table=gr.Dataframe(headers=["ID","用户","引擎","积分","状态","详情","时间"],interactive=False)
                with gr.Tab("🔐 操作审计"):
                    audit_btn=gr.Button("查看审计记录"); audit_table=gr.Dataframe(headers=["ID","管理员","操作","目标用户","详情","时间"],interactive=False)
        refresh_btn.click(admin_refresh,[user_id,session_token,csrf_token],[admin_stat,users_table,logs_table])
        search_btn.click(search,[user_id,session_token,csrf_token,keyword],users_table)
        all_btn.click(users_all,[user_id,session_token,csrf_token],users_table)
        adjust_btn.click(adjust,[user_id,session_token,csrf_token,target_id,amount,mode],[user_message,users_table,logs_table])
        set_btn.click(set_credits,[user_id,session_token,csrf_token,target_id,set_amount],[user_message,users_table,logs_table])
        disable_btn.click(lambda uid,tok,csrf,t: disable(uid,tok,csrf,t,True),[user_id,session_token,csrf_token,target_id],[user_message,users_table])
        enable_btn.click(lambda uid,tok,csrf,t: disable(uid,tok,csrf,t,False),[user_id,session_token,csrf_token,target_id],[user_message,users_table])
        recharge_refresh.click(recharge_orders,[user_id,session_token,csrf_token,recharge_filter],recharge_table)
        recharge_filter.change(recharge_orders,[user_id,session_token,csrf_token,recharge_filter],recharge_table)
        approve_btn.click(lambda uid,tok,csrf,oid,note: recharge_review(uid,tok,csrf,oid,True,note),[user_id,session_token,csrf_token,order_id,admin_note],[recharge_message,recharge_table])
        reject_btn.click(lambda uid,tok,csrf,oid,note: recharge_review(uid,tok,csrf,oid,False,note),[user_id,session_token,csrf_token,order_id,admin_note],[recharge_message,recharge_table])
        usage_btn.click(usage,[user_id,session_token,csrf_token],logs_table)
        audit_btn.click(audits,[user_id,session_token,csrf_token],audit_table)
        gr.on(inputs=[browser_session],outputs=[user_id,session_token,csrf_token,panel,admin_message,admin_stat,users_table,logs_table,audit_table],fn=restore)

if __name__=="__main__": admin_app.launch(server_name="0.0.0.0",server_port=7861)
