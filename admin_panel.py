from __future__ import annotations

import gradio as gr
from account_system import admin_users, admin_usage, add_credits, set_credits, get_user, usage_for_user


def _users(keyword=""):
    rows = admin_users()
    keyword = (keyword or "").strip().lower()
    if keyword:
        rows = [r for r in rows if keyword in r["username"].lower()]
    return [[r["id"], r["username"], r["credits"], "管理员" if r["is_admin"] else "用户", r["created_at"], r["last_login"] or "-"] for r in rows]


def _usage(user_id):
    try:
        uid = int(user_id)
    except Exception:
        return []
    rows = usage_for_user(uid)
    return [[r["id"], r["provider"], r["credits"], "成功" if r["success"] else "失败", r["detail"] or "", r["created_at"]] for r in rows]


def _adjust(user_id, amount, mode):
    try:
        uid = int(user_id)
        n = int(amount)
    except Exception:
        return "❌ 用户 ID 和额度必须是数字。", _users()
    user = get_user(uid)
    if not user:
        return "❌ 用户不存在。", _users()
    if user["is_admin"]:
        return "❌ 不能修改管理员额度。", _users()
    if mode == "增加":
        add_credits(uid, abs(n))
    else:
        add_credits(uid, -abs(n))
    return f"✅ {user['username']} 当前额度：{get_user(uid)['credits']}", _users()


def _set(user_id, amount):
    try:
        uid, n = int(user_id), int(amount)
    except Exception:
        return "❌ 用户 ID 和额度必须是数字。", _users()
    user = get_user(uid)
    if not user:
        return "❌ 用户不存在。", _users()
    if user["is_admin"]:
        return "❌ 不能修改管理员额度。", _users()
    set_credits(uid, n)
    return f"✅ {user['username']} 当前额度：{get_user(uid)['credits']}", _users()


def build_admin_panel():
    with gr.Column() as panel:
        gr.Markdown("# 🛠 管理员后台\n用户、额度和 AI 使用情况统一管理")
        with gr.Row():
            stat = gr.Markdown()
        def stats_text():
            rows = admin_users(); logs = admin_usage()
            total = len(rows); active = sum(not r.get("disabled", 0) for r in rows)
            credits = sum(r["credits"] for r in rows)
            success = sum(r["success"] for r in logs)
            gemini = sum(r["credits"] for r in logs if r["provider"] == "gemini" and r["success"])
            openai = sum(r["credits"] for r in logs if r["provider"] == "openai" and r["success"])
            local = sum(1 for r in logs if r["provider"] == "local" and r["success"])
            return f"**用户 {total}**　**可用 {active}**　**剩余额度 {credits}**　**成功修复 {success}**　|　Gemini **{gemini}**　OpenAI **{openai}**　LaMa **{local}**"
        refresh = gr.Button("🔄 刷新统计")
        refresh.click(stats_text, outputs=stat)
        gr.Markdown("## 👥 用户管理")
        with gr.Row():
            search = gr.Textbox(label="搜索用户", placeholder="输入用户名")
            search_btn = gr.Button("🔎 搜索")
            all_btn = gr.Button("显示全部")
        users = gr.Dataframe(headers=["ID", "用户名", "额度", "角色", "注册时间", "最后登录"], interactive=False)
        search_btn.click(_users, search, users)
        all_btn.click(lambda: _users(), outputs=users)
        with gr.Row():
            uid = gr.Number(label="用户 ID", precision=0)
            amount = gr.Number(label="额度", precision=0)
            mode = gr.Radio(["增加", "扣除"], value="增加", label="操作")
            adjust = gr.Button("执行额度调整", variant="primary")
        msg = gr.Markdown()
        adjust.click(_adjust, [uid, amount, mode], [msg, users])
        with gr.Row():
            set_amount = gr.Number(label="设置为指定额度", precision=0)
            set_btn = gr.Button("设置额度")
        set_btn.click(_set, [uid, set_amount], [msg, users])
        gr.Markdown("## 📋 用户详细使用记录")
        usage_btn = gr.Button("查看该用户记录")
        usage = gr.Dataframe(headers=["ID", "引擎", "扣除额度", "状态", "详情", "时间"], interactive=False)
        usage_btn.click(_usage, uid, usage)
        panel.visible = True
        panel.render()
    return panel
