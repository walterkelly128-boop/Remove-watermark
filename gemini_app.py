from __future__ import annotations

import traceback
import numpy as np
import gradio as gr
from PIL import Image

from app import CSS, EDITOR_JS, _pil, auto_detect, decode_mask, reset_editor
from ai_provider_engine import RCImageEngine
from core.inpaint_engine import InpaintEngine
import account_system as accounts


local_engine = InpaintEngine()
PROVIDER_COST = {"local": 0, "gemini": 1, "openai": 2}


def _engine_status(provider: str) -> tuple[str, str]:
    if provider == "local":
        return "● 已就绪", "本地 LaMa · CPU · 免费"
    api_provider = "gemini" if provider == "gemini" else "openai"
    engine = RCImageEngine(api_provider)
    if not engine.api_key:
        return "○ 未配置", f"{engine.model} · 管理员尚未配置 API"
    label = "Gemini" if provider == "gemini" else "OpenAI"
    return "● 已配置", f"{label} · {engine.model} · 每次 {PROVIDER_COST[provider]} 次额度"


def refresh_status():
    local_s, local_d = _engine_status("local")
    gemini_s, gemini_d = _engine_status("gemini")
    openai_s, openai_d = _engine_status("openai")
    return local_s, local_d, gemini_s, gemini_d, openai_s, openai_d


def auth_login(username, password):
    user, message = accounts.login(username, password)
    if not user:
        return None, gr.update(visible=True), gr.update(visible=False), f"❌ {message}", ""
    return user["id"], gr.update(visible=False), gr.update(visible=True), f"✅ 欢迎回来，**{user['username']}**", account_text(user)


def auth_register(username, password):
    ok, message = accounts.register(username, password)
    if ok:
        return f"✅ {message} 现在可以登录。", gr.update(value=username)
    return f"❌ {message}", gr.update()


def logout():
    return None, gr.update(visible=True), gr.update(visible=False), "", ""


def account_text(user):
    if not user:
        return ""
    role = "管理员" if user["is_admin"] else "普通用户"
    return f"### 👤 {user['username']}  ·  {role}\n**剩余额度：{user['credits']} 次**"


def refresh_account(user_id):
    user = accounts.get_user(user_id) if user_id else None
    if not user:
        return ""
    return account_text(user)


def ai_restore(provider, image, mask_data, user_id):
    pil = _pil(image)
    if pil is None:
        raise gr.Error("请先上传图片。")
    if not user_id:
        raise gr.Error("请先登录账户。")

    mask = decode_mask(mask_data, pil.size)
    pixels = int(np.count_nonzero(mask))
    if pixels == 0:
        raise gr.Error("Mask 是空的：请先自动识别，或用画笔涂满需要修复的区域。")

    cost = PROVIDER_COST.get(provider, 1)
    reserved = False
    try:
        ok, message = accounts.consume_credit(user_id, provider, cost)
        if not ok:
            raise gr.Error(f"{message} 当前需要 {cost} 次额度。")
        reserved = cost > 0

        if provider == "local":
            mask_image = Image.fromarray(mask.astype(np.uint8), "L")
            result = local_engine.run(pil, mask_image)
        else:
            engine = RCImageEngine(provider)
            if not engine.available:
                raise RuntimeError("管理员尚未配置第三方 API。")
            result = engine.repair(pil, mask)

        accounts.log_usage(user_id, provider, cost, True, "repair success")
        return result
    except gr.Error:
        if reserved:
            accounts.refund_credit(user_id, provider, cost)
            accounts.log_usage(user_id, provider, cost, False, "credit refunded")
        raise
    except Exception as exc:
        if reserved:
            accounts.refund_credit(user_id, provider, cost)
        accounts.log_usage(user_id, provider, cost, False, str(exc))
        traceback.print_exc()
        raise gr.Error(f"修复失败：{type(exc).__name__}: {exc}") from exc


def admin_refresh(user_id):
    user = accounts.get_user(user_id) if user_id else None
    if not user or not user["is_admin"]:
        return [], [], "❌ 无管理员权限。"
    users = accounts.admin_users()
    logs = accounts.admin_usage()
    return users, logs, f"✅ 管理后台：{len(users)} 个账户，{len(logs)} 条使用记录。"


def admin_add_credit(user_id, target_id, amount):
    user = accounts.get_user(user_id) if user_id else None
    if not user or not user["is_admin"]:
        return "❌ 无管理员权限。", [], []
    try:
        accounts.add_credits(int(target_id), int(amount))
        return f"✅ 已为用户 ID {target_id} 增加 {int(amount)} 次额度。", accounts.admin_users(), accounts.admin_usage()
    except Exception as exc:
        return f"❌ 操作失败：{exc}", accounts.admin_users(), accounts.admin_usage()


def admin_set_credit(user_id, target_id, amount):
    user = accounts.get_user(user_id) if user_id else None
    if not user or not user["is_admin"]:
        return "❌ 无管理员权限。", [], []
    try:
        accounts.set_credits(int(target_id), int(amount))
        return f"✅ 已将用户 ID {target_id} 的额度设置为 {max(0, int(amount))}。", accounts.admin_users(), accounts.admin_usage()
    except Exception as exc:
        return f"❌ 操作失败：{exc}", accounts.admin_users(), accounts.admin_usage()


CARD_CSS = """
.engine-card { border: 1px solid var(--border-color-primary); border-radius: 14px; padding: 16px; min-height: 145px; cursor: pointer; transition: .18s ease; }
.engine-card:hover { border-color: var(--primary-500); transform: translateY(-2px); }
.engine-card.selected { border: 2px solid var(--primary-500); box-shadow: 0 0 0 2px rgba(99,102,241,.10); }
.engine-icon { font-size: 28px; margin-bottom: 8px; }
.engine-name { font-size: 18px; font-weight: 700; }
.engine-status { margin-top: 10px; font-weight: 600; }
.engine-desc { font-size: 13px; opacity: .72; margin-top: 5px; }
"""

with gr.Blocks(title="AI 图片智能修复", theme=gr.themes.Soft(), css=CSS + CARD_CSS, head=EDITOR_JS) as demo:
    user_id = gr.State(None)

    with gr.Column(visible=True) as auth_panel:
        gr.Markdown("# 🔐 AI 图片智能修复")
        gr.Markdown("登录账户后使用 AI 修复。新用户注册赠送免费额度。")
        with gr.Tabs():
            with gr.Tab("登录"):
                login_user = gr.Textbox(label="用户名")
                login_pass = gr.Textbox(label="密码", type="password")
                login_btn = gr.Button("登录", variant="primary")
            with gr.Tab("注册"):
                reg_user = gr.Textbox(label="用户名（3-32位）")
                reg_pass = gr.Textbox(label="密码（至少6位）", type="password")
                reg_btn = gr.Button("注册", variant="primary")
        auth_message = gr.Markdown()

    with gr.Column(visible=False) as app_panel:
        with gr.Row():
            with gr.Column(scale=5):
                gr.Markdown("# AI 图片智能修复")
                gr.Markdown("自动识别候选区域 + 手动画笔/橡皮擦 Mask。支持本地 CPU LaMa、Gemini 和 OpenAI。")
            with gr.Column(scale=2):
                account_info = gr.Markdown()
                logout_btn = gr.Button("退出登录")

        with gr.Tabs():
            with gr.Tab("🖼️ 图片修复"):
                with gr.Row():
                    with gr.Column():
                        source = gr.Image(label="原图", type="pil")
                        with gr.Row():
                            auto_btn = gr.Button("✨ 自动识别候选区域", variant="primary")
                            clear_btn = gr.Button("清除 Mask")
                        status = gr.Markdown("上传图片后开始。")
                    with gr.Column():
                        preview = gr.Image(label="自动识别预览（红色=候选区域）", type="pil")

                gr.Markdown("## Mask 编辑器")
                editor = gr.HTML(label="Mask 编辑器")
                mask_data = gr.Textbox(label="", elem_id="mask-data", visible=True, container=False)

                gr.Markdown("## 选择 AI 修复引擎")
                with gr.Row():
                    with gr.Column(elem_classes=["engine-card"]):
                        gr.Markdown("### 🖥️ 本地 LaMa")
                        local_status = gr.Markdown("● 已就绪")
                        local_desc = gr.Markdown("本地 CPU · 免费 · 不消耗额度")
                        local_btn = gr.Button("选择本地", variant="primary")
                    with gr.Column(elem_classes=["engine-card"]):
                        gr.Markdown("### ✨ Gemini")
                        gemini_status = gr.Markdown("○ 检测中…")
                        gemini_desc = gr.Markdown("Gemini 图像模型 · 每次 1 次额度")
                        gemini_btn = gr.Button("选择 Gemini")
                    with gr.Column(elem_classes=["engine-card"]):
                        gr.Markdown("### ◉ OpenAI")
                        openai_status = gr.Markdown("○ 检测中…")
                        openai_desc = gr.Markdown("OpenAI 图像模型 · 每次 2 次额度")
                        openai_btn = gr.Button("选择 OpenAI")

                selected = gr.State("local")
                selected_text = gr.Markdown("**当前引擎：本地 LaMa（免费）**")
                restore_btn = gr.Button("🚀 开始修复", variant="primary", elem_id="restore-btn")
                result = gr.Image(label="修复结果", type="pil", format="png")

            with gr.Tab("📊 使用记录"):
                usage_table = gr.Dataframe(headers=["时间", "引擎", "额度", "成功", "详情"], interactive=False)
                refresh_usage_btn = gr.Button("刷新记录")

            with gr.Tab("⚙️ 管理后台", visible=False) as admin_tab:
                admin_message = gr.Markdown()
                with gr.Row():
                    target_id = gr.Number(label="用户 ID", precision=0)
                    amount = gr.Number(label="额度", value=10, precision=0)
                with gr.Row():
                    add_btn = gr.Button("增加额度", variant="primary")
                    set_btn = gr.Button("设置额度")
                users_table = gr.Dataframe(headers=["ID", "用户名", "额度", "管理员", "注册时间", "最后登录"], interactive=False)
                logs_table = gr.Dataframe(headers=["ID", "用户", "引擎", "额度", "成功", "详情", "时间"], interactive=False)
                refresh_admin_btn = gr.Button("刷新后台数据")

    login_btn.click(auth_login, inputs=[login_user, login_pass], outputs=[user_id, auth_panel, app_panel, auth_message, account_info])
    reg_btn.click(auth_register, inputs=[reg_user, reg_pass], outputs=[auth_message, login_user])
    logout_btn.click(logout, outputs=[user_id, auth_panel, app_panel, auth_message, account_info])

    source.change(reset_editor, inputs=source, outputs=[editor, mask_data])
    auto_btn.click(auto_detect, inputs=source, outputs=[preview, editor, status, mask_data], show_progress="minimal")
    clear_btn.click(reset_editor, inputs=source, outputs=[editor, mask_data])
    demo.load(refresh_status, outputs=[local_status, local_desc, gemini_status, gemini_desc, openai_status, openai_desc])
    local_btn.click(lambda: ("local", "**当前引擎：本地 LaMa（免费）**"), outputs=[selected, selected_text])
    gemini_btn.click(lambda: ("gemini", "**当前引擎：Gemini（1 次额度）**"), outputs=[selected, selected_text])
    openai_btn.click(lambda: ("openai", "**当前引擎：OpenAI（2 次额度）**"), outputs=[selected, selected_text])
    restore_btn.click(ai_restore, inputs=[selected, source, mask_data, user_id], outputs=result)

    refresh_usage_btn.click(
        lambda uid: [[r["created_at"], r["provider"], r["credits"], "成功" if r["success"] else "失败", r["detail"]] for r in accounts.usage_for_user(uid)] if uid else [],
        inputs=user_id, outputs=usage_table,
    )
    refresh_admin_btn.click(admin_refresh, inputs=user_id, outputs=[users_table, logs_table, admin_message])
    add_btn.click(admin_add_credit, inputs=[user_id, target_id, amount], outputs=[admin_message, users_table, logs_table])
    set_btn.click(admin_set_credit, inputs=[user_id, target_id, amount], outputs=[admin_message, users_table, logs_table])


if __name__ == "__main__":
    demo.launch(server_name="0.0.0.0", server_port=7860, show_error=True)
