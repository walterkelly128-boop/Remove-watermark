from __future__ import annotations

import os
import tempfile
from io import BytesIO

import gradio as gr

import gemini_app as base
from image_compress import compress_image, format_size

SITE_CSS = r"""
.gradio-container{max-width:1180px!important;margin:auto!important}
.zf-header{border-bottom:1px solid #e8ebf2;padding:14px 0;margin-bottom:12px}
.zf-logo{font-size:25px;font-weight:850;letter-spacing:-.7px;color:#151a2b}.zf-logo span{color:#5b5bd6}
.zf-nav{color:#667085;font-size:14px}.zf-account{border:1px solid #e4e7ec;border-radius:12px;padding:8px 14px;background:#fff}
.zf-hero{text-align:center;padding:64px 20px 50px;border-radius:28px;background:linear-gradient(180deg,#f6f7ff,#fff);border:1px solid #eceeff;margin-bottom:28px}
.zf-hero h1{font-size:52px!important;line-height:1.08!important;letter-spacing:-2.5px;margin:16px 0!important}.zf-hero p{font-size:18px;color:#667085;line-height:1.8}
.zf-pill{display:inline-block;padding:7px 13px;border-radius:999px;background:#eeedff;color:#5148c9;font-weight:700;font-size:13px}
.zf-card{border:1px solid #e4e7ec;border-radius:20px;padding:24px;min-height:190px;background:#fff;transition:.2s}.zf-card:hover{box-shadow:0 14px 36px rgba(16,24,40,.08);transform:translateY(-2px)}
.zf-card h3{margin:12px 0 8px;font-size:21px}.zf-card p{color:#667085;line-height:1.7}.zf-tool{border:1px solid #e4e7ec;border-radius:22px;padding:24px;margin-top:24px;background:#fff}
.zf-tool-title{text-align:center}.zf-tool-title h2{font-size:32px;margin-bottom:6px}.zf-tool-title p{color:#667085}.zf-muted{color:#667085}
.zf-hidden{display:none!important}
@media(max-width:700px){.zf-hero h1{font-size:39px!important}.zf-nav{display:none}}
"""


def _original_size(image):
    filename = getattr(image, "filename", None)
    if filename and os.path.isfile(filename):
        try:
            return os.path.getsize(filename)
        except OSError:
            pass
    probe = BytesIO()
    image.save(probe, format="PNG")
    return len(probe.getvalue())


def compress_for_web(image, quality, output_format):
    if image is None:
        raise gr.Error("请先上传图片。")
    try:
        original_bytes = _original_size(image)
        data, fmt = compress_image(image, quality, output_format)
        suffix = ".jpg" if fmt == "JPEG" else ".webp" if fmt == "WEBP" else ".png"
        fd, path = tempfile.mkstemp(prefix="zolfox-compress-", suffix=suffix)
        with os.fdopen(fd, "wb") as f:
            f.write(data)
        ratio = max(0.0, (1 - len(data) / original_bytes) * 100) if original_bytes else 0.0
        return path, f"**原图：** {format_size(original_bytes)}　　**压缩后：** {format_size(len(data))}　　**体积减少：** {ratio:.1f}%"
    except Exception as exc:
        raise gr.Error("压缩失败，请更换图片或参数后重试。") from exc


def do_login(username, password, totp, request: gr.Request = None):
    uid, token, csrf, _auth, _app, _logout, msg, account, admin = base.auth_login(username, password, totp, request)
    return uid, token, csrf, gr.update(visible=False), gr.update(visible=True), msg, account, admin


def do_logout(token):
    base.logout(token)
    return None, None, None, gr.update(visible=True), gr.update(visible=False), "", "", gr.update(visible=False)


def show_home():
    return gr.update(visible=True), gr.update(visible=False), gr.update(visible=False)


def show_remove():
    return gr.update(visible=False), gr.update(visible=True), gr.update(visible=False)


def show_compress():
    return gr.update(visible=False), gr.update(visible=False), gr.update(visible=True)


with gr.Blocks(title="ZOLFOX Tools · 在线工具箱", theme=gr.themes.Soft(), css=SITE_CSS + base.CSS, head=base.EDITOR_JS) as demo:
    user_id = gr.State(None)
    session_token = gr.State(None)
    csrf_token = gr.State(None)

    # Header: login/register stays in the upper-right area of the homepage.
    with gr.Row(elem_classes=["zf-header"]):
        gr.HTML('<div class="zf-logo"><span>ZOLFOX</span> Tools</div>')
        gr.Markdown("图片工具　　PDF 工具　　AI 工具　　更多工具", elem_classes=["zf-nav"])
        with gr.Column(scale=0, min_width=190):
            with gr.Row():
                login_open = gr.Button("登录 / 注册", size="sm")
                logout_btn = gr.Button("退出登录", size="sm", visible=False)

    # Compact login/register panel opened from the upper-right button.
    with gr.Column(visible=False, elem_classes=["zf-tool"]) as auth_panel:
        with gr.Tabs():
            with gr.Tab("登录"):
                login_user = gr.Textbox(label="用户名")
                login_pass = gr.Textbox(label="密码", type="password")
                login_totp = gr.Textbox(label="管理员二次验证码（可选）", type="password")
                login_btn = gr.Button("登录", variant="primary")
            with gr.Tab("注册"):
                reg_user = gr.Textbox(label="用户名（3-32位）")
                reg_pass = gr.Textbox(label="密码（至少8位）", type="password")
                reg_btn = gr.Button("注册", variant="primary")
        auth_message = gr.Markdown()

    with gr.Row(visible=False) as account_bar:
        account_info = gr.Markdown(elem_classes=["zf-account"])
        account_refresh = gr.Button("刷新额度", size="sm")

    with gr.Column(visible=True) as home_view:
        gr.HTML('<div class="zf-hero"><div class="zf-pill">ZOLFOX Tools · 在线工具箱</div><h1>简单、快速、实用的<br><span style="color:#5b5bd6">在线工具</span></h1><p>图片、PDF、AI 与更多常用工具，持续更新中。<br>无需安装软件，打开浏览器即可使用。</p></div>')
        gr.Markdown("## 图片工具")
        with gr.Row():
            with gr.Column(elem_classes=["zf-card"]):
                gr.Markdown("### ✨ 图片去水印\n自动识别或手动选择需要修复的区域，使用本地 AI、Gemini 或 OpenAI 自然重建图片内容。")
                remove_open = gr.Button("立即使用 →", variant="primary")
            with gr.Column(elem_classes=["zf-card"]):
                gr.Markdown("### 📦 图片压缩\n快速压缩 JPG、PNG、WebP 图片，在尽量保持画质的同时减小文件体积。")
                compress_open = gr.Button("立即使用 →", variant="primary")
        gr.Markdown("## 更多工具")
        with gr.Row():
            for title, desc in [("📄 PDF 转 Word","将 PDF 文档转换为可编辑文件。"),("🧩 PDF 合并","多个 PDF 快速合并。"),("🤖 AI 图片增强","提升图片清晰度与细节。")]:
                with gr.Column(elem_classes=["zf-card"]):
                    gr.Markdown(f"### {title}\n{desc}\n\n**即将上线**")

    # Same-page watermark tool. There is no separate /remove-watermark page.
    with gr.Column(visible=False, elem_classes=["zf-tool"]) as remove_view:
        with gr.Row():
            gr.Markdown("# 🖼️ 图片去水印")
            remove_back = gr.Button("← 返回工具首页", size="sm")
        gr.Markdown("自动识别候选区域，也可以使用 Mask 编辑器精确指定需要修复的位置。")
        with gr.Row():
            with gr.Column():
                source = gr.Image(label="原图", type="pil")
                with gr.Row():
                    auto_btn = gr.Button("✨ 自动识别候选区域", variant="primary")
                    clear_btn = gr.Button("清除 Mask")
                status = gr.Markdown("上传图片后开始。")
            with gr.Column():
                preview = gr.Image(label="识别预览", type="pil")
        gr.Markdown("## Mask 编辑器")
        editor = gr.HTML(label="Mask 编辑器")
        mask_data = gr.Textbox(label="", elem_id="mask-data", visible=True, container=False)
        gr.Markdown("## AI 修复引擎")
        with gr.Row():
            with gr.Column(elem_classes=["zf-card"]):
                gr.Markdown("### 🖥️ 本地 LaMa")
                local_status = gr.Markdown("● 已就绪")
                local_desc = gr.Markdown("本地 CPU · 免费")
                local_btn = gr.Button("选择本地", variant="primary")
            with gr.Column(elem_classes=["zf-card"]):
                gr.Markdown("### ✨ Gemini")
                gemini_status = gr.Markdown("○ 检测中…")
                gemini_desc = gr.Markdown("每次 1 次额度")
                gemini_btn = gr.Button("选择 Gemini")
            with gr.Column(elem_classes=["zf-card"]):
                gr.Markdown("### ◉ OpenAI")
                openai_status = gr.Markdown("○ 检测中…")
                openai_desc = gr.Markdown("每次 2 次额度")
                openai_btn = gr.Button("选择 OpenAI")
        selected = gr.State("local")
        selected_text = gr.Markdown("**当前引擎：本地 LaMa（免费）**")
        restore_btn = gr.Button("🚀 开始修复", variant="primary")
        result = gr.Image(label="修复结果", type="pil", format="png")

    # Same-page compression tool.
    with gr.Column(visible=False, elem_classes=["zf-tool"]) as compress_view:
        with gr.Row():
            gr.Markdown("# 📦 图片压缩")
            compress_back = gr.Button("← 返回工具首页", size="sm")
        gr.Markdown("快速压缩 JPG、PNG、WebP 图片。")
        with gr.Row():
            with gr.Column():
                compress_input = gr.Image(label="上传图片", type="pil")
                compress_quality = gr.Slider(10, 95, value=80, step=1, label="压缩质量（越低体积越小）")
                compress_format = gr.Radio(["保持原格式", "JPG", "PNG", "WebP"], value="保持原格式", label="输出格式")
                compress_btn = gr.Button("📦 开始压缩", variant="primary")
            with gr.Column():
                compress_output = gr.File(label="压缩结果")
                compress_info = gr.Markdown("上传图片后开始。")

    # Logged-in account area: recharge, usage, and admin remain available without leaving this page.
    with gr.Column(visible=False, elem_classes=["zf-tool"]) as account_panel:
        with gr.Tabs():
            with gr.Tab("💰 充值额度"):
                gr.Markdown("### 手动充值\n选择套餐提交申请，管理员确认收款后手动增加额度。")
                gr.Markdown(base.recharge.payment_instructions())
                recharge_choice = gr.Radio(base.recharge.package_choices(), label="充值套餐", value=base.recharge.package_choices()[0] if base.recharge.package_choices() else None)
                recharge_note = gr.Textbox(label="付款说明 / 流水号（可选）", max_lines=2)
                recharge_submit_btn = gr.Button("提交充值申请", variant="primary")
                recharge_message = gr.Markdown()
                recharge_table = gr.Dataframe(headers=["订单号","额度","金额","状态","付款说明","申请时间","审核时间"], interactive=False)
            with gr.Tab("📊 使用记录"):
                usage_table = gr.Dataframe(headers=["时间","引擎","额度","成功","详情"], interactive=False)
                refresh_usage_btn = gr.Button("刷新记录")
            with gr.Tab("⚙️ 管理后台", visible=False) as admin_tab:
                admin_stat = gr.Markdown()
                admin_refresh = gr.Button("刷新统计")
                gr.Markdown("### 用户管理")
                users_table = gr.Dataframe(headers=["ID","用户名","额度","状态","角色","注册时间","最后登录"], interactive=False)
                with gr.Row():
                    target_id = gr.Number(label="用户 ID", precision=0)
                    amount = gr.Number(label="额度", value=10, precision=0)
                    mode = gr.Radio(["增加","扣除"], value="增加", label="操作")
                    adjust_btn = gr.Button("执行")
                admin_message = gr.Markdown()
                gr.Markdown("### 使用记录")
                logs_table = gr.Dataframe(headers=["ID","用户","引擎","额度","状态","详情","时间"], interactive=False)

    # Header account actions.
    login_open.click(lambda: gr.update(visible=True), outputs=auth_panel)
    login_btn.click(do_login, [login_user, login_pass, login_totp], [user_id, session_token, csrf_token, auth_panel, logout_btn, auth_message, account_info, admin_tab])
    reg_btn.click(base.auth_register, [reg_user, reg_pass], [auth_message, login_user])
    logout_btn.click(do_logout, [session_token], [user_id, session_token, csrf_token, auth_panel, logout_btn, auth_message, account_info, admin_tab])
    logout_btn.click(lambda: gr.update(visible=False), outputs=account_bar)
    login_btn.click(lambda: gr.update(visible=True), outputs=account_bar)
    login_btn.click(lambda: gr.update(visible=True), outputs=account_panel)
    account_refresh.click(base.refresh_status, outputs=[])

    # Tool navigation without changing URL or opening a separate page.
    remove_open.click(show_remove, outputs=[home_view, remove_view, compress_view])
    compress_open.click(show_compress, outputs=[home_view, remove_view, compress_view])
    remove_back.click(show_home, outputs=[home_view, remove_view, compress_view])
    compress_back.click(show_home, outputs=[home_view, remove_view, compress_view])

    # Watermark events.
    source.change(base.reset_editor, source, [editor, mask_data])
    auto_btn.click(base.auto_detect, source, [preview, editor, status, mask_data], show_progress="minimal")
    clear_btn.click(base.reset_editor, source, [editor, mask_data])
    demo.load(base.refresh_status, [local_status, local_desc, gemini_status, gemini_desc, openai_status, openai_desc])
    local_btn.click(lambda: ("local", "**当前引擎：本地 LaMa（免费）**"), outputs=[selected, selected_text])
    gemini_btn.click(lambda: ("gemini", "**当前引擎：Gemini（1 次额度）**"), outputs=[selected, selected_text])
    openai_btn.click(lambda: ("openai", "**当前引擎：OpenAI（2 次额度）**"), outputs=[selected, selected_text])
    restore_btn.click(base.ai_restore, [selected, source, mask_data, user_id, session_token, csrf_token], result)

    # Compression events.
    compress_btn.click(compress_for_web, [compress_input, compress_quality, compress_format], [compress_output, compress_info])

    # Account events.
    refresh_usage_btn.click(base.usage_view, [user_id, session_token, csrf_token], usage_table)
    recharge_submit_btn.click(base.recharge_submit, [user_id, session_token, csrf_token, recharge_choice, recharge_note], [recharge_message, recharge_table])
    admin_refresh.click(base.admin_refresh_view, [user_id, session_token, csrf_token], [admin_stat, users_table, logs_table])
    adjust_btn.click(base.admin_adjust, [user_id, session_token, csrf_token, target_id, amount, mode], [admin_message, users_table, logs_table])


if __name__ == "__main__":
    demo.launch(server_name="0.0.0.0", server_port=int(os.getenv("PORT", "7860")), show_error=True)
