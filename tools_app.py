from __future__ import annotations

import os
import tempfile
from io import BytesIO

import gradio as gr
from fastapi import FastAPI
from gradio import mount_gradio_app

import gemini_app as base
from image_compress import compress_image, format_size

SITE_CSS = r"""
.gradio-container{max-width:1180px!important;margin:auto!important}
.zf-header{border-bottom:1px solid #e8ebf2;padding:14px 0;margin-bottom:18px}
.zf-logo{font-size:25px;font-weight:850;letter-spacing:-.7px;color:#151a2b}.zf-logo span{color:#5b5bd6}
.zf-nav{color:#667085;font-size:14px}.zf-account{border:1px solid #e4e7ec;border-radius:12px;padding:8px 14px;background:#fff}
.zf-member{width:100%;max-width:430px;margin:0 0 18px auto;border:1px solid #e4e7ec;border-radius:18px;padding:18px;background:#fff;box-shadow:0 16px 40px rgba(16,24,40,.10)}
.zf-member-title{font-size:18px;font-weight:800;margin-bottom:10px}.zf-balance{font-size:28px;font-weight:850;color:#5148c9;margin:8px 0 16px}.zf-history{font-size:13px;color:#667085;line-height:1.8}
.zf-hero{text-align:center;padding:70px 20px 56px;border-radius:28px;background:linear-gradient(180deg,#f6f7ff,#fff);border:1px solid #eceeff;margin-bottom:30px}
.zf-hero h1{font-size:52px!important;line-height:1.08!important;letter-spacing:-2.5px;margin:16px 0!important}.zf-hero p{font-size:18px;color:#667085;line-height:1.8}
.zf-pill{display:inline-block;padding:7px 13px;border-radius:999px;background:#eeedff;color:#5148c9;font-weight:700;font-size:13px}
.zf-card{border:1px solid #e4e7ec;border-radius:20px;padding:24px;min-height:190px;background:#fff;transition:.2s}.zf-card:hover{box-shadow:0 14px 36px rgba(16,24,40,.08);transform:translateY(-2px)}
.zf-card h3{margin:12px 0 8px;font-size:21px}.zf-card p{color:#667085;line-height:1.7}.zf-tool{border:1px solid #e4e7ec;border-radius:22px;padding:24px;margin-top:20px;background:#fff}
.zf-muted{color:#667085}.zf-link button{min-height:42px}
.zf-inner-page{width:100%;max-width:none!important;margin:0!important;padding:8px 3vw 48px!important;box-sizing:border-box}
.zf-inner-page .zf-tool,.zf-inner-page .engine-card{width:100%;box-sizing:border-box}
@media(max-width:700px){.zf-hero h1{font-size:39px!important}.zf-nav{display:none}.zf-inner-page{padding-left:14px!important;padding-right:14px!important}.zf-member{max-width:none}}
"""

INNER_CSS = SITE_CSS + r"""
.gradio-container{max-width:none!important;width:100%!important;margin:0!important;padding-left:0!important;padding-right:0!important}
body{overflow-x:hidden}
"""

BROWSER_STATE_SECRET = os.getenv("BROWSER_STATE_SECRET", "zolfox-browser-session-v1")
BROWSER_STATE_KEY = "zolfox_shared_session"


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


def _session_blob(token, csrf):
    if not token or not csrf:
        return ""
    return f"{token}|{csrf}"


def _member_markdown(user_id):
    if not user_id:
        return ""
    user = base.accounts.get_user(user_id)
    if not user:
        return ""
    lines = [
        f"### 👤 {user['username']}",
        f"<div class='zf-balance'>{int(user['credits'])} 积分</div>",
        "当前可用积分",
        "",
        "### 使用记录",
    ]
    rows = base.accounts.usage_for_user(user_id, 20)
    if not rows:
        lines.append("暂无使用记录")
    else:
        for row in rows:
            provider = {"gemini": "Gemini", "openai": "OpenAI", "local": "本地 LaMa"}.get(row["provider"], row["provider"])
            amount = int(row["credits"] or 0)
            cost = f"-{amount} 积分" if amount else "免费"
            status = "成功" if row["success"] else "失败"
            when = str(row["created_at"]).replace("T", " ")[:16]
            lines.append(f"- **{provider}**　{cost}　{status}　`{when}`")
    return "\n".join(lines)


def toggle_member(user_id, opened):
    if not user_id:
        return gr.update(visible=False), False, gr.update(visible=True)
    new_open = not bool(opened)
    return gr.update(visible=new_open), new_open, gr.update(visible=False)


def restore_session(browser_session):
    try:
        blob = (browser_session or "").strip()
        if not blob or "|" not in blob:
            raise ValueError("empty session")
        token, csrf = blob.split("|", 1)
        session, _ = base.accounts.validate_session(token, csrf)
        if not session:
            raise ValueError("invalid session")
        uid = int(session["user_id"])
        name = session["username"]
        return (
            uid, token, csrf,
            gr.update(visible=False), gr.update(visible=True), "",
            bool(session["is_admin"]), gr.update(value=name, visible=True),
            browser_session, gr.update(visible=False), False,
            gr.update(value=_member_markdown(uid)),
        )
    except Exception:
        return (
            None, None, None,
            gr.update(visible=False), gr.update(visible=False), "",
            False, gr.update(value="登录 / 注册", visible=True), "",
            gr.update(visible=False), False, gr.update(value=""),
        )


def do_login(username, password, totp, request: gr.Request = None):
    try:
        uid, token, csrf, auth_view, logout_view, _app_view, msg, account, admin = base.auth_login(username, password, totp, request)
        success = bool(uid and token and csrf)
        if success:
            return (
                uid, token, csrf,
                gr.update(visible=False), gr.update(visible=True), msg,
                bool(admin), gr.update(value=(username or "").strip(), visible=True),
                _session_blob(token, csrf), gr.update(visible=False), False,
                gr.update(value=_member_markdown(uid)),
            )
        return (
            None, None, None,
            gr.update(visible=True), gr.update(visible=False), msg,
            False, gr.update(value="登录 / 注册", visible=True), "",
            gr.update(visible=False), False, gr.update(value=""),
        )
    except Exception as exc:
        return (
            None, None, None,
            gr.update(visible=True), gr.update(visible=False), f"❌ 登录失败：{type(exc).__name__}: {exc}",
            False, gr.update(value="登录 / 注册", visible=True), "",
            gr.update(visible=False), False, gr.update(value=""),
        )


def do_logout(token):
    base.logout(token)
    return (
        None, None, None,
        gr.update(visible=False), gr.update(visible=False), "",
        False, gr.update(value="登录 / 注册", visible=True), "",
        gr.update(visible=False), False, gr.update(value=""),
    )


def add_header():
    user_id = gr.State(None)
    session_token = gr.State(None)
    csrf_token = gr.State(None)
    admin_state = gr.State(False)
    member_opened = gr.State(False)
    browser_session = gr.BrowserState("", storage_key=BROWSER_STATE_KEY, secret=BROWSER_STATE_SECRET)

    with gr.Row(elem_classes=["zf-header"]):
        gr.HTML('<div class="zf-logo"><span>ZOLFOX</span> Tools</div>')
        gr.Markdown("图片工具　　PDF 工具　　AI 工具　　更多工具", elem_classes=["zf-nav"])
        with gr.Column(scale=0, min_width=150):
            login_open = gr.Button("登录 / 注册", size="sm")

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

    with gr.Column(visible=False, elem_classes=["zf-member"]) as member_panel:
        member_info = gr.Markdown()
        logout_btn = gr.Button("退出登录")

    login_open.click(
        toggle_member,
        [user_id, member_opened],
        [member_panel, member_opened, auth_panel],
    )
    login_btn.click(
        do_login,
        [login_user, login_pass, login_totp],
        [user_id, session_token, csrf_token, auth_panel, login_open, auth_message, admin_state, login_open, browser_session, member_panel, member_opened, member_info],
    )
    reg_btn.click(base.auth_register, [reg_user, reg_pass], [auth_message, login_user])
    logout_btn.click(
        do_logout,
        [session_token],
        [user_id, session_token, csrf_token, auth_panel, login_open, auth_message, admin_state, login_open, browser_session, member_panel, member_opened, member_info],
    )
    gr.on(
        inputs=[browser_session],
        outputs=[user_id, session_token, csrf_token, auth_panel, login_open, auth_message, admin_state, login_open, browser_session, member_panel, member_opened, member_info],
        fn=restore_session,
    )
    return user_id, session_token, csrf_token


def home_demo():
    with gr.Blocks(title="ZOLFOX Tools · 在线工具箱", theme=gr.themes.Soft(), css=SITE_CSS) as demo:
        add_header()
        gr.HTML('<div class="zf-hero"><div class="zf-pill">ZOLFOX Tools · 在线工具箱</div><h1>简单、快速、实用的<br><span style="color:#5b5bd6">在线工具</span></h1><p>图片、PDF、AI 与更多常用工具，持续更新中。<br>无需安装软件，打开浏览器即可使用。</p></div>')
        gr.Markdown("## 图片工具")
        with gr.Row():
            with gr.Column(elem_classes=["zf-card"]):
                gr.Markdown("### ✨ 图片去水印\n自动识别或手动指定需要修复的区域，使用 AI 自然重建图片内容。")
                gr.HTML('<a class="zf-link" href="/remove-watermark/"><button>立即使用 →</button></a>')
            with gr.Column(elem_classes=["zf-card"]):
                gr.Markdown("### 📦 图片压缩\n快速压缩 JPG、PNG、WebP 图片，在尽量保持画质的同时减小文件体积。")
                gr.HTML('<a class="zf-link" href="/image-compress/"><button>立即使用 →</button></a>')
        gr.Markdown("## 更多工具")
        with gr.Row():
            for title, desc in [("📄 PDF 转 Word", "将 PDF 文档转换为可编辑文件。"), ("🧩 PDF 合并", "多个 PDF 快速合并。"), ("🤖 AI 图片增强", "提升图片清晰度与细节。")]:
                with gr.Column(elem_classes=["zf-card"]):
                    gr.Markdown(f"### {title}\n{desc}\n\n**即将上线**")
    return demo


def remove_demo():
    with gr.Blocks(title="图片去水印 · ZOLFOX Tools", theme=gr.themes.Soft(), css=INNER_CSS + base.CSS, head=base.EDITOR_JS) as demo:
        with gr.Column(elem_classes=["zf-inner-page"]):
            user_id, session_token, csrf_token = add_header()
            with gr.Row():
                gr.Markdown("# 🖼️ 图片去水印")
                gr.HTML('<a href="/"><button>← 工具首页</button></a>')
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
                    gr.Markdown("### 🖥️ 本地 LaMa\n本地 CPU · 免费")
                    local_status = gr.Markdown("● 已就绪")
                    local_desc = gr.Markdown("本地 CPU · 免费")
                    local_btn = gr.Button("选择本地", variant="primary")
                with gr.Column(elem_classes=["zf-card"]):
                    gr.Markdown("### ✨ Gemini\n每次 1 积分")
                    gemini_status = gr.Markdown("○ 检测中…")
                    gemini_desc = gr.Markdown("每次 1 积分")
                    gemini_btn = gr.Button("选择 Gemini")
                with gr.Column(elem_classes=["zf-card"]):
                    gr.Markdown("### ◉ OpenAI\n每次 3 积分")
                    openai_status = gr.Markdown("○ 检测中…")
                    openai_desc = gr.Markdown("每次 3 积分")
                    openai_btn = gr.Button("选择 OpenAI")
            selected = gr.State("local")
            selected_text = gr.Markdown("**当前引擎：本地 LaMa（免费）**")
            restore_btn = gr.Button("🚀 开始修复", variant="primary", elem_id="restore-btn")
            result = gr.Image(label="修复结果", type="pil", format="png")
            source.change(base.reset_editor, source, [editor, mask_data])
            auto_btn.click(base.auto_detect, source, [preview, editor, status, mask_data], show_progress="minimal")
            clear_btn.click(base.reset_editor, source, [editor, mask_data])
            demo.load(base.refresh_status, inputs=[], outputs=[local_status, local_desc, gemini_status, gemini_desc, openai_status, openai_desc])
            local_btn.click(lambda: ("local", "**当前引擎：本地 LaMa（免费）**"), outputs=[selected, selected_text])
            gemini_btn.click(lambda: ("gemini", "**当前引擎：Gemini（1 积分）**"), outputs=[selected, selected_text])
            openai_btn.click(lambda: ("openai", "**当前引擎：OpenAI（3 积分）**"), outputs=[selected, selected_text])
            restore_btn.click(base.ai_restore, [selected, source, mask_data, user_id, session_token, csrf_token], result)
    return demo


def compress_demo():
    with gr.Blocks(title="图片压缩 · ZOLFOX Tools", theme=gr.themes.Soft(), css=INNER_CSS) as demo:
        with gr.Column(elem_classes=["zf-inner-page"]):
            add_header()
            with gr.Row():
                gr.Markdown("# 📦 图片压缩")
                gr.HTML('<a href="/"><button>← 工具首页</button></a>')
            gr.Markdown("快速压缩 JPG、PNG、WebP 图片，在尽量保持画质的同时减小文件体积。")
            with gr.Row():
                with gr.Column(elem_classes=["zf-tool"]):
                    compress_input = gr.Image(label="上传图片", type="pil")
                    compress_quality = gr.Slider(10, 95, value=80, step=1, label="压缩质量（越低体积越小）")
                    compress_format = gr.Radio(["保持原格式", "JPG", "PNG", "WebP"], value="保持原格式", label="输出格式")
                    compress_btn = gr.Button("📦 开始压缩", variant="primary")
                with gr.Column(elem_classes=["zf-tool"]):
                    compress_output = gr.File(label="压缩结果")
                    compress_info = gr.Markdown("上传图片后开始。")
            compress_btn.click(compress_for_web, [compress_input, compress_quality, compress_format], [compress_output, compress_info])
    return demo


home = home_demo()
remove_watermark = remove_demo()
image_compress = compress_demo()
app = FastAPI(title="ZOLFOX Tools")
app = mount_gradio_app(app, remove_watermark, path="/remove-watermark")
app = mount_gradio_app(app, image_compress, path="/image-compress")
app = mount_gradio_app(app, home, path="/")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=int(os.getenv("PORT", "7860")))
