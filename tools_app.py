from __future__ import annotations

import os
import tempfile
from io import BytesIO

import gradio as gr

import gemini_app as base
from image_compress import compress_image, format_size


def _original_size(image):
    """Return the size of the original upload as accurately as Gradio allows."""
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
        info = (
            f"**原图：** {format_size(original_bytes)}　　"
            f"**压缩后：** {format_size(len(data))}　　"
            f"**体积减少：** {ratio:.1f}%"
        )
        return path, info
    except Exception as exc:
        raise gr.Error("压缩失败，请更换图片或参数后重试。") from exc


def refresh_account(uid, token, csrf):
    user, message = base.require_user(uid, token, csrf)
    return base.account_text(user) if user else ""


# Reuse the existing authenticated watermark application so account, credit,
# recharge and admin data stay in one SQLite database and one process.
with base.demo:
    with gr.Tab("📦 图片压缩"):
        gr.Markdown(
            "# 📦 图片压缩\n"
            "快速压缩 JPG、PNG、WebP 图片。图片仅用于当前处理，不写入业务数据库。"
        )
        with gr.Row():
            with gr.Column():
                compress_input = gr.Image(label="上传图片", type="pil")
                compress_quality = gr.Slider(
                    10, 95, value=80, step=1, label="压缩质量（越低体积越小）"
                )
                compress_format = gr.Radio(
                    ["保持原格式", "JPG", "PNG", "WebP"],
                    value="保持原格式",
                    label="输出格式",
                )
                compress_btn = gr.Button("📦 开始压缩", variant="primary")
            with gr.Column():
                compress_output = gr.File(label="压缩结果")
                compress_info = gr.Markdown("上传图片后开始。")
        compress_btn.click(
            compress_for_web,
            inputs=[compress_input, compress_quality, compress_format],
            outputs=[compress_output, compress_info],
        )

    # Keep the displayed balance current after a recharge is approved without
    # forcing the user to log out and log in again.
    recharge_refresh = getattr(base, "recharge_refresh_btn", None)
    account_info = getattr(base, "account_info", None)
    if recharge_refresh is not None and account_info is not None:
        recharge_refresh.click(
            refresh_account,
            inputs=[base.user_id, base.session_token, base.csrf_token],
            outputs=[account_info],
        )

if __name__ == "__main__":
    base.demo.launch(
        server_name="0.0.0.0",
        server_port=int(os.getenv("PORT", "7860")),
        show_error=False,
    )
