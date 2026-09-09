from __future__ import annotations

import os
import tempfile

import gradio as gr

import gemini_app as base
from image_compress import compress_image, format_size


def compress_for_web(image, quality, output_format):
    if image is None:
        raise gr.Error("请先上传图片。")
    try:
        data, fmt = compress_image(image, quality, output_format)
        original_bytes = 0
        if getattr(image, "filename", None) and os.path.exists(image.filename):
            original_bytes = os.path.getsize(image.filename)
        if not original_bytes:
            # Gradio supplies a PIL image; PNG encoding gives a stable comparison fallback.
            from io import BytesIO
            probe = BytesIO()
            image.save(probe, format="PNG")
            original_bytes = len(probe.getvalue())
        suffix = ".jpg" if fmt == "JPEG" else ".webp" if fmt == "WEBP" else ".png"
        fd, path = tempfile.mkstemp(prefix="zolfox-compress-", suffix=suffix)
        with os.fdopen(fd, "wb") as f:
            f.write(data)
        ratio = max(0.0, (1 - len(data) / original_bytes) * 100) if original_bytes else 0.0
        info = f"**原图：** {format_size(original_bytes)}　　**压缩后：** {format_size(len(data))}　　**体积减少：** {ratio:.1f}%"
        return path, info
    except Exception as exc:
        raise gr.Error(f"压缩失败：{type(exc).__name__}: {exc}") from exc


# Reuse the existing authenticated watermark application and add the first new ZOLFOX tool.
# This keeps the account/credit/admin system in one process and one Docker image.
with base.demo:
    with gr.Tab("📦 图片压缩"):
        gr.Markdown("# 📦 图片压缩\n快速压缩 JPG、PNG、WebP 图片。处理完成后只保留临时结果，不写入业务数据库。")
        with gr.Row():
            with gr.Column():
                compress_input = gr.Image(label="上传图片", type="pil")
                compress_quality = gr.Slider(10, 95, value=80, step=1, label="压缩质量")
                compress_format = gr.Radio(["保持原格式", "JPG", "PNG", "WebP"], value="保持原格式", label="输出格式")
                compress_btn = gr.Button("📦 开始压缩", variant="primary")
            with gr.Column():
                compress_output = gr.File(label="压缩结果")
                compress_info = gr.Markdown("上传图片后开始。")
        compress_btn.click(
            compress_for_web,
            inputs=[compress_input, compress_quality, compress_format],
            outputs=[compress_output, compress_info],
        )

if __name__ == "__main__":
    base.demo.launch(server_name="0.0.0.0", server_port=int(os.getenv("PORT", "7860")), show_error=False)
