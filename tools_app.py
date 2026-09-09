from __future__ import annotations

import os
import tempfile
from io import BytesIO

import gradio as gr
from fastapi import FastAPI
from fastapi.responses import HTMLResponse

import gemini_app as base
from image_compress import compress_image, format_size


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


HOME_HTML = r'''<!doctype html>
<html lang="zh-CN">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>ZOLFOX Tools - 在线工具箱</title>
<style>
:root{--bg:#f6f8fc;--card:#fff;--text:#172033;--muted:#667085;--line:#e7ebf3;--brand:#5b5bd6;--brand2:#7c5cff;--shadow:0 18px 50px rgba(26,35,68,.08)}
*{box-sizing:border-box}body{margin:0;background:var(--bg);color:var(--text);font-family:-apple-system,BlinkMacSystemFont,"Segoe UI","PingFang SC","Microsoft YaHei",sans-serif}
a{text-decoration:none;color:inherit}.nav{height:72px;background:rgba(255,255,255,.88);backdrop-filter:blur(14px);border-bottom:1px solid var(--line);display:flex;align-items:center}.navin{width:min(1180px,92%);margin:auto;display:flex;align-items:center;justify-content:space-between}.logo{font-size:22px;font-weight:800;letter-spacing:-.6px}.logo span{color:var(--brand)}.links{display:flex;gap:28px;color:#667085;font-size:14px}.links a:hover{color:var(--text)}.login{border:1px solid var(--line);padding:9px 17px;border-radius:10px;background:#fff;font-weight:600}
.hero{width:min(1180px,92%);margin:0 auto;padding:86px 0 56px;text-align:center}.eyebrow{display:inline-flex;padding:7px 13px;border-radius:999px;background:#eeedff;color:#5148c9;font-size:13px;font-weight:700}.hero h1{font-size:58px;line-height:1.08;letter-spacing:-2.8px;margin:20px 0 18px}.hero h1 em{font-style:normal;background:linear-gradient(90deg,var(--brand),var(--brand2));-webkit-background-clip:text;background-clip:text;color:transparent}.hero p{font-size:18px;color:var(--muted);margin:0 auto;max-width:650px;line-height:1.8}.search{margin:32px auto 0;max-width:650px;background:#fff;border:1px solid var(--line);box-shadow:var(--shadow);border-radius:15px;padding:5px;display:flex}.search input{flex:1;border:0;outline:0;padding:14px 16px;font-size:15px;background:transparent}.search button{border:0;border-radius:11px;background:#171a2b;color:white;padding:0 22px;font-weight:700}
.section{width:min(1180px,92%);margin:0 auto;padding:20px 0 72px}.section-head{display:flex;justify-content:space-between;align-items:end;margin-bottom:20px}.section h2{font-size:26px;margin:0 0 7px}.section-head p{margin:0;color:var(--muted);font-size:14px}.grid{display:grid;grid-template-columns:repeat(2,minmax(0,1fr));gap:18px}.card{background:var(--card);border:1px solid var(--line);border-radius:20px;padding:28px;min-height:215px;box-shadow:0 8px 28px rgba(26,35,68,.04);transition:.2s}.card:hover{transform:translateY(-4px);box-shadow:var(--shadow);border-color:#d9d7ff}.icon{width:52px;height:52px;border-radius:15px;background:#f0efff;display:grid;place-items:center;font-size:25px}.card h3{margin:19px 0 8px;font-size:20px}.card p{color:var(--muted);font-size:14px;line-height:1.7;margin:0 0 20px}.use{color:var(--brand);font-weight:700;font-size:14px}.soon{opacity:.72}.soon .icon{background:#f1f3f7}.footer{border-top:1px solid var(--line);padding:30px 0;color:#8a93a6;font-size:13px;text-align:center}
@media(max-width:700px){.links{display:none}.hero{padding:58px 0 38px}.hero h1{font-size:42px}.grid{grid-template-columns:1fr}.hero p{font-size:16px}}
</style>
</head>
<body>
<nav class="nav"><div class="navin"><a class="logo" href="/"><span>ZOLFOX</span> Tools</a><div class="links"><a href="#image">图片工具</a><a href="#pdf">PDF 工具</a><a href="#ai">AI 工具</a><a href="#more">更多工具</a></div><a class="login" href="/remove-watermark">开始使用</a></div></nav>
<section class="hero"><div class="eyebrow">ZOLFOX Tools · 在线工具箱</div><h1>简单、快速、实用的<br><em>在线工具</em></h1><p>图片、PDF、AI 与更多常用工具，持续更新中。无需安装软件，打开浏览器即可使用。</p><div class="search"><input placeholder="搜索你需要的工具……" aria-label="搜索工具"><button>搜索</button></div></section>
<section class="section" id="image"><div class="section-head"><div><h2>图片工具</h2><p>处理日常图片任务，简单直接</p></div></div><div class="grid">
<a class="card" href="/remove-watermark"><div class="icon">✨</div><h3>图片去水印</h3><p>自动识别或手动标记需要修复的区域，智能重建图片内容。</p><div class="use">立即使用 →</div></a>
<a class="card" href="/image-compress"><div class="icon">📦</div><h3>图片压缩</h3><p>快速压缩 JPG、PNG、WebP 图片，在尽量保持画质的同时减小文件体积。</p><div class="use">立即使用 →</div></a>
</div></section>
<section class="section" id="pdf"><div class="section-head"><div><h2>PDF 工具</h2><p>常用 PDF 工具正在准备中</p></div></div><div class="grid"><div class="card soon"><div class="icon">📄</div><h3>PDF 转 Word</h3><p>将 PDF 文档转换为可编辑的 Word 文件。</p><div class="use">即将上线</div></div><div class="card soon"><div class="icon">🧩</div><h3>PDF 合并</h3><p>多个 PDF 快速合并成一个文件。</p><div class="use">即将上线</div></div></div></section>
<section class="section" id="ai"><div class="section-head"><div><h2>AI 工具</h2><p>更多智能工具陆续加入</p></div></div><div class="grid"><div class="card soon"><div class="icon">🤖</div><h3>AI 图片增强</h3><p>提升图片清晰度与细节表现。</p><div class="use">即将上线</div></div><div class="card soon"><div class="icon">🪄</div><h3>AI 图片编辑</h3><p>使用自然语言完成图片编辑任务。</p><div class="use">即将上线</div></div></div></section>
<footer class="footer">© ZOLFOX Tools · 专注实用的在线工具</footer>
</body></html>'''


compression_demo = gr.Blocks(
    title="图片压缩 - ZOLFOX Tools",
    theme=gr.themes.Soft(),
    css="""
    .tool-wrap{max-width:1100px;margin:auto}.tool-title{text-align:center;padding:24px 0 10px}.tool-title h1{font-size:34px}.tool-title p{color:#667085}
    .back-link{display:inline-block;margin:8px 0;color:#5b5bd6;text-decoration:none;font-weight:600}
    """,
) 
with compression_demo:
    gr.HTML('<div class="tool-wrap"><a class="back-link" href="/">← 返回 ZOLFOX Tools</a><div class="tool-title"><h1>📦 图片压缩</h1><p>快速压缩 JPG、PNG、WebP 图片，减少文件体积。</p></div></div>')
    with gr.Row():
        with gr.Column():
            compress_input = gr.Image(label="上传图片", type="pil")
            compress_quality = gr.Slider(10, 95, value=80, step=1, label="压缩质量（越低体积越小）")
            compress_format = gr.Radio(["保持原格式", "JPG", "PNG", "WebP"], value="保持原格式", label="输出格式")
            compress_btn = gr.Button("📦 开始压缩", variant="primary")
        with gr.Column():
            compress_output = gr.File(label="压缩结果")
            compress_info = gr.Markdown("上传图片后开始。")
    compress_btn.click(compress_for_web, inputs=[compress_input, compress_quality, compress_format], outputs=[compress_output, compress_info])

    recharge_refresh = getattr(base, "recharge_refresh_btn", None)
    account_info = getattr(base, "account_info", None)
    if recharge_refresh is not None and account_info is not None:
        recharge_refresh.click(refresh_account, inputs=[base.user_id, base.session_token, base.csrf_token], outputs=[account_info])


web_app = FastAPI(title="ZOLFOX Tools")

@web_app.get("/", response_class=HTMLResponse)
def home():
    return HOME_HTML

# The existing watermark application remains the full-featured authenticated tool.
web_app = gr.mount_gradio_app(web_app, base.demo, path="/remove-watermark")
web_app = gr.mount_gradio_app(web_app, compression_demo, path="/image-compress")


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(web_app, host="0.0.0.0", port=int(os.getenv("PORT", "7860")))
