from __future__ import annotations

import os
import traceback
import numpy as np
import gradio as gr
from PIL import Image

from app import CSS, EDITOR_JS, _pil, auto_detect, decode_mask, reset_editor
from ai_provider_engine import RCImageEngine
from core.inpaint_engine import InpaintEngine


local_engine = InpaintEngine()


def _engine_status(provider: str) -> tuple[str, str]:
    if provider == "local":
        return "● 已就绪", "本地 LaMa · CPU · 无需 API"
    api_provider = "gemini" if provider == "gemini" else "openai"
    engine = RCImageEngine(api_provider)
    if not engine.api_key:
        return "○ 未配置", f"{engine.model} · 请设置 RC_API_KEY"
    if provider == "gemini":
        return "● 已配置", f"Gemini · {engine.model} · RCouyi API"
    return "● 已配置", f"OpenAI · {engine.model} · RCouyi API"


def refresh_status():
    local_s, local_d = _engine_status("local")
    gemini_s, gemini_d = _engine_status("gemini")
    openai_s, openai_d = _engine_status("openai")
    return local_s, local_d, gemini_s, gemini_d, openai_s, openai_d


def ai_restore(provider, image, mask_data):
    pil = _pil(image)
    if pil is None:
        raise gr.Error("请先上传图片。")

    mask = decode_mask(mask_data, pil.size)
    pixels = int(np.count_nonzero(mask))
    if pixels == 0:
        raise gr.Error("Mask 是空的：请先自动识别，或用画笔涂满需要修复的区域。")

    try:
        if provider == "local":
            mask_image = Image.fromarray(mask.astype(np.uint8), "L")
            return local_engine.run(pil, mask_image)

        engine = RCImageEngine(provider)
        if not engine.available:
            raise gr.Error(
                "没有配置第三方 API。请在 .env 设置 RC_API_KEY、RC_API_BASE_URL 和对应模型名，然后重启 Docker。"
            )
        return engine.repair(pil, mask)
    except gr.Error:
        raise
    except Exception as exc:
        traceback.print_exc()
        raise gr.Error(f"修复失败：{type(exc).__name__}: {exc}") from exc


CARD_CSS = """
.engine-card { border: 1px solid var(--border-color-primary); border-radius: 14px; padding: 16px; min-height: 145px; cursor: pointer; transition: .18s ease; }
.engine-card:hover { border-color: var(--primary-500); transform: translateY(-2px); }
.engine-card.selected { border: 2px solid var(--primary-500); box-shadow: 0 0 0 2px rgba(99,102,241,.10); }
.engine-icon { font-size: 28px; margin-bottom: 8px; }
.engine-name { font-size: 18px; font-weight: 700; }
.engine-status { margin-top: 10px; font-weight: 600; }
.engine-desc { font-size: 13px; opacity: .72; margin-top: 5px; }
"""

with gr.Blocks(
    title="AI 图片智能修复",
    theme=gr.themes.Soft(),
    css=CSS + CARD_CSS,
    head=EDITOR_JS,
) as demo:
    gr.Markdown("# AI 图片智能修复")
    gr.Markdown(
        "自动识别候选区域 + 手动画笔/橡皮擦 Mask。支持本地 CPU LaMa、Gemini 图像模型和 OpenAI 图像模型。"
    )

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
            local_desc = gr.Markdown("本地 CPU · 无需 API")
            local_btn = gr.Button("选择本地", variant="primary")
        with gr.Column(elem_classes=["engine-card"]):
            gr.Markdown("### ✨ Gemini")
            gemini_status = gr.Markdown("○ 检测中…")
            gemini_desc = gr.Markdown("Gemini 图像模型")
            gemini_btn = gr.Button("选择 Gemini")
        with gr.Column(elem_classes=["engine-card"]):
            gr.Markdown("### ◉ OpenAI")
            openai_status = gr.Markdown("○ 检测中…")
            openai_desc = gr.Markdown("OpenAI 图像模型")
            openai_btn = gr.Button("选择 OpenAI")

    selected = gr.State("local")
    selected_text = gr.Markdown("**当前引擎：本地 LaMa（CPU）**")
    restore_btn = gr.Button("🚀 开始修复", variant="primary", elem_id="restore-btn")
    result = gr.Image(label="修复结果", type="pil", format="png")

    gr.Markdown(
        "### API 配置\n"
        "本地 LaMa（CPU）无需 API。\n\n"
        "第三方接口默认：`https://api.rcouyi.com/v1`。\n\n"
        "`.env`：`RC_API_KEY`、`RC_GEMINI_MODEL`、`RC_OPENAI_MODEL`。"
    )

    source.change(reset_editor, inputs=source, outputs=[editor, mask_data])
    auto_btn.click(auto_detect, inputs=source, outputs=[preview, editor, status, mask_data], show_progress="minimal")
    clear_btn.click(reset_editor, inputs=source, outputs=[editor, mask_data])
    demo.load(refresh_status, outputs=[local_status, local_desc, gemini_status, gemini_desc, openai_status, openai_desc])
    local_btn.click(lambda: ("local", "**当前引擎：本地 LaMa（CPU）**"), outputs=[selected, selected_text])
    gemini_btn.click(lambda: ("gemini", "**当前引擎：Gemini**"), outputs=[selected, selected_text])
    openai_btn.click(lambda: ("openai", "**当前引擎：OpenAI**"), outputs=[selected, selected_text])
    restore_btn.click(ai_restore, inputs=[selected, source, mask_data], outputs=result)


if __name__ == "__main__":
    demo.launch(server_name="0.0.0.0", server_port=7860, show_error=True)
