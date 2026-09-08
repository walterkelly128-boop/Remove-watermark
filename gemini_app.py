from __future__ import annotations

import traceback
import numpy as np
import gradio as gr
from PIL import Image

from app import (
    CSS,
    EDITOR_JS,
    _pil,
    auto_detect,
    build_editor,
    decode_mask,
    reset_editor,
)
from gemini_engine import GeminiInpaintEngine


gemini = GeminiInpaintEngine()


def gemini_restore(image, mask_data):
    pil = _pil(image)
    if pil is None:
        raise gr.Error("请先上传图片。")
    mask = decode_mask(mask_data, pil.size)
    pixels = int(np.count_nonzero(mask))
    if pixels == 0:
        raise gr.Error("Mask 是空的：请先自动识别，或在编辑器中用画笔涂满水印区域。")
    if not gemini.available:
        raise gr.Error("没有配置 GEMINI_API_KEY。请先设置环境变量后重启 Docker。")
    try:
        return gemini.repair(pil, mask, use_crop=True)
    except Exception as exc:
        traceback.print_exc()
        raise gr.Error(f"Gemini 修复失败：{type(exc).__name__}: {exc}") from exc


with gr.Blocks(
    title="Gemini AI 图片智能修复",
    theme=gr.themes.Soft(),
    css=CSS,
    head=EDITOR_JS,
) as demo:
    gr.Markdown("# Gemini AI 图片智能修复")
    gr.Markdown(
        "使用 Gemini 3.1 Flash Image（Nano Banana 2）进行局部 AI 重建。"
        "自动检测只是生成候选 Mask，最终由你确认需要修复的区域。"
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

    with gr.Row():
        restore_btn = gr.Button("🚀 Gemini AI 智能修复", variant="primary", elem_id="restore-btn")

    result = gr.Image(label="Gemini 修复结果", type="pil", format="png")

    gr.Markdown(
        "### 配置\n"
        "Docker 环境变量：`GEMINI_API_KEY=你的Key`。\n\n"
        "可通过 `GEMINI_IMAGE_MODEL` 切换模型，默认 `gemini-3.1-flash-image`。"
    )

    source.change(reset_editor, inputs=source, outputs=[editor, mask_data])
    auto_btn.click(auto_detect, inputs=source, outputs=[preview, editor, status, mask_data], show_progress="minimal")
    clear_btn.click(reset_editor, inputs=source, outputs=[editor, mask_data])
    restore_btn.click(gemini_restore, inputs=[source, mask_data], outputs=result)


if __name__ == "__main__":
    demo.launch(server_name="0.0.0.0", server_port=7860, show_error=True)
