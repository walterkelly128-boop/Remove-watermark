from __future__ import annotations

import traceback
import numpy as np
import gradio as gr

from app import CSS, EDITOR_JS, _pil, auto_detect, decode_mask, reset_editor
from ai_provider_engine import RCImageEngine
from core.inpaint_engine import InpaintEngine


local_engine = InpaintEngine()


def ai_restore(provider, image, mask_data):
    pil = _pil(image)
    if pil is None:
        raise gr.Error("请先上传图片。")

    mask = decode_mask(mask_data, pil.size)
    pixels = int(np.count_nonzero(mask))
    if pixels == 0:
        raise gr.Error("Mask 是空的：请先自动识别，或用画笔涂满需要修复的区域。")

    try:
        if provider == "本地 LaMa（CPU）":
            return local_engine.run(pil, __import__("PIL").Image.fromarray(mask.astype("uint8"), "L"))

        api_provider = "gemini" if provider == "Gemini" else "openai"
        engine = RCImageEngine(api_provider)
        if not engine.available:
            raise gr.Error(
                "没有配置第三方 API。请在 .env 设置 RC_API_KEY、RC_API_BASE_URL 和对应模型名，然后重启 Docker。"
            )
        return engine.repair(pil, mask)
    except gr.Error:
        raise
    except Exception as exc:
        traceback.print_exc()
        raise gr.Error(f"{provider} 修复失败：{type(exc).__name__}: {exc}") from exc


with gr.Blocks(
    title="AI 图片智能修复",
    theme=gr.themes.Soft(),
    css=CSS,
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

    provider = gr.Dropdown(
        choices=["本地 LaMa（CPU）", "Gemini", "OpenAI"],
        value="本地 LaMa（CPU）",
        label="AI 修复引擎",
        info="本地无需 API；Gemini / OpenAI 通过 RCouyi OpenAI-compatible API 调用。",
    )

    restore_btn = gr.Button("🚀 开始修复", variant="primary", elem_id="restore-btn")
    result = gr.Image(label="修复结果", type="pil", format="png")

    gr.Markdown(
        "### API 配置\n"
        "本地 LaMa（CPU）无需 API。\n\n"
        "第三方接口默认：`https://api.rcouyi.com/v1`。\n\n"
        "`.env` 示例：\n"
        "`RC_API_KEY=你的第三方API Key`\n\n"
        "`RC_API_BASE_URL=https://api.rcouyi.com/v1`\n\n"
        "`RC_GEMINI_MODEL=你的Gemini图像模型名`\n\n"
        "`RC_OPENAI_MODEL=你的OpenAI图像模型名`\n\n"
        "程序不会把 Key 放到浏览器端。模型名称以第三方平台实际提供的模型为准。"
    )

    source.change(reset_editor, inputs=source, outputs=[editor, mask_data])
    auto_btn.click(auto_detect, inputs=source, outputs=[preview, editor, status, mask_data], show_progress="minimal")
    clear_btn.click(reset_editor, inputs=source, outputs=[editor, mask_data])
    restore_btn.click(ai_restore, inputs=[provider, source, mask_data], outputs=result)


if __name__ == "__main__":
    demo.launch(server_name="0.0.0.0", server_port=7860, show_error=True)
