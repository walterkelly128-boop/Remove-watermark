from __future__ import annotations

from pathlib import Path
import traceback

import numpy as np
import gradio as gr
from PIL import Image

from core.detector import detect_candidates, overlay_mask
from core.inpaint_engine import InpaintEngine

APP_ROOT = Path(__file__).resolve().parent
DATA_DIR = APP_ROOT / "data" / "output"
DATA_DIR.mkdir(parents=True, exist_ok=True)

engine = InpaintEngine()


def _pil(value):
    if isinstance(value, Image.Image):
        return value.convert("RGB")
    if isinstance(value, np.ndarray):
        return Image.fromarray(value).convert("RGB")
    return None


def _mask_layer(mask: np.ndarray) -> Image.Image:
    mask = np.where(np.asarray(mask) > 30, 255, 0).astype(np.uint8)
    rgba = np.zeros((mask.shape[0], mask.shape[1], 4), dtype=np.uint8)
    rgba[:, :, :3] = 255
    rgba[:, :, 3] = mask
    return Image.fromarray(rgba, "RGBA")


def _extract_mask(value, size):
    if value is None:
        return np.zeros((size[1], size[0]), dtype=np.uint8)
    if isinstance(value, dict):
        layers = value.get("layers") or []
        result = np.zeros((size[1], size[0]), dtype=np.uint8)
        for layer in layers:
            arr = np.asarray(layer)
            if arr.ndim == 3 and arr.shape[2] >= 4:
                arr = arr[:, :, 3]
            elif arr.ndim == 3:
                arr = np.max(arr[:, :, :3], axis=2)
            if arr.shape != result.shape:
                arr = np.asarray(Image.fromarray(arr.astype(np.uint8), "L").resize(size, Image.Resampling.NEAREST))
            result = np.maximum(result, arr.astype(np.uint8))
        return np.where(result > 30, 255, 0).astype(np.uint8)
    arr = np.asarray(value)
    if arr.ndim == 3 and arr.shape[2] >= 4:
        arr = arr[:, :, 3]
    elif arr.ndim == 3:
        arr = np.max(arr[:, :, :3], axis=2)
    if arr.shape != (size[1], size[0]):
        arr = np.asarray(Image.fromarray(arr.astype(np.uint8), "L").resize(size, Image.Resampling.NEAREST))
    return np.where(arr > 30, 255, 0).astype(np.uint8)


def _editor_value(image: Image.Image, mask: np.ndarray):
    background = image.convert("RGB")
    layer = _mask_layer(mask)
    composite = Image.alpha_composite(background.convert("RGBA"), layer)
    return {"background": background, "layers": [layer], "composite": composite}


def auto_detect(image):
    if image is None:
        return None, None, "⚠️ 请先上传图片。"
    try:
        pil = _pil(image)
        mask = np.asarray(detect_candidates(pil), dtype=np.uint8)
        if mask.shape != (pil.height, pil.width):
            mask = np.asarray(Image.fromarray(mask, "L").resize(pil.size, Image.Resampling.NEAREST))
        mask = np.where(mask > 30, 255, 0).astype(np.uint8)
        count = int(np.count_nonzero(mask))
        return overlay_mask(pil, mask), _editor_value(pil, mask), f"✅ 自动识别完成：{count:,} 个 Mask 像素。"
    except Exception as exc:
        traceback.print_exc()
        return None, None, f"❌ 自动识别失败：{type(exc).__name__}: {exc}"


def reset_editor(image):
    pil = _pil(image)
    if pil is None:
        return None
    return _editor_value(pil, np.zeros((pil.height, pil.width), dtype=np.uint8))


def restore(image, editor_value):
    pil = _pil(image)
    if pil is None:
        raise gr.Error("请先上传图片。")
    mask = _extract_mask(editor_value, pil.size)
    if not np.any(mask):
        raise gr.Error("没有修复区域，请先自动识别或用画笔添加 Mask。")
    return engine.run(pil, Image.fromarray(mask, "L"))


with gr.Blocks(title="AI 图片智能修复", theme=gr.themes.Soft()) as demo:
    gr.Markdown("# AI 图片智能修复\n自动识别候选区域 + 手动 Mask 精修 + CPU AI Inpainting")
    gr.Markdown("自动识别后直接编辑 Mask。画笔增加区域，橡皮擦删除区域。")

    with gr.Row():
        with gr.Column():
            source = gr.Image(label="原图", type="pil")
            with gr.Row():
                auto_btn = gr.Button("✨ 自动识别候选区域", variant="primary")
                clear_btn = gr.Button("清除 Mask")
            status = gr.Markdown("上传图片后开始。")
        with gr.Column():
            preview = gr.Image(label="自动识别预览", type="pil")

    gr.Markdown("## Mask 编辑器")
    mask_editor = gr.ImageEditor(
        label="Mask 编辑器（画笔增加 / 橡皮擦删除）",
        value=None,
        type="pil",
        image_mode="RGBA",
        sources=[],
        layers=False,
        brush=gr.Brush(colors=["#ffffff"], default_size=32, color_mode="fixed"),
        eraser=gr.Eraser(default_size=32),
        height=600,
    )

    with gr.Row():
        restore_btn = gr.Button("🚀 AI 智能修复", variant="primary", scale=2)
    result = gr.Image(label="修复结果", type="pil", format="png")

    source.change(reset_editor, inputs=source, outputs=mask_editor)
    auto_btn.click(auto_detect, inputs=source, outputs=[preview, mask_editor, status], show_progress="minimal")
    clear_btn.click(reset_editor, inputs=source, outputs=mask_editor)
    restore_btn.click(restore, inputs=[source, mask_editor], outputs=result)

if __name__ == "__main__":
    demo.launch(server_name="0.0.0.0", server_port=7860, show_error=True)
