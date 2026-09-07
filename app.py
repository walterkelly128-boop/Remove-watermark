from __future__ import annotations

import os
from pathlib import Path
import numpy as np
import gradio as gr
from PIL import Image, ImageOps

from core.detector import detect_candidates, overlay_mask
from core.inpaint_engine import InpaintEngine

APP_ROOT = Path(__file__).resolve().parent
DATA_DIR = APP_ROOT / "data" / "output"
DATA_DIR.mkdir(parents=True, exist_ok=True)

engine = InpaintEngine()


def to_mask_array(value, size):
    """Convert an ImageEditor value or image into a grayscale mask."""
    if value is None:
        return np.zeros((size[1], size[0]), dtype=np.uint8)
    if isinstance(value, Image.Image):
        return np.array(value.convert("L").resize(size, Image.Resampling.NEAREST))
    if isinstance(value, dict):
        # Gradio ImageEditor generally exposes composite/background/layers.
        candidate = value.get("composite") or value.get("background")
        if candidate is not None:
            if isinstance(candidate, np.ndarray):
                arr = candidate
                if arr.ndim == 3 and arr.shape[2] >= 4:
                    arr = arr[:, :, 3]
                elif arr.ndim == 3:
                    arr = np.max(arr[:, :, :3], axis=2)
                return np.array(Image.fromarray(arr.astype(np.uint8)).convert("L").resize(size, Image.Resampling.NEAREST))
            if isinstance(candidate, Image.Image):
                return np.array(candidate.convert("L").resize(size, Image.Resampling.NEAREST))
        layers = value.get("layers") or []
        if layers:
            acc = Image.new("L", size, 0)
            for layer in layers:
                if isinstance(layer, np.ndarray):
                    layer_img = Image.fromarray(layer.astype(np.uint8))
                elif isinstance(layer, Image.Image):
                    layer_img = layer
                else:
                    continue
                if layer_img.mode in ("RGBA", "LA"):
                    alpha = layer_img.getchannel("A")
                else:
                    alpha = layer_img.convert("L")
                alpha = alpha.resize(size, Image.Resampling.NEAREST)
                acc = Image.fromarray(np.maximum(np.array(acc), np.array(alpha)).astype(np.uint8))
            return np.array(acc)
    return np.zeros((size[1], size[0]), dtype=np.uint8)


def auto_detect(image):
    if image is None:
        return None, None, "请先上传图片。"
    pil = image if isinstance(image, Image.Image) else Image.fromarray(image)
    mask = detect_candidates(pil)
    preview = overlay_mask(pil, mask)
    mask_img = Image.fromarray(mask).convert("L")
    count = int((mask > 0).sum())
    if count == 0:
        msg = "未找到高置信度候选区域。可以使用下方画笔手动补充 Mask。"
    else:
        msg = f"已生成候选 Mask：{count:,} 个像素。请检查红色区域，确认无误后进行 AI 修复。"
    return preview, mask_img, msg


def make_editor_value(mask):
    if mask is None:
        return None
    m = mask.convert("L")
    # White = target area. Display as grayscale so users can paint/edit it.
    return m


def restore(image, mask_value):
    if image is None:
        raise gr.Error("请先上传图片。")
    pil = image if isinstance(image, Image.Image) else Image.fromarray(image)
    mask = to_mask_array(mask_value, pil.size)
    # Normalize to a clean binary mask with a small safety dilation.
    mask = (mask > 30).astype(np.uint8) * 255
    if int(mask.sum()) == 0:
        raise gr.Error("没有检测到修复区域，请先自动识别或在 Mask 编辑器中涂抹。")
    mask_img = Image.fromarray(mask, mode="L")
    result = engine.run(pil, mask_img)
    return result


def auto_to_editor(image):
    preview, mask, msg = auto_detect(image)
    return preview, make_editor_value(mask), msg


def clear_mask(image):
    if image is None:
        return None
    pil = image if isinstance(image, Image.Image) else Image.fromarray(image)
    return Image.new("L", pil.size, 0)

with gr.Blocks(title="AI 图片智能修复", theme=gr.themes.Soft()) as demo:
    gr.Markdown("# AI 图片智能修复\n自动识别候选区域 + 手动补充/擦除 Mask + CPU AI Inpainting")
    gr.Markdown("仅处理你拥有或获授权修改的图片。自动识别是候选检测，请在修复前检查 Mask。")

    with gr.Row():
        with gr.Column():
            source = gr.Image(label="原图", type="pil")
            auto_btn = gr.Button("✨ 自动识别候选区域", variant="primary")
            status = gr.Markdown("上传图片后开始。")

        with gr.Column():
            preview = gr.Image(label="自动识别预览（红色=待修复）", type="pil")

    gr.Markdown("## Mask 编辑\n白色区域 = AI 要重建的区域；黑色区域 = 保留。可用画笔增加区域，用橡皮擦删除误识别区域。")
    mask_editor = gr.ImageEditor(
        label="Mask 编辑器",
        type="pil",
        image_mode="L",
        sources=[],
        brush=gr.Brush(colors=["#ffffff"], default_size=24),
        eraser=gr.Eraser(default_size=24),
        height=520,
    )

    with gr.Row():
        clear_btn = gr.Button("清除 Mask")
        restore_btn = gr.Button("🚀 AI 智能修复", variant="primary")

    result = gr.Image(label="修复结果", type="pil", format="png")

    auto_btn.click(auto_to_editor, inputs=source, outputs=[preview, mask_editor, status])
    clear_btn.click(clear_mask, inputs=source, outputs=mask_editor)
    restore_btn.click(restore, inputs=[source, mask_editor], outputs=result)

if __name__ == "__main__":
    demo.launch(server_name="0.0.0.0", server_port=7860, show_error=True)
