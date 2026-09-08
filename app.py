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


def _to_array(value):
    if isinstance(value, Image.Image):
        return np.asarray(value)
    if isinstance(value, np.ndarray):
        return value
    return None


def _alpha_mask(layer, size):
    arr = _to_array(layer)
    if arr is None:
        return np.zeros((size[1], size[0]), dtype=np.uint8)

    if arr.ndim == 3 and arr.shape[2] >= 4:
        mask = arr[:, :, 3]
    elif arr.ndim == 3:
        mask = np.max(arr[:, :, :3], axis=2)
    elif arr.ndim == 2:
        mask = arr
    else:
        return np.zeros((size[1], size[0]), dtype=np.uint8)

    mask = np.asarray(mask, dtype=np.uint8)
    if mask.shape != (size[1], size[0]):
        mask = np.asarray(
            Image.fromarray(mask, mode="L").resize(size, Image.Resampling.NEAREST)
        )
    return np.where(mask > 30, 255, 0).astype(np.uint8)


def _editor_value(image: Image.Image, mask: np.ndarray):
    """Create an EditorValue with the original image and one editable mask layer."""
    background = image.convert("RGB")
    alpha = Image.fromarray(mask.astype(np.uint8), mode="L")

    layer = Image.new("RGBA", background.size, (255, 255, 255, 0))
    layer.putalpha(alpha)
    composite = Image.alpha_composite(background.convert("RGBA"), layer)

    return {
        "background": background,
        "layers": [layer],
        "composite": composite,
    }


def to_mask_array(value, size):
    """Read only editable layers; never treat the original background as a mask."""
    if value is None:
        return np.zeros((size[1], size[0]), dtype=np.uint8)

    if isinstance(value, dict):
        layers = value.get("layers") or []
        mask = np.zeros((size[1], size[0]), dtype=np.uint8)
        for layer in layers:
            mask = np.maximum(mask, _alpha_mask(layer, size))
        return mask

    return _alpha_mask(value, size)


def auto_detect(image):
    """Run detector and initialize the editor with the detected mask."""
    if image is None:
        return None, None, "⚠️ 请先上传图片。"

    try:
        pil = image if isinstance(image, Image.Image) else Image.fromarray(image)
        pil = pil.convert("RGB")

        # Keep detector input predictable and always return a valid EditorValue.
        mask = detect_candidates(pil)
        mask = np.asarray(mask, dtype=np.uint8)
        if mask.shape != (pil.height, pil.width):
            mask = np.asarray(
                Image.fromarray(mask, mode="L").resize(
                    pil.size, Image.Resampling.NEAREST
                )
            )
        mask = np.where(mask > 30, 255, 0).astype(np.uint8)

        preview = overlay_mask(pil, mask)
        count = int(np.count_nonzero(mask))
        editor_value = _editor_value(pil, mask)

        if count == 0:
            msg = (
                "⚠️ 自动识别没有找到候选区域。"
                "你仍然可以在 Mask 编辑器中用画笔手动添加区域。"
            )
        else:
            msg = (
                f"✅ 自动识别完成：{count:,} 个 Mask 像素。"
                "红色/白色区域就是当前待修复区域，可继续画笔增加或使用橡皮擦删除误选。"
            )

        return preview, editor_value, msg

    except Exception as exc:
        # Never let a detector exception make the button appear to do nothing.
        print("[auto_detect] ERROR:")
        traceback.print_exc()
        return None, None, f"❌ 自动识别失败：{type(exc).__name__}: {exc}"


def reset_editor(image):
    if image is None:
        return None
    pil = image if isinstance(image, Image.Image) else Image.fromarray(image)
    pil = pil.convert("RGB")
    empty = np.zeros((pil.height, pil.width), dtype=np.uint8)
    return _editor_value(pil, empty)


def restore(image, editor_value):
    if image is None:
        raise gr.Error("请先上传图片。")

    pil = image if isinstance(image, Image.Image) else Image.fromarray(image)
    pil = pil.convert("RGB")
    mask = to_mask_array(editor_value, pil.size)

    if not np.any(mask):
        raise gr.Error("没有修复区域，请先点击自动识别或用画笔涂抹。")

    return engine.run(pil, Image.fromarray(mask, mode="L"))


with gr.Blocks(title="AI 图片智能修复", theme=gr.themes.Soft()) as demo:
    gr.Markdown(
        "# AI 图片智能修复\n"
        "自动识别候选区域 + 手动精修 Mask + CPU AI Inpainting"
    )
    gr.Markdown(
        "上传一次原图即可。点击自动识别后，Mask 编辑器会直接显示原图和检测结果。"
        "画笔增加区域，橡皮擦删除误选。"
    )

    with gr.Row():
        with gr.Column():
            source = gr.Image(label="原图", type="pil")
            auto_btn = gr.Button("✨ 自动识别候选区域", variant="primary")
            status = gr.Markdown("上传图片后开始。")
        with gr.Column():
            preview = gr.Image(label="自动识别预览（红色=待修复）", type="pil")

    gr.Markdown("## Mask 编辑器")
    mask_editor = gr.ImageEditor(
        label="Mask 编辑器",
        value=None,
        type="pil",
        image_mode="RGBA",
        sources=[],
        # Keep a single editable layer. Avoid LayerOptions so this works reliably
        # with the pinned Gradio 5.44.1 environment.
        layers=False,
        brush=gr.Brush(
            colors=["#ffffff"],
            default_size=24,
            color_mode="fixed",
        ),
        eraser=gr.Eraser(default_size=24),
        height=520,
    )

    with gr.Row():
        clear_btn = gr.Button("清除全部 Mask")
        restore_btn = gr.Button("🚀 AI 智能修复", variant="primary")

    result = gr.Image(label="修复结果", type="pil", format="png")

    source.change(reset_editor, inputs=source, outputs=mask_editor)
    auto_btn.click(
        auto_detect,
        inputs=source,
        outputs=[preview, mask_editor, status],
        show_progress="minimal",
    )
    clear_btn.click(reset_editor, inputs=source, outputs=mask_editor)
    restore_btn.click(restore, inputs=[source, mask_editor], outputs=result)


if __name__ == "__main__":
    demo.launch(server_name="0.0.0.0", server_port=7860, show_error=True)
