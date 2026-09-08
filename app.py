from __future__ import annotations

from pathlib import Path
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
    """Create a real Gradio EditorValue: original background + editable mask layer."""
    background = image.convert("RGB")
    alpha = Image.fromarray(mask.astype(np.uint8), mode="L")

    # White pixels are the repair mask; transparent pixels reveal the original image.
    layer = Image.new("RGBA", background.size, (255, 255, 255, 0))
    layer.putalpha(alpha)
    composite = Image.alpha_composite(background.convert("RGBA"), layer)

    return {
        "background": background,
        "layers": [layer],
        "composite": composite,
    }


def to_mask_array(value, size):
    """Read the current editable layer(s), never the original background."""
    if value is None:
        return np.zeros((size[1], size[0]), dtype=np.uint8)

    if isinstance(value, dict):
        layers = value.get("layers") or []
        mask = np.zeros((size[1], size[0]), dtype=np.uint8)
        for layer in layers:
            mask = np.maximum(mask, _alpha_mask(layer, size))
        return mask

    # Backward-compatible fallback for a direct image value.
    return _alpha_mask(value, size)


def auto_detect(image):
    if image is None:
        return None, None, "请先上传图片。"

    pil = image if isinstance(image, Image.Image) else Image.fromarray(image)
    mask = detect_candidates(pil)
    preview = overlay_mask(pil, mask)
    count = int(np.count_nonzero(mask))

    if count == 0:
        msg = "未找到高置信度候选区域。可以直接用画笔增加需要修复的区域。"
    else:
        msg = (
            f"已自动识别 {count:,} 个 Mask 像素。"
            "现在编辑器中的底图就是原图，白色区域就是自动检测结果；"
            "可用画笔增加、橡皮擦删除。"
        )

    # IMPORTANT: return the EditorValue itself, not gr.update().
    return preview, _editor_value(pil, mask), msg


def reset_editor(image):
    if image is None:
        return None
    pil = image if isinstance(image, Image.Image) else Image.fromarray(image)
    empty = np.zeros((pil.height, pil.width), dtype=np.uint8)
    return _editor_value(pil, empty)


def restore(image, editor_value):
    if image is None:
        raise gr.Error("请先上传图片。")

    pil = image if isinstance(image, Image.Image) else Image.fromarray(image)
    mask = to_mask_array(editor_value, pil.size)

    if not np.any(mask):
        raise gr.Error("没有修复区域，请先点击自动识别或用画笔涂抹。")

    result = engine.run(pil, Image.fromarray(mask, mode="L"))
    return result


with gr.Blocks(title="AI 图片智能修复", theme=gr.themes.Soft()) as demo:
    gr.Markdown("# AI 图片智能修复\n自动识别候选区域 + 手动精修 Mask + CPU AI Inpainting")
    gr.Markdown(
        "上传一次原图即可。点击自动识别后，Mask 编辑器会直接显示原图和自动检测区域。"
        "白色 = AI 修复区域；橡皮擦 = 取消误选。"
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
        layers=gr.LayerOptions(
            allow_additional_layers=False,
            layers=["自动检测"],
        ),
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

    # Uploading/changing the source initializes the editor with that same image.
    source.change(reset_editor, inputs=source, outputs=mask_editor)
    auto_btn.click(auto_detect, inputs=source, outputs=[preview, mask_editor, status])
    clear_btn.click(reset_editor, inputs=source, outputs=mask_editor)
    restore_btn.click(restore, inputs=[source, mask_editor], outputs=result)


if __name__ == "__main__":
    demo.launch(server_name="0.0.0.0", server_port=7860, show_error=True)
