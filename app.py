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


def _as_mask_array(value, size):
    """Convert PIL/NumPy/Gradio ImageEditor values to a binary grayscale mask."""
    if value is None:
        return np.zeros((size[1], size[0]), dtype=np.uint8)

    if isinstance(value, Image.Image):
        arr = np.asarray(value)
    elif isinstance(value, np.ndarray):
        arr = value
    else:
        return np.zeros((size[1], size[0]), dtype=np.uint8)

    if arr.ndim == 3:
        # For an RGBA drawing layer, alpha is the safest representation of
        # the painted area. For RGB/RGBA composites, use brightness when alpha
        # is fully opaque; this prevents a black background from becoming a mask.
        if arr.shape[2] >= 4:
            alpha = arr[:, :, 3].astype(np.uint8)
            rgb = arr[:, :, :3].astype(np.uint8)
            if np.any(rgb > 10):
                arr = np.max(rgb, axis=2)
            else:
                arr = alpha
        else:
            arr = np.max(arr[:, :, :3], axis=2)
    elif arr.ndim != 2:
        return np.zeros((size[1], size[0]), dtype=np.uint8)

    arr = np.asarray(arr, dtype=np.uint8)
    if arr.shape != (size[1], size[0]):
        arr = np.asarray(
            Image.fromarray(arr, mode="L").resize(size, Image.Resampling.NEAREST)
        )
    return arr


def to_mask_array(value, size):
    """Robustly extract painted pixels from all common Gradio ImageEditor formats."""
    if value is None:
        return np.zeros((size[1], size[0]), dtype=np.uint8)

    if isinstance(value, (Image.Image, np.ndarray)):
        return _as_mask_array(value, size)

    if not isinstance(value, dict):
        return np.zeros((size[1], size[0]), dtype=np.uint8)

    # IMPORTANT: ImageEditor stores manual strokes in `layers`. Prefer those
    # over `composite`, because composite can contain the black background.
    layers = value.get("layers") or []
    acc = np.zeros((size[1], size[0]), dtype=np.uint8)
    for layer in layers:
        layer_mask = _as_mask_array(layer, size)
        acc = np.maximum(acc, layer_mask)

    if np.any(acc > 30):
        return acc

    # Fallback for versions that flatten the editor into composite/background.
    for key in ("composite", "background"):
        candidate = value.get(key)
        if candidate is None:
            continue
        candidate_mask = _as_mask_array(candidate, size)
        if np.any(candidate_mask > 30):
            return candidate_mask

    return acc


def _editor_value(mask: Image.Image):
    """Create an ImageEditor value with an explicit black background and white mask layer."""
    black = Image.new("L", mask.size, 0)
    white_mask = mask.convert("L")
    return {
        "background": black,
        "layers": [white_mask],
        "composite": white_mask,
    }


def auto_detect(image):
    if image is None:
        return None, None, "请先上传图片。"
    pil = image if isinstance(image, Image.Image) else Image.fromarray(image)
    mask = detect_candidates(pil)
    preview = overlay_mask(pil, mask)
    mask_img = Image.fromarray(mask, mode="L")
    count = int(np.count_nonzero(mask))
    if count == 0:
        msg = "未找到高置信度候选区域。请直接在 Mask 编辑器中用画笔涂白需要去除的水印。"
    else:
        msg = f"已生成候选 Mask：{count:,} 个像素。请检查红色区域，可用画笔增加、橡皮擦删除。"
    return preview, _editor_value(mask_img), msg


def restore(image, mask_value):
    if image is None:
        raise gr.Error("请先上传图片。")

    pil = image if isinstance(image, Image.Image) else Image.fromarray(image)
    mask = to_mask_array(mask_value, pil.size)
    mask = (mask > 30).astype(np.uint8) * 255

    if not np.any(mask):
        raise gr.Error("没有检测到修复区域，请先点击「自动识别候选区域」，或在 Mask 编辑器中用白色画笔涂抹水印。")

    result = engine.run(pil, Image.fromarray(mask, mode="L"))
    return result


def clear_mask(image):
    if image is None:
        return None
    pil = image if isinstance(image, Image.Image) else Image.fromarray(image)
    return _editor_value(Image.new("L", pil.size, 0))


with gr.Blocks(title="AI 图片智能修复", theme=gr.themes.Soft()) as demo:
    gr.Markdown("# AI 图片智能修复\n自动识别候选区域 + 手动画笔/橡皮擦 Mask + CPU AI Inpainting")
    gr.Markdown("仅处理你拥有或获授权修改的图片。自动识别只是候选检测，请在修复前检查 Mask。")

    with gr.Row():
        with gr.Column():
            source = gr.Image(label="原图", type="pil")
            auto_btn = gr.Button("✨ 自动识别候选区域", variant="primary")
            status = gr.Markdown("上传图片后开始。")
        with gr.Column():
            preview = gr.Image(label="自动识别预览（红色=待修复）", type="pil")

    gr.Markdown("## Mask 编辑\n白色区域 = AI 要重建；黑色区域 = 保留。用画笔增加区域，用橡皮擦删除误识别区域。")
    mask_editor = gr.ImageEditor(
        label="Mask 编辑器",
        type="pil",
        image_mode="L",
        sources=[],
        brush=gr.Brush(colors=["#ffffff"], default_size=24, color_mode="fixed"),
        eraser=gr.Eraser(default_size=24),
        height=520,
    )

    with gr.Row():
        clear_btn = gr.Button("清除 Mask")
        restore_btn = gr.Button("🚀 AI 智能修复", variant="primary")

    result = gr.Image(label="修复结果", type="pil", format="png")

    auto_btn.click(auto_detect, inputs=source, outputs=[preview, mask_editor, status])
    clear_btn.click(clear_mask, inputs=source, outputs=mask_editor)
    restore_btn.click(restore, inputs=[source, mask_editor], outputs=result)


if __name__ == "__main__":
    demo.launch(server_name="0.0.0.0", server_port=7860, show_error=True)
