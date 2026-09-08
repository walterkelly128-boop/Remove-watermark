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
    """Extract a binary mask from PIL/NumPy editor layers."""
    if value is None:
        return np.zeros((size[1], size[0]), dtype=np.uint8)

    if isinstance(value, Image.Image):
        arr = np.asarray(value)
    elif isinstance(value, np.ndarray):
        arr = value
    else:
        return np.zeros((size[1], size[0]), dtype=np.uint8)

    if arr.ndim == 3:
        if arr.shape[2] >= 4:
            alpha = arr[:, :, 3].astype(np.uint8)
            rgb = arr[:, :, :3].astype(np.uint8)
            # A transparent brush layer should be interpreted from alpha.
            arr = alpha if np.any(alpha < 250) or not np.any(rgb > 10) else np.max(rgb, axis=2)
        else:
            arr = np.max(arr[:, :, :3], axis=2)
    elif arr.ndim != 2:
        return np.zeros((size[1], size[0]), dtype=np.uint8)

    arr = np.asarray(arr, dtype=np.uint8)
    if arr.shape != (size[1], size[0]):
        arr = np.asarray(Image.fromarray(arr, mode="L").resize(size, Image.Resampling.NEAREST))
    return arr


def _layer_mask(layer, size):
    """Return only painted pixels from one ImageEditor layer."""
    if layer is None:
        return np.zeros((size[1], size[0]), dtype=np.uint8)

    if isinstance(layer, Image.Image):
        arr = np.asarray(layer)
    elif isinstance(layer, np.ndarray):
        arr = layer
    else:
        return _as_mask_array(layer, size)

    if arr.ndim == 3 and arr.shape[2] >= 4:
        # Gradio's layer alpha is authoritative: erased pixels become transparent.
        alpha = arr[:, :, 3].astype(np.uint8)
        mask = alpha
    else:
        mask = _as_mask_array(arr, size)

    if mask.shape != (size[1], size[0]):
        mask = np.asarray(Image.fromarray(mask, mode="L").resize(size, Image.Resampling.NEAREST))
    return (mask > 30).astype(np.uint8) * 255


def to_mask_array(value, size):
    """Extract the current visible mask without resurrecting erased pixels."""
    if value is None:
        return np.zeros((size[1], size[0]), dtype=np.uint8)

    if isinstance(value, (Image.Image, np.ndarray)):
        return (_as_mask_array(value, size) > 30).astype(np.uint8) * 255

    if not isinstance(value, dict):
        return np.zeros((size[1], size[0]), dtype=np.uint8)

    layers = value.get("layers") or []
    acc = np.zeros((size[1], size[0]), dtype=np.uint8)
    for layer in layers:
        # Eraser edits the layer transparency. Rebuild from the current layer,
        # not by preserving pixels from an earlier composite.
        acc = np.maximum(acc, _layer_mask(layer, size))

    if np.any(acc):
        return acc

    # Only use composite as a last fallback. Never use background because the
    # original image must not become a repair mask.
    composite = value.get("composite")
    if composite is not None:
        return (_as_mask_array(composite, size) > 30).astype(np.uint8) * 255
    return acc


def _editor_value(mask: Image.Image, background: Image.Image | None = None):
    """Provide the original image as editor background plus an editable white mask layer."""
    mask = mask.convert("L")
    alpha = np.asarray(mask, dtype=np.uint8)
    layer = Image.new("RGBA", mask.size, (255, 255, 255, 0))
    layer.putalpha(Image.fromarray(alpha, mode="L"))
    background = (background.convert("RGB") if background is not None else Image.new("RGB", mask.size, (0, 0, 0)))
    # Composite must include the background. Some Gradio ImageEditor versions\n    # render the editor from composite rather than background + layers.\n    composite = Image.alpha_composite(background.convert("RGBA"), layer)\n    return {\n        "background": background,\n        "layers": [layer],\n        "composite": composite,\n    }


def auto_detect(image):
    if image is None:
        return None, None, "请先上传图片。"
    pil = image if isinstance(image, Image.Image) else Image.fromarray(image)
    mask = detect_candidates(pil)
    preview = overlay_mask(pil, mask)
    mask_img = Image.fromarray(mask, mode="L")
    count = int(np.count_nonzero(mask))
    if count == 0:
        msg = "未找到高置信度候选区域。请直接在 Mask 编辑器中用画笔涂白需要去除的区域。"
    else:
        msg = f"已生成候选 Mask：{count:,} 个像素。白色区域可继续增加，橡皮擦可以直接取消误选。"
    return preview, gr.update(value=_editor_value(mask_img, pil)), msg


def restore(image, mask_value):
    if image is None:
        raise gr.Error("请先上传图片。")

    pil = image if isinstance(image, Image.Image) else Image.fromarray(image)
    mask = to_mask_array(mask_value, pil.size)

    if not np.any(mask):
        raise gr.Error("没有检测到修复区域，请先自动识别或在 Mask 编辑器中用画笔涂抹。")

    result = engine.run(pil, Image.fromarray(mask, mode="L"))
    return result


def clear_mask(image):
    if image is None:
        return None
    pil = image if isinstance(image, Image.Image) else Image.fromarray(image)
    return gr.update(value=_editor_value(Image.new("L", pil.size, 0), pil))


with gr.Blocks(title="AI 图片智能修复", theme=gr.themes.Soft()) as demo:
    gr.Markdown("# AI 图片智能修复\n自动识别候选区域 + 可撤销 Mask 编辑 + CPU AI Inpainting")
    gr.Markdown("仅处理你拥有或获授权修改的图片。白色 = 修复，透明/黑色 = 保留。")

    with gr.Row():
        with gr.Column():
            source = gr.Image(label="原图", type="pil")
            auto_btn = gr.Button("✨ 自动识别候选区域", variant="primary")
            status = gr.Markdown("上传图片后开始。")
        with gr.Column():
            preview = gr.Image(label="自动识别预览（红色=待修复）", type="pil")

    gr.Markdown("## Mask 编辑\n白色区域 = AI 要重建；擦除后区域会立即恢复为保留状态。编辑器的撤销按钮也可以撤销上一步。")
    mask_editor = gr.ImageEditor(
        label="Mask 编辑器（原图底图 + 白色修复区域）",
        type="pil",
        image_mode="RGBA",
        sources=[],
        brush=gr.Brush(colors=["#ffffff"], default_size=24, color_mode="fixed"),
        eraser=gr.Eraser(default_size=24),
        height=520,
    )

    with gr.Row():
        clear_btn = gr.Button("清除全部 Mask")
        restore_btn = gr.Button("🚀 AI 智能修复", variant="primary")

    result = gr.Image(label="修复结果", type="pil", format="png")

    source.change(clear_mask, inputs=source, outputs=mask_editor)
    auto_btn.click(auto_detect, inputs=source, outputs=[preview, mask_editor, status])
    clear_btn.click(clear_mask, inputs=source, outputs=mask_editor)
    restore_btn.click(restore, inputs=[source, mask_editor], outputs=result)


if __name__ == "__main__":
    demo.launch(server_name="0.0.0.0", server_port=7860, show_error=True)
