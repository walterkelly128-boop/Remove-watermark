from __future__ import annotations

import base64
import io
import json
from pathlib import Path

import gradio as gr
import numpy as np
from PIL import Image

from core.detector import detect_candidates, overlay_mask
from core.inpaint_engine import InpaintEngine

APP_ROOT = Path(__file__).resolve().parent
DATA_DIR = APP_ROOT / "data" / "output"
DATA_DIR.mkdir(parents=True, exist_ok=True)

engine = InpaintEngine()


def _image_data_uri(image: Image.Image) -> str:
    image = image.convert("RGB")
    buf = io.BytesIO()
    image.save(buf, format="PNG", optimize=True)
    encoded = base64.b64encode(buf.getvalue()).decode("ascii")
    return f"data:image/png;base64,{encoded}"


def _mask_data_uri(mask: np.ndarray) -> str:
    mask = np.asarray(mask, dtype=np.uint8)
    image = Image.fromarray(mask, mode="L")
    buf = io.BytesIO()
    image.save(buf, format="PNG", optimize=True)
    encoded = base64.b64encode(buf.getvalue()).decode("ascii")
    return f"data:image/png;base64,{encoded}"


def _empty_mask(image: Image.Image) -> np.ndarray:
    return np.zeros((image.height, image.width), dtype=np.uint8)


def _payload(image: Image.Image, mask: np.ndarray) -> str:
    return json.dumps(
        {
            "image": _image_data_uri(image),
            "mask": _mask_data_uri(mask),
        },
        separators=(",", ":"),
    )


def _decode_mask(payload: str | None, size: tuple[int, int]) -> np.ndarray:
    if not payload:
        return np.zeros((size[1], size[0]), dtype=np.uint8)

    try:
        data = json.loads(payload)
        uri = data.get("mask", "")
        if "," not in uri:
            raise ValueError("invalid mask data")
        raw = base64.b64decode(uri.split(",", 1)[1])
        image = Image.open(io.BytesIO(raw)).convert("L")
        image = image.resize(size, Image.Resampling.NEAREST)
        arr = np.asarray(image, dtype=np.uint8)
        return np.where(arr > 30, 255, 0).astype(np.uint8)
    except Exception as exc:
        raise gr.Error(f"Mask 数据无效：{exc}") from exc


EDITOR_HTML = r'''
<div id="mask-editor-root" class="mask-editor-root">
  <div class="mask-toolbar">
    <button type="button" id="mask-brush" class="mask-tool active">🖌 画笔</button>
    <button type="button" id="mask-eraser" class="mask-tool">🧽 橡皮擦</button>
    <label class="mask-size-label">大小 <input id="mask-size" type="range" min="4" max="160" value="32" step="2"><span id="mask-size-value">32</span> px</label>
    <button type="button" id="mask-undo" class="mask-tool">↶ 撤销</button>
    <button type="button" id="mask-clear" class="mask-tool danger">清除 Mask</button>
  </div>
  <div class="mask-help">自动识别区域显示为红色。画笔增加区域，橡皮擦直接删除误选区域；右键也可临时使用橡皮擦。</div>
  <div id="mask-canvas-box" class="mask-canvas-box">
    <canvas id="mask-canvas"></canvas>
  </div>
  <div id="mask-empty" class="mask-empty">请先上传图片。</div>
</div>
'''

EDITOR_JS = r'''
(() => {
  const ROOT = "#mask-editor-root";
  const PAYLOAD = "#mask-payload textarea";
  let image = new Image();
  let imageReady = false;
  let maskCanvas = document.createElement("canvas");
  let maskCtx = maskCanvas.getContext("2d");
  let viewCanvas = null;
  let viewCtx = null;
  let overlayCanvas = document.createElement("canvas");
  let overlayCtx = overlayCanvas.getContext("2d");
  let tool = "brush";
  let size = 32;
  let drawing = false;
  let rightButton = false;
  let history = [];
  let lastPayload = "";
  let initialized = false;

  function payloadEl() {
    return document.querySelector(PAYLOAD);
  }

  function rootEl() {
    return document.querySelector(ROOT);
  }

  function ensureInit() {
    const root = rootEl();
    if (!root) return false;
    if (initialized && viewCanvas && viewCanvas.isConnected) return true;

    viewCanvas = root.querySelector("#mask-canvas");
    viewCtx = viewCanvas.getContext("2d");
    root.querySelector("#mask-brush").addEventListener("click", () => setTool("brush"));
    root.querySelector("#mask-eraser").addEventListener("click", () => setTool("eraser"));
    root.querySelector("#mask-undo").addEventListener("click", undo);
    root.querySelector("#mask-clear").addEventListener("click", clearMask);

    const slider = root.querySelector("#mask-size");
    const value = root.querySelector("#mask-size-value");
    slider.addEventListener("input", () => {
      size = Number(slider.value);
      value.textContent = String(size);
    });

    viewCanvas.addEventListener("pointerdown", startDraw);
    viewCanvas.addEventListener("pointermove", draw);
    viewCanvas.addEventListener("pointerup", endDraw);
    viewCanvas.addEventListener("pointercancel", endDraw);
    viewCanvas.addEventListener("contextmenu", (event) => event.preventDefault());

    initialized = true;
    return true;
  }

  function setTool(next) {
    tool = next;
    const root = rootEl();
    if (!root) return;
    root.querySelector("#mask-brush").classList.toggle("active", tool === "brush");
    root.querySelector("#mask-eraser").classList.toggle("active", tool === "eraser");
    viewCanvas.style.cursor = tool === "eraser" ? "cell" : "crosshair";
  }

  function snapshot() {
    if (!maskCanvas.width || !maskCanvas.height) return;
    try {
      history.push(maskCtx.getImageData(0, 0, maskCanvas.width, maskCanvas.height));
      if (history.length > 20) history.shift();
    } catch (_) {}
  }

  function undo() {
    if (!history.length) return;
    const state = history.pop();
    maskCtx.putImageData(state, 0, 0);
    render();
    sendPayload();
  }

  function clearMask() {
    if (!maskCanvas.width) return;
    snapshot();
    maskCtx.clearRect(0, 0, maskCanvas.width, maskCanvas.height);
    render();
    sendPayload();
  }

  function canvasPoint(event) {
    const rect = viewCanvas.getBoundingClientRect();
    return {
      x: (event.clientX - rect.left) * (viewCanvas.width / rect.width),
      y: (event.clientY - rect.top) * (viewCanvas.height / rect.height),
    };
  }

  function startDraw(event) {
    if (!imageReady) return;
    event.preventDefault();
    drawing = true;
    rightButton = event.button === 2;
    viewCanvas.setPointerCapture(event.pointerId);
    snapshot();
    const p = canvasPoint(event);
    paintAt(p.x, p.y, true);
  }

  function draw(event) {
    if (!drawing) return;
    event.preventDefault();
    const p = canvasPoint(event);
    paintAt(p.x, p.y, false);
  }

  function endDraw(event) {
    if (!drawing) return;
    drawing = false;
    try { viewCanvas.releasePointerCapture(event.pointerId); } catch (_) {}
    sendPayload();
  }

  function paintAt(x, y, dot) {
    const erase = rightButton || tool === "eraser";
    maskCtx.save();
    maskCtx.globalCompositeOperation = erase ? "destination-out" : "source-over";
    maskCtx.strokeStyle = "rgba(255,255,255,1)";
    maskCtx.fillStyle = "rgba(255,255,255,1)";
    maskCtx.lineWidth = size;
    maskCtx.lineCap = "round";
    maskCtx.lineJoin = "round";
    if (dot) {
      maskCtx.beginPath();
      maskCtx.arc(x, y, size / 2, 0, Math.PI * 2);
      maskCtx.fill();
    } else {
      if (!paintAt.last) paintAt.last = {x, y};
      maskCtx.beginPath();
      maskCtx.moveTo(paintAt.last.x, paintAt.last.y);
      maskCtx.lineTo(x, y);
      maskCtx.stroke();
    }
    paintAt.last = {x, y};
    maskCtx.restore();
    render();
  }

  function render() {
    if (!viewCanvas || !imageReady) return;
    viewCtx.clearRect(0, 0, viewCanvas.width, viewCanvas.height);
    viewCtx.drawImage(image, 0, 0, viewCanvas.width, viewCanvas.height);

    overlayCanvas.width = viewCanvas.width;
    overlayCanvas.height = viewCanvas.height;
    overlayCtx.clearRect(0, 0, overlayCanvas.width, overlayCanvas.height);
    overlayCtx.fillStyle = "rgba(255, 45, 45, 0.55)";
    overlayCtx.fillRect(0, 0, overlayCanvas.width, overlayCanvas.height);
    overlayCtx.globalCompositeOperation = "destination-in";
    overlayCtx.drawImage(maskCanvas, 0, 0, overlayCanvas.width, overlayCanvas.height);
    overlayCtx.globalCompositeOperation = "source-over";
    viewCtx.drawImage(overlayCanvas, 0, 0);
  }

  function loadPayload(raw) {
    if (!raw || raw === lastPayload) return;
    let data;
    try { data = JSON.parse(raw); } catch (_) { return; }
    if (!data.image) return;
    lastPayload = raw;

    const root = rootEl();
    if (!root) return;
    root.querySelector("#mask-empty").style.display = "none";

    imageReady = false;
    image = new Image();
    image.onload = () => {
      viewCanvas.width = image.naturalWidth;
      viewCanvas.height = image.naturalHeight;
      maskCanvas.width = image.naturalWidth;
      maskCanvas.height = image.naturalHeight;
      overlayCanvas.width = image.naturalWidth;
      overlayCanvas.height = image.naturalHeight;
      maskCtx.clearRect(0, 0, maskCanvas.width, maskCanvas.height);
      if (data.mask) {
        const maskImage = new Image();
        maskImage.onload = () => {
          maskCtx.clearRect(0, 0, maskCanvas.width, maskCanvas.height);
          maskCtx.drawImage(maskImage, 0, 0, maskCanvas.width, maskCanvas.height);
          imageReady = true;
          history = [];
          paintAt.last = null;
          render();
        };
        maskImage.src = data.mask;
      } else {
        imageReady = true;
        history = [];
        render();
      }
    };
    image.src = data.image;
  }

  function sendPayload() {
    const el = payloadEl();
    if (!el || !imageReady) return;
    const payload = JSON.stringify({
      image: image.src,
      mask: maskCanvas.toDataURL("image/png")
    });
    lastPayload = payload;
    el.value = payload;
    el.dispatchEvent(new Event("input", {bubbles: true}));
    el.dispatchEvent(new Event("change", {bubbles: true}));
  }

  function poll() {
    ensureInit();
    const el = payloadEl();
    if (el && el.value) loadPayload(el.value);
  }

  const observer = new MutationObserver(() => ensureInit());
  observer.observe(document.documentElement, {childList: true, subtree: true});
  setInterval(poll, 250);
  setTimeout(poll, 300);
})();
'''

EDITOR_CSS = r'''
.mask-editor-root { width: 100%; border: 1px solid var(--border-color-primary); border-radius: 12px; overflow: hidden; background: var(--background-fill-primary); }
.mask-toolbar { display: flex; gap: 8px; align-items: center; flex-wrap: wrap; padding: 10px; border-bottom: 1px solid var(--border-color-primary); }
.mask-tool { border: 1px solid var(--border-color-primary); border-radius: 8px; padding: 7px 12px; background: var(--button-secondary-background-fill); color: var(--body-text-color); cursor: pointer; }
.mask-tool.active { background: var(--button-primary-background-fill); color: var(--button-primary-text-color); border-color: var(--button-primary-border-color); }
.mask-tool.danger { color: #d33; }
.mask-size-label { display: flex; align-items: center; gap: 6px; font-size: 14px; }
.mask-size-label input { width: 130px; }
.mask-help { padding: 8px 12px; color: var(--body-text-color-subdued); font-size: 13px; }
.mask-canvas-box { width: 100%; min-height: 300px; display: flex; justify-content: center; align-items: center; overflow: auto; background: #222; padding: 12px; box-sizing: border-box; }
#mask-canvas { display: block; max-width: 100%; height: auto; touch-action: none; cursor: crosshair; box-shadow: 0 0 0 1px rgba(255,255,255,.15); }
.mask-empty { padding: 80px 20px; text-align: center; color: var(--body-text-color-subdued); }
#mask-payload { display: none !important; }
'''


def auto_detect(image):
    if image is None:
        return None, "请先上传图片。", ""
    pil = image if isinstance(image, Image.Image) else Image.fromarray(image)
    mask = detect_candidates(pil)
    preview = overlay_mask(pil, mask)
    count = int(np.count_nonzero(mask))
    if count:
        status = f"已自动识别 {count:,} 个 Mask 像素。红色区域就是当前待修复区域，可用橡皮擦直接删除误选。"
    else:
        status = "未找到高置信度候选区域，可以直接使用画笔增加需要修复的区域。"
    return preview, status, _payload(pil, mask)


def reset_editor(image):
    if image is None:
        return ""
    pil = image if isinstance(image, Image.Image) else Image.fromarray(image)
    return _payload(pil, _empty_mask(pil))


def clear_payload(image):
    return reset_editor(image)


def restore(image, payload):
    if image is None:
        raise gr.Error("请先上传图片。")
    pil = image if isinstance(image, Image.Image) else Image.fromarray(image)
    mask = _decode_mask(payload, pil.size)
    if not np.any(mask):
        raise gr.Error("没有修复区域，请先自动识别或用画笔涂抹。")
    result = engine.run(pil, Image.fromarray(mask, mode="L"))
    return result


with gr.Blocks(
    title="AI 图片智能修复",
    theme=gr.themes.Soft(),
    css=EDITOR_CSS,
    js=EDITOR_JS,
) as demo:
    gr.Markdown("# AI 图片智能修复\n自动识别 + 手动 Mask 精修 + CPU AI Inpainting")
    gr.Markdown("原图只需上传一次。自动识别后的区域直接进入 Mask 编辑器；画笔增加、橡皮擦删除误选区域。")

    with gr.Row():
        with gr.Column():
            source = gr.Image(label="原图", type="pil")
            auto_btn = gr.Button("✨ 自动识别候选区域", variant="primary")
            status = gr.Markdown("上传图片后开始。")
        with gr.Column():
            preview = gr.Image(label="自动识别预览（红色=待修复）", type="pil")

    gr.Markdown("## Mask 编辑器")
    gr.HTML(EDITOR_HTML)
    mask_payload = gr.Textbox(value="", elem_id="mask-payload", show_label=False, container=False)

    with gr.Row():
        clear_btn = gr.Button("清除全部 Mask")
        restore_btn = gr.Button("🚀 AI 智能修复", variant="primary")

    result = gr.Image(label="修复结果", type="pil", format="png")

    source.change(reset_editor, inputs=source, outputs=mask_payload)
    auto_btn.click(auto_detect, inputs=source, outputs=[preview, status, mask_payload])
    clear_btn.click(clear_payload, inputs=source, outputs=mask_payload)
    restore_btn.click(restore, inputs=[source, mask_payload], outputs=result)


if __name__ == "__main__":
    demo.launch(server_name="0.0.0.0", server_port=7860, show_error=True)
