from __future__ import annotations

from pathlib import Path
import base64
import io
import json
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


def _b64_png(image: Image.Image, mode: str | None = None) -> str:
    if mode:
        image = image.convert(mode)
    buf = io.BytesIO()
    image.save(buf, format="PNG")
    return base64.b64encode(buf.getvalue()).decode("ascii")


# The editor HTML contains only data + markup. The controller is installed once
# from Blocks(head=...) so it does not depend on Gradio executing <script> tags
# inside dynamically returned gr.HTML content.
EDITOR_JS = r"""
<script>
(() => {
  if (window.__wmMaskEditorInstalled) return;
  window.__wmMaskEditorInstalled = true;

  function b64json(s) {
    try {
      const bin = atob(s);
      const bytes = Uint8Array.from(bin, c => c.charCodeAt(0));
      return JSON.parse(new TextDecoder().decode(bytes));
    } catch (e) {
      console.error('Mask editor payload error', e);
      return null;
    }
  }

  function findMaskBox() {
    return document.querySelector('#mask-data textarea') || document.querySelector('#mask-data input');
  }

  function syncMask(state) {
    const box = findMaskBox();
    if (!box || !state.maskCanvas) return;
    const value = state.maskCanvas.toDataURL('image/png');
    box.value = value;
    box.dispatchEvent(new Event('input', {bubbles:true}));
    box.dispatchEvent(new Event('change', {bubbles:true}));
  }

  function install(root) {
    if (!root || root.dataset.ready === '1') return;
    const payload = b64json(root.dataset.payload || '');
    if (!payload || !payload.image || !payload.mask) return;

    const img = root.querySelector('.wm-image');
    const overlay = root.querySelector('.wm-canvas');
    if (!img || !overlay) return;

    root.dataset.ready = '1';
    const ctx = overlay.getContext('2d');
    const maskCanvas = document.createElement('canvas');
    const mctx = maskCanvas.getContext('2d', {willReadFrequently:true});
    maskCanvas.width = payload.w;
    maskCanvas.height = payload.h;

    const state = {maskCanvas, mctx, payload, overlay, ctx, img, mode:'paint', drawing:false, last:null, scale:1, undo:[], redo:[]};
    root.__wmState = state;

    function fit() {
      const maxW = Math.min(root.clientWidth || 1000, 1100);
      state.scale = Math.min(maxW / payload.w, 560 / payload.h, 1);
      const cssW = Math.max(1, Math.round(payload.w * state.scale));
      const cssH = Math.max(1, Math.round(payload.h * state.scale));
      root.querySelector('.wm-stage').style.width = cssW + 'px';
      root.querySelector('.wm-stage').style.height = cssH + 'px';
      overlay.width = payload.w;
      overlay.height = payload.h;
      overlay.style.width = cssW + 'px';
      overlay.style.height = cssH + 'px';
      img.style.width = cssW + 'px';
      img.style.height = cssH + 'px';
      render();
    }

    function render() {
      ctx.clearRect(0, 0, overlay.width, overlay.height);
      const d = mctx.getImageData(0, 0, payload.w, payload.h).data;
      const out = ctx.createImageData(payload.w, payload.h);
      for (let i=0; i<d.length; i+=4) {
        const a = d[i] > 30 ? 105 : 0;
        out.data[i] = 255;
        out.data[i+1] = 40;
        out.data[i+2] = 40;
        out.data[i+3] = a;
      }
      ctx.putImageData(out, 0, 0);
    }

    function point(e) {
      const r = overlay.getBoundingClientRect();
      return {
        x: Math.max(0, Math.min(payload.w, (e.clientX-r.left) / state.scale)),
        y: Math.max(0, Math.min(payload.h, (e.clientY-r.top) / state.scale))
      };
    }

    function snapshot() {
      state.undo.push(mctx.getImageData(0, 0, payload.w, payload.h));
      if (state.undo.length > 40) state.undo.shift();
      state.redo.length = 0;
    }

    function stroke(a,b) {
      mctx.save();
      mctx.lineCap = 'round';
      mctx.lineJoin = 'round';
      mctx.lineWidth = Number(root.querySelector('.wm-size').value) || 32;
      if (state.mode === 'paint') {
        mctx.globalCompositeOperation = 'source-over';
        mctx.strokeStyle = '#fff';
      } else {
        mctx.globalCompositeOperation = 'destination-out';
        mctx.strokeStyle = '#000';
      }
      mctx.beginPath();
      mctx.moveTo(a.x,a.y);
      mctx.lineTo(b.x,b.y);
      mctx.stroke();
      mctx.restore();
      render();
    }

    overlay.addEventListener('pointerdown', e => {
      e.preventDefault();
      snapshot();
      state.drawing = true;
      state.last = point(e);
      stroke(state.last, state.last);
      overlay.setPointerCapture(e.pointerId);
    });

    overlay.addEventListener('pointermove', e => {
      if (!state.drawing) return;
      e.preventDefault();
      const q = point(e);
      stroke(state.last,q);
      state.last = q;
    });

    function end(e) {
      if (!state.drawing) return;
      state.drawing = false;
      state.last = null;
      if (e && overlay.hasPointerCapture(e.pointerId)) overlay.releasePointerCapture(e.pointerId);
      syncMask(state);
    }
    overlay.addEventListener('pointerup', end);
    overlay.addEventListener('pointercancel', end);
    overlay.addEventListener('lostpointercapture', end);

    root.querySelectorAll('.wm-tool[data-mode]').forEach(btn => btn.addEventListener('click', () => {
      state.mode = btn.dataset.mode;
      root.querySelectorAll('.wm-tool[data-mode]').forEach(x => x.classList.remove('active'));
      btn.classList.add('active');
    }));

    root.querySelector('.wm-size').addEventListener('input', e => {
      root.querySelector('.wm-size-label').textContent = e.target.value;
    });

    root.querySelector('[data-action="undo"]').addEventListener('click', () => {
      if (!state.undo.length) return;
      state.redo.push(mctx.getImageData(0,0,payload.w,payload.h));
      mctx.putImageData(state.undo.pop(),0,0);
      render(); syncMask(state);
    });

    root.querySelector('[data-action="redo"]').addEventListener('click', () => {
      if (!state.redo.length) return;
      state.undo.push(mctx.getImageData(0,0,payload.w,payload.h));
      mctx.putImageData(state.redo.pop(),0,0);
      render(); syncMask(state);
    });

    root.querySelector('[data-action="clear"]').addEventListener('click', () => {
      snapshot();
      mctx.clearRect(0,0,payload.w,payload.h);
      render(); syncMask(state);
    });

    img.onload = () => fit();
    img.src = 'data:image/png;base64,' + payload.image;

    const maskImg = new Image();
    maskImg.onload = () => {
      mctx.clearRect(0,0,payload.w,payload.h);
      mctx.drawImage(maskImg,0,0,payload.w,payload.h);
      render();
      syncMask(state);
    };
    maskImg.src = 'data:image/png;base64,' + payload.mask;

    if (img.complete) fit();
  }

  function scan() {
    document.querySelectorAll('.wm-editor[data-payload]').forEach(install);
  }

  const observer = new MutationObserver(scan);
  observer.observe(document.documentElement, {childList:true, subtree:true});
  scan();
  window.addEventListener('resize', scan);
})();
</script>
"""


def build_editor(image: Image.Image, mask: np.ndarray) -> str:
    image = image.convert("RGB")
    mask = np.where(np.asarray(mask) > 30, 255, 0).astype(np.uint8)
    w, h = image.size
    payload = {
        "w": w,
        "h": h,
        "image": _b64_png(image),
        "mask": _b64_png(Image.fromarray(mask, "L")),
    }
    encoded = base64.b64encode(
        json.dumps(payload, separators=(",", ":")).encode("utf-8")
    ).decode("ascii")

    return f'''<div class="wm-editor" data-payload="{encoded}">
  <div class="wm-stage">
    <img class="wm-image" alt="原图" draggable="false">
    <canvas class="wm-canvas"></canvas>
  </div>
  <div class="wm-toolbar">
    <button type="button" class="wm-tool active" data-mode="paint">🖌 画笔</button>
    <button type="button" class="wm-tool" data-mode="erase">🧽 橡皮擦</button>
    <label>笔刷 <input class="wm-size" type="range" min="4" max="200" value="32"><span class="wm-size-label">32</span></label>
    <button type="button" class="wm-tool" data-action="undo">↶ 撤销</button>
    <button type="button" class="wm-tool" data-action="redo">↷ 重做</button>
    <button type="button" class="wm-tool" data-action="clear">清空</button>
    <span class="wm-help">红色 = Mask　画笔增加　橡皮擦删除</span>
  </div>
</div>'''


def decode_mask(data, size):
    if not data:
        return np.zeros((size[1], size[0]), dtype=np.uint8)
    try:
        if data.startswith("data:image"):
            data = data.split(",", 1)[1]
        arr = np.asarray(Image.open(io.BytesIO(base64.b64decode(data))).convert("L"))
        if arr.shape != (size[1], size[0]):
            arr = np.asarray(Image.fromarray(arr).resize(size, Image.Resampling.NEAREST))
        return np.where(arr > 30, 255, 0).astype(np.uint8)
    except Exception:
        traceback.print_exc()
        return np.zeros((size[1], size[0]), dtype=np.uint8)


def auto_detect(image):
    if image is None:
        return None, None, "⚠️ 请先上传图片。", ""
    try:
        pil = _pil(image)
        mask = np.asarray(detect_candidates(pil), dtype=np.uint8)
        if mask.shape != (pil.height, pil.width):
            mask = np.asarray(Image.fromarray(mask, "L").resize(pil.size, Image.Resampling.NEAREST))
        mask = np.where(mask > 30, 255, 0).astype(np.uint8)
        return (
            overlay_mask(pil, mask),
            build_editor(pil, mask),
            f"✅ 自动识别完成：{int(np.count_nonzero(mask)):,} 个 Mask 像素。",
            _b64_png(Image.fromarray(mask, "L")),
        )
    except Exception as exc:
        traceback.print_exc()
        return None, None, f"❌ 自动识别失败：{type(exc).__name__}: {exc}", ""


def reset_editor(image):
    pil = _pil(image)
    if pil is None:
        return None, ""
    mask = np.zeros((pil.height, pil.width), dtype=np.uint8)
    return build_editor(pil, mask), _b64_png(Image.fromarray(mask, "L"))


def restore(image, mask_data):
    pil = _pil(image)
    if pil is None:
        raise gr.Error("请先上传图片。")
    mask = decode_mask(
        "data:image/png;base64," + mask_data
        if mask_data and not mask_data.startswith("data:")
        else mask_data,
        pil.size,
    )
    if not np.any(mask):
        raise gr.Error("没有修复区域，请先自动识别或用画笔添加 Mask。")
    return engine.run(pil, Image.fromarray(mask, "L"))


CSS = '''
.wm-editor{border:1px solid #d9d9d9;border-radius:12px;padding:12px;background:#fafafa}
.wm-stage{position:relative;margin:0 auto;overflow:hidden;line-height:0;background:#222;border-radius:8px;user-select:none}
.wm-image{position:absolute;left:0;top:0;display:block;max-width:none;object-fit:fill;user-select:none;-webkit-user-drag:none}
.wm-canvas{position:absolute;left:0;top:0;display:block;max-width:none;cursor:crosshair;touch-action:none}
.wm-toolbar{display:flex;flex-wrap:wrap;gap:8px;align-items:center;margin-top:10px}
.wm-tool{border:1px solid #ccc;background:white;border-radius:7px;padding:7px 12px;cursor:pointer}
.wm-tool.active{background:#111;color:white}
.wm-size{width:130px}
.wm-help{color:#666;font-size:13px;margin-left:auto}
'''

with gr.Blocks(
    title="AI 图片智能修复",
    theme=gr.themes.Soft(),
    css=CSS,
    head=EDITOR_JS,
) as demo:
    gr.Markdown("# AI 图片智能修复\n自动识别候选区域 + 自定义 Mask 编辑 + CPU AI Inpainting")
    gr.Markdown("自动识别后，右侧显示检测预览；下方编辑器显示原图 + 红色 Mask，可直接画笔增加或橡皮擦删除。")
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
    mask_data = gr.Textbox(label="", elem_id="mask-data", visible=False)
    restore_btn = gr.Button("🚀 AI 智能修复", variant="primary")
    result = gr.Image(label="修复结果", type="pil", format="png")

    source.change(reset_editor, inputs=source, outputs=[editor, mask_data])
    auto_btn.click(
        auto_detect,
        inputs=source,
        outputs=[preview, editor, status, mask_data],
        show_progress="minimal",
    )
    clear_btn.click(reset_editor, inputs=source, outputs=[editor, mask_data])
    restore_btn.click(restore, inputs=[source, mask_data], outputs=result)


if __name__ == "__main__":
    demo.launch(server_name="0.0.0.0", server_port=7860, show_error=True)
