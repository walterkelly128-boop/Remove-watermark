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


def _png_data(image: Image.Image) -> str:
    buf = io.BytesIO()
    image.save(buf, format="PNG")
    return "data:image/png;base64," + base64.b64encode(buf.getvalue()).decode("ascii")


def _mask_b64(mask: np.ndarray) -> str:
    mask = np.where(np.asarray(mask) > 30, 255, 0).astype(np.uint8)
    buf = io.BytesIO()
    Image.fromarray(mask, "L").save(buf, format="PNG")
    return base64.b64encode(buf.getvalue()).decode("ascii")


def build_editor(image: Image.Image, mask: np.ndarray) -> str:
    image_url = _png_data(image)
    mask_url = "data:image/png;base64," + _mask_b64(mask)
    w, h = image.size
    payload = json.dumps({"w": w, "h": h, "image": image_url, "mask": mask_url}, ensure_ascii=False)
    safe_payload = payload.replace("&", "&amp;").replace("'", "&apos;")
    return f'''<div class="wm-editor" data-payload='{safe_payload}'>
<canvas class="wm-canvas"></canvas>
<div class="wm-toolbar">
<button type="button" class="wm-tool active" data-mode="paint">🖌 画笔</button>
<button type="button" class="wm-tool" data-mode="erase">🧽 橡皮擦</button>
<label>笔刷 <input class="wm-size" type="range" min="4" max="200" value="32"><span class="wm-size-label">32</span></label>
<button type="button" class="wm-tool" data-action="undo">↶ 撤销</button>
<button type="button" class="wm-tool" data-action="redo">↷ 重做</button>
<button type="button" class="wm-tool" data-action="clear">清空</button>
<span class="wm-help">画笔=增加 Mask　橡皮擦=删除 Mask</span>
</div></div>
<script>
(() => {{
 const root = document.currentScript.previousElementSibling;
 if (!root || root.dataset.ready) return; root.dataset.ready='1';
 const p = JSON.parse(root.dataset.payload.replaceAll('&amp;', '&').replaceAll('&apos;', "'"));
 const c = root.querySelector('.wm-canvas'), ctx=c.getContext('2d');
 const img=new Image(), mask=new Image();
 let mode='paint', drawing=false, last=null, undo=[], redo=[];
 let scale=1, maskCanvas=document.createElement('canvas'), mctx=maskCanvas.getContext('2d');
 img.onload=()=>{{ fit(); mask.src=p.mask; }};
 mask.onload=()=>{{ maskCanvas.width=p.w; maskCanvas.height=p.h; mctx.clearRect(0,0,p.w,p.h); mctx.drawImage(mask,0,0,p.w,p.h); draw(); sync(); }};
 img.src=p.image;
 function fit(){{ const maxW=Math.min(root.clientWidth||900,1100), maxH=560; scale=Math.min(maxW/p.w,maxH/p.h); c.width=Math.max(1,Math.round(p.w*scale)); c.height=Math.max(1,Math.round(p.h*scale)); draw(); }}
 function pos(e){{ const r=c.getBoundingClientRect(); return {{x:(e.clientX-r.left)/scale,y:(e.clientY-r.top)/scale}}; }}
 function draw(){{ if(!img.complete)return; ctx.clearRect(0,0,c.width,c.height); ctx.drawImage(img,0,0,c.width,c.height); ctx.save(); ctx.globalAlpha=.42; ctx.drawImage(maskCanvas,0,0,c.width,c.height); ctx.restore(); }}
 function snapshot(){{ undo.push(maskCanvas.toDataURL()); if(undo.length>30)undo.shift(); redo=[]; }}
 function restoreData(url){{ const x=new Image(); x.onload=()=>{{mctx.clearRect(0,0,p.w,p.h);mctx.drawImage(x,0,0,p.w,p.h);draw();sync();}}; x.src=url; }}
 function stroke(a,b){{ mctx.save(); mctx.lineCap='round'; mctx.lineJoin='round'; mctx.lineWidth=Number(root.querySelector('.wm-size').value); if(mode==='paint'){{mctx.globalCompositeOperation='source-over';mctx.strokeStyle='rgba(255,0,0,1)';}} else {{mctx.globalCompositeOperation='destination-out';mctx.strokeStyle='rgba(0,0,0,1)';}} mctx.beginPath();mctx.moveTo(a.x,a.y);mctx.lineTo(b.x,b.y);mctx.stroke();mctx.restore();draw(); }}
 c.addEventListener('pointerdown',e=>{{e.preventDefault();snapshot();drawing=true;last=pos(e);stroke(last,last);c.setPointerCapture(e.pointerId);}});
 c.addEventListener('pointermove',e=>{{if(!drawing)return;const q=pos(e);stroke(last,q);last=q;}});
 const end=e=>{{drawing=false;if(e&&c.hasPointerCapture(e.pointerId))c.releasePointerCapture(e.pointerId);sync();}};
 c.addEventListener('pointerup',end); c.addEventListener('pointercancel',end);
 root.querySelectorAll('.wm-tool').forEach(b=>b.onclick=()=>{{if(b.dataset.mode){{mode=b.dataset.mode;root.querySelectorAll('.wm-tool').forEach(x=>x.classList.remove('active'));b.classList.add('active');}} else if(b.dataset.action==='clear'){{snapshot();mctx.clearRect(0,0,p.w,p.h);draw();sync();}} else if(b.dataset.action==='undo'&&undo.length){{redo.push(maskCanvas.toDataURL());restoreData(undo.pop());}} else if(b.dataset.action==='redo'&&redo.length){{undo.push(maskCanvas.toDataURL());restoreData(redo.pop());}}}});
 const range=root.querySelector('.wm-size'); range.oninput=()=>root.querySelector('.wm-size-label').textContent=range.value;
 function sync(){{const b=maskCanvas.toDataURL('image/png'); const box=document.querySelector('#mask-data textarea')||document.querySelector('#mask-data input'); if(box){{box.value=b;box.dispatchEvent(new Event('input',{{bubbles:true}}));box.dispatchEvent(new Event('change',{{bubbles:true}}));}}}}
 window.addEventListener('resize',fit);
}})();
</script>'''


def decode_mask(data, size):
    if not data:
        return np.zeros((size[1], size[0]), dtype=np.uint8)
    try:
        if data.startswith("data:image"):
            data = data.split(",", 1)[1]
        raw = base64.b64decode(data)
        arr = np.asarray(Image.open(io.BytesIO(raw)).convert("L"))
        if arr.shape != (size[1], size[0]):
            arr = np.asarray(Image.fromarray(arr).resize(size, Image.Resampling.NEAREST))
        return np.where(arr > 30, 255, 0).astype(np.uint8)
    except Exception:
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
        preview_image = overlay_mask(pil, mask)
        count = int(np.count_nonzero(mask))
        return preview_image, build_editor(pil, mask), f"✅ 自动识别完成：{count:,} 个 Mask 像素。", _mask_b64(mask)
    except Exception as exc:
        traceback.print_exc()
        return None, None, f"❌ 自动识别失败：{type(exc).__name__}: {exc}", ""


def reset_editor(image):
    pil = _pil(image)
    if pil is None:
        return None, ""
    mask = np.zeros((pil.height, pil.width), dtype=np.uint8)
    return build_editor(pil, mask), _mask_b64(mask)


def restore(image, mask_data):
    pil = _pil(image)
    if pil is None:
        raise gr.Error("请先上传图片。")
    mask = decode_mask("data:image/png;base64," + mask_data if mask_data and not mask_data.startswith("data:") else mask_data, pil.size)
    if not np.any(mask):
        raise gr.Error("没有修复区域，请先自动识别或用画笔添加 Mask。")
    return engine.run(pil, Image.fromarray(mask, "L"))


CSS = '''
.wm-editor{border:1px solid #d9d9d9;border-radius:12px;padding:12px;background:#fafafa}.wm-canvas{display:block;max-width:100%;margin:auto;background:#222;border-radius:8px;cursor:crosshair;touch-action:none}.wm-toolbar{display:flex;flex-wrap:wrap;gap:8px;align-items:center;margin-top:10px}.wm-tool{border:1px solid #ccc;background:white;border-radius:7px;padding:7px 12px;cursor:pointer}.wm-tool.active{background:#111;color:white}.wm-size{width:130px}.wm-help{color:#666;font-size:13px;margin-left:auto}
'''

with gr.Blocks(title="AI 图片智能修复", theme=gr.themes.Soft(), css=CSS) as demo:
    gr.Markdown("# AI 图片智能修复\n自动识别候选区域 + 自定义 Mask 编辑 + CPU AI Inpainting")
    gr.Markdown("使用独立 Canvas 编辑器：自动识别后右侧显示检测预览，同时下方可以直接画笔/橡皮擦编辑 Mask。")
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
    with gr.Row():
        restore_btn = gr.Button("🚀 AI 智能修复", variant="primary")
    result = gr.Image(label="修复结果", type="pil", format="png")

    source.change(reset_editor, inputs=source, outputs=[editor, mask_data])
    auto_btn.click(auto_detect, inputs=source, outputs=[preview, editor, status, mask_data], show_progress="minimal")
    clear_btn.click(reset_editor, inputs=source, outputs=[editor, mask_data])
    restore_btn.click(restore, inputs=[source, mask_data], outputs=result)

if __name__ == "__main__":
    demo.launch(server_name="0.0.0.0", server_port=7860, show_error=True)
