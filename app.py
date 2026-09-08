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


def build_editor(image: Image.Image, mask: np.ndarray) -> str:
    image = image.convert("RGB")
    mask = np.where(np.asarray(mask) > 30, 255, 0).astype(np.uint8)
    w, h = image.size
    payload = {"w": w, "h": h, "image": _b64_png(image), "mask": _b64_png(Image.fromarray(mask, "L"))}
    encoded = base64.b64encode(json.dumps(payload, separators=(",", ":")).encode("utf-8")).decode("ascii")
    return f'''<div class="wm-editor" data-payload="{encoded}">
<canvas class="wm-canvas"></canvas>
<div class="wm-toolbar">
<button type="button" class="wm-tool active" data-mode="paint">🖌 画笔</button>
<button type="button" class="wm-tool" data-mode="erase">🧽 橡皮擦</button>
<label>笔刷 <input class="wm-size" type="range" min="4" max="200" value="32"><span class="wm-size-label">32</span></label>
<button type="button" class="wm-tool" data-action="undo">↶ 撤销</button>
<button type="button" class="wm-tool" data-action="redo">↷ 重做</button>
<button type="button" class="wm-tool" data-action="clear">清空</button>
<span class="wm-help">红色 = 当前 Mask　画笔增加　橡皮擦删除</span>
</div></div>
<script>
(() => {{
 const root=document.currentScript?.parentElement;
 if(!root||root.dataset.ready==='1')return; root.dataset.ready='1';
 let p; try{{p=JSON.parse(atob(root.dataset.payload));}}catch(e){{console.error(e);return;}}
 const c=root.querySelector('.wm-canvas'),ctx=c.getContext('2d');
 const mc=document.createElement('canvas'),mctx=mc.getContext('2d',{{willReadFrequently:true}});
 const img=new Image(),mi=new Image(); let mode='paint',drawing=false,last=null,scale=1; const undo=[],redo=[];
 function fit(){{scale=Math.min(Math.min(root.clientWidth||900,1100)/p.w,560/p.h,1);c.width=Math.max(1,Math.round(p.w*scale));c.height=Math.max(1,Math.round(p.h*scale));render();}}
 function render(){{if(!img.complete||!img.naturalWidth)return;ctx.clearRect(0,0,c.width,c.height);ctx.drawImage(img,0,0,c.width,c.height);const d=mctx.getImageData(0,0,p.w,p.h).data;const ov=document.createElement('canvas');ov.width=p.w;ov.height=p.h;const oc=ov.getContext('2d');const out=oc.createImageData(p.w,p.h);for(let i=0;i<d.length;i+=4){{const a=d[i];out.data[i]=255;out.data[i+1]=40;out.data[i+2]=40;out.data[i+3]=a>30?105:0;}}oc.putImageData(out,0,0);ctx.drawImage(ov,0,0,c.width,c.height);}}
 function point(e){{const r=c.getBoundingClientRect();return{{x:Math.max(0,Math.min(p.w,(e.clientX-r.left)/scale)),y:Math.max(0,Math.min(p.h,(e.clientY-r.top)/scale))}};}}
 function snap(){{undo.push(mctx.getImageData(0,0,p.w,p.h));if(undo.length>30)undo.shift();redo.length=0;}}
 function sync(){{const box=document.querySelector('#mask-data textarea')||document.querySelector('#mask-data input');if(box){{box.value=mc.toDataURL('image/png');box.dispatchEvent(new Event('input',{{bubbles:true}}));box.dispatchEvent(new Event('change',{{bubbles:true}}));}}}}
 function stroke(a,b){{mctx.save();mctx.lineCap='round';mctx.lineJoin='round';mctx.lineWidth=Number(root.querySelector('.wm-size').value)||32;if(mode==='paint'){{mctx.globalCompositeOperation='source-over';mctx.strokeStyle='white';}}else{{mctx.globalCompositeOperation='destination-out';mctx.strokeStyle='rgba(0,0,0,1)';}}mctx.beginPath();mctx.moveTo(a.x,a.y);mctx.lineTo(b.x,b.y);mctx.stroke();mctx.restore();render();}}
 c.addEventListener('pointerdown',e=>{{e.preventDefault();snap();drawing=true;last=point(e);stroke(last,last);c.setPointerCapture(e.pointerId);}});
 c.addEventListener('pointermove',e=>{{if(!drawing)return;e.preventDefault();const q=point(e);stroke(last,q);last=q;}});
 const end=e=>{{if(!drawing)return;drawing=false;last=null;if(e&&c.hasPointerCapture(e.pointerId))c.releasePointerCapture(e.pointerId);sync();}};c.addEventListener('pointerup',end);c.addEventListener('pointercancel',end);
 root.querySelectorAll('.wm-tool').forEach(b=>b.addEventListener('click',()=>{{const m=b.dataset.mode,a=b.dataset.action;if(m){{mode=m;root.querySelectorAll('.wm-tool[data-mode]').forEach(x=>x.classList.remove('active'));b.classList.add('active');}}else if(a==='clear'){{snap();mctx.clearRect(0,0,p.w,p.h);render();sync();}}else if(a==='undo'&&undo.length){{redo.push(mctx.getImageData(0,0,p.w,p.h));mctx.putImageData(undo.pop(),0,0);render();sync();}}else if(a==='redo'&&redo.length){{undo.push(mctx.getImageData(0,0,p.w,p.h));mctx.putImageData(redo.pop(),0,0);render();sync();}}}}));
 const range=root.querySelector('.wm-size');range.addEventListener('input',()=>root.querySelector('.wm-size-label').textContent=range.value);
 img.onload=()=>{{mc.width=p.w;mc.height=p.h;mi.onload=()=>{{mctx.clearRect(0,0,p.w,p.h);mctx.drawImage(mi,0,0,p.w,p.h);fit();sync();}};mi.src='data:image/png;base64,'+p.mask;}};img.src='data:image/png;base64,'+p.image;
 window.addEventListener('resize',fit);
}})();
</script>'''


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
        return overlay_mask(pil, mask), build_editor(pil, mask), f"✅ 自动识别完成：{int(np.count_nonzero(mask)):,} 个 Mask 像素。", _b64_png(Image.fromarray(mask, "L"))
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
    mask = decode_mask("data:image/png;base64," + mask_data if mask_data and not mask_data.startswith("data:") else mask_data, pil.size)
    if not np.any(mask):
        raise gr.Error("没有修复区域，请先自动识别或用画笔添加 Mask。")
    return engine.run(pil, Image.fromarray(mask, "L"))

CSS='''.wm-editor{border:1px solid #d9d9d9;border-radius:12px;padding:12px;background:#fafafa}.wm-canvas{display:block;width:100%;max-width:1100px;height:auto;margin:auto;background:#222;border-radius:8px;cursor:crosshair;touch-action:none}.wm-toolbar{display:flex;flex-wrap:wrap;gap:8px;align-items:center;margin-top:10px}.wm-tool{border:1px solid #ccc;background:white;border-radius:7px;padding:7px 12px;cursor:pointer}.wm-tool.active{background:#111;color:white}.wm-size{width:130px}.wm-help{color:#666;font-size:13px;margin-left:auto}'''

with gr.Blocks(title="AI 图片智能修复",theme=gr.themes.Soft(),css=CSS) as demo:
    gr.Markdown("# AI 图片智能修复\n自动识别候选区域 + 自定义 Mask 编辑 + CPU AI Inpainting")
    gr.Markdown("自动识别后，右侧显示检测预览；下方编辑器显示原图 + 红色 Mask，可直接画笔增加或橡皮擦删除。")
    with gr.Row():
        with gr.Column():
            source=gr.Image(label="原图",type="pil")
            with gr.Row():
                auto_btn=gr.Button("✨ 自动识别候选区域",variant="primary")
                clear_btn=gr.Button("清除 Mask")
            status=gr.Markdown("上传图片后开始。")
        with gr.Column():
            preview=gr.Image(label="自动识别预览（红色=候选区域）",type="pil")
    gr.Markdown("## Mask 编辑器")
    editor=gr.HTML(label="Mask 编辑器")
    mask_data=gr.Textbox(label="",elem_id="mask-data",visible=False)
    restore_btn=gr.Button("🚀 AI 智能修复",variant="primary")
    result=gr.Image(label="修复结果",type="pil",format="png")
    source.change(reset_editor,inputs=source,outputs=[editor,mask_data])
    auto_btn.click(auto_detect,inputs=source,outputs=[preview,editor,status,mask_data],show_progress="minimal")
    clear_btn.click(reset_editor,inputs=source,outputs=[editor,mask_data])
    restore_btn.click(restore,inputs=[source,mask_data],outputs=result)

if __name__=="__main__":
    demo.launch(server_name="0.0.0.0",server_port=7860,show_error=True)
