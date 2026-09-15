import base64
import io
import os

import numpy as np
from fastapi import FastAPI, File, UploadFile
from fastapi.responses import RedirectResponse, Response
from PIL import Image
from gradio import mount_gradio_app

import tools_app
import admin_app
from core.detector import detect_candidates, overlay_mask
from app import build_editor, _mask_rle


def _png_b64(image: Image.Image) -> str:
    buf = io.BytesIO()
    image.save(buf, format="PNG")
    return base64.b64encode(buf.getvalue()).decode("ascii")


def _detect_upload(data: bytes):
    image = Image.open(io.BytesIO(data)).convert("RGB")
    mask = np.asarray(detect_candidates(image), dtype=np.uint8)
    mask = np.where(mask > 30, 255, 0).astype(np.uint8)
    preview = overlay_mask(image, mask)
    return {
        "ok": True,
        "width": image.width,
        "height": image.height,
        "pixels": int(np.count_nonzero(mask)),
        "preview": _png_b64(preview),
        "editor": build_editor(image, mask),
        "mask": _mask_rle(mask),
    }


DIRECT_DETECT_JS = r'''<script id="zolfox-direct-detect-v4">
(() => {
  if (window.__zolfoxDirectDetectV4) return;
  window.__zolfoxDirectDetectV4 = true;
  console.log('[zolfox-direct-detect-v4] loaded');
  const API = '/remove-watermark/api/detect';
  const q = s => document.querySelector(s);
  const button = () => q('#wm-auto-detect button') || q('#wm-auto-detect');
  const sourceInput = () => q('#wm-source input[type="file"]');
  const status = () => q('#wm-status');
  const previewRoot = () => q('#wm-preview');
  const editorRoot = () => q('#wm-editor');
  function setStatus(text) { const el=status(); if(el) el.textContent=text; }
  function showPreview(data) {
    const root=previewRoot(); if(!root) throw new Error('找不到 #wm-preview');
    let img=root.querySelector('.zf-detect-preview');
    if(!img){img=document.createElement('img');img.className='zf-detect-preview';img.alt='自动识别候选区域预览';img.style.cssText='display:block;width:100%;max-width:100%;max-height:620px;object-fit:contain;border-radius:8px';root.appendChild(img)}
    img.src='data:image/png;base64,'+data.preview;
  }
  function setNativeValue(el,value){const p=el instanceof HTMLTextAreaElement?HTMLTextAreaElement.prototype:HTMLInputElement.prototype;const s=Object.getOwnPropertyDescriptor(p,'value')?.set;if(s)s.call(el,value);else el.value=value;}
  function setMask(value){const box=q('#mask-data textarea')||q('#mask-data input');if(!box)return;setNativeValue(box,value);box.dispatchEvent(new Event('input',{bubbles:true}));box.dispatchEvent(new Event('change',{bubbles:true}));}
  function setEditor(html){const root=editorRoot();if(root)root.innerHTML=html||'';}
  async function detect(){
    const btn=button();if(btn?.dataset.zfBusy==='1')return;
    const input=sourceInput();const file=input?.files?.[0];
    console.log('[zolfox-direct-detect-v4] click',{input:!!input,file:file?.name||null});
    if(!file){setStatus('⚠️ 请先上传图片。');return;}
    if(btn){btn.dataset.zfBusy='1';btn.disabled=true;}
    try{
      setStatus('⏳ 正在自动识别候选区域，请稍候…');
      const form=new FormData();form.append('file',file,file.name||'source.png');
      const response=await fetch(API+'?_='+Date.now(),{method:'POST',body:form,credentials:'same-origin',cache:'no-store',headers:{'Cache-Control':'no-cache'}});
      const text=await response.text();let data;try{data=JSON.parse(text)}catch(_){throw new Error('服务器返回非 JSON：'+text.slice(0,200))}
      if(!response.ok||!data.ok)throw new Error(data.detail||('HTTP '+response.status));
      showPreview(data);setMask(data.mask||'');setEditor(data.editor||'');setStatus('✅ 自动识别完成：'+Number(data.pixels||0).toLocaleString()+' 个 Mask 像素。');
      console.log('[zolfox-direct-detect-v4] success',data.width,data.height,data.pixels);
    }catch(err){console.error('[zolfox-direct-detect-v4] failed',err);setStatus('❌ 自动识别失败：'+(err?.message||err));alert('自动识别失败：'+(err?.message||err));}
    finally{if(btn){btn.disabled=false;delete btn.dataset.zfBusy;}}
  }
  function intercept(e){const b=button();if(!b||!(e.target===b||b.contains(e.target)))return;e.preventDefault();e.stopPropagation();e.stopImmediatePropagation();if(e.type==='click')detect();return false;}
  function install(){const b=button();if(!b||b.dataset.zfDirectV4==='1')return;b.dataset.zfDirectV4='1';['pointerdown','mousedown','click'].forEach(t=>b.addEventListener(t,intercept,true));console.log('[zolfox-direct-detect-v4] installed');}
  const observer=new MutationObserver(install);observer.observe(document.documentElement,{childList:true,subtree:true});install();
})();
</script>'''


def create_app():
    app = FastAPI(title="ZOLFOX Tools")
    @app.get("/admin", include_in_schema=False)
    def admin_root(): return RedirectResponse(url="/admin/", status_code=307)
    @app.post("/remove-watermark/api/detect")
    async def detect_api(file: UploadFile = File(...)):
        try:
            data=await file.read()
            if not data:return {"ok":False,"detail":"文件为空"}
            if len(data)>25*1024*1024:return {"ok":False,"detail":"图片不能超过 25MB"}
            return _detect_upload(data)
        except Exception as exc:
            import traceback;traceback.print_exc();return {"ok":False,"detail":f"{type(exc).__name__}: {exc}"}
    app=mount_gradio_app(app,admin_app.admin_app,path="/admin")
    app=mount_gradio_app(app,tools_app.home,path="/")
    app=mount_gradio_app(app,tools_app.remove_watermark,path="/remove-watermark")
    app=mount_gradio_app(app,tools_app.image_compress,path="/image-compress")
    return app

app=create_app()

@app.middleware("http")
async def inject_direct_detector(request,call_next):
    response=await call_next(request)
    if request.url.path.rstrip("/")!="/remove-watermark":return response
    content_type=response.headers.get("content-type","")
    if "text/html" not in content_type:return response
    body=b"".join([chunk async for chunk in response.body_iterator])
    script=DIRECT_DETECT_JS.encode("utf-8")
    inserted=False
    for marker in (b"</head>",b"</HEAD>",b"<body",b"<BODY"):
        pos=body.find(marker)
        if pos>=0:
            body=body[:pos]+script+body[pos:];inserted=True;break
    if not inserted:body+=script
    headers=dict(response.headers);headers.pop("content-length",None);headers.pop("content-encoding",None)
    headers["cache-control"]="no-store, no-cache, must-revalidate, max-age=0";headers["x-zolfox-direct-detect"]="v4"
    return Response(content=body,status_code=response.status_code,headers=headers,media_type="text/html")

if __name__=="__main__":
    import uvicorn;uvicorn.run(app,host="0.0.0.0",port=int(os.getenv("PORT","7860")))
