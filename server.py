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


# IMPORTANT:
# The watermark detector must not depend on Gradio's queue/event transport.
# The browser uploads the image through Gradio, but the actual detection call
# is handled by a plain FastAPI endpoint. This avoids the queue/join problem
# that was causing the button to appear to do nothing under a mounted app.

def _png_b64(image: Image.Image) -> str:
    buf = io.BytesIO()
    image.save(buf, format="PNG")
    return base64.b64encode(buf.getvalue()).decode("ascii")


@app_placeholder = None


def _detect_upload(data: bytes):
    image = Image.open(io.BytesIO(data)).convert("RGB")
    mask = np.asarray(detect_candidates(image), dtype=np.uint8)
    mask = np.where(mask > 30, 255, 0).astype(np.uint8)
    pixels = int(np.count_nonzero(mask))
    preview = overlay_mask(image, mask)
    editor = build_editor(image, mask)
    return {
        "ok": True,
        "width": image.width,
        "height": image.height,
        "pixels": pixels,
        "preview": _png_b64(preview),
        "editor": editor,
        "mask": _mask_rle(mask),
    }


DETECT_JS = r"""
<script>
(() => {
  if (window.__zolfoxDirectDetectInstalled) return;
  window.__zolfoxDirectDetectInstalled = true;

  const API = '/remove-watermark/api/detect';

  function visible(el) {
    if (!el) return false;
    const r = el.getBoundingClientRect();
    return r.width > 0 && r.height > 0;
  }

  function findButton() {
    return [...document.querySelectorAll('button')].find(b =>
      (b.textContent || '').replace(/\s+/g, '').includes('自动识别候选区域')
    );
  }

  function findLabeledComponent(labelText) {
    const labels = [...document.querySelectorAll('label')];
    const label = labels.find(x => (x.textContent || '').includes(labelText));
    if (!label) return null;
    return label.closest('[data-testid]') || label.parentElement?.parentElement || label.parentElement;
  }

  function findSourceImage() {
    const candidates = [...document.querySelectorAll('img')].filter(visible);
    const src = candidates.find(img => {
      const s = img.getAttribute('src') || '';
      return s.includes('/gradio_api/file') || s.startsWith('blob:') || s.startsWith('data:image');
    });
    return src || candidates[0] || null;
  }

  function findPreviewImage() {
    const comp = findLabeledComponent('识别预览');
    if (comp) {
      const img = comp.querySelector('img');
      if (img) return img;
      const created = document.createElement('img');
      created.style.maxWidth = '100%';
      created.style.maxHeight = '620px';
      created.style.objectFit = 'contain';
      comp.appendChild(created);
      return created;
    }
    const imgs = [...document.querySelectorAll('img')].filter(visible);
    return imgs[1] || null;
  }

  function setMask(value) {
    const box = document.querySelector('#mask-data textarea, #mask-data input');
    if (!box) return;
    const proto = box instanceof HTMLTextAreaElement ? HTMLTextAreaElement.prototype : HTMLInputElement.prototype;
    const setter = Object.getOwnPropertyDescriptor(proto, 'value')?.set;
    if (setter) setter.call(box, value); else box.value = value;
    box.dispatchEvent(new Event('input', {bubbles: true}));
    box.dispatchEvent(new Event('change', {bubbles: true}));
  }

  function setEditor(html) {
    const comp = findLabeledComponent('Mask 编辑器');
    if (!comp) return;
    const old = comp.querySelector('.wm-editor');
    if (old) old.remove();
    const holder = document.createElement('div');
    holder.innerHTML = html;
    const editor = holder.firstElementChild;
    if (editor) comp.appendChild(editor);
  }

  function setStatus(text) {
    const comp = findLabeledComponent('状态');
    if (comp) {
      const md = comp.querySelector('.prose, [data-testid="markdown"]');
      if (md) md.textContent = text;
    }
    const all = [...document.querySelectorAll('div, p, span')];
    const status = all.find(x => (x.textContent || '').trim() === '上传图片后开始。');
    if (status) status.textContent = text;
  }

  async function runDetect(e) {
    if (e) {
      e.preventDefault();
      e.stopPropagation();
      if (e.stopImmediatePropagation) e.stopImmediatePropagation();
    }

    const btn = findButton();
    if (btn) {
      btn.disabled = true;
      btn.dataset.zfBusy = '1';
    }

    try {
      const source = findSourceImage();
      if (!source || !source.src) throw new Error('请先上传图片');

      setStatus('⏳ 正在自动识别候选区域，请稍候…');
      const imageResponse = await fetch(source.src, {credentials: 'same-origin'});
      if (!imageResponse.ok) throw new Error('无法读取已上传图片（HTTP ' + imageResponse.status + '）');
      const blob = await imageResponse.blob();

      const form = new FormData();
      form.append('file', blob, 'source.png');
      const response = await fetch(API, {method: 'POST', body: form, credentials: 'same-origin'});
      const data = await response.json();
      if (!response.ok || !data.ok) throw new Error(data.detail || '自动识别失败');

      const preview = findPreviewImage();
      if (!preview) throw new Error('找不到识别预览区域');
      preview.src = 'data:image/png;base64,' + data.preview;
      preview.style.display = 'block';
      preview.style.maxWidth = '100%';
      preview.style.maxHeight = '620px';
      preview.style.objectFit = 'contain';

      setMask(data.mask || '');
      setEditor(data.editor || '');
      setStatus('✅ 自动识别完成：' + Number(data.pixels || 0).toLocaleString() + ' 个 Mask 像素。');
      console.log('[zolfox-direct-detect] success', data.width, data.height, data.pixels);
    } catch (err) {
      console.error('[zolfox-direct-detect] failed', err);
      setStatus('❌ 自动识别失败：' + (err?.message || err));
      alert('自动识别失败：' + (err?.message || err));
    } finally {
      const b = findButton();
      if (b) {
        b.disabled = false;
        delete b.dataset.zfBusy;
      }
    }
  }

  function install() {
    const btn = findButton();
    if (!btn || btn.dataset.zfDirect === '1') return;
    btn.dataset.zfDirect = '1';
    btn.addEventListener('click', runDetect, true);
    console.log('[zolfox-direct-detect] installed');
  }

  const observer = new MutationObserver(install);
  observer.observe(document.documentElement, {childList: true, subtree: true});
  install();
})();
</script>
"""


def create_app():
    app = FastAPI(title="ZOLFOX Tools")

    @app.get("/admin", include_in_schema=False)
    def admin_root():
        return RedirectResponse(url="/admin/", status_code=307)

    @app.post("/remove-watermark/api/detect")
    async def detect_api(file: UploadFile = File(...)):
        try:
            data = await file.read()
            if not data:
                return Response('{"ok":false,"detail":"文件为空"}', status_code=400, media_type="application/json")
            if len(data) > 25 * 1024 * 1024:
                return Response('{"ok":false,"detail":"图片不能超过 25MB"}', status_code=413, media_type="application/json")
            return _detect_upload(data)
        except Exception as exc:
            import traceback
            traceback.print_exc()
            return Response(
                '{"ok":false,"detail":' + __import__('json').dumps(f'{type(exc).__name__}: {exc}', ensure_ascii=False) + '}',
                status_code=500,
                media_type="application/json",
            )

    # Direct, top-level mounts. No nested Gradio mount.
    app = mount_gradio_app(app, admin_app.admin_app, path="/admin")
    app = mount_gradio_app(app, tools_app.home, path="/")
    app = mount_gradio_app(app, tools_app.remove_watermark, path="/remove-watermark")
    app = mount_gradio_app(app, tools_app.image_compress, path="/image-compress")
    return app


app = create_app()

# Inject the direct-detection browser code into the watermark page only.
# This intentionally bypasses Gradio's queue/join transport for this one action.
@app.middleware("http")
async def inject_direct_detector(request, call_next):
    response = await call_next(request)
    if request.url.path.rstrip("/") != "/remove-watermark":
        return response
    content_type = response.headers.get("content-type", "")
    if "text/html" not in content_type:
        return response
    body = b"".join([chunk async for chunk in response.body_iterator])
    marker = b"</head>"
    injection = DETECT_JS.encode("utf-8")
    if marker in body:
        body = body.replace(marker, injection + marker, 1)
    headers = dict(response.headers)
    headers.pop("content-length", None)
    headers.pop("content-encoding", None)
    return Response(content=body, status_code=response.status_code, headers=headers, media_type="text/html")


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=int(os.getenv("PORT", "7860")))
