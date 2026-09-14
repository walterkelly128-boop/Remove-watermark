from pathlib import Path
import re

# Patch the Gradio page so watermark detection never registers a Gradio event.
path = Path('/app/tools_app.py')
text = path.read_text(encoding='utf-8')

old_event = 'auto_btn.click(base.auto_detect,source,[preview,editor,status,mask_data],queue=False,show_progress="hidden")'
if old_event in text:
    text = text.replace(old_event, '', 1)
elif 'auto_btn.click(base.auto_detect' in text:
    raise SystemExit('unexpected auto_detect event registration format')

replacements = {
    'source=gr.Image(label="原图",type="pil")': 'source=gr.Image(label="原图",type="pil",elem_id="wm-source")',
    'auto_btn=gr.Button("✨ 自动识别候选区域",variant="primary")': 'auto_btn=gr.Button("✨ 自动识别候选区域",variant="primary",elem_id="wm-auto-detect")',
    'clear_btn=gr.Button("清除 Mask")': 'clear_btn=gr.Button("清除 Mask",elem_id="wm-clear-mask")',
    'status=gr.Markdown("上传图片后开始。")': 'status=gr.Markdown("上传图片后开始。",elem_id="wm-status")',
    'gr.Column(): preview=gr.Image(label="识别预览",type="pil")': 'gr.Column(): preview=gr.Image(label="识别预览",type="pil",elem_id="wm-preview")',
    'gr.Markdown("## Mask 编辑器"); editor=gr.HTML(label="Mask 编辑器");': 'gr.Markdown("## Mask 编辑器"); editor=gr.HTML(label="Mask 编辑器",elem_id="wm-editor");',
}
for old, new in replacements.items():
    if old in text and new not in text:
        text = text.replace(old, new, 1)

if 'source.change(base.reset_editor,source,[editor,mask_data],queue=False)' not in text:
    raise SystemExit('source reset handler not found')
if 'clear_btn.click(base.reset_editor,source,[editor,mask_data],queue=False)' not in text:
    raise SystemExit('clear mask handler not found')

DIRECT_DETECT_HEAD = r'''<script>
(() => {
  if (window.__zolfoxDirectDetectV2) return;
  window.__zolfoxDirectDetectV2 = true;
  const API = '/remove-watermark/api/detect';
  const $ = (sel) => document.querySelector(sel);

  function setNativeValue(el, value) {
    if (!el) return;
    const proto = el instanceof HTMLTextAreaElement ? HTMLTextAreaElement.prototype : HTMLInputElement.prototype;
    const setter = Object.getOwnPropertyDescriptor(proto, 'value')?.set;
    if (setter) setter.call(el, value); else el.value = value;
    el.dispatchEvent(new Event('input', {bubbles: true}));
    el.dispatchEvent(new Event('change', {bubbles: true}));
  }
  function setStatus(text) {
    const el = $('#wm-status');
    if (el) el.textContent = text;
  }
  function findSourceImage() {
    const root = $('#wm-source');
    if (!root) return null;
    const imgs = [...root.querySelectorAll('img')];
    return imgs.find(img => {
      const src = img.currentSrc || img.src || '';
      return src && (src.includes('/gradio_api/file') || src.startsWith('blob:') || src.startsWith('data:image'));
    }) || imgs.find(img => img.currentSrc || img.src) || null;
  }
  function setPreview(src) {
    const root = $('#wm-preview');
    if (!root) return false;
    let img = root.querySelector('img');
    if (!img) {
      img = document.createElement('img');
      img.style.maxWidth = '100%';
      img.style.maxHeight = '620px';
      img.style.objectFit = 'contain';
      root.appendChild(img);
    }
    img.src = src;
    img.style.display = 'block';
    return true;
  }
  function setEditor(html) {
    const root = $('#wm-editor');
    if (!root) return false;
    root.innerHTML = html || '';
    return true;
  }
  async function detect() {
    const root = $('#wm-auto-detect');
    const button = root?.querySelector('button') || root;
    if (button?.dataset.zfBusy === '1') return;
    if (button) { button.dataset.zfBusy = '1'; button.disabled = true; }
    try {
      const source = findSourceImage();
      if (!source) throw new Error('请先上传图片');
      const src = source.currentSrc || source.src;
      if (!src) throw new Error('无法读取上传图片');
      setStatus('⏳ 正在自动识别候选区域，请稍候…');
      const imageResponse = await fetch(src, {credentials: 'same-origin', cache: 'no-store'});
      if (!imageResponse.ok) throw new Error('读取图片失败：HTTP ' + imageResponse.status);
      const blob = await imageResponse.blob();
      const form = new FormData();
      form.append('file', blob, 'source.png');
      const response = await fetch(API, {method: 'POST', body: form, credentials: 'same-origin', cache: 'no-store'});
      const data = await response.json();
      if (!response.ok || !data.ok) throw new Error(data.detail || '自动识别失败');
      if (!setPreview('data:image/png;base64,' + data.preview)) throw new Error('找不到识别预览区域');
      setNativeValue($('#mask-data textarea') || $('#mask-data input'), data.mask || '');
      setEditor(data.editor || '');
      setStatus('✅ 自动识别完成：' + Number(data.pixels || 0).toLocaleString() + ' 个 Mask 像素。');
      console.log('[zolfox-direct-detect-v2] success', data.width, data.height, data.pixels);
    } catch (err) {
      console.error('[zolfox-direct-detect-v2] failed', err);
      setStatus('❌ 自动识别失败：' + (err?.message || err));
      alert('自动识别失败：' + (err?.message || err));
    } finally {
      if (button) { button.disabled = false; delete button.dataset.zfBusy; }
    }
  }
  function install() {
    const root = $('#wm-auto-detect');
    const button = root?.querySelector('button') || root;
    if (!button || button.dataset.zfDirect === '1') return;
    button.dataset.zfDirect = '1';
    button.addEventListener('click', (event) => {
      event.preventDefault();
      event.stopPropagation();
      event.stopImmediatePropagation();
      detect();
    }, true);
    console.log('[zolfox-direct-detect-v2] installed');
  }
  const observer = new MutationObserver(install);
  observer.observe(document.documentElement, {childList: true, subtree: true});
  install();
})();
</script>'''

marker = 'def remove_demo():'
if marker not in text:
    raise SystemExit('remove_demo marker not found')
if 'DIRECT_DETECT_HEAD = ' not in text:
    text = text.replace(marker, 'DIRECT_DETECT_HEAD = ' + repr(DIRECT_DETECT_HEAD) + '\n\n' + marker, 1)
old_head = 'head=base.EDITOR_JS) as demo:'
new_head = 'head=base.EDITOR_JS + DIRECT_DETECT_HEAD) as demo:'
if old_head in text:
    text = text.replace(old_head, new_head, 1)
elif new_head not in text:
    raise SystemExit('remove_demo head argument not found')
path.write_text(text, encoding='utf-8')

# Remove the older HTTP-response JS injector from server.py. The detector JS
# now lives in the Gradio Blocks head, which is reliable and avoids duplicate
# capture listeners fighting over the same button.
server_path = Path('/app/server.py')
server = server_path.read_text(encoding='utf-8')
pattern = r'\n# Inject the direct-detection browser code into the watermark page only\..*?(?=\n\nif __name__ == "__main__":)'
server_new, count = re.subn(pattern, '', server, flags=re.S)
if count:
    server_path.write_text(server_new, encoding='utf-8')

print('patched watermark detection: no Gradio auto-detect event, stable DOM ids, direct FastAPI JS in Gradio head, old server injector removed', flush=True)
