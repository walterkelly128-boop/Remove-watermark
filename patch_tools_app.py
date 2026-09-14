from pathlib import Path

path = Path('/app/tools_app.py')
text = path.read_text(encoding='utf-8')

# This patch deliberately removes the automatic-detection Gradio event.
# Automatic detection is handled by the plain FastAPI endpoint in server.py.
# Keeping the button free of a Gradio .click() handler prevents queue/join from
# being used for detection at all.
old_event = 'auto_btn.click(base.auto_detect,source,[preview,editor,status,mask_data],queue=False,show_progress="hidden")'
if old_event in text:
    text = text.replace(old_event, '', 1)
elif 'auto_btn.click(base.auto_detect' in text:
    raise SystemExit('unexpected auto_detect event registration format')

# Give the direct browser implementation stable DOM targets.
replacements = {
    'source=gr.Image(label="原图",type="pil")': 'source=gr.Image(label="原图",type="pil",elem_id="wm-source")',
    'auto_btn=gr.Button("✨ 自动识别候选区域",variant="primary")': 'auto_btn=gr.Button("✨ 自动识别候选区域",variant="primary",elem_id="wm-auto-detect")',
    'clear_btn=gr.Button("清除 Mask")': 'clear_btn=gr.Button("清除 Mask",elem_id="wm-clear-mask")',
    'status=gr.Markdown("上传图片后开始。")': 'status=gr.Markdown("上传图片后开始。",elem_id="wm-status")',
    'gr.Column(): preview=gr.Image(label="识别预览",type="pil")': 'gr.Column(): preview=gr.Image(label="识别预览",type="pil",elem_id="wm-preview")',
    'gr.Markdown("## Mask 编辑器"); editor=gr.HTML(label="Mask 编辑器");': 'gr.Markdown("## Mask 编辑器"); editor=gr.HTML(label="Mask 编辑器",elem_id="wm-editor");',
}
for old, new in replacements.items():
    if old not in text:
        raise SystemExit(f'expected UI fragment not found: {old}')
    text = text.replace(old, new, 1)

# The upload/reset handler is still useful for clearing the editor, but it is
# not involved in automatic detection. Keep it synchronous where supported.
old_source = 'source.change(base.reset_editor,source,[editor,mask_data],queue=False)'
if old_source not in text:
    raise SystemExit('source reset handler not found')

old_clear = 'clear_btn.click(base.reset_editor,source,[editor,mask_data],queue=False)'
if old_clear not in text:
    raise SystemExit('clear mask handler not found')

DIRECT_DETECT_HEAD = r'''\n<script>\n(() => {\n  if (window.__zolfoxDirectDetectV2) return;\n  window.__zolfoxDirectDetectV2 = true;\n\n  const API = '/remove-watermark/api/detect';\n\n  const $ = (sel) => document.querySelector(sel);\n\n  function setNativeValue(el, value) {\n    if (!el) return;\n    const proto = el instanceof HTMLTextAreaElement ? HTMLTextAreaElement.prototype : HTMLInputElement.prototype;\n    const setter = Object.getOwnPropertyDescriptor(proto, 'value')?.set;\n    if (setter) setter.call(el, value); else el.value = value;\n    el.dispatchEvent(new Event('input', {bubbles: true}));\n    el.dispatchEvent(new Event('change', {bubbles: true}));\n  }\n\n  function setStatus(text) {\n    const el = $('#wm-status');\n    if (el) el.textContent = text;\n  }\n\n  function findSourceImage() {\n    const root = $('#wm-source');\n    if (!root) return null;\n    const imgs = [...root.querySelectorAll('img')];\n    return imgs.find(img => {\n      const src = img.currentSrc || img.src || '';\n      return src && (src.includes('/gradio_api/file') || src.startsWith('blob:') || src.startsWith('data:image'));\n    }) || imgs.find(img => img.currentSrc || img.src) || null;\n  }\n\n  function setPreview(src) {\n    const root = $('#wm-preview');\n    if (!root) return false;\n    let img = root.querySelector('img');\n    if (!img) {\n      img = document.createElement('img');\n      img.style.maxWidth = '100%';\n      img.style.maxHeight = '620px';\n      img.style.objectFit = 'contain';\n      root.appendChild(img);\n    }\n    img.src = src;\n    img.style.display = 'block';\n    return true;\n  }\n\n  function setEditor(html) {\n    const root = $('#wm-editor');\n    if (!root) return false;\n    root.innerHTML = html || '';\n    return true;\n  }\n\n  async function detect() {\n    const button = $('#wm-auto-detect button') || $('#wm-auto-detect');\n    if (button?.dataset.zfBusy === '1') return;\n    if (button) { button.dataset.zfBusy = '1'; button.disabled = true; }\n\n    try {\n      const source = findSourceImage();\n      if (!source) throw new Error('请先上传图片');\n      const src = source.currentSrc || source.src;\n      if (!src) throw new Error('无法读取上传图片');\n\n      setStatus('⏳ 正在自动识别候选区域，请稍候…');\n      const imageResponse = await fetch(src, {credentials: 'same-origin', cache: 'no-store'});\n      if (!imageResponse.ok) throw new Error('读取图片失败：HTTP ' + imageResponse.status);\n      const blob = await imageResponse.blob();\n\n      const form = new FormData();\n      form.append('file', blob, 'source.png');\n      const response = await fetch(API, {\n        method: 'POST',\n        body: form,\n        credentials: 'same-origin',\n        cache: 'no-store'\n      });\n      const data = await response.json();\n      if (!response.ok || !data.ok) throw new Error(data.detail || '自动识别失败');\n\n      if (!setPreview('data:image/png;base64,' + data.preview)) {\n        throw new Error('找不到识别预览区域');\n      }\n      setNativeValue($('#mask-data textarea') || $('#mask-data input'), data.mask || '');\n      setEditor(data.editor || '');\n      setStatus('✅ 自动识别完成：' + Number(data.pixels || 0).toLocaleString() + ' 个 Mask 像素。');\n      console.log('[zolfox-direct-detect-v2] success', data.width, data.height, data.pixels);\n    } catch (err) {\n      console.error('[zolfox-direct-detect-v2] failed', err);\n      setStatus('❌ 自动识别失败：' + (err?.message || err));\n      alert('自动识别失败：' + (err?.message || err));\n    } finally {\n      if (button) { button.disabled = false; delete button.dataset.zfBusy; }\n    }\n  }\n\n  function install() {\n    const root = $('#wm-auto-detect');\n    const button = root?.querySelector('button') || root;\n    if (!button || button.dataset.zfDirect === '1') return;\n    button.dataset.zfDirect = '1';\n    button.addEventListener('click', (event) => {\n      event.preventDefault();\n      event.stopPropagation();\n      event.stopImmediatePropagation();\n      detect();\n    }, true);\n    console.log('[zolfox-direct-detect-v2] installed');\n  }\n\n  const observer = new MutationObserver(install);\n  observer.observe(document.documentElement, {childList: true, subtree: true});\n  install();\n})();\n</script>\n'''

# Inject the direct handler into the Gradio page itself. This is more reliable
# than modifying the HTTP response after Gradio has generated its HTML.
marker = 'def remove_demo():'
if marker not in text:
    raise SystemExit('remove_demo marker not found')
if 'DIRECT_DETECT_HEAD' not in text:
    text = text.replace(marker, 'DIRECT_DETECT_HEAD = ' + repr(DIRECT_DETECT_HEAD) + '\n\n' + marker, 1)

old_head = 'head=base.EDITOR_JS) as demo:'
new_head = 'head=base.EDITOR_JS + DIRECT_DETECT_HEAD) as demo:'
if old_head not in text:
    raise SystemExit('remove_demo head argument not found')
text = text.replace(old_head, new_head, 1)

path.write_text(text, encoding='utf-8')
print('patched tools_app.py: removed Gradio auto-detect event, added stable DOM ids, and injected direct FastAPI detector JS', flush=True)
