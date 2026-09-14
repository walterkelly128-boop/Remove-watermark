from pathlib import Path

path = Path('/app/tools_app.py')
text = path.read_text(encoding='utf-8')
old = 'auto_btn.click(base.auto_detect, source, [preview, editor, status, mask_data], show_progress="minimal")'
new = 'auto_btn.click(base.auto_detect, source, [preview, editor, status, mask_data], queue=False, show_progress="minimal")'
if old not in text:
    raise SystemExit('auto_detect event registration not found')
text = text.replace(old, new, 1)
text = text.replace('source.change(base.reset_editor, source, [editor, mask_data])', 'source.change(base.reset_editor, source, [editor, mask_data], queue=False)', 1)
text = text.replace('clear_btn.click(base.reset_editor, source, [editor, mask_data])', 'clear_btn.click(base.reset_editor, source, [editor, mask_data], queue=False)', 1)
path.write_text(text, encoding='utf-8')
print('patched tools_app.py: auto_detect/source/clear use queue=False', flush=True)
