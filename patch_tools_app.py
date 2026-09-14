from pathlib import Path

path = Path('/app/tools_app.py')
text = path.read_text(encoding='utf-8')

# The source file may already contain the fix. Keep this build-time patch
# idempotent so Docker rebuilds never fail just because Git already has it.
old_auto = 'auto_btn.click(base.auto_detect, source, [preview, editor, status, mask_data], show_progress="minimal")'
new_auto = 'auto_btn.click(base.auto_detect, source, [preview, editor, status, mask_data], queue=False, show_progress="hidden")'
if old_auto in text:
    text = text.replace(old_auto, new_auto, 1)
elif 'auto_btn.click(base.auto_detect, source, [preview, editor, status, mask_data], queue=False' in text:
    pass
else:
    raise SystemExit('auto_detect event registration not found')

old_source = 'source.change(base.reset_editor, source, [editor, mask_data])'
if old_source in text:
    text = text.replace(old_source, 'source.change(base.reset_editor, source, [editor, mask_data], queue=False)', 1)

old_clear = 'clear_btn.click(base.reset_editor, source, [editor, mask_data])'
if old_clear in text:
    text = text.replace(old_clear, 'clear_btn.click(base.reset_editor, source, [editor, mask_data], queue=False)', 1)

path.write_text(text, encoding='utf-8')
print('patched tools_app.py: auto_detect/source/clear queue configuration verified', flush=True)
