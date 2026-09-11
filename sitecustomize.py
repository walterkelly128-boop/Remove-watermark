"""Small Gradio compatibility patch for the mounted tools app.

The remove-watermark page is built in tools_app.py, which imports
base.auto_detect from app.py.  Its event registration historically omitted
queue=False, so Gradio can put candidate detection behind queue/join when the
app is mounted under /remove-watermark/.  Python imports sitecustomize during
startup, allowing us to apply a narrowly-scoped default without rewriting the
large tools_app.py file.
"""

try:
    import gradio as gr

    _original_click = gr.Button.click

    def _click_with_auto_detect_direct(self, *args, **kwargs):
        fn = kwargs.get("fn")
        if fn is None and args:
            # Gradio's click(fn, ...) signature has fn as the first positional
            # argument in the supported versions used by this project.
            fn = args[0]
        if getattr(fn, "__name__", "") == "auto_detect":
            kwargs.setdefault("queue", False)
            kwargs.setdefault("show_progress", "minimal")
        return _original_click(self, *args, **kwargs)

    gr.Button.click = _click_with_auto_detect_direct
except Exception:
    # Never prevent the application from starting if Gradio changes its API.
    pass
