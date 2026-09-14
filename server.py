import os

from fastapi import FastAPI
from fastapi.responses import RedirectResponse
from gradio import mount_gradio_app

import tools_app
import admin_app

# IMPORTANT:
# Do not mount tools_app.app at / and then mount another Gradio app inside it.
# That creates a nested FastAPI/Gradio mount and can break Gradio event routing
# under a subpath (the page loads, uploads work, but button callbacks never
# reach the Python function). Mount every Gradio Blocks instance directly on
# the single top-level FastAPI application instead.

# The mounted remove-watermark Blocks needs its queue initialized before mount.
try:
    tools_app.remove_watermark.queue(default_concurrency_limit=1)
    print("[startup] remove-watermark Gradio queue initialized", flush=True)
except Exception as exc:
    print(f"[startup] remove-watermark queue init failed: {type(exc).__name__}: {exc}", flush=True)

app = FastAPI(title="ZOLFOX Tools")

@app.get("/admin", include_in_schema=False)
def admin_root():
    return RedirectResponse(url="/admin/", status_code=307)

# Direct, top-level mounts. No nested Gradio mount.
app = mount_gradio_app(app, admin_app.admin_app, path="/admin")
app = mount_gradio_app(app, tools_app.home, path="/")
app = mount_gradio_app(app, tools_app.remove_watermark, path="/remove-watermark")
app = mount_gradio_app(app, tools_app.image_compress, path="/image-compress")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=int(os.getenv("PORT", "7860")))
