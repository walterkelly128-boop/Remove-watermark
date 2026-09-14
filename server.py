import os

from fastapi import FastAPI
from fastapi.responses import RedirectResponse
from gradio import mount_gradio_app

import tools_app
import admin_app

# Gradio mounted under FastAPI needs its queue infrastructure initialized before
# mount_gradio_app(). Individual lightweight events such as auto-detect still use
# queue=False, but the Blocks queue itself must exist for the mounted app to
# initialize and handle its API dependencies correctly.
try:
    tools_app.remove_watermark.queue(default_concurrency_limit=1)
    print("[startup] remove-watermark Gradio queue initialized", flush=True)
except Exception as exc:
    print(f"[startup] remove-watermark queue init failed: {type(exc).__name__}: {exc}", flush=True)

app = FastAPI(title="ZOLFOX Tools")

# Make both /admin and /admin/ work. Gradio's mounted app uses the trailing-slash
# form internally, while users should be able to type /admin directly.
@app.get("/admin", include_in_schema=False)
def admin_root():
    return RedirectResponse(url="/admin/", status_code=307)

app = mount_gradio_app(app, admin_app.admin_app, path="/admin")
app.mount("/", tools_app.app)

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=int(os.getenv("PORT", "7860")))
