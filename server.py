import os

from fastapi import FastAPI
from gradio import mount_gradio_app

import tools_app
import admin_app

# Build a clean parent application. Mount /admin first, then mount the
# existing ZOLFOX application at /. This avoids the root Gradio mount
# intercepting /admin requests.
app = FastAPI(title="ZOLFOX Tools")
app = mount_gradio_app(app, admin_app.admin_app, path="/admin")
app.mount("/", tools_app.app)

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=int(os.getenv("PORT", "7860")))
