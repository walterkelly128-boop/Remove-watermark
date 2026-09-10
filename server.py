import os

from fastapi import FastAPI
from fastapi.responses import RedirectResponse
from gradio import mount_gradio_app

import tools_app
import admin_app

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
