import os

from fastapi import FastAPI
from gradio import mount_gradio_app

import tools_app
import admin_app

app = tools_app.app
app = mount_gradio_app(app, admin_app.admin_app, path="/admin")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=int(os.getenv("PORT", "7860")))
