import os

from fastapi import FastAPI
from gradio import mount_gradio_app

import tools_app
import admin_app

# tools_app already mounts the homepage at "/". In Starlette/FastAPI route
# matching is ordered, so the catch-all homepage mount must come AFTER /admin.
app = tools_app.app
app = mount_gradio_app(app, admin_app.admin_app, path="/admin")

# Move the /admin mount before the root (/) mount so /admin is not swallowed
# by the homepage Gradio application.
routes = app.router.routes
admin_routes = [r for r in routes if getattr(r, "path", "") == "/admin"]
other_routes = [r for r in routes if getattr(r, "path", "") != "/admin"]
root_routes = [r for r in other_routes if getattr(r, "path", "") == "/"]
non_root_routes = [r for r in other_routes if getattr(r, "path", "") != "/"]
app.router.routes = admin_routes + non_root_routes + root_routes

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=int(os.getenv("PORT", "7860")))
