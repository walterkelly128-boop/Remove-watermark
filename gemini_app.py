import gradio as gr

# Existing application implementation is loaded from app.py.
# Keep the original authentication, quota, recharge, Gemini, OpenAI and LaMa
# callbacks intact while exposing stable DOM IDs for the direct detector.
from app import *

# This module intentionally remains a compatibility entry point for the
# watermark-removal UI. The production UI is assembled in app.py/tools_app.py.
