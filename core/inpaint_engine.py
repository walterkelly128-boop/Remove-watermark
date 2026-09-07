from __future__ import annotations

from PIL import Image
from simple_lama_inpainting import SimpleLama


class InpaintEngine:
    def __init__(self):
        self._model = None

    def _get_model(self):
        if self._model is None:
            self._model = SimpleLama()
        return self._model

    def run(self, image: Image.Image, mask: Image.Image) -> Image.Image:
        model = self._get_model()
        return model(image.convert("RGB"), mask.convert("L"))
