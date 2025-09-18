import logging
from pathlib import Path
from typing import Literal

from PIL import Image
from pydantic import BaseModel

from inpainting_service.inpainting import StableDiffusion


class StableDiffusionParams(BaseModel):
    type: Literal["stable_diffusion"]
    model_or_checkpoint: str
    cache_dir: Path


LoadParams = StableDiffusionParams


class Service:
    def __init__(self):
        self.model = None

    def load(self, params: LoadParams):
        self.model = None
        logging.info("Loading model...")

        try:
            match params.type:
                case "stable_diffusion":
                    self.model = StableDiffusion(
                        cache_dir=params.cache_dir,
                        model_or_checkpoint=params.model_or_checkpoint,
                    )
        except Exception:
            logging.warning("Could not load LLM model", exc_info=True)
            logging.info("Image inpaint functionality will be disabled")

    def health(self):
        if self.model is None:
            return False
        return True

    def inpaint(self, prompt: str, image: Image.Image, mask: Image.Image):
        if not self.health():
            raise Exception("Service is not healthy")

        return self.model.inpaint(prompt, image, mask)
