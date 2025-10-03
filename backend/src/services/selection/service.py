import logging
from typing import Literal

from PIL import Image
from pydantic import BaseModel

from .object_detection import GroundingDINOSAM


class GroundingDinoSAMParams(BaseModel):
    type: Literal["grounding_dino_sam"]
    grounding_dino_model: str
    sam_model: str
    cache_dir: str


LoadParams = GroundingDinoSAMParams


class Service:
    def __init__(self):
        self.model = None

    def load(self, params: LoadParams):
        self.model = None
        logging.info("Loading model...")

        try:
            match params.type:
                case "grounding_dino_sam":
                    self.model = GroundingDINOSAM(
                        cache_dir=params.cache_dir,
                        gd_model=params.grounding_dino_model,
                        sam_model=params.sam_model,
                    )
        except Exception:
            logging.warning("Could not load LLM model", exc_info=True)
            logging.info("Image selection functionality will be disabled")

    def health(self):
        if self.model is None:
            return False
        return True

    def image_selection(self, prompt: str, image: Image.Image, threshold: float):
        if not self.health():
            raise Exception("Service is not healthy")

        return self.model.image_selection(prompt, image, threshold)
