import logging
from pathlib import Path
from typing import Literal

from PIL import Image
from pydantic import BaseModel

from .face_swapping import FaceSwapper


class FaceSwappingParams(BaseModel):
    type: Literal["face_swapping"]
    insightface_model: str = ""
    inswapper_model: str = ""
    cache_dir: Path


LoadParams = FaceSwappingParams


class Service:
    def __init__(self):
        self.model = None

    def load(self, params: LoadParams):
        self.model = None
        logging.info("Loading model...")

        try:
            match params.type:
                case "face_swapping":
                    self.model = FaceSwapper(
                        cache_dir=params.cache_dir,
                        insightface_model=params.insightface_model,
                        inswapper_model=params.inswapper_model,
                    )
        except Exception:
            logging.warning("Could not load LLM model", exc_info=True)
            logging.info("Face swapping functionality will be disabled")

    def health(self):
        if self.model is None:
            return False
        return True

    def swap_face(
        self, source_image: Image.Image, target_image: Image.Image, face_index: int
    ):
        if not self.health():
            raise Exception("Service is not healthy")

        return self.model.swap_face(source_image, target_image, face_index)

    def analyze_faces_with_preview(self, source_image: Image.Image):
        if not self.health():
            raise Exception("Service is not healthy")

        return self.model.analyze_faces_with_preview(source_image)
