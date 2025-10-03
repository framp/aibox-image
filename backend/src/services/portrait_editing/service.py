import logging
from pathlib import Path
from typing import Literal

from PIL import Image
from pydantic import BaseModel

from .portrait_editing import LivePortraitEngine


class LivePortraitParams(BaseModel):
    type: Literal["liveportrait"]
    cache_dir: Path


LoadParams = LivePortraitParams


class Service:
    def __init__(self):
        self.model = None

    def load(self, params: LoadParams):
        self.model = None
        logging.info("Loading LivePortrait model...")

        try:
            match params.type:
                case "liveportrait":
                    self.model = LivePortraitEngine(cache_dir=params.cache_dir)
                    logging.info("LivePortrait model loaded and ready")
        except Exception:
            logging.warning("Could not load LivePortrait model", exc_info=True)
            logging.info("Portrait editing functionality will be disabled")

    def health(self):
        if self.model is None:
            return False
        return True

    def edit_expression(self, image: Image.Image, expression_params: dict):
        if not self.health():
            raise Exception("Service is not healthy")

        return self.model.edit_expression(image, expression_params)
