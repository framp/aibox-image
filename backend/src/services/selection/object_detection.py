import logging
from pathlib import Path
from typing import Protocol, cast

import numpy as np
import torch
from PIL import Image
from transformers import (
    AutoModelForZeroShotObjectDetection,
    AutoProcessor,
    Sam2Model,
    Sam2Processor,
)

DEVICE = (
    "mps"
    if torch.backends.mps.is_available()
    else "cuda"
    if torch.cuda.is_available()
    else "cpu"
)

GD_THRESHOLD = 0.25
GD_TEXT_THRESHOLD = 0.15


class ObjectDetector(Protocol):
    def image_selection(
        self, prompt: str, image: Image.Image, threshold: float
    ) -> list[Image.Image]: ...


class GroundingDINOSAM(ObjectDetector):
    def __init__(self, cache_dir: Path, gd_model: str, sam_model: str):
        logging.info(f"Loading Grounding DINO {gd_model} on {DEVICE}...")
        self.gd_processor = AutoProcessor.from_pretrained(gd_model, cache_dir=cache_dir)
        self.gd_model = AutoModelForZeroShotObjectDetection.from_pretrained(
            gd_model, cache_dir=cache_dir
        ).to(DEVICE)
        logging.info("Grounding DINO loaded successfully")

        logging.info(f"Loading SAM2 {sam_model} on {DEVICE}...")
        self.sam_processor = Sam2Processor.from_pretrained(
            sam_model, cache_dir=cache_dir
        )
        self.sam_model = Sam2Model.from_pretrained(sam_model, cache_dir=cache_dir).to(
            DEVICE
        )
        logging.info("SAM2 loaded successfully")

    def image_selection(self, prompt: str, image: Image.Image, threshold: float):
        inputs = self.gd_processor(
            images=image,
            text=[prompt],
            return_tensors="pt",
        ).to(DEVICE)

        with torch.no_grad():
            gd_outputs = self.gd_model(**inputs)

        gd_results = self.gd_processor.post_process_grounded_object_detection(
            gd_outputs,
            inputs.input_ids,
            threshold=threshold,
            text_threshold=GD_TEXT_THRESHOLD,
            target_sizes=[image.size[::-1]],
        )

        if (
            not gd_results
            or "boxes" not in gd_results[0]
            or len(gd_results[0]["boxes"]) == 0
        ):
            logging.info("No objects detected by Grounding DINO")
            return cast(list[Image.Image], [])

        boxes = gd_results[0]["boxes"]

        input_boxes = [boxes.tolist()]

        sam_inputs = self.sam_processor(
            images=image,
            input_boxes=input_boxes,
            return_tensors="pt",
        ).to(DEVICE)

        with torch.no_grad():
            sam_outputs = self.sam_model(**sam_inputs)

        masks = self.sam_processor.post_process_masks(
            sam_outputs.pred_masks.cpu(), sam_inputs["original_sizes"]
        )[0]

        # return masks
        mask_images: list[Image.Image] = []
        for i in range(masks.shape[1]):
            mask = masks[0, i].numpy()
            mask = (mask * 255).astype(np.uint8)

            image = Image.fromarray(mask, mode="L")
            mask_images.append(image)

        return mask_images
