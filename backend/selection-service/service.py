import argparse
import logging
import signal
import sys
import time
import traceback
from pathlib import Path
from typing import List

import cv2
import msgpack
import numpy as np
import torch
import zmq
from aibox_image_lib.transport import BaseRequest, BaseResponse, Transport
from aibox_image_lib.transport.zmq import ZmqTransport
from PIL import Image
from pydantic import BaseModel
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
GROUNDING_DINO_MODEL = "IDEA-Research/grounding-dino-base"
SAM_MODEL = "facebook/sam2.1-hiera-large"

GD_THRESHOLD = 0.25
GD_TEXT_THRESHOLD = 0.15


class ImageSelectionRequest(BaseRequest):
    prompt: str
    image_bytes: bytes
    threshold: float = 0.25


class ImageSelectionResponse(BaseResponse):
    status: str
    masks: List[bytes]


class Service:
    def __init__(self, cache_dir: Path):
        self.cache_dir = cache_dir
        self.supported_formats = {".jpg", ".jpeg", ".png", ".bmp", ".tiff", ".webp"}

        self._init_models()

    def _init_models(self):
        try:
            logging.info(
                f"Loading Grounding DINO {GROUNDING_DINO_MODEL} on {DEVICE}..."
            )
            self.gd_processor = AutoProcessor.from_pretrained(
                GROUNDING_DINO_MODEL, cache_dir=self.cache_dir
            )
            self.gd_model = AutoModelForZeroShotObjectDetection.from_pretrained(
                GROUNDING_DINO_MODEL, cache_dir=self.cache_dir
            ).to(DEVICE)
            logging.info("Grounding DINO loaded successfully")

            logging.info(f"Loading SAM2 {SAM_MODEL} on {DEVICE}...")
            self.sam_processor = Sam2Processor.from_pretrained(
                SAM_MODEL, cache_dir=self.cache_dir
            )
            self.sam_model = Sam2Model.from_pretrained(
                SAM_MODEL, cache_dir=self.cache_dir
            ).to(DEVICE)
            logging.info("SAM2 loaded successfully")

        except Exception:
            logging.warning("Could not load LLM model", exc_info=True)
            logging.info("Image selection functionality will be disabled")

    def image_selection(self, prompt: str, image_bytes: bytes, threshold: float = 0.25):
        import io

        image = Image.open(io.BytesIO(image_bytes)).convert("RGB")

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
            return []

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

        masks_bytes = []
        for i in range(masks.shape[1]):
            mask = masks[0, i].numpy()
            mask = (mask * 255).astype(np.uint8)
            is_success, buffer = cv2.imencode(".png", mask)
            if not is_success:
                raise RuntimeError(f"Failed to encode mask {i}")

            masks_bytes.append(buffer.tobytes())

        return masks_bytes


def signal_handler(sig, frame):
    logging.info("Received shutdown signal")
    sys.exit(0)


def main():
    parser = argparse.ArgumentParser(description="LLM Text Processing Service")
    parser.add_argument(
        "--port",
        type=int,
        default=5558,
        help="Port to bind the service (default: 5558)",
    )
    parser.add_argument("--debug", action="store_true", help="Enable debug logging")
    parser.add_argument(
        "--cache-dir",
        type=str,
        default="../../model-cache",
        help="Model cache directory",
    )

    args = parser.parse_args()

    logging.basicConfig(level=logging.DEBUG if args.debug else logging.INFO)

    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)

    transport = ZmqTransport()
    service = Service(cache_dir=args.cache_dir)

    @transport.handler("image_selection")
    def image_selection(request: ImageSelectionRequest):
        masks = service.image_selection(
            request.prompt, request.image_bytes, request.threshold
        )
        return ImageSelectionResponse(status="success", masks=masks)

    try:
        transport.start(port=args.port)
    except KeyboardInterrupt:
        logging.info("Shutdown requested")
    except Exception:
        logging.error("Service error", exc_info=True)
    finally:
        transport.stop()


if __name__ == "__main__":
    main()
