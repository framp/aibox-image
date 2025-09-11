import argparse
import signal
import sys
import time
from pathlib import Path

import cv2
import msgpack
import numpy as np
import torch
import zmq
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
GROUNDING_DINO_MODEL = "IDEA-Research/grounding-dino-base"
SAM_MODEL = "facebook/sam2.1-hiera-large"

GD_THRESHOLD = 0.25
GD_TEXT_THRESHOLD = 0.15


class Service:
    def __init__(self, port: int = 5557):
        self.port = port
        self.context = zmq.Context()
        self.socket = None
        self.running = False

        self.supported_formats = {".jpg", ".jpeg", ".png", ".bmp", ".tiff", ".webp"}

        self._init_models()

    def _init_models(self):
        try:
            print(f"Loading Grounding DINO {GROUNDING_DINO_MODEL} on {DEVICE}...")
            self.gd_processor = AutoProcessor.from_pretrained(GROUNDING_DINO_MODEL)
            self.gd_model = AutoModelForZeroShotObjectDetection.from_pretrained(
                GROUNDING_DINO_MODEL
            ).to(DEVICE)
            print("Grounding DINO loaded successfully")

            print(f"Loading SAM2 {SAM_MODEL} on {DEVICE}...")
            self.sam_processor = Sam2Processor.from_pretrained(SAM_MODEL)
            self.sam_model = Sam2Model.from_pretrained(SAM_MODEL).to(DEVICE)
            print("SAM2 loaded successfully")

        except Exception as e:
            print(f"Warning: Could not load LLM model: {e}")
            import traceback

            traceback.print_exc()
            print("Image selection functionality will be disabled")

    def image_selection(self, prompt: str, image_path: str, threshold: float = 0.25):
        image_path = Path(image_path)

        if not image_path.exists():
            raise FileNotFoundError(f"Image file not found: {image_path}")

        if image_path.suffix.lower() not in self.supported_formats:
            raise ValueError(f"Unsupported image format: {image_path.suffix}")

        image = Image.open(image_path).convert("RGB")

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
            print("No objects detected by Grounding DINO")
            return []

        # Extract bounding boxes
        boxes = gd_results[0]["boxes"]

        # --- Prepare SAM2 inputs ---
        input_boxes = [boxes.tolist()]  # add batch dim -> shape (1, num_boxes, 4)

        # Prepare SAM2 inputs
        sam_inputs = self.sam_processor(
            images=image,
            input_boxes=input_boxes,
            return_tensors="pt",
        ).to(DEVICE)

        # --- SAM2: generate masks ---
        with torch.no_grad():
            sam_outputs = self.sam_model(**sam_inputs)

        # Resize masks to original image size
        masks = self.sam_processor.post_process_masks(
            sam_outputs.pred_masks.cpu(), sam_inputs["original_sizes"]
        )[0]  # shape: (1, num_masks, H, W)

        # --- Encode masks as PNGs ---
        masks_bytes = []
        for i in range(masks.shape[1]):
            mask = masks[0, i].numpy()  # 2D mask

            # Convert to alpha mask
            mask = (mask * 255).astype(np.uint8)

            # Encode as PNG
            is_success, buffer = cv2.imencode(".png", mask)
            if not is_success:
                raise RuntimeError(f"Failed to encode mask {i}")

            masks_bytes.append(buffer.tobytes())

        return masks_bytes

    def start(self):
        print(f"Starting Selection Service on port {self.port}")

        self.socket = self.context.socket(zmq.REP)
        self.socket.bind(f"tcp://*:{self.port}")
        self.running = True

        print("Selection Service ready! Waiting for requests...")

        while self.running:
            try:
                message = self.socket.recv()
                request = msgpack.unpackb(message, raw=False)

                print(f"Received request: {request.get('action', 'unknown')}")

                if request.get("action") == "image_selection":
                    prompt = request.get("prompt")
                    image_path = request.get("image_path")
                    threshold = request.get("threshold", 0.25)

                    try:
                        selections = self.image_selection(prompt, image_path, threshold)

                        response = {"status": "success", "masks": selections}
                        print(f"Extracted {len(selections)} selections from image")
                    except Exception as e:
                        response = {"status": "error", "message": str(e)}
                        print(f"Error processing request: {e}")

                elif request.get("action") == "health":
                    try:
                        gd_device = (
                            str(self.gd_model.device)
                            if self.gd_model
                            else "unavailable"
                        )
                    except:
                        gd_device = "unavailable"

                    try:
                        sam_device = (
                            str(self.sam_model.device)
                            if self.sam_model
                            else "unavailable"
                        )
                    except:
                        sam_device = "unavailable"

                    response = {
                        "status": "healthy",
                        "service": "llm",
                        "sam_model": SAM_MODEL,
                        "sam_device": sam_device,
                        "gd_model": GROUNDING_DINO_MODEL,
                        "gd_device": gd_device,
                        "timestamp": time.time(),
                    }

                elif request.get("action") == "shutdown":
                    response = {
                        "status": "shutting_down",
                        "message": "Selection Service shutting down",
                    }
                    self.running = False

                else:
                    response = {
                        "status": "error",
                        "message": f"Unknown action: {request.get('action')}",
                    }

                self.socket.send(msgpack.packb(response, use_bin_type=True))

            except KeyboardInterrupt:
                print("\nReceived interrupt signal")
                break
            except Exception as e:
                print(f"Error processing request: {e}")
                try:
                    error_response = {
                        "status": "error",
                        "message": f"Service error: {str(e)}",
                    }
                    self.socket.send(msgpack.packb(error_response, use_bin_type=True))
                except:
                    pass

        self.stop()

    def stop(self):
        print("Stopping Selection Service...")
        self.running = False
        if self.socket:
            self.socket.close()
        self.context.term()
        print("Selection Service stopped")


def signal_handler(sig, frame):
    print("\nReceived shutdown signal")
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

    args = parser.parse_args()

    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)

    service = Service(port=args.port)

    try:
        service.start()
    except KeyboardInterrupt:
        print("\nShutdown requested")
    except Exception as e:
        print(f"Service error: {e}")
    finally:
        service.stop()


if __name__ == "__main__":
    main()
