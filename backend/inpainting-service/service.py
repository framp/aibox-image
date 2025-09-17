import argparse
import io
import os
import signal
import sys
import time
from pathlib import Path
from typing import Optional

import msgpack
import torch
import zmq
from diffusers import StableDiffusionInpaintPipeline
from dotenv import load_dotenv
from huggingface_hub import login
from PIL import Image

DEVICE = (
    "mps"
    if torch.backends.mps.is_available()
    else "cuda"
    if torch.cuda.is_available()
    else "cpu"
)
SD_MODEL = "stable-diffusion-v1-5/stable-diffusion-inpainting"

load_dotenv()
login(token=os.environ["HF_TOKEN"])


def get_dimensions_divisible_by_8(width: int, height: int) -> tuple[int, int]:
    """Round dimensions to closest numbers divisible by 8."""
    return ((width + 4) // 8) * 8, ((height + 4) // 8) * 8


class Service:
    def __init__(
        self, cache_dir: Path, port: int = 5557, custom_checkpoint: Optional[str] = None
    ):
        self.port = port
        self.context = zmq.Context()
        self.socket: Optional[zmq.Socket] = None
        self.running = False
        self.custom_checkpoint = custom_checkpoint
        self.model: Optional[StableDiffusionInpaintPipeline] = None
        self.model_name: str = ""
        self.cache_dir = cache_dir

        self.supported_formats = {".jpg", ".jpeg", ".png", ".bmp", ".tiff", ".webp"}

        self._init_models()

    def _init_models(self) -> None:
        try:
            if self.custom_checkpoint:
                print(
                    f"Loading custom checkpoint: {self.custom_checkpoint} on {DEVICE}..."
                )
                self.model_name = self.custom_checkpoint
                # Check if it's a single file (.safetensors or .ckpt)
                if self.custom_checkpoint.endswith((".safetensors", ".ckpt")):
                    self.model = StableDiffusionInpaintPipeline.from_single_file(
                        self.custom_checkpoint,
                        torch_dtype=torch.float16,
                        cache_dir=self.cache_dir,
                    ).to(DEVICE)
                else:
                    # It's a directory or repo ID
                    self.model = StableDiffusionInpaintPipeline.from_pretrained(
                        self.custom_checkpoint,
                        torch_dtype=torch.float16,
                        cache_dir=self.cache_dir,
                    ).to(DEVICE)
            else:
                print(f"Loading {SD_MODEL} on {DEVICE}...")
                self.model_name = SD_MODEL
                self.model = StableDiffusionInpaintPipeline.from_pretrained(
                    SD_MODEL,
                    torch_dtype=torch.float16,
                    cache_dir=self.cache_dir,
                ).to(DEVICE)

            self.model.safety_checker = None
            self.model.requires_safety_checker = False

            print("Model loaded successfully")
        except Exception as e:
            print(f"Warning: Could not load model: {e}")
            import traceback

            traceback.print_exc()
            self.model = None

    def inpaint(self, prompt: str, image_bytes: bytes, mask: bytes) -> bytes:
        if self.model is None:
            raise RuntimeError("Model not loaded")

        image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
        mask = Image.open(io.BytesIO(mask)).convert("RGB")

        original_width, original_height = image.size
        gen_width, gen_height = get_dimensions_divisible_by_8(
            original_width, original_height
        )

        if (gen_width, gen_height) != (original_width, original_height):
            image = image.resize((gen_width, gen_height))
            mask = mask.resize((gen_width, gen_height))
        mask = mask.convert("L")

        # Use Stable Diffusion inpainting parameters
        result = self.model(
            prompt=prompt,
            image=image,
            mask_image=mask,
            width=gen_width,
            height=gen_height,
            num_inference_steps=20,
            guidance_scale=7.5,
            generator=torch.Generator(DEVICE).manual_seed(0),
        ).images[0]

        mask_resized = mask
        if (gen_width, gen_height) != (original_width, original_height):
            result = result.resize((original_width, original_height))
            mask_resized = mask.resize((original_width, original_height))

        original_image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
        final_image = Image.composite(result, original_image, mask_resized)

        byte_array = io.BytesIO()
        final_image.save(byte_array, format="PNG")
        return byte_array.getvalue()

    def start(self) -> None:
        print(f"Starting Inpaint Service on port {self.port}")

        self.socket = self.context.socket(zmq.REP)
        self.socket.bind(f"tcp://*:{self.port}")
        self.running = True

        print("Inpaint Service ready! Waiting for requests...")

        while self.running:
            try:
                message = self.socket.recv()
                request = msgpack.unpackb(message, raw=False)

                print(f"Received request: {request.get('action', 'unknown')}")

                if request.get("action") == "inpaint":
                    prompt = request.get("prompt")
                    mask = request.get("mask")
                    image_bytes = request.get("image_bytes")

                    try:
                        image = self.inpaint(prompt, image_bytes, mask)

                        response = {"status": "success", "image": image}
                        print(f"Successfully inpainted image")
                    except Exception as e:
                        response = {"status": "error", "message": str(e)}
                        print(f"Error processing request: {e}")

                elif request.get("action") == "health":
                    try:
                        device = str(self.model.device) if self.model else "unavailable"
                    except:
                        device = "unavailable"

                    response = {
                        "status": "healthy",
                        "service": "llm",
                        "model": self.model_name,
                        "device": device,
                        "timestamp": time.time(),
                    }

                elif request.get("action") == "shutdown":
                    response = {
                        "status": "shutting_down",
                        "message": "Inpainting Service shutting down",
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

    def stop(self) -> None:
        print("Stopping Inpainting Service...")
        self.running = False
        if self.socket:
            self.socket.close()
        self.context.term()
        print("Inpainting Service stopped")


def signal_handler(sig, frame) -> None:
    print("\nReceived shutdown signal")
    sys.exit(0)


def main() -> None:
    parser = argparse.ArgumentParser(description="Inpainting Service")
    parser.add_argument(
        "--port",
        type=int,
        default=5559,
        help="Port to bind the service (default: 5559)",
    )
    parser.add_argument("--debug", action="store_true", help="Enable debug logging")
    parser.add_argument(
        "--custom-checkpoint",
        type=str,
        help="Custom checkpoint model to use while keeping VAE, UNet from base model",
    )
    parser.add_argument(
        "--cache-dir",
        type=str,
        default="../../model-cache",
        help="Model cache directory",
    )

    args = parser.parse_args()

    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)

    service = Service(
        cache_dir=args.cache_dir,
        port=args.port,
        custom_checkpoint=args.custom_checkpoint,
    )

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
