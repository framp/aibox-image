import argparse
import io
import os
import signal
import sys
import time

import msgpack
import torch
import zmq
from diffusers import FluxFillPipeline, StableDiffusionInpaintPipeline
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
# MODEL = "black-forest-labs/FLUX.1-Fill-dev"
MODEL = "stabilityai/stable-diffusion-2-inpainting"
IMG_WIDTH = 512
IMG_HEIGHT = 512

load_dotenv()
login(token=os.environ["HF_TOKEN"])


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
            print(f"Loading {MODEL} on {DEVICE}...")
            # self.model = FluxFillPipeline.from_pretrained(
            #     MODEL, torch_dtype=torch.bfloat16
            # ).to(DEVICE)
            self.model = StableDiffusionInpaintPipeline.from_pretrained(
                "runwayml/stable-diffusion-inpainting", torch_dtype=torch.float16
            ).to(DEVICE)

            print("Optimizing model...")
            # self.model.vae.enable_slicing()
            # self.model.vae.enable_tiling()

            # self.model.to(torch.float16)

            print("Model loaded successfully")
        except Exception as e:
            print(f"Warning: Could not load LLM model: {e}")
            import traceback

            traceback.print_exc()
            print("Image inpainting functionality will be disabled")

    def inpaint(self, prompt: str, image_path: str, mask: bytes):
        image = Image.open(image_path).convert("RGB")
        mask = Image.open(io.BytesIO(mask)).convert("RGB")

        image = image.resize((IMG_WIDTH, IMG_HEIGHT))
        mask = mask.resize((IMG_WIDTH, IMG_HEIGHT))

        with torch.inference_mode():
            result = self.model(
                prompt=prompt,
                image=image,
                mask_image=mask,
                width=IMG_WIDTH,
                height=IMG_HEIGHT,
                num_inference_steps=15,
                generator=torch.Generator(DEVICE).manual_seed(0),
            ).images[0]

        result.save("result.png", "PNG")

        byte_array = io.BytesIO()
        result.save(byte_array, format="PNG")
        return byte_array.getvalue()

    def start(self):
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
                    image_path = request.get("image_path")

                    try:
                        image = self.inpaint(prompt, image_path, mask)

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
                        "model": MODEL,
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

    def stop(self):
        print("Stopping Inpainting Service...")
        self.running = False
        if self.socket:
            self.socket.close()
        self.context.term()
        print("Inpainting Service stopped")


def signal_handler(sig, frame):
    print("\nReceived shutdown signal")
    sys.exit(0)


def main():
    parser = argparse.ArgumentParser(description="Inpainting Service")
    parser.add_argument(
        "--port",
        type=int,
        default=5559,
        help="Port to bind the service (default: 5559)",
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
