import logging
from pathlib import Path
from typing import Protocol

import torch
from diffusers import StableDiffusionUpscalePipeline
from PIL import Image

DEVICE = (
    "mps"
    if torch.backends.mps.is_available()
    else "cuda"
    if torch.cuda.is_available()
    else "cpu"
)


def get_dimensions_divisible_by_8(width: int, height: int) -> tuple[int, int]:
    """Round dimensions to closest numbers divisible by 8."""
    return ((width + 4) // 8) * 8, ((height + 4) // 8) * 8


class Upscaler(Protocol):
    def upscale(self, prompt: str, image: Image.Image) -> Image.Image: ...


class StableDiffusion(Upscaler):
    def __init__(self, cache_dir: Path, model_or_checkpoint: str):
        if model_or_checkpoint.endswith((".safetensors", ".ckpt", ".pth")):
            logging.info(
                f"Loading custom checkpoint: {model_or_checkpoint} on {DEVICE}..."
            )
            self.model = StableDiffusionUpscalePipeline.from_single_file(
                model_or_checkpoint,
                torch_dtype=torch.float16,
                cache_dir=cache_dir,
            ).to(DEVICE)
        else:
            logging.info(f"Loading {model_or_checkpoint} on {DEVICE}...")
            self.model = StableDiffusionUpscalePipeline.from_pretrained(
                model_or_checkpoint,
                torch_dtype=torch.float16,
                cache_dir=cache_dir,
            ).to(DEVICE)

        logging.info("Model loaded successfully")

    def upscale(self, prompt: str, image: Image.Image):
        if self.model is None:
            raise RuntimeError("Model not loaded")

        width, height = image.size
        width, height = get_dimensions_divisible_by_8(width, height)
        image = image.resize((width, height))

        # Use Stable Diffusion upscaling parameters
        result = self.model(
            prompt=prompt,
            image=image,
            num_inference_steps=5,
            guidance_scale=7.5,
            generator=torch.Generator(DEVICE).manual_seed(0),
        ).images[0]

        return result
