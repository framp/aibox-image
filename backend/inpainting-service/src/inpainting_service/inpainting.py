import logging
from pathlib import Path
from typing import Protocol

import torch
from diffusers import StableDiffusionInpaintPipeline

# from dotenv import load_dotenv
# from huggingface_hub import login
from PIL import Image

DEVICE = (
    "mps"
    if torch.backends.mps.is_available()
    else "cuda"
    if torch.cuda.is_available()
    else "cpu"
)

# load_dotenv()
# login(token=os.environ["HF_TOKEN"])


def get_dimensions_divisible_by_8(width: int, height: int) -> tuple[int, int]:
    """Round dimensions to closest numbers divisible by 8."""
    return ((width + 4) // 8) * 8, ((height + 4) // 8) * 8


class Inpainter(Protocol):
    def inpaint(
        self, prompt: str, image: Image.Image, mask: Image.Image
    ) -> Image.Image: ...


class StableDiffusion(Inpainter):
    def __init__(self, cache_dir: Path, model_or_checkpoint: str):
        if model_or_checkpoint.endswith((".safetensors", ".ckpt")):
            logging.info(
                f"Loading custom checkpoint: {model_or_checkpoint} on {DEVICE}..."
            )
            self.model = StableDiffusionInpaintPipeline.from_single_file(
                model_or_checkpoint,
                torch_dtype=torch.float16,
                cache_dir=cache_dir,
            ).to(DEVICE)
        else:
            logging.info(f"Loading {model_or_checkpoint} on {DEVICE}...")
            self.model = StableDiffusionInpaintPipeline.from_pretrained(
                model_or_checkpoint,
                torch_dtype=torch.float16,
                cache_dir=cache_dir,
            ).to(DEVICE)

        logging.info("Model loaded successfully")

    def inpaint(self, prompt: str, image: Image.Image, mask: Image.Image):
        if self.model is None:
            raise RuntimeError("Model not loaded")

        original_image = image.copy()
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

        final_image = Image.composite(result, original_image, mask_resized)

        return final_image
