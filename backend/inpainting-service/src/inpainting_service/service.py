import os
from pathlib import Path
from typing import Optional

import torch
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
    def __init__(self, cache_dir: Path, custom_checkpoint: Optional[str] = None):
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

            print("Model loaded successfully")
        except Exception as e:
            print(f"Warning: Could not load model: {e}")
            import traceback

            traceback.print_exc()
            self.model = None

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
