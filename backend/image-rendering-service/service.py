#!/usr/bin/env python3

import os
import sys
import json
import time
import signal
import argparse
from pathlib import Path
from typing import Dict, Optional, Union, List
import torch
import zmq
import msgpack
from PIL import Image
import io
import base64
from diffusers import StableDiffusionPipeline, FluxPipeline, AutoPipelineForText2Image
from diffusers import UNet2DConditionModel, FluxTransformer2DModel, AutoencoderKL, DDIMScheduler, FlowMatchEulerDiscreteScheduler
from transformers import CLIPTextModel, CLIPTokenizer, T5EncoderModel, T5TokenizerFast
import safetensors.torch


class ImageRenderingService:
    def __init__(self, port: int = 5558):
        self.port = port
        self.context = zmq.Context()
        self.socket = None
        self.running = False

        self.current_pipeline = None
        self.current_model_info = None  # Store current model type and path
        self.device = None
        self.torch_dtype = None

        # Model configurations
        self.model_configs = {
            "stable-diffusion": {
                "default_model_id": "runwayml/stable-diffusion-v1-5",
                "pipeline_class": StableDiffusionPipeline
            },
            "stable-diffusion-xl": {
                "default_model_id": "stabilityai/stable-diffusion-xl-base-1.0",
                "pipeline_class": AutoPipelineForText2Image
            },
            "flux": {
                "default_model_id": "black-forest-labs/FLUX.1-schnell",
                "pipeline_class": FluxPipeline
            },
            "flux-dev": {
                "default_model_id": "black-forest-labs/FLUX.1-dev",
                "pipeline_class": FluxPipeline
            }
        }

        self._init_device()

    def _init_device(self):
        """Initialize device and dtype settings"""
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.torch_dtype = torch.float16 if torch.cuda.is_available() else torch.float32
        print(f"Device: {self.device}, dtype: {self.torch_dtype}")

    def _load_from_checkpoint(self, checkpoint_path: str, model_type: str, pipeline_class, vae_path: Optional[str] = None):
        """Load pipeline from checkpoint file with individual components"""
        print(f"Loading {model_type} from checkpoint: {checkpoint_path}")

        if model_type.startswith("flux"):
            return self._load_flux_from_checkpoint(checkpoint_path, vae_path)
        else:
            return self._load_stable_diffusion_from_checkpoint(checkpoint_path, vae_path)

    def _load_flux_from_checkpoint(self, checkpoint_path: str, vae_path: Optional[str] = None):
        """Load Flux pipeline from checkpoint"""
        try:
            # For Flux models, we need to load components separately
            print("Loading Flux components from default model...")
            base_model = "black-forest-labs/FLUX.1-schnell"  # Use as base for other components

            # Load VAE (custom or default)
            if vae_path:
                print(f"Loading custom VAE from: {vae_path}")
                if Path(vae_path).suffix in ['.safetensors', '.ckpt', '.bin']:
                    # Load from single checkpoint file
                    print("Loading VAE from single file")
                    vae = AutoencoderKL.from_pretrained(base_model, subfolder="vae", torch_dtype=self.torch_dtype)
                    # Load and apply VAE weights from checkpoint
                    if vae_path.endswith('.safetensors'):
                        vae_state_dict = safetensors.torch.load_file(vae_path)
                    else:
                        vae_state_dict = torch.load(vae_path, map_location="cpu")
                    vae.load_state_dict(vae_state_dict, strict=False)
                else:
                    # Load from directory or HF model ID
                    vae = AutoencoderKL.from_pretrained(vae_path, torch_dtype=self.torch_dtype)
            else:
                vae = AutoencoderKL.from_pretrained(base_model, subfolder="vae", torch_dtype=self.torch_dtype)
            text_encoder = CLIPTextModel.from_pretrained(base_model, subfolder="text_encoder", torch_dtype=self.torch_dtype)
            text_encoder_2 = T5EncoderModel.from_pretrained(base_model, subfolder="text_encoder_2", torch_dtype=self.torch_dtype)
            tokenizer = CLIPTokenizer.from_pretrained(base_model, subfolder="tokenizer")
            tokenizer_2 = T5TokenizerFast.from_pretrained(base_model, subfolder="tokenizer_2")
            scheduler = FlowMatchEulerDiscreteScheduler.from_pretrained(base_model, subfolder="scheduler")

            # Load transformer from checkpoint
            print(f"Loading transformer weights from {checkpoint_path}")
            if checkpoint_path.endswith('.safetensors'):
                state_dict = safetensors.torch.load_file(checkpoint_path)
            else:
                state_dict = torch.load(checkpoint_path, map_location="cpu")

            # Create transformer and load weights
            transformer = FluxTransformer2DModel.from_pretrained(base_model, subfolder="transformer", torch_dtype=self.torch_dtype)
            transformer.load_state_dict(state_dict, strict=False)

            # Create pipeline
            pipeline = FluxPipeline(
                vae=vae,
                text_encoder=text_encoder,
                text_encoder_2=text_encoder_2,
                tokenizer=tokenizer,
                tokenizer_2=tokenizer_2,
                transformer=transformer,
                scheduler=scheduler,
            )

            if torch.cuda.is_available():
                pipeline = pipeline.to(self.device)

            return pipeline

        except Exception as e:
            print(f"Failed to load Flux from checkpoint: {e}")
            # Fallback to default model
            print("Falling back to default Flux model...")
            return FluxPipeline.from_pretrained("black-forest-labs/FLUX.1-schnell", torch_dtype=self.torch_dtype)

    def _load_stable_diffusion_from_checkpoint(self, checkpoint_path: str, vae_path: Optional[str] = None):
        """Load Stable Diffusion pipeline from checkpoint"""
        try:
            # Load base components
            print("Loading Stable Diffusion components from default model...")
            base_model = "runwayml/stable-diffusion-v1-5"

            # Load VAE (custom or default)
            if vae_path:
                print(f"Loading custom VAE from: {vae_path}")
                if Path(vae_path).suffix in ['.safetensors', '.ckpt', '.bin']:
                    # Load from single checkpoint file
                    print("Loading VAE from single file")
                    vae = AutoencoderKL.from_pretrained(base_model, subfolder="vae", torch_dtype=self.torch_dtype)
                    # Load and apply VAE weights from checkpoint
                    if vae_path.endswith('.safetensors'):
                        vae_state_dict = safetensors.torch.load_file(vae_path)
                    else:
                        vae_state_dict = torch.load(vae_path, map_location="cpu")
                    vae.load_state_dict(vae_state_dict, strict=False)
                else:
                    # Load from directory or HF model ID
                    vae = AutoencoderKL.from_pretrained(vae_path, torch_dtype=self.torch_dtype)
            else:
                vae = AutoencoderKL.from_pretrained(base_model, subfolder="vae", torch_dtype=self.torch_dtype)
            text_encoder = CLIPTextModel.from_pretrained(base_model, subfolder="text_encoder", torch_dtype=self.torch_dtype)
            tokenizer = CLIPTokenizer.from_pretrained(base_model, subfolder="tokenizer")
            scheduler = DDIMScheduler.from_pretrained(base_model, subfolder="scheduler")

            # Load UNet from checkpoint
            print(f"Loading UNet weights from {checkpoint_path}")
            if checkpoint_path.endswith('.safetensors'):
                state_dict = safetensors.torch.load_file(checkpoint_path)
            else:
                state_dict = torch.load(checkpoint_path, map_location="cpu")

            # Create UNet and load weights
            unet = UNet2DConditionModel.from_pretrained(base_model, subfolder="unet", torch_dtype=self.torch_dtype)

            # Filter state dict for UNet keys (checkpoint might contain other components)
            unet_state_dict = {k.replace("model.diffusion_model.", ""): v for k, v in state_dict.items()
                              if "model.diffusion_model" in k or k.startswith("unet")}
            if not unet_state_dict:
                unet_state_dict = state_dict  # Use full state dict if no specific keys found

            unet.load_state_dict(unet_state_dict, strict=False)

            # Create pipeline
            pipeline = StableDiffusionPipeline(
                vae=vae,
                text_encoder=text_encoder,
                tokenizer=tokenizer,
                unet=unet,
                scheduler=scheduler,
                safety_checker=None,
                requires_safety_checker=False,
            )

            pipeline = pipeline.to(self.device)
            return pipeline

        except Exception as e:
            print(f"Failed to load Stable Diffusion from checkpoint: {e}")
            # Fallback to default model
            print("Falling back to default Stable Diffusion model...")
            return StableDiffusionPipeline.from_pretrained(
                "runwayml/stable-diffusion-v1-5",
                torch_dtype=self.torch_dtype,
                safety_checker=None,
                requires_safety_checker=False
            )

    def _load_model(self, model_type: str, model_path: Optional[str] = None, vae_path: Optional[str] = None):
        """Load a specific model if not already loaded"""
        model_type = model_type.lower()

        if model_type not in self.model_configs:
            raise ValueError(f"Unsupported model type: {model_type}. Available: {list(self.model_configs.keys())}")

        config = self.model_configs[model_type]
        model_id = model_path if model_path else config["default_model_id"]
        pipeline_class = config["pipeline_class"]

        # Check if we already have this model loaded (including VAE check)
        if (self.current_model_info and
            self.current_model_info.get('type') == model_type and
            self.current_model_info.get('path') == model_id and
            self.current_model_info.get('vae_path') == vae_path and
            self.current_pipeline is not None):
            print(f"Model {model_id} with VAE {vae_path} already loaded, skipping...")
            return

        # Unload current model if different
        if self.current_pipeline is not None:
            print("Unloading current model...")
            del self.current_pipeline
            torch.cuda.empty_cache() if torch.cuda.is_available() else None

        print(f"Loading {model_type} model: {model_id}")

        try:
            # Check if model_id is a single checkpoint file
            if model_id and Path(model_id).suffix in ['.safetensors', '.ckpt', '.bin']:
                self.current_pipeline = self._load_from_checkpoint(model_id, model_type, pipeline_class, vae_path)
            else:
                # Load pipeline based on model type for directories or HF repos
                if model_type.startswith("flux"):
                    self.current_pipeline = pipeline_class.from_pretrained(
                        model_id,
                        torch_dtype=self.torch_dtype,
                    )
                    if torch.cuda.is_available():
                        self.current_pipeline = self.current_pipeline.to(self.device)
                else:
                    self.current_pipeline = pipeline_class.from_pretrained(
                        model_id,
                        torch_dtype=self.torch_dtype,
                        safety_checker=None,
                        requires_safety_checker=False
                    )
                    self.current_pipeline = self.current_pipeline.to(self.device)

            # Enable memory efficient attention if available
            if hasattr(self.current_pipeline, 'enable_attention_slicing'):
                self.current_pipeline.enable_attention_slicing()

            if hasattr(self.current_pipeline, 'enable_model_cpu_offload') and torch.cuda.is_available():
                self.current_pipeline.enable_model_cpu_offload()

            # Store current model info
            self.current_model_info = {
                'type': model_type,
                'path': model_id,
                'vae_path': vae_path
            }

            print(f"{model_type} model loaded successfully")

        except Exception as e:
            print(f"Error loading model {model_id}: {e}")
            import traceback
            traceback.print_exc()
            raise

    def generate_image(self, prompt: str, model_type: str = "stable-diffusion",
                      model_path: Optional[str] = None, vae_path: Optional[str] = None,
                      negative_prompt: Optional[str] = None, width: int = 512, height: int = 512,
                      num_inference_steps: int = 20, guidance_scale: float = 7.5,
                      seed: Optional[int] = None) -> Optional[str]:
        """Generate image and return base64 encoded string"""

        # Load model if needed
        self._load_model(model_type, model_path, vae_path)

        if self.current_pipeline is None:
            return None

        try:
            # Set seed for reproducibility
            generator = None
            if seed is not None:
                generator = torch.Generator(device=self.device).manual_seed(seed)

            # Adjust parameters based on model type
            generation_args = {
                "prompt": prompt,
                "width": width,
                "height": height,
                "num_inference_steps": num_inference_steps,
                "generator": generator
            }

            # Add model-specific parameters
            if model_type.startswith("flux"):
                # Flux models don't use guidance_scale and negative_prompt the same way
                if hasattr(self.current_pipeline, 'guidance_scale'):
                    generation_args["guidance_scale"] = guidance_scale
            else:
                generation_args["guidance_scale"] = guidance_scale
                if negative_prompt:
                    generation_args["negative_prompt"] = negative_prompt

            print(f"Generating image with prompt: '{prompt[:50]}...'")

            # Generate image
            result = self.current_pipeline(**generation_args)
            image = result.images[0]

            # Convert to base64
            buffer = io.BytesIO()
            image.save(buffer, format='PNG')
            img_str = base64.b64encode(buffer.getvalue()).decode()

            print(f"Image generated successfully ({width}x{height})")
            return img_str

        except Exception as e:
            print(f"Error generating image: {e}")
            import traceback
            traceback.print_exc()
            return None

    def save_image_from_base64(self, base64_str: str, output_path: str) -> bool:
        """Save base64 encoded image to file"""
        try:
            img_data = base64.b64decode(base64_str)
            image = Image.open(io.BytesIO(img_data))

            output_path = Path(output_path)
            output_path.parent.mkdir(parents=True, exist_ok=True)

            image.save(output_path)
            print(f"Image saved to: {output_path}")
            return True

        except Exception as e:
            print(f"Error saving image: {e}")
            return False

    def start(self):
        print(f"Starting Image Rendering Service on port {self.port}")

        self.socket = self.context.socket(zmq.REP)
        self.socket.bind(f"tcp://*:{self.port}")
        self.running = True

        print("Image Rendering Service ready! Waiting for requests...")

        while self.running:
            try:
                message = self.socket.recv()
                request = msgpack.unpackb(message, raw=False)

                print(f"Received request: {request.get('action', 'unknown')}")

                if request.get('action') == 'generate_image':
                    prompt = request.get('prompt')
                    if not prompt:
                        response = {
                            'status': 'error',
                            'message': 'Missing prompt parameter'
                        }
                    else:
                        try:
                            # Extract parameters with defaults
                            model_type = request.get('model_type', 'stable-diffusion')
                            model_path = request.get('model_path')
                            vae_path = request.get('vae_path')
                            negative_prompt = request.get('negative_prompt')
                            width = request.get('width', 512)
                            height = request.get('height', 512)
                            num_inference_steps = request.get('num_inference_steps', 20)
                            guidance_scale = request.get('guidance_scale', 7.5)
                            seed = request.get('seed')

                            # Generate image
                            image_b64 = self.generate_image(
                                prompt=prompt,
                                model_type=model_type,
                                model_path=model_path,
                                vae_path=vae_path,
                                negative_prompt=negative_prompt,
                                width=width,
                                height=height,
                                num_inference_steps=num_inference_steps,
                                guidance_scale=guidance_scale,
                                seed=seed
                            )

                            if image_b64:
                                response = {
                                    'status': 'success',
                                    'image': image_b64,
                                    'format': 'base64_png',
                                    'model_info': self.current_model_info
                                }
                                print(f"Generated image for prompt: '{prompt[:30]}...'")
                            else:
                                response = {
                                    'status': 'error',
                                    'message': 'Failed to generate image'
                                }

                        except Exception as e:
                            response = {
                                'status': 'error',
                                'message': str(e)
                            }
                            print(f"Error processing request: {e}")

                elif request.get('action') == 'generate_and_save':
                    prompt = request.get('prompt')
                    output_path = request.get('output_path')

                    if not prompt or not output_path:
                        response = {
                            'status': 'error',
                            'message': 'Missing prompt or output_path parameter'
                        }
                    else:
                        try:
                            # Extract parameters with defaults
                            model_type = request.get('model_type', 'stable-diffusion')
                            model_path = request.get('model_path')
                            vae_path = request.get('vae_path')
                            negative_prompt = request.get('negative_prompt')
                            width = request.get('width', 512)
                            height = request.get('height', 512)
                            num_inference_steps = request.get('num_inference_steps', 20)
                            guidance_scale = request.get('guidance_scale', 7.5)
                            seed = request.get('seed')

                            # Generate image
                            image_b64 = self.generate_image(
                                prompt=prompt,
                                model_type=model_type,
                                model_path=model_path,
                                vae_path=vae_path,
                                negative_prompt=negative_prompt,
                                width=width,
                                height=height,
                                num_inference_steps=num_inference_steps,
                                guidance_scale=guidance_scale,
                                seed=seed
                            )

                            if image_b64:
                                # Save image to file
                                success = self.save_image_from_base64(image_b64, output_path)
                                if success:
                                    response = {
                                        'status': 'success',
                                        'output_path': str(output_path),
                                        'model_info': self.current_model_info,
                                        'message': 'Image generated and saved successfully'
                                    }
                                else:
                                    response = {
                                        'status': 'error',
                                        'message': 'Image generated but failed to save'
                                    }
                            else:
                                response = {
                                    'status': 'error',
                                    'message': 'Failed to generate image'
                                }

                        except Exception as e:
                            response = {
                                'status': 'error',
                                'message': str(e)
                            }
                            print(f"Error processing request: {e}")

                elif request.get('action') == 'health':
                    response = {
                        'status': 'healthy',
                        'service': 'image-rendering',
                        'current_model': self.current_model_info,
                        'device': str(self.device),
                        'timestamp': time.time()
                    }

                elif request.get('action') == 'shutdown':
                    response = {
                        'status': 'shutting_down',
                        'message': 'Image Rendering Service shutting down'
                    }
                    self.running = False

                else:
                    response = {
                        'status': 'error',
                        'message': f"Unknown action: {request.get('action')}"
                    }

                self.socket.send(msgpack.packb(response, use_bin_type=True))

            except KeyboardInterrupt:
                print("\nReceived interrupt signal")
                break
            except Exception as e:
                print(f"Error processing request: {e}")
                try:
                    error_response = {
                        'status': 'error',
                        'message': f"Service error: {str(e)}"
                    }
                    self.socket.send(msgpack.packb(error_response, use_bin_type=True))
                except:
                    pass

        self.stop()

    def stop(self):
        print("Stopping Image Rendering Service...")
        self.running = False
        if self.socket:
            self.socket.close()
        self.context.term()
        print("Image Rendering Service stopped")


def signal_handler(sig, frame):
    print("\nReceived shutdown signal")
    sys.exit(0)


def main():
    parser = argparse.ArgumentParser(description="Image Rendering Service")
    parser.add_argument(
        "--port",
        type=int,
        default=5558,
        help="Port to bind the service (default: 5558)"
    )
    parser.add_argument(
        "--debug",
        action="store_true",
        help="Enable debug logging"
    )

    args = parser.parse_args()

    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)

    service = ImageRenderingService(port=args.port)

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
