#!/usr/bin/env python3

import argparse
import sys
import time
import zmq
import msgpack
import base64
from pathlib import Path
from PIL import Image
import io


class ImageRenderingClient:
    def __init__(self, host: str = "localhost", port: int = 5558):
        self.host = host
        self.port = port
        self.context = zmq.Context()
        self.socket = None

    def connect(self):
        """Connect to the image rendering service"""
        self.socket = self.context.socket(zmq.REQ)
        self.socket.connect(f"tcp://{self.host}:{self.port}")
        print(f"Connected to image rendering service at {self.host}:{self.port}")

    def disconnect(self):
        """Disconnect from the service"""
        if self.socket:
            self.socket.close()
        self.context.term()

    def send_request(self, request: dict) -> dict:
        """Send a request to the service and return the response"""
        if not self.socket:
            raise RuntimeError("Not connected to service. Call connect() first.")

        # Send request
        self.socket.send(msgpack.packb(request, use_bin_type=True))

        # Receive response
        response_data = self.socket.recv()
        response = msgpack.unpackb(response_data, raw=False)

        return response

    def generate_image(self, prompt: str, output_path: str = None, **kwargs) -> bool:
        """
        Generate an image with the given prompt

        Args:
            prompt: Text prompt for image generation
            output_path: Optional path to save the image
            **kwargs: Additional parameters like model_type, model_path, width, height, etc.

        Returns:
            bool: Success status
        """
        print(f"Generating image for prompt: '{prompt}'")

        if output_path:
            # Use generate_and_save action
            request = {
                'action': 'generate_and_save',
                'prompt': prompt,
                'output_path': output_path,
                **kwargs
            }
        else:
            # Use generate_image action
            request = {
                'action': 'generate_image',
                'prompt': prompt,
                **kwargs
            }

        try:
            response = self.send_request(request)

            if response.get('status') == 'success':
                if output_path:
                    print(f"Image saved to: {response.get('output_path')}")
                else:
                    # Save image data if no output path specified
                    image_data = response.get('image')
                    if image_data:
                        # Generate default filename
                        timestamp = int(time.time())
                        default_path = f"generated_image_{timestamp}.png"

                        # Decode and save image
                        img_bytes = base64.b64decode(image_data)
                        image = Image.open(io.BytesIO(img_bytes))
                        image.save(default_path)
                        print(f"Image saved to: {default_path}")

                # Print model info if available
                model_info = response.get('model_info')
                if model_info:
                    print(f"Generated using: {model_info['type']} ({model_info['path']})")

                return True
            else:
                print(f"Error: {response.get('message', 'Unknown error')}")
                return False

        except Exception as e:
            print(f"Error communicating with service: {e}")
            return False

    def health_check(self) -> bool:
        """Check if the service is healthy"""
        try:
            request = {'action': 'health'}
            response = self.send_request(request)

            if response.get('status') == 'healthy':
                print("Service is healthy")
                current_model = response.get('current_model')
                if current_model:
                    print(f"Current model: {current_model['type']} ({current_model['path']})")
                else:
                    print("No model currently loaded")
                print(f"Device: {response.get('device')}")
                return True
            else:
                print("Service is not healthy")
                return False

        except Exception as e:
            print(f"Health check failed: {e}")
            return False


def main():
    parser = argparse.ArgumentParser(description="Image Rendering CLI")
    parser.add_argument(
        "prompt",
        help="Text prompt for image generation"
    )
    parser.add_argument(
        "-o", "--output",
        help="Output file path (default: auto-generated filename)"
    )
    parser.add_argument(
        "--host",
        default="localhost",
        help="Service host (default: localhost)"
    )
    parser.add_argument(
        "--port",
        type=int,
        default=5558,
        help="Service port (default: 5558)"
    )
    parser.add_argument(
        "--model-type",
        choices=["stable-diffusion", "stable-diffusion-xl", "flux", "flux-dev"],
        default="stable-diffusion",
        help="Model type to use (default: stable-diffusion)"
    )
    parser.add_argument(
        "--model-path",
        help="Custom model path (local directory or HuggingFace model ID)"
    )
    parser.add_argument(
        "--vae-path",
        help="Custom VAE path (local directory or HuggingFace model ID)"
    )
    parser.add_argument(
        "--width",
        type=int,
        default=512,
        help="Image width (default: 512)"
    )
    parser.add_argument(
        "--height",
        type=int,
        default=512,
        help="Image height (default: 512)"
    )
    parser.add_argument(
        "--steps",
        type=int,
        default=20,
        help="Number of inference steps (default: 20)"
    )
    parser.add_argument(
        "--guidance",
        type=float,
        default=7.5,
        help="Guidance scale (default: 7.5)"
    )
    parser.add_argument(
        "--negative-prompt",
        help="Negative prompt for generation"
    )
    parser.add_argument(
        "--seed",
        type=int,
        help="Random seed for reproducible generation"
    )
    parser.add_argument(
        "--health",
        action="store_true",
        help="Check service health and exit"
    )

    args = parser.parse_args()

    # Create client and connect
    client = ImageRenderingClient(host=args.host, port=args.port)

    try:
        client.connect()

        if args.health:
            # Just do health check
            success = client.health_check()
            sys.exit(0 if success else 1)

        # Prepare generation parameters
        generation_params = {
            'model_type': args.model_type,
            'width': args.width,
            'height': args.height,
            'num_inference_steps': args.steps,
            'guidance_scale': args.guidance,
        }

        # Add optional parameters
        if args.model_path:
            generation_params['model_path'] = args.model_path
        if args.vae_path:
            generation_params['vae_path'] = args.vae_path
        if args.negative_prompt:
            generation_params['negative_prompt'] = args.negative_prompt
        if args.seed is not None:
            generation_params['seed'] = args.seed

        # Generate image
        success = client.generate_image(
            prompt=args.prompt,
            output_path=args.output,
            **generation_params
        )

        sys.exit(0 if success else 1)

    except KeyboardInterrupt:
        print("\nInterrupted by user")
        sys.exit(1)
    except Exception as e:
        print(f"Error: {e}")
        sys.exit(1)
    finally:
        client.disconnect()


if __name__ == "__main__":
    main()
