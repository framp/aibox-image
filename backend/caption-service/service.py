#!/usr/bin/env python3

import os
import sys
import json
import time
import signal
import argparse
from pathlib import Path
from typing import Dict, Optional
from PIL import Image
import torch
import zmq
import msgpack
from transformers import AutoProcessor, AutoModelForCausalLM


class FlorenceService:
    def __init__(self, port: int = 5556, model_name: str = "microsoft/Florence-2-base"):
        self.port = port
        self.model_name = model_name
        self.context = zmq.Context()
        self.socket = None
        self.running = False
        
        self.processor = None
        self.model = None
        self.device = None
        self.torch_dtype = None
        
        self.supported_formats = {'.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.webp'}
        self.caption_prompt = "<MORE_DETAILED_CAPTION>"
        
        self._init_models()
    
    def _init_models(self):
        try:
            print(f"Loading Florence-2 model: {self.model_name}")
            
            self.device = "cuda:0" if torch.cuda.is_available() else "cpu"
            self.torch_dtype = torch.float16 if torch.cuda.is_available() else torch.float32
            
            print(f"Device: {self.device}, dtype: {self.torch_dtype}")
            
            print("Loading Florence-2 processor...")
            self.processor = AutoProcessor.from_pretrained(
                self.model_name, 
                trust_remote_code=True,
            )
            print("Processor loaded successfully")
            
            print("Loading Florence-2 model...")
            self.model = AutoModelForCausalLM.from_pretrained(
                self.model_name, 
                torch_dtype=self.torch_dtype, 
                trust_remote_code=True,
            ).to(self.device)
            
            print("Model loaded successfully")
            print(f"Florence-2 model loaded successfully on {self.device}")
            
        except Exception as e:
            print(f"Warning: Could not load Florence-2 model: {e}")
            import traceback
            traceback.print_exc()
            print("Florence-2 functionality will be disabled")
            self.processor = None
            self.model = None
    
    def generate_caption(self, image_path: str) -> str:
        if self.processor is None or self.model is None:
            return "No caption available - Florence-2 model not loaded"
        
        try:
            image_path = Path(image_path)
            
            if not image_path.exists():
                raise FileNotFoundError(f"Image file not found: {image_path}")
            
            if image_path.suffix.lower() not in self.supported_formats:
                raise ValueError(f"Unsupported image format: {image_path.suffix}")
            
            image = Image.open(image_path).convert('RGB')
            
            max_size = 512
            if max(image.width, image.height) > max_size:
                if image.width > image.height:
                    new_width = max_size
                    new_height = int((image.height / image.width) * max_size)
                else:
                    new_height = max_size
                    new_width = int((image.width / image.height) * max_size)
                image = image.resize((new_width, new_height), Image.Resampling.LANCZOS)
            
            inputs = self.processor(
                text=self.caption_prompt, 
                images=image, 
                return_tensors="pt"
            ).to(self.device, self.torch_dtype)
            
            generated_ids = self.model.generate(
                input_ids=inputs["input_ids"],
                pixel_values=inputs["pixel_values"],
                max_new_tokens=1024,
                num_beams=3
            )
            
            generated_text = self.processor.batch_decode(
                generated_ids, 
                skip_special_tokens=False
            )[0]
            
            parsed_answer = self.processor.post_process_generation(
                generated_text, 
                task=self.caption_prompt, 
                image_size=(image.width, image.height)
            )
            
            if isinstance(parsed_answer, dict) and self.caption_prompt in parsed_answer:
                return str(parsed_answer[self.caption_prompt])
            else:
                return str(parsed_answer)
                
        except Exception as e:
            print(f"Error generating caption: {e}")
            return f"Caption generation failed: {e}"
    
    def start(self):
        print(f"Starting Florence Service on port {self.port}")
        
        self.socket = self.context.socket(zmq.REP)
        self.socket.bind(f"tcp://*:{self.port}")
        self.running = True
        
        print("Florence Service ready! Waiting for requests...")
        
        while self.running:
            try:
                message = self.socket.recv()
                request = msgpack.unpackb(message, raw=False)
                
                print(f"Received request: {request.get('action', 'unknown')}")
                
                if request.get('action') == 'generate_caption':
                    image_path = request.get('image_path')
                    if not image_path:
                        response = {
                            'status': 'error',
                            'message': 'Missing image_path parameter'
                        }
                    else:
                        try:
                            caption = self.generate_caption(image_path)
                            
                            response = {
                                'status': 'success',
                                'caption': caption
                            }
                            print(f"Generated caption: {caption[:50]}...")
                        except Exception as e:
                            response = {
                                'status': 'error',
                                'message': str(e)
                            }
                            print(f"Error processing request: {e}")
                
                elif request.get('action') == 'health':
                    response = {
                        'status': 'healthy',
                        'service': 'florence',
                        'model': self.model_name,
                        'device': str(self.device),
                        'timestamp': time.time()
                    }
                
                elif request.get('action') == 'shutdown':
                    response = {
                        'status': 'shutting_down',
                        'message': 'Florence Service shutting down'
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
        print("Stopping Florence Service...")
        self.running = False
        if self.socket:
            self.socket.close()
        self.context.term()
        print("Florence Service stopped")


def signal_handler(sig, frame):
    print("\nReceived shutdown signal")
    sys.exit(0)


def main():
    parser = argparse.ArgumentParser(description="Florence Image Captioning Service")
    parser.add_argument(
        "--port", 
        type=int, 
        default=5556,
        help="Port to bind the service (default: 5556)"
    )
    parser.add_argument(
        "--model",
        type=str,
        default="microsoft/Florence-2-base",
        help="Florence model to use (default: microsoft/Florence-2-base)"
    )
    parser.add_argument(
        "--debug",
        action="store_true",
        help="Enable debug logging"
    )
    
    args = parser.parse_args()
    
    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)
    
    service = FlorenceService(port=args.port, model_name=args.model)
    
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