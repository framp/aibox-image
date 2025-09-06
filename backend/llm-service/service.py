#!/usr/bin/env python3

import os
import sys
import json
import time
import signal
import argparse
import re
from typing import List, Dict, Optional, Tuple
from pathlib import Path
import torch
import numpy as np
import zmq
import msgpack
from PIL import Image
from transformers import AutoTokenizer, AutoModelForCausalLM, AutoProcessor, AutoModel
from sentence_transformers import SentenceTransformer


llm_model_name = "Qwen/Qwen3-4B"
siglip2_model_name = "google/siglip2-base-patch16-224"
embedding_model_name = "Qwen/Qwen3-Embedding-0.6B"

_tokenizer = None
_llm_model = None

_siglip2_processor = None
_siglip2_model = None

_embedding_model = None


def get_tokenizer():
    global _tokenizer
    if _tokenizer is None:
        _tokenizer = AutoTokenizer.from_pretrained(llm_model_name)
    return _tokenizer


def get_llm_model():
    global _llm_model
    if _llm_model is None:
        _llm_model = AutoModelForCausalLM.from_pretrained(
            llm_model_name, torch_dtype="auto", device_map="auto"
        )
    return _llm_model


def get_siglip2_processor():
    global _siglip2_processor
    if _siglip2_processor is None:
        _siglip2_processor = AutoProcessor.from_pretrained(siglip2_model_name)
    return _siglip2_processor


def get_siglip2_model():
    global _siglip2_model
    if _siglip2_model is None:
        _siglip2_model = AutoModel.from_pretrained(
            siglip2_model_name, 
            torch_dtype=torch.float16, 
            device_map="auto"
        )
    return _siglip2_model


def get_embedding_model():
    global _embedding_model
    if _embedding_model is None:
        _embedding_model = SentenceTransformer(embedding_model_name)
    return _embedding_model


def generate_response(messages, enable_thinking=True, max_new_tokens=32768):
    tokenizer = get_tokenizer()
    model = get_llm_model()
    text = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
        enable_thinking=enable_thinking,
    )
    model_inputs = tokenizer([text], return_tensors="pt").to(model.device)

    generated_ids = model.generate(**model_inputs, max_new_tokens=max_new_tokens)
    output_ids = generated_ids[0][len(model_inputs.input_ids[0]) :].tolist()

    if enable_thinking:
        try:
            index = len(output_ids) - output_ids[::-1].index(151668)
        except ValueError:
            index = 0

        thinking_content = tokenizer.decode(
            output_ids[:index], skip_special_tokens=True
        ).strip("\n")
        content = tokenizer.decode(output_ids[index:], skip_special_tokens=True).strip(
            "\n"
        )
    else:
        thinking_content = ""
        content = tokenizer.decode(output_ids, skip_special_tokens=True).strip("\n")
    return (
        thinking_content,
        content,
    )


def generate_image_embedding(image_path: str) -> Optional[np.ndarray]:
    try:
        image_path = Path(image_path)
        if not image_path.exists():
            print(f"Image file not found: {image_path}")
            return None
        
        processor = get_siglip2_processor()
        model = get_siglip2_model()
        
        if processor is None or model is None:
            print("SigLIP2 models not available")
            return None
        
        with Image.open(image_path) as img:
            if img.mode != 'RGB':
                img = img.convert('RGB')
            
            inputs = processor(images=img, return_tensors="pt")
            
            inputs = {k: v.to(model.device) for k, v in inputs.items()}
            
            with torch.no_grad():
                image_features = model.get_image_features(**inputs)
                
                embedding = image_features.cpu().numpy().squeeze()
                
                print(f"Generated SigLIP2 embedding: shape={embedding.shape}, dtype={embedding.dtype}")
                return embedding
                
    except Exception as e:
        print(f"Error generating image embedding: {e}")
        import traceback
        traceback.print_exc()
        return None


def generate_text_embedding(text: str, prompt_name: str = "query") -> Optional[np.ndarray]:
    try:
        model = get_embedding_model()
        
        if model is None:
            print("Text embedding model not available")
            return None
        
        if prompt_name:
            embedding = model.encode([text], prompt_name=prompt_name)
        else:
            embedding = model.encode([text])
        
        embedding_array = embedding[0] if len(embedding.shape) > 1 else embedding
        
        print(f"Generated text embedding: shape={embedding_array.shape}, dtype={embedding_array.dtype}")
        return embedding_array
        
    except Exception as e:
        print(f"Error generating text embedding: {e}")
        import traceback
        traceback.print_exc()
        return None


class LLMService:
    STOP_WORDS = {
        'a', 'an', 'the',
        'in', 'on', 'at', 'by', 'for', 'with', 'from', 'to', 'of', 'into', 'over', 'under',
        'and', 'or', 'but', 'so', 'yet',
        'very', 'quite', 'some', 'many', 'much', 'more', 'most', 'all', 'any',
        'image', 'photo', 'picture', 'shot', 'view', 'scene', 'background',
        'tags', 'tag', 'keywords', 'keyword', 'description', 'caption',
        'one', 'two', 'three', 'four', 'five', 'six', 'seven', 'eight', 'nine', 'ten',
        'is', 'are', 'was', 'were', 'be', 'been', 'being', 'have', 'has', 'had',
        'it', 'its', 'this', 'that', 'these', 'those', 'they', 'them', 'their',
        'can', 'will', 'would', 'could', 'should', 'may', 'might', 'must',
        'there', 'here', 'where', 'when', 'how', 'what', 'which', 'who', 'why'
    }
    
    def __init__(self, port: int = 5557):
        self.port = port
        self.context = zmq.Context()
        self.socket = None
        self.running = False
        
        self._init_models()
    
    def _init_models(self):
        try:
            print(f"Loading LLM model: {llm_model_name}")
            
            print("Loading LLM tokenizer...")
            tokenizer = get_tokenizer()
            print("Tokenizer loaded successfully")
            
            print("Loading LLM model...")
            model = get_llm_model()
            print("Model loaded successfully")
            print(f"LLM model loaded successfully on device: {model.device}")
            
        except Exception as e:
            print(f"Warning: Could not load LLM model: {e}")
            import traceback
            traceback.print_exc()
            print("LLM functionality will be disabled")
        
        try:
            print(f"Loading SigLIP2 model: {siglip2_model_name}")
            
            print("Loading SigLIP2 processor...")
            processor = get_siglip2_processor()
            print("SigLIP2 processor loaded successfully")
            
            print("Loading SigLIP2 model...")
            model = get_siglip2_model()
            print(f"SigLIP2 model loaded successfully on device: {model.device}")
            
        except Exception as e:
            print(f"Warning: Could not load SigLIP2 model: {e}")
            import traceback
            traceback.print_exc()
            print("Image embedding functionality will be disabled")
        
        try:
            print(f"Loading text embedding model: {embedding_model_name}")
            
            print("Loading text embedding model...")
            model = get_embedding_model()
            print(f"Text embedding model loaded successfully")
            
        except Exception as e:
            print(f"Warning: Could not load text embedding model: {e}")
            import traceback
            traceback.print_exc()
            print("Text embedding functionality will be disabled")
    
    def extract_tags(self, caption: str, num_tags: int = 15) -> List[str]:
        try:
            tokenizer = get_tokenizer()
            model = get_llm_model()
            if tokenizer is None or model is None:
                return []
            
            messages = [
                {
                    "role": "user",
                    "content": f"Extract {num_tags} relevant single-word tags from this image description: {caption}\n\nProvide only the tags separated by commas, no explanations:"
                }
            ]
            
            thinking_content, content = generate_response(messages, enable_thinking=False, max_new_tokens=200)
            
            print(f"Generated text: '{content}'")
            
            if not content:
                return []
            
            generated_text = re.sub(r'^(tags?:?\s*)', '', content, flags=re.IGNORECASE)
            
            raw_words = []
            for separator in [',', '\n', ';', '•', '-', '|', ' ']:
                if separator in generated_text:
                    parts = generated_text.split(separator)
                    for part in parts:
                        raw_words.extend(part.split())
                    break
            else:
                raw_words = generated_text.split()
            
            cleaned_tags = []
            seen_tags = set()
            
            for word in raw_words:
                clean_word = re.sub(r'[^\w]', '', word.strip().lower())
                clean_word = re.sub(r'\d', '', clean_word)
                
                if (clean_word and 
                    len(clean_word) > 2 and 
                    ' ' not in clean_word and
                    not clean_word.isdigit() and
                    clean_word not in self.STOP_WORDS and
                    clean_word not in seen_tags):
                    cleaned_tags.append(clean_word)
                    seen_tags.add(clean_word)
                    if len(cleaned_tags) >= num_tags:
                        break
            
            return cleaned_tags
            
        except Exception as e:
            print(f"Error extracting tags with LLM: {e}")
            return []
    
    def refine_caption(self, caption: str) -> str:
        try:
            tokenizer = get_tokenizer()
            model = get_llm_model()
            if tokenizer is None or model is None:
                return caption
            
            messages = [
                {
                    "role": "user",
                    "content": f"Improve this image caption by making it more descriptive and natural while keeping the same meaning:\n\nOriginal: {caption}\n\nImproved:"
                }
            ]
            
            thinking_content, refined_caption = generate_response(messages, enable_thinking=False, max_new_tokens=150)
            
            if not refined_caption or len(refined_caption.strip()) < 10:
                return caption
            
            return refined_caption.strip()
            
        except Exception as e:
            print(f"Error refining caption with LLM: {e}")
            return caption
    
    def start(self):
        print(f"Starting LLM Service on port {self.port}")
        
        self.socket = self.context.socket(zmq.REP)
        self.socket.bind(f"tcp://*:{self.port}")
        self.running = True
        
        print("LLM Service ready! Waiting for requests...")
        
        while self.running:
            try:
                message = self.socket.recv()
                request = msgpack.unpackb(message, raw=False)
                
                print(f"Received request: {request.get('action', 'unknown')}")
                
                if request.get('action') == 'extract_tags':
                    caption = request.get('caption')
                    num_tags = request.get('num_tags', 15)
                    
                    if not caption:
                        response = {
                            'status': 'error',
                            'message': 'Missing caption parameter'
                        }
                    else:
                        try:
                            tags = self.extract_tags(caption, num_tags)
                            
                            response = {
                                'status': 'success',
                                'tags': tags
                            }
                            print(f"Extracted {len(tags)} tags from caption")
                        except Exception as e:
                            response = {
                                'status': 'error',
                                'message': str(e)
                            }
                            print(f"Error processing request: {e}")
                
                elif request.get('action') == 'refine_caption':
                    caption = request.get('caption')
                    
                    if not caption:
                        response = {
                            'status': 'error',
                            'message': 'Missing caption parameter'
                        }
                    else:
                        try:
                            refined_caption = self.refine_caption(caption)
                            
                            response = {
                                'status': 'success',
                                'refined_caption': refined_caption
                            }
                            print(f"Refined caption: {refined_caption[:50]}...")
                        except Exception as e:
                            response = {
                                'status': 'error',
                                'message': str(e)
                            }
                            print(f"Error processing request: {e}")
                
                elif request.get('action') == 'generate_embedding':
                    image_path = request.get('image_path')
                    
                    if not image_path:
                        response = {
                            'status': 'error',
                            'message': 'Missing image_path parameter'
                        }
                    else:
                        try:
                            embedding = generate_image_embedding(image_path)
                            
                            if embedding is not None:
                                embedding_list = embedding.tolist()
                                response = {
                                    'status': 'success',
                                    'embedding': embedding_list,
                                    'embedding_shape': embedding.shape,
                                    'model': siglip2_model_name
                                }
                                print(f"Generated image embedding: shape={embedding.shape}")
                            else:
                                response = {
                                    'status': 'error',
                                    'message': 'Failed to generate image embedding'
                                }
                        except Exception as e:
                            response = {
                                'status': 'error',
                                'message': str(e)
                            }
                            print(f"Error processing embedding request: {e}")
                
                elif request.get('action') == 'generate_text_embedding':
                    text = request.get('text')
                    prompt_name = request.get('prompt_name', 'query')
                    
                    if not text:
                        response = {
                            'status': 'error',
                            'message': 'Missing text parameter'
                        }
                    else:
                        try:
                            embedding = generate_text_embedding(text, prompt_name)
                            
                            if embedding is not None:
                                embedding_list = embedding.tolist()
                                response = {
                                    'status': 'success',
                                    'embedding': embedding_list,
                                    'embedding_shape': embedding.shape,
                                    'model': embedding_model_name
                                }
                                print(f"Generated text embedding: shape={embedding.shape}")
                            else:
                                response = {
                                    'status': 'error',
                                    'message': 'Failed to generate text embedding'
                                }
                        except Exception as e:
                            response = {
                                'status': 'error',
                                'message': str(e)
                            }
                            print(f"Error processing text embedding request: {e}")
                
                elif request.get('action') == 'health':
                    try:
                        llm_model = get_llm_model()
                        llm_device = str(llm_model.device) if llm_model else "unavailable"
                    except:
                        llm_device = "unavailable"
                    
                    try:
                        siglip2_model = get_siglip2_model()
                        siglip2_device = str(siglip2_model.device) if siglip2_model else "unavailable"
                    except:
                        siglip2_device = "unavailable"
                    
                    try:
                        embedding_model = get_embedding_model()
                        embedding_status = "available" if embedding_model else "unavailable"
                    except:
                        embedding_status = "unavailable"
                    
                    response = {
                        'status': 'healthy',
                        'service': 'llm',
                        'llm_model': llm_model_name,
                        'llm_device': llm_device,
                        'siglip2_model': siglip2_model_name,
                        'siglip2_device': siglip2_device,
                        'embedding_model': embedding_model_name,
                        'embedding_status': embedding_status,
                        'timestamp': time.time()
                    }
                
                elif request.get('action') == 'shutdown':
                    response = {
                        'status': 'shutting_down',
                        'message': 'LLM Service shutting down'
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
        print("Stopping LLM Service...")
        self.running = False
        if self.socket:
            self.socket.close()
        self.context.term()
        print("LLM Service stopped")


def signal_handler(sig, frame):
    print("\nReceived shutdown signal")
    sys.exit(0)


def main():
    parser = argparse.ArgumentParser(description="LLM Text Processing Service")
    parser.add_argument(
        "--port", 
        type=int, 
        default=5557,
        help="Port to bind the service (default: 5557)"
    )
    parser.add_argument(
        "--debug",
        action="store_true",
        help="Enable debug logging"
    )
    
    args = parser.parse_args()
    
    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)
    
    service = LLMService(port=args.port)
    
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