#!/usr/bin/env python3

import os
import sys
import json
import time
import signal
import argparse
from pathlib import Path
from typing import List, Dict, Optional, Tuple, NamedTuple
from PIL import Image
import numpy as np
import zmq
import msgpack
from deepface import DeepFace
from sklearn.metrics.pairwise import cosine_similarity


class FaceInfo(NamedTuple):
    bbox: Tuple[int, int, int, int]
    confidence: float
    age: Optional[int]
    gender: Optional[str]
    race: Optional[str]
    emotion: Optional[str]
    embedding: Optional[List[float]]


class FaceDetectionService:
    def __init__(self, port: int = 5555):
        self.port = port
        self.context = zmq.Context()
        self.socket = None
        self.running = False
        self.detector_backend = 'mtcnn'
        self.embedding_model_name = 'ArcFace'
        
        self._warmup_models()
    
    def _warmup_models(self):
        try:
            print("Warming up DeepFace models...")
            
            test_image = np.ones((100, 100, 3), dtype=np.uint8) * 128
            temp_path = "/tmp/warmup_face_test.jpg"
            Image.fromarray(test_image).save(temp_path)
            
            try:
                DeepFace.analyze(
                    img_path=temp_path,
                    actions=[],
                    detector_backend=self.detector_backend,
                    enforce_detection=False
                )
            except Exception as e:
                print(f"Warmup analyze failed (expected): {e}")
            
            try:
                DeepFace.represent(
                    img_path=temp_path,
                    detector_backend=self.detector_backend,
                    model_name=self.embedding_model_name,
                    enforce_detection=False
                )
            except Exception as e:
                print(f"Warmup represent failed (expected): {e}")
            
            if os.path.exists(temp_path):
                os.remove(temp_path)
                
            print("DeepFace models warmed up successfully")
            
        except Exception as e:
            print(f"Warning: Model warmup failed: {e}")
    
    def detect_faces(self, image_path: str) -> List[FaceInfo]:
        try:
            image_path = Path(image_path)
            
            if not image_path.exists():
                raise FileNotFoundError(f"Image file not found: {image_path}")
            
            print(f"Processing image: {image_path}")
            
            try:
                with Image.open(image_path) as img:
                    if img.mode != 'RGB':
                        img = img.convert('RGB')
                    
                    import tempfile
                    with tempfile.NamedTemporaryFile(suffix='.jpg', delete=False) as temp_file:
                        img.save(temp_file.name, 'JPEG')
                        temp_image_path = temp_file.name
                
                print(f"Using {self.detector_backend} detector and {self.embedding_model_name} model")
                embedding_obj = DeepFace.represent(
                    img_path=temp_image_path,
                    detector_backend=self.detector_backend,
                    model_name=self.embedding_model_name,
                    enforce_detection=False,
                )
                
                import os
                if os.path.exists(temp_image_path):
                    os.remove(temp_image_path)
                    
            except Exception as img_error:
                print(f"Image preprocessing failed: {img_error}")
                try:
                    print(f"Using {self.detector_backend} detector and {self.embedding_model_name} model (fallback)")
                    embedding_obj = DeepFace.represent(
                        img_path=str(image_path),
                        detector_backend=self.detector_backend,
                        model_name=self.embedding_model_name,
                        enforce_detection=False,
                    )
                except Exception as fallback_error:
                    print(f"Fallback also failed: {fallback_error}")
                    return []
            
            print(f"DeepFace.represent returned: {type(embedding_obj)}")
            
            faces = []
            if embedding_obj and isinstance(embedding_obj, list):
                print(f"Found {len(embedding_obj)} face embeddings")
                for i, emb_data in enumerate(embedding_obj):
                    if isinstance(emb_data, dict) and 'embedding' in emb_data:
                        is_real = emb_data.get('is_real', True)
                        
                        facial_area = emb_data.get('facial_area', {})
                        if facial_area:
                            bbox = (
                                int(facial_area.get('x', 0)),
                                int(facial_area.get('y', 0)),
                                int(facial_area.get('w', 100)),
                                int(facial_area.get('h', 100))
                            )
                            confidence = float(facial_area.get('confidence', 0.9))
                        else:
                            bbox = (0, 0, 100, 100)
                            confidence = 0.9
                        
                        face_info = FaceInfo(
                            bbox=bbox,
                            confidence=confidence,
                            age=None,
                            gender=None,
                            race=None,
                            emotion=None,
                            embedding=emb_data['embedding']
                        )
                        faces.append(face_info)
                        print(f"Processed face {i+1}: embedding dims={len(emb_data['embedding'])}, is_real={is_real}")
                    else:
                        print(f"Skipping invalid embedding data at index {i}: {type(emb_data)}")
                        
                return faces
            else:
                print("No embeddings found or invalid format")
                return []
            
        except Exception as e:
            print(f"Error detecting faces: {e}")
            import traceback
            traceback.print_exc()
            return []
    
    def extract_faces(self, image_path: str) -> List[Dict]:
        faces = self.detect_faces(image_path)
        result = []
        
        for i, face in enumerate(faces):
            if face.embedding:
                face_data = {
                    'face_id': i + 1,
                    'bbox': face.bbox,
                    'confidence': face.confidence,
                    'embedding': face.embedding,
                    'embedding_model': self.embedding_model_name
                }
                result.append(face_data)
        
        return result
    
    def compare_embeddings(self, embedding1: List[float], embedding2: List[float], 
                          metric: str = "cosine") -> Dict:
        try:
            emb1 = np.array(embedding1).reshape(1, -1)
            emb2 = np.array(embedding2).reshape(1, -1)
            
            if metric == "cosine":
                similarity = float(cosine_similarity(emb1, emb2)[0][0])
                distance = float(1 - similarity)
                threshold = 0.68
                verified = bool(distance < threshold)
                
            elif metric == "euclidean":
                distance = float(np.linalg.norm(emb1 - emb2))
                threshold = 4.15
                verified = bool(distance < threshold)
                similarity = float(1 / (1 + distance))
                
            elif metric == "euclidean_l2":
                emb1_norm = emb1 / np.linalg.norm(emb1)
                emb2_norm = emb2 / np.linalg.norm(emb2)
                distance = float(np.linalg.norm(emb1_norm - emb2_norm))
                threshold = 1.13
                verified = bool(distance < threshold)
                similarity = float(1 / (1 + distance))
                
            elif metric == "angular":
                cos_sim = float(cosine_similarity(emb1, emb2)[0][0])
                cos_sim = float(np.clip(cos_sim, -1, 1))
                distance = float(np.arccos(cos_sim) / np.pi)
                threshold = 0.4
                verified = bool(distance < threshold)
                similarity = float(1 - distance)
                
            else:
                raise ValueError(f"Unknown metric: {metric}")
            
            return {
                'verified': verified,
                'distance': distance,
                'similarity': similarity,
                'threshold': float(threshold),
                'metric': metric,
                'model': self.embedding_model_name
            }
            
        except Exception as e:
            return {
                'error': f"Failed to compare embeddings: {str(e)}",
                'verified': bool(False),
                'distance': float('inf'),
                'similarity': float(0.0),
                'threshold': None,
                'metric': metric,
                'model': self.embedding_model_name
            }
    
    def start(self):
        print(f"Starting Face Detection Service on port {self.port}")
        
        self.socket = self.context.socket(zmq.REP)
        self.socket.bind(f"tcp://*:{self.port}")
        self.running = True
        
        print("Face Detection Service ready! Waiting for requests...")
        
        while self.running:
            try:
                message = self.socket.recv()
                request = msgpack.unpackb(message, raw=False)
                
                print(f"Received request: {request.get('action', 'unknown')}")
                
                if request.get('action') == 'detect_faces':
                    image_path = request.get('image_path')
                    if not image_path:
                        response = {
                            'status': 'error',
                            'message': 'Missing image_path parameter'
                        }
                    else:
                        try:
                            faces_data = self.extract_faces(image_path)
                            
                            response = {
                                'status': 'success',
                                'faces': faces_data
                            }
                            print(f"Detected {len(faces_data)} faces")
                        except Exception as e:
                            response = {
                                'status': 'error',
                                'message': str(e)
                            }
                            print(f"Error processing request: {e}")
                
                elif request.get('action') == 'extract_faces':
                    image_path = request.get('image_path')
                    
                    if not image_path:
                        response = {
                            'status': 'error',
                            'message': 'Missing image_path parameter'
                        }
                    else:
                        try:
                            faces_data = self.extract_faces(image_path)
                            response = {
                                'status': 'success',
                                'faces': faces_data
                            }
                            print(f"Extracted {len(faces_data)} faces with embeddings")
                        except Exception as e:
                            response = {
                                'status': 'error',
                                'message': str(e)
                            }
                            print(f"Error extracting faces: {e}")
                
                elif request.get('action') == 'compare_embeddings':
                    embedding1 = request.get('embedding1')
                    embedding2 = request.get('embedding2')
                    metric = request.get('metric', 'cosine')
                    
                    if not embedding1 or not embedding2:
                        response = {
                            'status': 'error',
                            'message': 'Missing embedding1 or embedding2 parameter'
                        }
                    else:
                        try:
                            comparison_result = self.compare_embeddings(embedding1, embedding2, metric)
                            response = {
                                'status': 'success',
                                'comparison': comparison_result
                            }
                            print(f"Compared embeddings: verified={comparison_result.get('verified', False)}")
                        except Exception as e:
                            response = {
                                'status': 'error',
                                'message': str(e)
                            }
                            print(f"Error comparing embeddings: {e}")
                
                elif request.get('action') == 'health':
                    response = {
                        'status': 'healthy',
                        'service': 'face-detection',
                        'timestamp': time.time()
                    }
                
                elif request.get('action') == 'shutdown':
                    response = {
                        'status': 'shutting_down',
                        'message': 'Face Detection Service shutting down'
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
        print("Stopping Face Detection Service...")
        self.running = False
        if self.socket:
            self.socket.close()
        self.context.term()
        print("Face Detection Service stopped")


def signal_handler(sig, frame):
    print("\nReceived shutdown signal")
    sys.exit(0)


def main():
    parser = argparse.ArgumentParser(description="Face Detection Service")
    parser.add_argument(
        "--port", 
        type=int, 
        default=5555,
        help="Port to bind the service (default: 5555)"
    )
    parser.add_argument(
        "--debug",
        action="store_true",
        help="Enable debug logging"
    )
    
    args = parser.parse_args()
    
    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)
    
    service = FaceDetectionService(port=args.port)
    
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