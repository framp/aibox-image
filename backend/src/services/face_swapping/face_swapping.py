import hashlib
import logging
import os
from pathlib import Path
from typing import List, Optional, Protocol

import cv2
import insightface
import numpy as np
import torch
from gfpgan import GFPGANer
from huggingface_hub import hf_hub_download
from insightface.app.common import Face
from insightface.utils import face_align
from PIL import Image


class FaceSwapperProtocol(Protocol):
    def swap_face(
        self, source_image: Image.Image, target_image: Image.Image, face_index: int
    ) -> Image.Image: ...


def get_image_md5hash(image: np.ndarray) -> str:
    """Get MD5 hash of image array"""
    return hashlib.md5(image.tobytes()).hexdigest()


def get_providers():
    """Get optimal execution providers based on available hardware"""
    try:
        if torch.cuda.is_available():
            providers = ["CUDAExecutionProvider"]
        elif torch.backends.mps.is_available():
            providers = ["CoreMLExecutionProvider"]
        else:
            providers = ["CPUExecutionProvider"]
    except Exception as e:
        logging.debug(f"ExecutionProviderError: {e}. EP is set to CPU.")
        providers = ["CPUExecutionProvider"]
    return providers


def analyze_faces(img_data: np.ndarray, det_size=(640, 640), face_analyser=None):
    """Analyze faces in image data"""
    if face_analyser is None:
        return []

    faces = []
    try:
        faces = face_analyser.get(img_data)
    except Exception as e:
        logging.error(f"No faces found: {e}")

    # Try halving det_size if no faces are found
    if len(faces) == 0 and det_size[0] > 320 and det_size[1] > 320:
        det_size_half = (det_size[0] // 2, det_size[1] // 2)
        logging.info("Trying to halve 'det_size' parameter")
        face_analyser.prepare(ctx_id=0, det_size=det_size_half)
        try:
            faces = face_analyser.get(img_data)
        except Exception as e:
            logging.error(f"No faces found with halved det_size: {e}")

    return faces


def sort_faces_by_order(faces: List[Face], order: str = "large-small"):
    """Sort faces by specified order"""
    if order == "left-right":
        return sorted(faces, key=lambda x: x.bbox[0])
    elif order == "right-left":
        return sorted(faces, key=lambda x: x.bbox[0], reverse=True)
    elif order == "top-bottom":
        return sorted(faces, key=lambda x: x.bbox[1])
    elif order == "bottom-top":
        return sorted(faces, key=lambda x: x.bbox[1], reverse=True)
    elif order == "small-large":
        return sorted(
            faces, key=lambda x: (x.bbox[2] - x.bbox[0]) * (x.bbox[3] - x.bbox[1])
        )
    else:  # "large-small" by default
        return sorted(
            faces,
            key=lambda x: (x.bbox[2] - x.bbox[0]) * (x.bbox[3] - x.bbox[1]),
            reverse=True,
        )


def get_face_by_index(
    faces: List[Face], face_index: int, order: str = "large-small"
) -> Optional[Face]:
    """Get face by index from sorted faces"""
    if not faces:
        return None

    faces_sorted = sort_faces_by_order(faces, order)

    if face_index >= len(faces_sorted):
        logging.info(
            f"Requested face index ({face_index}) is out of bounds (max available index is {len(faces_sorted) - 1})"
        )
        return None

    return faces_sorted[face_index]


def download_inswapper_model(cache_dir: Path) -> str:
    """Download inswapper_128.onnx model from Hugging Face"""
    model_filename = "inswapper_128.onnx"
    model_path = cache_dir / "inswapper" / model_filename

    if model_path.exists():
        logging.info(f"Model already exists at: {model_path}")
        return str(model_path)

    logging.info("Downloading inswapper_128.onnx from Hugging Face...")
    model_path.parent.mkdir(parents=True, exist_ok=True)

    try:
        downloaded_path = hf_hub_download(
            repo_id="ezioruan/inswapper_128.onnx",
            filename=model_filename,
            cache_dir=str(cache_dir / "huggingface"),
            local_dir=str(model_path.parent),
            local_dir_use_symlinks=False,
        )
        logging.info(f"Successfully downloaded model to: {downloaded_path}")
        return downloaded_path
    except Exception as e:
        logging.error(f"Failed to download model from Hugging Face: {e}")
        raise


def download_gfpgan_model(cache_dir: Path) -> str:
    """Download GFPGANv1.4.pth model from Hugging Face"""
    model_filename = "GFPGANv1.4.pth"
    model_path = cache_dir / "gfpgan" / model_filename

    if model_path.exists():
        logging.info(f"GFPGAN model already exists at: {model_path}")
        return str(model_path)

    logging.info("Downloading GFPGANv1.4.pth from Hugging Face...")
    model_path.parent.mkdir(parents=True, exist_ok=True)

    try:
        downloaded_path = hf_hub_download(
            repo_id="gmk123/GFPGAN",
            filename=model_filename,
            cache_dir=str(cache_dir / "huggingface"),
            local_dir=str(model_path.parent),
            local_dir_use_symlinks=False,
        )
        logging.info(f"Successfully downloaded GFPGAN model to: {downloaded_path}")
        return downloaded_path
    except Exception as e:
        logging.error(f"Failed to download GFPGAN model from Hugging Face: {e}")
        raise


class FaceSwapper(FaceSwapperProtocol):
    def __init__(
        self, cache_dir: Path, insightface_model: str = "", inswapper_model: str = ""
    ):
        logging.info(
            f"Initializing face swapper with insightface: {insightface_model}, inswapper: {inswapper_model}"
        )
        self.cache_dir = cache_dir
        self.insightface_model = insightface_model
        self.inswapper_model = inswapper_model

        # Initialize providers
        self.providers = get_providers()
        logging.info(f"Using providers: {self.providers}")

        # Initialize face analysis model
        self.face_analyser = None
        self.face_swapper_model = None

        # Cache for analyzed faces
        self.source_faces_cache = {}
        self.target_faces_cache = {}

        # Initialize models
        self._init_models()

    def _init_models(self):
        """Initialize face analysis and swapping models"""
        try:
            # Initialize face analysis model
            insightface_path = self.cache_dir / "insightface"
            insightface_path.mkdir(parents=True, exist_ok=True)

            logging.info("Loading face analysis model...")
            self.face_analyser = insightface.app.FaceAnalysis(
                name="buffalo_l", providers=self.providers, root=str(insightface_path)
            )
            self.face_analyser.prepare(ctx_id=0, det_size=(640, 640))

            # Initialize face swapper model (Inswapper)
            inswapper_path = self.inswapper_model

            if not inswapper_path or not os.path.exists(inswapper_path):
                logging.info(
                    "Inswapper model not found locally, downloading from Hugging Face..."
                )
                try:
                    inswapper_path = download_inswapper_model(self.cache_dir)
                except Exception as e:
                    logging.error(f"Failed to download inswapper model: {e}")
                    inswapper_path = None

            if inswapper_path and os.path.exists(inswapper_path):
                logging.info(f"Loading inswapper model from: {inswapper_path}")
                self.face_swapper_model = insightface.model_zoo.get_model(
                    inswapper_path, providers=self.providers
                )
            else:
                logging.warning("Inswapper model not found or could not be downloaded")

            self.gfpgan_model = GFPGANer(
                model_path=download_gfpgan_model(self.cache_dir),
                upscale=1,  # keep size, just enhance quality
                arch="clean",
                channel_multiplier=2,
                bg_upsampler=None,
            )

        except Exception as e:
            logging.error(f"Failed to initialize models: {e}")
            raise

    def swap_face(
        self, source_image: Image.Image, target_image: Image.Image, face_index: int
    ) -> Image.Image:
        """Perform face swapping between source and target images"""
        logging.info(f"Face swapping requested with face index: {face_index}")

        if self.face_analyser is None or self.face_swapper_model is None:
            logging.error("Models not properly initialized")
            return target_image

        try:
            # Convert PIL images to OpenCV format
            source_cv = cv2.cvtColor(np.array(source_image), cv2.COLOR_RGB2BGR)
            target_cv = cv2.cvtColor(np.array(target_image), cv2.COLOR_RGB2BGR)

            # Analyze source image for faces
            source_hash = get_image_md5hash(source_cv)
            if source_hash not in self.source_faces_cache:
                logging.info("Analyzing source image...")
                source_faces = analyze_faces(
                    source_cv, face_analyser=self.face_analyser
                )
                self.source_faces_cache[source_hash] = source_faces
            else:
                logging.info("Using cached source faces...")
                source_faces = self.source_faces_cache[source_hash]

            # Analyze target image for faces
            target_hash = get_image_md5hash(target_cv)
            if target_hash not in self.target_faces_cache:
                logging.info("Analyzing target image...")
                target_faces = analyze_faces(
                    target_cv, face_analyser=self.face_analyser
                )
                self.target_faces_cache[target_hash] = target_faces
            else:
                logging.info("Using cached target faces...")
                target_faces = self.target_faces_cache[target_hash]

            # Check if faces were found
            if not source_faces:
                logging.warning("No faces found in source image")
                return target_image

            if not target_faces:
                logging.warning("No faces found in target image")
                return target_image

            # Get source face (use first face if multiple)
            source_face = get_face_by_index(source_faces, 0)
            if source_face is None:
                logging.warning("Could not get source face")
                return target_image

            # Get target face by index
            target_face = get_face_by_index(target_faces, face_index)
            if target_face is None:
                logging.warning(f"Could not get target face at index {face_index}")
                return target_image

            # Optional: align
            source_aligned = face_align.norm_crop(
                source_cv, source_face.kps, image_size=128
            )
            target_aligned = face_align.norm_crop(
                target_cv, target_face.kps, image_size=128
            )

            aligned_dir = self.cache_dir / "aligned_faces"
            aligned_dir.mkdir(parents=True, exist_ok=True)

            # Save as PNG
            cv2.imwrite(str(aligned_dir / "source_aligned.png"), source_aligned)
            cv2.imwrite(
                str(aligned_dir / f"target_aligned_{face_index}.png"), target_aligned
            )

            # Perform face swapping with paste_back=True (full-frame compositing)
            logging.info("Performing face swap...")
            result_cv = self.face_swapper_model.get(
                target_cv, target_face, source_face, paste_back=True
            )

            cropped_faces, restored_faces, restored_img = self.gfpgan_model.enhance(
                result_cv, has_aligned=False, paste_back=True, weight=0.5
            )

            # Use restored_img as the final output
            if restored_img is None:
                # fallback: take first restored face
                result_enhanced = restored_faces[0]
            else:
                result_enhanced = restored_img

            # Ensure uint8
            if result_enhanced.dtype != np.uint8:
                if result_enhanced.max() <= 1.0:
                    result_enhanced = (result_enhanced * 255).astype(np.uint8)
                else:
                    result_enhanced = result_enhanced.astype(np.uint8)

            # Convert result back to PIL Image
            result_image = Image.fromarray(
                cv2.cvtColor(result_enhanced, cv2.COLOR_BGR2RGB)
            )

            logging.info("Face swap completed successfully")
            return result_image

        except Exception as e:
            logging.error(f"Face swapping failed: {e}")
            return target_image

    def analyze_faces_with_preview(self, source_image: Image.Image):
        """Analyze faces in source image and return face info with preview"""
        logging.info("Analyzing faces for preview...")

        if self.face_analyser is None:
            logging.error("Face analyzer not initialized")
            return [], source_image

        try:
            # Convert PIL image to OpenCV format
            source_cv = cv2.cvtColor(np.array(source_image), cv2.COLOR_RGB2BGR)

            # Analyze faces
            source_hash = get_image_md5hash(source_cv)
            if source_hash not in self.source_faces_cache:
                logging.info("Analyzing source image for preview...")
                source_faces = analyze_faces(
                    source_cv, face_analyser=self.face_analyser
                )
                self.source_faces_cache[source_hash] = source_faces
            else:
                logging.info("Using cached source faces for preview...")
                source_faces = self.source_faces_cache[source_hash]

            if not source_faces:
                logging.warning("No faces found in source image")
                return [], source_image

            # Sort faces by size (largest first)
            sorted_faces = sort_faces_by_order(source_faces, "large-small")

            # Create preview image with face bounding boxes
            preview_cv = source_cv.copy()
            faces_info = []

            for i, face in enumerate(sorted_faces):
                # Get bounding box coordinates
                bbox = face.bbox.astype(int)
                x1, y1, x2, y2 = bbox

                # Draw bounding box
                color = (
                    (0, 255, 0) if i == 0 else (255, 0, 0)
                )  # Green for first face, red for others
                cv2.rectangle(preview_cv, (x1, y1), (x2, y2), color, 2)

                # Add face index label
                cv2.putText(
                    preview_cv,
                    f"Face {i}",
                    (x1, y1 - 10),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.5,
                    color,
                    1,
                )

                # Store face info
                faces_info.append(
                    {
                        "index": i,
                        "bbox": [int(x1), int(y1), int(x2), int(y2)],
                        "confidence": float(face.det_score),
                        "is_primary": i == 0,
                    }
                )

            # Convert back to PIL Image
            preview_image = Image.fromarray(cv2.cvtColor(preview_cv, cv2.COLOR_BGR2RGB))

            logging.info(f"Found {len(sorted_faces)} faces in source image")
            return faces_info, preview_image

        except Exception as e:
            logging.error(f"Face analysis failed: {e}")
            return [], source_image
