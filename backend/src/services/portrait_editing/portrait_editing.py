import copy
import json
import logging
import os

# Add LivePortrait to Python path
import sys
from pathlib import Path

import cv2
import numpy as np
import torch
from huggingface_hub import snapshot_download
from PIL import Image
from scipy.interpolate import CubicSpline

liveportrait_path = Path(__file__).parent.parent.parent / "LivePortrait"
if liveportrait_path.exists():
    sys.path.insert(0, str(liveportrait_path))

from backend.LivePortrait.src.config.inference_config import InferenceConfig
from backend.LivePortrait.src.live_portrait_wrapper import LivePortraitWrapper
from backend.LivePortrait.src.utils.camera import get_rotation_matrix

DEVICE = (
    "mps"
    if torch.backends.mps.is_available()
    else "cuda"
    if torch.cuda.is_available()
    else "cpu"
)


def tensor2pil(image):
    return Image.fromarray(
        np.clip(255.0 * image.cpu().numpy().squeeze(), 0, 255).astype(np.uint8)
    )


def pil2tensor(image):
    return torch.from_numpy(np.array(image).astype(np.float32) / 255.0).unsqueeze(0)


def rgb_crop(rgb, region):
    return rgb[region[1] : region[3], region[0] : region[2]]


def get_rgb_size(rgb):
    return rgb.shape[1], rgb.shape[0]


def create_transform_matrix(x, y, scale=1):
    return np.float32([[scale, 0, x], [0, scale, y]])


def calc_crop_limit(center, img_size, crop_size):
    pos = center - crop_size / 2
    if pos < 0:
        crop_size += pos * 2
        pos = 0

    pos2 = pos + crop_size

    if img_size < pos2:
        crop_size -= (pos2 - img_size) * 2
        pos2 = img_size
        pos = pos2 - crop_size

    return pos, pos2, crop_size


def interpolate_dicts(
    from_dict, to_dict, interpolations_num, interpolation_type="linear"
):
    def linear_interpolate(val1, val2, alpha):
        return val1 + (val2 - val1) * alpha

    def nearest_neighbor_interpolate(val1, val2, alpha):
        return val1 if alpha < 0.5 else val2

    keys = list(from_dict.keys())
    from_values = np.array([from_dict[key] for key in keys])
    to_values = np.array([to_dict[key] for key in keys])
    interpolated_dicts = []

    if interpolation_type == "cubic":
        cs = CubicSpline([0, 1], np.vstack([from_values, to_values]), axis=0)

    for i in range(interpolations_num):
        alpha = i / (interpolations_num - 1)
        interpolated_dict = {}

        for key in keys:
            if interpolation_type == "linear":
                interpolated_dict[key] = linear_interpolate(
                    from_dict[key], to_dict[key], alpha
                )
            elif interpolation_type == "nearest":
                interpolated_dict[key] = nearest_neighbor_interpolate(
                    from_dict[key], to_dict[key], alpha
                )
            elif interpolation_type == "cubic":
                interpolated_dict[key] = cs(alpha)[keys.index(key)]

        interpolated_dicts.append(interpolated_dict)

    return interpolated_dicts


def update_expression_json(new_dict):
    default_keys = {
        "rotate_pitch": 0,
        "rotate_yaw": 0,
        "rotate_roll": 0,
        "blink": 0,
        "eyebrow": 0,
        "wink": 0,
        "pupil_x": 0,
        "pupil_y": 0,
        "aaa": 0,
        "eee": 0,
        "woo": 0,
        "smile": 0,
        "src_weight": 0,
    }

    for key in new_dict:
        if key in default_keys:
            default_keys[key] = new_dict[key]

    return default_keys


class PreparedSrcImg:
    def __init__(self, src_rgb, crop_trans_m, x_s_info, f_s_user, x_s_user, mask_ori):
        self.src_rgb = src_rgb
        self.crop_trans_m = crop_trans_m
        self.x_s_info = x_s_info
        self.f_s_user = f_s_user
        self.x_s_user = x_s_user
        self.mask_ori = mask_ori


class ExpressionSet:
    def __init__(self, erst=None, es=None):
        if es != None:
            self.e = copy.deepcopy(es.e)
            self.r = copy.deepcopy(es.r)
            self.s = copy.deepcopy(es.s)
            self.t = copy.deepcopy(es.t)
        elif erst != None:
            self.e = erst[0]
            self.r = erst[1]
            self.s = erst[2]
            self.t = erst[3]
        else:
            self.e = torch.from_numpy(np.zeros((1, 21, 3))).float().to(device=DEVICE)
            self.r = torch.Tensor([0, 0, 0])
            self.s = 0
            self.t = 0

    def div(self, value):
        self.e /= value
        self.r /= value
        self.s /= value
        self.t /= value

    def add(self, other):
        self.e += other.e
        self.r += other.r
        self.s += other.s
        self.t += other.t

    def sub(self, other):
        self.e -= other.e
        self.r -= other.r
        self.s -= other.s
        self.t -= other.t

    def mul(self, value):
        self.e *= value
        self.r *= value
        self.s *= value
        self.t *= value


class LivePortraitEngine:
    def __init__(self, cache_dir: Path):
        self.cache_dir = cache_dir
        self.bbox_model = None
        self.mask_img = None
        self.models_dir = None

        # Initialize immediately, not lazily
        logging.info("Initializing LivePortrait engine...")
        self._ensure_models_downloaded()

        # Initialize pipeline immediately
        logging.info("Loading LivePortrait pipeline...")
        self.pipeline = self._create_pipeline()
        logging.info("LivePortrait engine fully initialized")

    def _ensure_models_downloaded(self):
        """Download LivePortrait models if not already present"""
        if self.models_dir is None:
            logging.info("Downloading LivePortrait models...")
            try:
                self.models_dir = Path(
                    snapshot_download(
                        repo_id="KwaiVGI/LivePortrait",
                        cache_dir=self.cache_dir,
                        ignore_patterns=["*.git*", "README.md", "docs/*"],
                    )
                )
                logging.info(f"Models downloaded to: {self.models_dir}")
            except Exception as e:
                raise RuntimeError(f"Failed to download LivePortrait models: {e}")

    def detect_face(self, image_rgb):
        crop_factor = 1.7
        bbox_drop_size = 10

        if self.bbox_model is None:
            # Use InsightFace for face detection
            from src.utils.face_analysis_diy import FaceAnalysisDIY

            # Set up InsightFace model path
            insightface_model_path = self.models_dir / "insightface"
            if not insightface_model_path.exists():
                raise FileNotFoundError(
                    f"InsightFace models not found in {self.models_dir}"
                )

            self.bbox_model = FaceAnalysisDIY(
                name="buffalo_l",
                root=str(insightface_model_path),
                allowed_modules=["detection"],
            )
            self.bbox_model.prepare(ctx_id=0, det_size=(640, 640))

        # Convert RGB to BGR for InsightFace
        image_bgr = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2BGR)

        # Detect faces
        faces = self.bbox_model.get(image_bgr, max_face_num=1)

        if not faces:
            return None

        # Use the largest face
        face = faces[0]
        bbox = face.bbox  # [x1, y1, x2, y2]

        x1, y1, x2, y2 = bbox
        bbox_w = x2 - x1
        bbox_h = y2 - y1

        w, h = get_rgb_size(image_rgb)

        if min(bbox_w, bbox_h) > bbox_drop_size and bbox_w > 0 and bbox_h > 0:
            crop_w = bbox_w * crop_factor
            crop_h = bbox_h * crop_factor

            crop_w = max(crop_h, crop_w)
            crop_h = crop_w

            kernel_x = x1 + bbox_w / 2
            kernel_y = y1 + bbox_h / 2

            new_x1, new_x2, crop_w = calc_crop_limit(kernel_x, w, crop_w)

            if crop_w < crop_h:
                crop_h = crop_w

            new_y1, new_y2, crop_h = calc_crop_limit(kernel_y, h, crop_h)

            if crop_h < crop_w:
                crop_w = crop_h
                new_x1, new_x2, crop_w = calc_crop_limit(kernel_x, w, crop_w)

            return [int(new_x1), int(new_y1), int(new_x2), int(new_y2)]

        logging.warning("Failed to detect face!")
        return None

    def crop_face(self, rgb_img):
        region = self.detect_face(rgb_img)
        face_image = rgb_crop(rgb_img, region)
        return face_image, region

    def _create_pipeline(self):
        """Create and return the LivePortrait pipeline"""
        # Setup inference config with model paths from downloaded models and config from cloned repo
        liveportrait_repo = Path(__file__).parent.parent.parent / "LivePortrait"
        liveportrait_dir = self.models_dir / "liveportrait"
        base_models_dir = liveportrait_dir / "base_models"
        retargeting_models_dir = liveportrait_dir / "retargeting_models"

        inference_cfg = InferenceConfig(
            models_config=str(liveportrait_repo / "src" / "config" / "models.yaml"),
            checkpoint_F=str(base_models_dir / "appearance_feature_extractor.pth"),
            checkpoint_M=str(base_models_dir / "motion_extractor.pth"),
            checkpoint_G=str(base_models_dir / "spade_generator.pth"),
            checkpoint_W=str(base_models_dir / "warping_module.pth"),
            checkpoint_S=str(
                retargeting_models_dir / "stitching_retargeting_module.pth"
            ),
        )

        return LivePortraitWrapper(inference_cfg)

    def get_pipeline(self):
        """Get the already initialized pipeline"""
        return self.pipeline

    def prepare_src_image(self, img):
        h, w = img.shape[:2]
        input_shape = [256, 256]
        if h != input_shape[0] or w != input_shape[1]:
            x = cv2.resize(
                img, (input_shape[0], input_shape[1]), interpolation=cv2.INTER_LINEAR
            )
        else:
            x = img.copy()

        if x.ndim == 3:
            x = x[np.newaxis].astype(np.float32) / 255.0
        elif x.ndim == 4:
            x = x.astype(np.float32) / 255.0
        else:
            raise ValueError(f"img ndim should be 3 or 4: {x.ndim}")
        x = np.clip(x, 0, 1)
        x = torch.from_numpy(x).permute(0, 3, 1, 2)
        x = x.to(DEVICE)
        return x

    def get_mask(self):
        if self.mask_img is None:
            # Look for mask template in LivePortrait resources
            mask_path = (
                liveportrait_path / "src" / "utils" / "resources" / "mask_template.png"
            )
            if mask_path.exists():
                self.mask_img = cv2.imread(str(mask_path), cv2.IMREAD_COLOR)
            else:
                # Create a default mask if not found
                self.mask_img = np.ones((256, 256, 3), dtype=np.uint8) * 255
        return self.mask_img

    def prepare_source(self, source_image):
        logging.info("Preparing source image...")
        engine = self.get_pipeline()

        # Convert PIL image to numpy array if needed
        if isinstance(source_image, Image.Image):
            source_image_np = np.array(source_image)
        else:
            source_image_np = source_image

        img_rgb = source_image_np
        face_img, crop_region = self.crop_face(img_rgb)

        scale = face_img.shape[0] / 512.0
        crop_trans_m = create_transform_matrix(crop_region[0], crop_region[1], scale)
        mask_ori = cv2.warpAffine(
            self.get_mask(), crop_trans_m, get_rgb_size(img_rgb), cv2.INTER_LINEAR
        )
        mask_ori = mask_ori.astype(np.float32) / 255.0

        i_s = self.prepare_src_image(face_img)
        x_s_info = engine.get_kp_info(i_s)
        f_s_user = engine.extract_feature_3d(i_s)
        x_s_user = engine.transform_keypoint(x_s_info)

        return PreparedSrcImg(
            img_rgb, crop_trans_m, x_s_info, f_s_user, x_s_user, mask_ori
        )

    def calc_fe(
        self,
        x_d_new,
        eyes,
        eyebrow,
        wink,
        pupil_x,
        pupil_y,
        mouth,
        eee,
        woo,
        smile,
        rotate_pitch,
        rotate_yaw,
        rotate_roll,
    ):
        x_d_new[0, 20, 1] += smile * -0.01
        x_d_new[0, 14, 1] += smile * -0.02
        x_d_new[0, 17, 1] += smile * 0.0065
        x_d_new[0, 17, 2] += smile * 0.003
        x_d_new[0, 13, 1] += smile * -0.00275
        x_d_new[0, 16, 1] += smile * -0.00275
        x_d_new[0, 3, 1] += smile * -0.0035
        x_d_new[0, 7, 1] += smile * -0.0035

        x_d_new[0, 19, 1] += mouth * 0.001
        x_d_new[0, 19, 2] += mouth * 0.0001
        x_d_new[0, 17, 1] += mouth * -0.0001
        rotate_pitch -= mouth * 0.05

        x_d_new[0, 20, 2] += eee * -0.001
        x_d_new[0, 20, 1] += eee * -0.001
        x_d_new[0, 14, 1] += eee * -0.001

        x_d_new[0, 14, 1] += woo * 0.001
        x_d_new[0, 3, 1] += woo * -0.0005
        x_d_new[0, 7, 1] += woo * -0.0005
        x_d_new[0, 17, 2] += woo * -0.0005

        x_d_new[0, 11, 1] += wink * 0.001
        x_d_new[0, 13, 1] += wink * -0.0003
        x_d_new[0, 17, 0] += wink * 0.0003
        x_d_new[0, 17, 1] += wink * 0.0003
        x_d_new[0, 3, 1] += wink * -0.0003
        rotate_roll -= wink * 0.1
        rotate_yaw -= wink * 0.1

        if 0 < pupil_x:
            x_d_new[0, 11, 0] += pupil_x * 0.0007
            x_d_new[0, 15, 0] += pupil_x * 0.001
        else:
            x_d_new[0, 11, 0] += pupil_x * 0.001
            x_d_new[0, 15, 0] += pupil_x * 0.0007

        x_d_new[0, 11, 1] += pupil_y * -0.001
        x_d_new[0, 15, 1] += pupil_y * -0.001
        eyes -= pupil_y / 2.0

        x_d_new[0, 11, 1] += eyes * -0.001
        x_d_new[0, 13, 1] += eyes * 0.0003
        x_d_new[0, 15, 1] += eyes * -0.001
        x_d_new[0, 16, 1] += eyes * 0.0003

        if 0 < eyebrow:
            x_d_new[0, 1, 1] += eyebrow * 0.001
            x_d_new[0, 2, 1] += eyebrow * -0.001
        else:
            x_d_new[0, 1, 0] += eyebrow * -0.001
            x_d_new[0, 2, 0] += eyebrow * 0.001
            x_d_new[0, 1, 1] += eyebrow * 0.0003
            x_d_new[0, 2, 1] += eyebrow * -0.0003

        return torch.Tensor([rotate_pitch, rotate_yaw, rotate_roll])

    def expression_run(self, psi, expression_json):
        rotate_yaw = -expression_json["rotate_yaw"]

        pipeline = self.get_pipeline()

        s_info = psi.x_s_info
        s_exp = s_info["exp"] * expression_json["src_weight"]
        s_exp[0, 5] = s_info["exp"][0, 5]
        s_exp += s_info["kp"]

        es = ExpressionSet()

        es.r = self.calc_fe(
            es.e,
            expression_json["blink"],
            expression_json["eyebrow"],
            expression_json["wink"],
            expression_json["pupil_x"],
            expression_json["pupil_y"],
            expression_json["aaa"],
            expression_json["eee"],
            expression_json["woo"],
            expression_json["smile"],
            expression_json["rotate_pitch"],
            rotate_yaw,
            expression_json["rotate_roll"],
        )

        new_rotate = get_rotation_matrix(
            s_info["pitch"] + es.r[0], s_info["yaw"] + es.r[1], s_info["roll"] + es.r[2]
        )
        x_d_new = (s_info["scale"] * (1 + es.s)) * (
            (s_exp + es.e) @ new_rotate
        ) + s_info["t"]

        x_d_new = pipeline.stitching(psi.x_s_user, x_d_new)

        crop_out = pipeline.warp_decode(psi.f_s_user, psi.x_s_user, x_d_new)
        crop_out = pipeline.parse_output(crop_out["out"])[0]

        crop_with_fullsize = cv2.warpAffine(
            crop_out, psi.crop_trans_m, get_rgb_size(psi.src_rgb), cv2.INTER_LINEAR
        )
        out = np.clip(
            psi.mask_ori * crop_with_fullsize + (1 - psi.mask_ori) * psi.src_rgb, 0, 255
        ).astype(np.uint8)

        return Image.fromarray(out)

    def edit_expression(
        self, image: Image.Image, expression_params: dict
    ) -> Image.Image:
        """Main method to edit portrait expression"""
        psi = self.prepare_source(image)

        expression_json = update_expression_json(expression_params)

        result_image = self.expression_run(psi, expression_json)

        return result_image
