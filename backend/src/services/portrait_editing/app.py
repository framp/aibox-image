import io
from typing import Literal

from PIL import Image

from backend.lib.transport import BaseRequest, BaseResponse, Transport

from .service import LoadParams, Service


class LoadRequest(BaseRequest, LoadParams):
    pass


class LoadResponse(BaseResponse):
    status: str


class HealthResponse(BaseResponse):
    status: Literal["healthy", "unhealthy"]


class EditExpressionRequest(BaseRequest):
    image_bytes: bytes
    rotate_pitch: float
    rotate_yaw: float
    rotate_roll: float
    blink: float
    eyebrow: float
    wink: float
    pupil_x: float
    pupil_y: float
    aaa: float
    eee: float
    woo: float
    smile: float
    src_weight: float


class EditExpressionResponse(BaseResponse):
    status: str
    image: bytes


def register_use_cases(transport: Transport, service: Service):
    @transport.handler()
    def load(request: LoadRequest):
        service.load(params=request)
        return BaseResponse()

    @transport.handler()
    def health(_: BaseRequest):
        healthy = service.health()
        return HealthResponse(status="healthy" if healthy else "unhealthy")

    @transport.handler("edit_expression")
    def edit_expression(request: EditExpressionRequest):
        image = Image.open(io.BytesIO(request.image_bytes)).convert("RGB")

        expression_params = {
            "rotate_pitch": request.rotate_pitch,
            "rotate_yaw": request.rotate_yaw,
            "rotate_roll": request.rotate_roll,
            "blink": request.blink,
            "eyebrow": request.eyebrow,
            "wink": request.wink,
            "pupil_x": request.pupil_x,
            "pupil_y": request.pupil_y,
            "aaa": request.aaa,
            "eee": request.eee,
            "woo": request.woo,
            "smile": request.smile,
            "src_weight": request.src_weight,
        }

        result_image = service.edit_expression(image, expression_params)

        buffer = io.BytesIO()
        result_image.save(buffer, format="PNG")
        image_bytes = buffer.getvalue()

        return EditExpressionResponse(status="success", image=image_bytes)
