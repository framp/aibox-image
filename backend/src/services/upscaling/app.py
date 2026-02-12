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


class UpscaleRequest(BaseRequest):
    prompt: str
    image_bytes: bytes


class UpscaleResponse(BaseResponse):
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

    @transport.handler("upscale")
    def upscale(request: UpscaleRequest):
        image = Image.open(io.BytesIO(request.image_bytes)).convert("RGB")

        image = service.upscale(request.prompt, image)

        buffer = io.BytesIO()
        image.save(buffer, format="PNG")
        image_bytes = buffer.getvalue()

        return UpscaleResponse(status="success", image=image_bytes)
