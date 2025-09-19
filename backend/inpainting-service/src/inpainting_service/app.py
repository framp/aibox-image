import io
from typing import Literal

from aibox_image_lib.transport import BaseRequest, BaseResponse, Transport
from PIL import Image

from inpainting_service.service import LoadParams, Service


class LoadRequest(BaseRequest, LoadParams):
    pass


class LoadResponse(BaseResponse):
    status: str


class HealthResponse(BaseResponse):
    status: Literal["healthy", "unhealthy"]


class InpaintRequest(BaseRequest):
    prompt: str
    image_bytes: bytes
    mask: bytes


class InpaintResponse(BaseResponse):
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

    @transport.handler("inpaint")
    def inpaint(request: InpaintRequest):
        image = Image.open(io.BytesIO(request.image_bytes)).convert("RGB")
        mask = Image.open(io.BytesIO(request.mask)).convert("RGB")

        image = service.inpaint(request.prompt, image, mask)

        buffer = io.BytesIO()
        image.save(buffer, format="PNG")
        image_bytes = buffer.getvalue()

        return InpaintResponse(status="success", image=image_bytes)
