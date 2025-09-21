import io
from typing import Literal

from aibox_image_lib.transport import BaseRequest, BaseResponse, Transport
from PIL import Image

from portrait_editing_service.service import LoadParams, Service


class LoadRequest(BaseRequest, LoadParams):
    pass


class LoadResponse(BaseResponse):
    status: str


class HealthResponse(BaseResponse):
    status: Literal["healthy", "unhealthy"]


class EditExpressionRequest(BaseRequest):
    image_bytes: bytes
    expression_params: dict


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

        result_image = service.edit_expression(image, request.expression_params)

        buffer = io.BytesIO()
        result_image.save(buffer, format="PNG")
        image_bytes = buffer.getvalue()

        return EditExpressionResponse(status="success", image=image_bytes)
