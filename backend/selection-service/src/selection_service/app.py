import io

from aibox_image_lib.transport import BaseRequest, BaseResponse, Transport
from PIL import Image

from selection_service.service import Service


class ImageSelectionRequest(BaseRequest):
    prompt: str
    image_bytes: bytes
    threshold: float = 0.25


class ImageSelectionResponse(BaseResponse):
    status: str
    masks: list[bytes]


def register_use_cases(transport: Transport, service: Service):
    @transport.handler("image_selection")
    def image_selection(request: ImageSelectionRequest):
        image = Image.open(io.BytesIO(request.image_bytes)).convert("RGB")

        masks = service.image_selection(request.prompt, image, request.threshold)

        mask_bytes = []
        for mask in masks:
            buffer = io.BytesIO()
            mask.save(buffer, format="PNG")
            mask_bytes.append(buffer.getvalue())

        return ImageSelectionResponse(status="success", masks=mask_bytes)
