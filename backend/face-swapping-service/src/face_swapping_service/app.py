import io
from typing import Literal

from aibox_image_lib.transport import BaseRequest, BaseResponse, Transport
from PIL import Image

from face_swapping_service.service import LoadParams, Service


class LoadRequest(BaseRequest, LoadParams):
    pass


class LoadResponse(BaseResponse):
    status: str


class HealthResponse(BaseResponse):
    status: Literal["healthy", "unhealthy"]


class FaceSwapRequest(BaseRequest):
    source_image_bytes: bytes
    target_image_bytes: bytes
    face_index: int


class FaceSwapResponse(BaseResponse):
    status: str
    image: bytes


class AnalyzeFacesRequest(BaseRequest):
    image_bytes: bytes


class AnalyzeFacesResponse(BaseResponse):
    status: str
    faces: list
    preview_image: bytes


def register_use_cases(transport: Transport, service: Service):
    @transport.handler()
    def load(request: LoadRequest):
        service.load(params=request)
        return BaseResponse()

    @transport.handler()
    def health(_: BaseRequest):
        healthy = service.health()
        return HealthResponse(status="healthy" if healthy else "unhealthy")

    @transport.handler("face_swap")
    def face_swap(request: FaceSwapRequest):
        source_image = Image.open(io.BytesIO(request.source_image_bytes)).convert("RGB")
        target_image = Image.open(io.BytesIO(request.target_image_bytes)).convert("RGB")

        result_image = service.swap_face(source_image, target_image, request.face_index)

        buffer = io.BytesIO()
        result_image.save(buffer, format="PNG")
        image_bytes = buffer.getvalue()

        return FaceSwapResponse(status="success", image=image_bytes)

    @transport.handler("analyze_faces")
    def analyze_faces(request: AnalyzeFacesRequest):
        source_image = Image.open(io.BytesIO(request.image_bytes)).convert("RGB")

        faces_info, preview_image = service.analyze_faces_with_preview(source_image)

        buffer = io.BytesIO()
        preview_image.save(buffer, format="PNG")
        preview_bytes = buffer.getvalue()

        return AnalyzeFacesResponse(status="success", faces=faces_info, preview_image=preview_bytes)
