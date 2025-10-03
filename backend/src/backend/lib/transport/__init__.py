import logging
from abc import ABC, abstractmethod
from typing import Callable, Type, TypeVar

from pydantic import BaseModel


class BaseRequest(BaseModel):
    action: str


class BaseResponse(BaseModel):
    pass


Req = TypeVar("Req", bound=BaseRequest)
Resp = TypeVar("Resp", bound=BaseResponse)


class Transport(ABC):
    def __init__(self):
        # name -> (function, request_model)
        self._handlers: dict[
            str, tuple[Callable[[BaseRequest], BaseResponse], Type[BaseRequest]]
        ] = {}

    def handler(self, name: str | None = None):
        """
        Marks a method as a handler.
        """

        def decorator(func: Callable):
            # Infer request type: first parameter
            params = list(func.__annotations__.items())
            if len(params) != 1:
                raise ValueError("Handler must have exactly one parameter (request)")

            request_type = params[0][1]
            if not issubclass(request_type, BaseRequest):
                raise TypeError(
                    f"Handler parameter must subclass BaseRequest, got {request_type}"
                )

            handler_name = name or func.__name__
            self._handlers[handler_name] = (func, request_type)
            return func

        return decorator

    def _handle(self, raw_request: bytes) -> bytes:
        request = self._parse(raw_request, BaseRequest)
        logging.info(f"Handling request {request.action}")

        func, req_model = self._handlers[request.action]

        req_obj = self._parse(raw_request, req_model)
        resp_obj = func(req_obj)

        if not isinstance(resp_obj, BaseResponse):
            raise TypeError(
                f"Handler {func.__name__} must return a BaseResponse subclass"
            )

        serialized = self._serialize(resp_obj)
        logging.info(f"Handled request {request.action}")

        return serialized

    @abstractmethod
    def _parse(self, request: bytes, model: type[Req]) -> Req:
        """
        Parse a request from the transport protocol.
        """
        ...

    @abstractmethod
    def _serialize(self, response: BaseResponse) -> bytes:
        """
        Serialize a request to the transport protocol.
        """
        ...
