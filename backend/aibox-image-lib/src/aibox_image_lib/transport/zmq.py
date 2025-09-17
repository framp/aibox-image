import logging
from typing import Type, cast

import msgpack
import zmq

from . import BaseResponse, Req, Transport


class ZmqTransport(Transport):
    def __init__(self):
        super().__init__()

        self.socket = None
        self.context = None

    def _parse(self, request: bytes, model: Type[Req]) -> Req:
        data = msgpack.unpackb(request, raw=False)
        return model(**data)

    def _serialize(self, response: BaseResponse) -> bytes:
        return cast(bytes, msgpack.packb(response.model_dump(), use_bin_type=True))

    def start(self, port: int):
        if self.context or self.socket:
            raise ValueError("ZmqTransport already started")

        logging.info("Starting ZmqTransport...")
        self.context = zmq.Context()
        self.socket = self.context.socket(zmq.REP)
        self.socket.bind(f"tcp://*:{port}")

        logging.info(f"ZmqTransport started on port {port}")
        while self.context:
            message = self.socket.recv()
            response = self._handle(message)
            self.socket.send(response)

    def stop(self):
        logging.info("Stopping ZmqTransport...")

        if self.socket:
            self.socket.close()
            self.socket = None
        if self.context:
            self.context.term()
            self.context = None

        logging.info("ZmqTransport stopped")
