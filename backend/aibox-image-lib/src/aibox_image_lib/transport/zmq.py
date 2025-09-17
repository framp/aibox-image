from typing import Type

import msgpack
import zmq

from . import ReqT, RespT, Transport


class ZmqTransport(Transport):
    def __init__(self):
        super().__init__()

        self.socket = None
        self.context = None

    def _parse(self, raw: bytes, model: Type[ReqT]) -> ReqT:
        data = msgpack.unpackb(raw, raw=False)
        return model(**data)

    def _serialize(self, response: RespT) -> bytes:
        return msgpack.packb(response.model_dump(), use_bin_type=True)

    def start(self, port: int):
        if self.context or self.socket:
            raise ValueError("ZmqTransport already started")

        print("Starting ZmqTransport...")
        self.context = zmq.Context()
        self.socket = self.context.socket(zmq.REP)
        self.socket.bind(f"tcp://*:{port}")

        print(f"ZmqTransport started on port {port}")
        while self.context:
            message = self.socket.recv()
            response = self._handle(message)
            self.socket.send(response)

    def stop(self):
        print("Stopping ZmqTransport...")

        if self.socket:
            self.socket.close()
            self.socket = None
        if self.context:
            self.context.term()
            self.context = None

        print("ZmqTransport stopped")
