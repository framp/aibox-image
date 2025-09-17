import argparse
import logging
import signal
import sys

from aibox_image_lib.transport.zmq import ZmqTransport

from selection_service.app import register_use_cases
from selection_service.service import Service


def signal_handler(sig, frame):
    logging.info("Received shutdown signal")
    sys.exit(0)


def main():
    parser = argparse.ArgumentParser(description="LLM Text Processing Service")
    parser.add_argument(
        "--port",
        type=int,
        default=5558,
        help="Port to bind the service (default: 5558)",
    )
    parser.add_argument("--debug", action="store_true", help="Enable debug logging")

    args = parser.parse_args()

    logging.basicConfig(level=logging.DEBUG if args.debug else logging.INFO)

    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)

    transport = ZmqTransport()
    service = Service()
    register_use_cases(transport, service)

    try:
        transport.start(port=args.port)
    except KeyboardInterrupt:
        logging.info("Shutdown requested")
    except Exception:
        logging.error("Service error", exc_info=True)
    finally:
        transport.stop()


if __name__ == "__main__":
    main()
