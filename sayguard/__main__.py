"""python -m sayguard serve [--config configs/site.yaml | configs/warehouse.yaml] [--port 8000]"""
from __future__ import annotations

import argparse
import logging


def main() -> None:
    ap = argparse.ArgumentParser(prog="sayguard")
    sub = ap.add_subparsers(dest="cmd", required=True)
    s = sub.add_parser("serve", help="run the multi-camera pipeline and dashboard")
    s.add_argument("--config", default=None)
    s.add_argument("--host", default="127.0.0.1")
    s.add_argument("--port", type=int, default=8000)
    a = ap.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")

    if a.cmd == "serve":
        import uvicorn

        from .config import load_config
        from .pipeline import make_pipeline
        from .server import create_app

        cfg = load_config(a.config)
        app = create_app(make_pipeline(cfg))
        # Open MJPEG tiles never finish on their own; without a timeout the first Ctrl+C waits
        # forever and the second one force-quits past the shutdown handler (pipe.stop()).
        uvicorn.run(app, host=a.host, port=a.port, log_level="warning", timeout_graceful_shutdown=3)


if __name__ == "__main__":
    main()
