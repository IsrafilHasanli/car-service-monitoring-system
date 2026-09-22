from __future__ import annotations

import logging
import os

from app.config import settings
from app.vision.pipeline import ANPRPipeline

os.environ["ORT_DISABLE_TENSORRT"] = "1"


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
    ANPRPipeline(settings).run_forever()


if __name__ == "__main__":
    main()
