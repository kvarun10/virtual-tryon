"""Console + file logging and a small JSON-lines metric log."""

from __future__ import annotations

import json
import logging
import time
from pathlib import Path
from typing import Any, Dict, Optional

_FORMAT = "%(asctime)s | %(levelname)-7s | %(name)s | %(message)s"


def get_logger(name: str = "dressup", log_file: Optional[Path] = None, level: int = logging.INFO) -> logging.Logger:
    logger = logging.getLogger(name)
    logger.setLevel(level)
    if not any(isinstance(h, logging.StreamHandler) and not isinstance(h, logging.FileHandler) for h in logger.handlers):
        stream = logging.StreamHandler()
        stream.setFormatter(logging.Formatter(_FORMAT, "%H:%M:%S"))
        logger.addHandler(stream)
    if log_file is not None:
        log_file = Path(log_file)
        log_file.parent.mkdir(parents=True, exist_ok=True)
        if not any(isinstance(h, logging.FileHandler) and Path(h.baseFilename) == log_file.resolve() for h in logger.handlers):
            fh = logging.FileHandler(log_file, encoding="utf-8")
            fh.setFormatter(logging.Formatter(_FORMAT))
            logger.addHandler(fh)
    logger.propagate = False
    return logger


class JsonlLogger:
    """Append one JSON object per line; survives Colab disconnects because each line is flushed."""

    def __init__(self, path: Path):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)

    def log(self, record: Dict[str, Any]) -> None:
        record = {"time": time.time(), **record}
        with open(self.path, "a", encoding="utf-8") as fh:
            fh.write(json.dumps(record, default=str) + "\n")

    def read(self):
        if not self.path.exists():
            return []
        with open(self.path, "r", encoding="utf-8") as fh:
            return [json.loads(line) for line in fh if line.strip()]
