from __future__ import annotations

import logging
from pathlib import Path


def setup_logging(level: str = "INFO", log_file: Path | str = Path("logs/recon_risk.log")) -> None:
    root = logging.getLogger()
    if root.handlers:
        return

    lvl = getattr(logging, level.upper(), logging.INFO)
    Path(log_file).parent.mkdir(parents=True, exist_ok=True)

    formatter = logging.Formatter(
        fmt="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    stream_handler = logging.StreamHandler()
    stream_handler.setLevel(lvl)
    stream_handler.setFormatter(formatter)

    file_handler = logging.FileHandler(log_file, encoding="utf-8")
    file_handler.setLevel(lvl)
    file_handler.setFormatter(formatter)

    root.setLevel(lvl)
    root.addHandler(stream_handler)
    root.addHandler(file_handler)


def get_logger(name: str) -> logging.Logger:
    return logging.getLogger(name)

