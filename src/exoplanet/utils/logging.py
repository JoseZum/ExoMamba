"""Logger de texto + writer de TensorBoard."""

from __future__ import annotations

import logging
import sys
from pathlib import Path
from typing import Any


def setup_logger(name: str, log_file: Path | None = None, level: int = logging.INFO) -> logging.Logger:
    """Crea un logger que escribe a stdout y opcionalmente a un archivo."""
    logger = logging.getLogger(name)
    logger.setLevel(level)
    logger.handlers.clear()

    fmt = logging.Formatter("%(asctime)s | %(levelname)s | %(message)s", datefmt="%H:%M:%S")

    sh = logging.StreamHandler(sys.stdout)
    sh.setFormatter(fmt)
    logger.addHandler(sh)

    if log_file is not None:
        fh = logging.FileHandler(log_file, mode="w", encoding="utf-8")
        fh.setFormatter(fmt)
        logger.addHandler(fh)

    logger.propagate = False
    return logger


class TensorBoardWriter:
    """Wrapper liviano sobre SummaryWriter para que sea trivial de mockear en tests."""

    def __init__(self, log_dir: Path | str | None) -> None:
        self.enabled = log_dir is not None
        self._writer: Any = None
        if self.enabled:
            from torch.utils.tensorboard import SummaryWriter

            self._writer = SummaryWriter(log_dir=str(log_dir))

    def add_scalar(self, tag: str, value: float, step: int) -> None:
        if self.enabled:
            self._writer.add_scalar(tag, value, step)

    def add_scalars(self, prefix: str, values: dict[str, float], step: int) -> None:
        for k, v in values.items():
            self.add_scalar(f"{prefix}/{k}", v, step)

    def close(self) -> None:
        if self.enabled:
            self._writer.close()
