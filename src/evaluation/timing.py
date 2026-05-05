"""Timing and memory helpers."""

from __future__ import annotations

import time
from contextlib import contextmanager
from dataclasses import dataclass
from typing import Iterator


@dataclass
class Timer:
    elapsed_s: float = 0.0


@contextmanager
def measure_time() -> Iterator[Timer]:
    timer = Timer()
    start = time.perf_counter()
    try:
        yield timer
    finally:
        timer.elapsed_s = time.perf_counter() - start


def get_peak_gpu_mb(reset: bool = False) -> float:
    """Return peak CUDA memory in MB, or 0 when torch/CUDA is unavailable."""
    try:
        import torch

        if not torch.cuda.is_available():
            return 0.0
        peak = torch.cuda.max_memory_allocated() / (1024 * 1024)
        if reset:
            torch.cuda.reset_peak_memory_stats()
        return float(peak)
    except Exception:
        return 0.0

