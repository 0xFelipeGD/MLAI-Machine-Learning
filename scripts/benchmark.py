#!/usr/bin/env python3
"""
scripts/benchmark.py — Measure AGRO inference latency on the Pi.

Loads the AGRO pipeline and runs N iterations on a synthetic frame, then
reports avg / p50 / p95 / p99 latency and equivalent FPS.

    python scripts/benchmark.py --iterations 100
"""

from __future__ import annotations

import argparse
import logging
import statistics
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np  # noqa: E402


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--iterations", type=int, default=100)
    p.add_argument("--warmup", type=int, default=5)
    return p.parse_args()


def percentile(values: list[float], q: float) -> float:
    if not values:
        return 0.0
    s = sorted(values)
    k = (len(s) - 1) * q
    f = int(k)
    c = min(f + 1, len(s) - 1)
    if f == c:
        return s[f]
    return s[f] + (s[c] - s[f]) * (k - f)


def bench_pipeline(name: str, pipeline, frame: np.ndarray, iterations: int, warmup: int) -> dict:
    print(f"\n[{name}] warming up ({warmup} runs)...")
    for _ in range(warmup):
        pipeline.process(frame)
    print(f"[{name}] running {iterations} iterations...")
    times: list[float] = []
    for _ in range(iterations):
        t0 = time.perf_counter()
        pipeline.process(frame)
        times.append((time.perf_counter() - t0) * 1000.0)
    return {
        "avg_ms": round(statistics.mean(times), 2),
        "p50_ms": round(percentile(times, 0.50), 2),
        "p95_ms": round(percentile(times, 0.95), 2),
        "p99_ms": round(percentile(times, 0.99), 2),
        "min_ms": round(min(times), 2),
        "max_ms": round(max(times), 2),
        "fps": round(1000.0 / max(0.01, statistics.mean(times)), 2),
    }


def main() -> int:
    logging.basicConfig(level=logging.WARNING)
    args = parse_args()

    rng = np.random.default_rng(42)
    frame = rng.integers(0, 255, size=(480, 640, 3), dtype=np.uint8)

    from engine.agro.pipeline import AgroPipeline
    ap = AgroPipeline()
    ap.save_frames = False
    ap.save_annotated = False
    stats = bench_pipeline("AGRO", ap, frame, args.iterations, args.warmup)
    print(f"  {stats}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
