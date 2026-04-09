#!/usr/bin/env python3
"""
scratch_test.py — quick smoke check that the three MLAI .tflite files load
and produce output on a dummy input.

Run from anywhere; the script locates models/ relative to its own location
(the project root, two levels up). Missing models are reported but don't
crash the run, so you can use this script incrementally as you produce
each model:

    python tests/scratch_test.py     # from project root
    python scratch_test.py           # from tests/
    python /abs/path/scratch_test.py # from anywhere
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import tensorflow as tf

# tests/scratch_test.py is one level inside the project root.
PROJECT_ROOT = Path(__file__).resolve().parent.parent

MODELS = [
    "models/indust/padim_toothbrush.tflite",
    "models/agro/fruit_detector.tflite",
    "models/agro/fruit_quality.tflite",
]


def check(rel_path: str) -> str:
    """Return 'ok' / 'skip' / 'fail' for one model file."""
    full_path = PROJECT_ROOT / rel_path
    if not full_path.exists():
        print(f"  [skip] {rel_path}  (file not found)")
        return "skip"
    try:
        interp = tf.lite.Interpreter(model_path=str(full_path))
        interp.allocate_tensors()
        inp = interp.get_input_details()[0]
        outs = interp.get_output_details()
        fake = np.zeros(inp["shape"], dtype=inp["dtype"])
        interp.set_tensor(inp["index"], fake)
        interp.invoke()
        # Convert numpy int wrappers to plain Python ints for a clean repr.
        shape_pretty = [int(d) for d in inp["shape"]]
        print(
            f"  [ok]   {rel_path}  "
            f"input={shape_pretty} {inp['dtype'].__name__}  "
            f"outputs={len(outs)}"
        )
        return "ok"
    except Exception as e:
        print(f"  [FAIL] {rel_path}  {type(e).__name__}: {e}")
        return "fail"


def main() -> int:
    print(f"Project root: {PROJECT_ROOT}")
    results = [check(p) for p in MODELS]
    n_ok = results.count("ok")
    n_skip = results.count("skip")
    n_fail = results.count("fail")
    print()
    print(f"Summary: {n_ok} ok, {n_skip} skipped, {n_fail} failed")
    if n_skip:
        print("(skipped files don't exist yet — see training/README.md §3 and §4)")
    return 1 if n_fail else 0


if __name__ == "__main__":
    sys.exit(main())
