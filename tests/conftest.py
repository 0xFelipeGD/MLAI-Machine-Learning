"""tests/conftest.py — pytest configuration for the MLAI tests folder.

The smoke-test script `scratch_test.py` is a standalone helper, not a
pytest test module — but its filename happens to match pytest's default
`*_test.py` collection pattern. Tell pytest to skip it during collection
so `python3 -m pytest tests/` doesn't try to import it.

MLAI_NO_ENGINE=1 tells api/main.py's lifespan to skip starting the real
inference engine (which would try to open a camera and load .tflite
models). Tests that need STATE populated do it directly in-process.
"""
import os

os.environ.setdefault("MLAI_NO_ENGINE", "1")

collect_ignore = ["scratch_test.py"]
