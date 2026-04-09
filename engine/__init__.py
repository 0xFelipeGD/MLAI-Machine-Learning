"""MLAI inference engine — camera, preprocessing, calibration, measurement, and module pipelines."""

from pathlib import Path

# Project root resolved from this file's location.
# engine/__init__.py lives at <root>/engine/__init__.py
PROJECT_ROOT = Path(__file__).resolve().parent.parent

__version__ = "1.0.0"
