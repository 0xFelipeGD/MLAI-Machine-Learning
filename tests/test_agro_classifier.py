"""tests/test_agro_classifier.py — QualityClassifier loads class order from labels.txt."""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))


def test_classifier_reads_labels_file_over_fallback(tmp_path):
    """labels.txt next to the .tflite is authoritative over the fallback list.

    This guards against the bug where config/agro/config.yaml listed
    [good, defective, unripe] but the trained model only outputs
    [defective, good] (alphabetical order from Keras).
    """
    from engine.agro.classifier import QualityClassifier

    # Fake .tflite path (file does not exist → classifier stays in mock mode,
    # but the labels file loading happens before the model-exists check).
    model_path = tmp_path / "fruit_quality.tflite"
    labels_path = tmp_path / "fruit_quality.labels.txt"
    labels_path.write_text("defective\ngood\n")

    clf = QualityClassifier()
    # Caller passes a wrong/stale list — labels.txt must win.
    clf.load(model_path, input_size=(224, 224), classes=["good", "defective", "unripe"])

    assert clf.classes == ["defective", "good"]


def test_classifier_falls_back_when_labels_missing(tmp_path):
    """When no labels.txt sits next to the .tflite, the classes arg is used."""
    from engine.agro.classifier import QualityClassifier

    model_path = tmp_path / "fruit_quality.tflite"  # no .labels.txt alongside

    clf = QualityClassifier()
    clf.load(model_path, input_size=(224, 224), classes=["good", "defective"])

    assert clf.classes == ["good", "defective"]
