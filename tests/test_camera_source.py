"""tests/test_camera_source.py — CameraService backend resolution + stream behaviour."""
from __future__ import annotations

import sys
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))


def test_resolve_backend_http_url():
    from engine.camera import _resolve_backend

    assert _resolve_backend("http://10.107.97.1:8080/video", has_picamera2=False) == "stream"


def test_resolve_backend_https_url():
    from engine.camera import _resolve_backend

    assert _resolve_backend("https://example.com/cam.mjpg", has_picamera2=True) == "stream"


def test_resolve_backend_rtsp_url():
    from engine.camera import _resolve_backend

    assert _resolve_backend("rtsp://10.107.97.1:554/stream", has_picamera2=True) == "stream"


def test_resolve_backend_picamera2_explicit():
    from engine.camera import _resolve_backend

    assert _resolve_backend("picamera2", has_picamera2=True) == "picamera2"


def test_resolve_backend_opencv_explicit():
    from engine.camera import _resolve_backend

    assert _resolve_backend("opencv", has_picamera2=True) == "opencv"


def test_resolve_backend_auto_with_picamera2_available():
    from engine.camera import _resolve_backend

    assert _resolve_backend("auto", has_picamera2=True) == "picamera2"


def test_resolve_backend_auto_without_picamera2():
    from engine.camera import _resolve_backend

    assert _resolve_backend("auto", has_picamera2=False) == "opencv"


def test_resolve_backend_rejects_unknown_string():
    from engine.camera import _resolve_backend

    with pytest.raises(ValueError, match="camera.source"):
        _resolve_backend("picam", has_picamera2=True)


def test_resolve_backend_rejects_empty_string():
    from engine.camera import _resolve_backend

    with pytest.raises(ValueError, match="camera.source"):
        _resolve_backend("", has_picamera2=True)


def test_resolve_backend_rejects_none():
    from engine.camera import _resolve_backend

    with pytest.raises(ValueError, match="camera.source"):
        _resolve_backend(None, has_picamera2=True)  # type: ignore[arg-type]
