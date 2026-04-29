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


def _make_service_with_source(source: str) -> "CameraService":
    """Helper: build a CameraService with a forced source, bypassing config."""
    from engine.camera import CameraService

    cam = CameraService.__new__(CameraService)
    cam.width = 640
    cam.height = 480
    cam.target_fps = 5
    cam.source = source
    cam._ccm = None
    cam._tuning_file = None
    cam._picam_controls = {}
    cam._cap = None
    cam._picam = None
    cam._latest = None
    import threading

    cam._lock = threading.Lock()
    cam._stop = threading.Event()
    cam._thread = None
    cam._frame_times = []
    cam._fps = 0.0
    cam._stream_fail_count = 0
    return cam


def test_open_backend_stream_uses_ffmpeg():
    cam = _make_service_with_source("http://10.107.97.1:8080/video")
    fake_cap = MagicMock()
    fake_cap.isOpened.return_value = True
    with patch("engine.camera.cv2.VideoCapture", return_value=fake_cap) as vc:
        cam._open_backend()
    vc.assert_called_once_with("http://10.107.97.1:8080/video", pytest.importorskip("cv2").CAP_FFMPEG)
    assert cam._cap is fake_cap
    assert cam._picam is None


def test_open_backend_opencv_local_unchanged():
    cam = _make_service_with_source("opencv")
    fake_cap = MagicMock()
    fake_cap.isOpened.return_value = True
    with patch("engine.camera.cv2.VideoCapture", return_value=fake_cap) as vc:
        cam._open_backend()
    vc.assert_called_once_with(0)
    assert cam._cap is fake_cap


def test_open_backend_stream_raises_when_open_fails():
    cam = _make_service_with_source("http://10.107.97.1:8080/video")
    fake_cap = MagicMock()
    fake_cap.isOpened.return_value = False
    with patch("engine.camera.cv2.VideoCapture", return_value=fake_cap):
        with pytest.raises(RuntimeError, match="stream"):
            cam._open_backend()


def test_redact_url_strips_credentials():
    from engine.camera import _redact_url

    assert _redact_url("rtsp://user:pass@10.107.97.1:554/stream") == "rtsp://***@10.107.97.1:554/stream"


def test_redact_url_passthrough_when_no_credentials():
    from engine.camera import _redact_url

    assert _redact_url("http://10.107.97.1:8080/video") == "http://10.107.97.1:8080/video"


def test_open_backend_stream_error_does_not_leak_credentials():
    cam = _make_service_with_source("rtsp://secretuser:secretpass@10.107.97.1:554/stream")
    fake_cap = MagicMock()
    fake_cap.isOpened.return_value = False
    with patch("engine.camera.cv2.VideoCapture", return_value=fake_cap):
        with pytest.raises(RuntimeError) as exc_info:
            cam._open_backend()
    err = str(exc_info.value)
    assert "secretpass" not in err
    assert "secretuser" not in err
    assert "10.107.97.1" in err  # host should still be present for debugging
