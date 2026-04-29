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
    cam._is_stream = False  # _open_backend overrides for stream backend
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


def test_stream_reconnect_after_threshold_failures():
    """30 consecutive failed reads on a stream backend trigger release+reopen."""
    from engine.camera import STREAM_RECONNECT_THRESHOLD

    cam = _make_service_with_source("http://10.107.97.1:8080/video")

    fake_cap_first = MagicMock()
    fake_cap_first.isOpened.return_value = True
    fake_cap_first.read.return_value = (False, None)

    fake_cap_second = MagicMock()
    fake_cap_second.isOpened.return_value = True
    fake_cap_second.read.return_value = (False, None)

    with patch(
        "engine.camera.cv2.VideoCapture",
        side_effect=[fake_cap_first, fake_cap_second],
    ) as vc:
        cam._open_backend()
        # Hit the threshold exactly: STREAM_RECONNECT_THRESHOLD failures
        # should not trigger a reconnect; the (N+1)th does.
        for _ in range(STREAM_RECONNECT_THRESHOLD):
            cam._grab_one()
        assert vc.call_count == 1, "should not reconnect at exactly threshold"
        cam._grab_one()
        assert vc.call_count == 2, "should reconnect after exceeding threshold"

    fake_cap_first.release.assert_called_once()


def test_stream_reconnect_counter_resets_on_success():
    """A successful read clears the failure counter."""
    import numpy as np
    from engine.camera import STREAM_RECONNECT_THRESHOLD

    cam = _make_service_with_source("http://10.107.97.1:8080/video")

    fake_cap = MagicMock()
    fake_cap.isOpened.return_value = True
    good_frame = np.zeros((480, 640, 3), dtype=np.uint8)
    # Pattern: 25 fails, 1 success, 25 fails — should NOT reconnect because
    # the success in the middle resets the counter.
    reads = [(False, None)] * 25 + [(True, good_frame)] + [(False, None)] * 25
    fake_cap.read.side_effect = reads

    with patch("engine.camera.cv2.VideoCapture", return_value=fake_cap) as vc:
        cam._open_backend()
        for _ in range(len(reads)):
            cam._grab_one()
        assert vc.call_count == 1, "counter must reset on success → no reconnect"


def test_stream_fail_counter_does_not_apply_to_picamera2():
    """The reconnect path is stream-only; opencv-local must not get rebuilt."""
    from engine.camera import STREAM_RECONNECT_THRESHOLD

    cam = _make_service_with_source("opencv")

    fake_cap = MagicMock()
    fake_cap.isOpened.return_value = True
    fake_cap.read.return_value = (False, None)

    with patch("engine.camera.cv2.VideoCapture", return_value=fake_cap) as vc:
        cam._open_backend()
        for _ in range(STREAM_RECONNECT_THRESHOLD + 5):
            cam._grab_one()
        assert vc.call_count == 1, "opencv-local must not auto-reconnect"
