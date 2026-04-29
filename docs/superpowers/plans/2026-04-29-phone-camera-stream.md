# Phone Camera Stream Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Let MLAI consume the Samsung S24 Ultra rear camera as its capture source by treating any URL string in `camera.source` as an MJPEG/RTSP stream over Wi-Fi, while leaving picamera2 and OpenCV-local backends untouched.

**Architecture:** Extend `engine/camera.py` with a third `stream` backend in `_open_backend()` that opens `cv2.VideoCapture(url, cv2.CAP_FFMPEG)`. Add a reconnect counter inside the existing `_loop` that fires only on the stream backend. Update `config/system_config.yaml`, `README.md`, and add a unit test file mocking `cv2.VideoCapture` to cover backend resolution and reconnect behaviour. No changes to API, WebSocket, inference engine, or frontend.

**Tech Stack:** Python 3.11, OpenCV 4.10 (with FFmpeg backend), pytest, IP Webcam Android app on the phone.

**Spec:** `docs/superpowers/specs/2026-04-29-phone-camera-stream-design.md`

**Concrete config target (this user's setup):** Pi seen by phone hotspot at `10.107.97.252`, so phone (gateway) is at `10.107.97.1`. Config will be `source: "http://10.107.97.1:8080/video"`. Other users substitute their own gateway IP, README documents how.

---

## File Structure

| File | Action | Responsibility |
|------|--------|----------------|
| `engine/camera.py` | Modify | Add URL detection, `stream` backend, reconnect logic |
| `config/system_config.yaml` | Modify | Document URL form for `camera.source` |
| `README.md` | Modify | §1 cross-reference, new sub-section, troubleshooting rows |
| `tests/test_camera_source.py` | Create | Unit tests for backend resolution, stream open args, reconnect |

No new module is introduced — `CameraService` already abstracts the backend, and the change is contained.

---

## Task 1: Backend resolution helper + URL detection

**Files:**
- Modify: `engine/camera.py` (add helper near top of file, around line 50, before `_load_camera_config`)
- Create: `tests/test_camera_source.py`

**Context:** The current `_open_backend` (lines 140-190) only knows three values for `self.source`: `"auto"`, `"picamera2"`, `"opencv"`. We need to recognise URLs (`http://`, `https://`, `rtsp://`) and return a new backend label `"stream"`. Pulling the resolution into a pure helper makes it trivially unit-testable.

- [ ] **Step 1: Write the failing test**

Create `tests/test_camera_source.py`:

```python
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
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_camera_source.py -v`

Expected: FAIL with `ImportError` on `_resolve_backend` (the function doesn't exist yet).

- [ ] **Step 3: Implement `_resolve_backend` helper in `engine/camera.py`**

Add this new module-level function below the existing `_load_camera_config` function (around line 60, just above `class CameraService`):

```python
import re

_URL_RE = re.compile(r"^(https?|rtsp)://", re.IGNORECASE)


def _resolve_backend(source: str, has_picamera2: bool) -> str:
    """Map a config `source` value to a concrete backend kind.

    Returns one of: "stream" | "picamera2" | "opencv".

    A URL (http://, https://, rtsp://) resolves to "stream".
    "auto" picks picamera2 if available, otherwise opencv.
    Any other string is taken at face value (must be "picamera2" or "opencv").
    """
    if isinstance(source, str) and _URL_RE.match(source):
        return "stream"
    if source == "auto":
        return "picamera2" if has_picamera2 else "opencv"
    return source
```

The `import re` should be added at the top of the file alongside the other stdlib imports (currently lines 21-27); don't duplicate it.

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/test_camera_source.py -v`

Expected: 7 passed.

- [ ] **Step 5: Commit**

```bash
git add engine/camera.py tests/test_camera_source.py
git commit -m "feat(camera): add _resolve_backend helper + URL detection

Recognises http://, https://, rtsp:// URLs in camera.source and
returns 'stream' as the backend kind. picamera2/opencv/auto paths
unchanged. Pure function, fully unit-tested via tests/test_camera_source.py."
```

---

## Task 2: Stream backend in `_open_backend`

**Files:**
- Modify: `engine/camera.py:140-190` (`_open_backend` method)
- Modify: `tests/test_camera_source.py`

**Context:** Once the helper resolves the backend kind, `_open_backend` needs an explicit `stream` branch that opens `cv2.VideoCapture(url, cv2.CAP_FFMPEG)`. The existing `opencv` branch (lines 184-190) keeps using `VideoCapture(0)` for local `/dev/video0` testing. They share the same `self._cap` attribute and the same downstream grab path.

- [ ] **Step 1: Write the failing test**

Append to `tests/test_camera_source.py`:

```python
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
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_camera_source.py -v`

Expected: 7 passed (Task 1 tests still pass), 3 failed (`test_open_backend_stream_uses_ffmpeg`, `test_open_backend_opencv_local_unchanged`, `test_open_backend_stream_raises_when_open_fails`) — current `_open_backend` doesn't branch on the stream kind.

- [ ] **Step 3: Modify `_open_backend` to add the stream branch**

Replace the body of `_open_backend` (currently `engine/camera.py:140-190`) with:

```python
    def _open_backend(self) -> None:
        backend = _resolve_backend(self.source, _HAS_PICAMERA2)

        if backend == "picamera2" and _HAS_PICAMERA2:
            picam_kwargs = {}
            if self._tuning_file:
                # Try to load a custom tuning file. Most common use:
                # tuning_file: "imx708.json" on a NoIR sensor to force
                # regular-RGB tuning (fixes the daytime blue cast).
                try:
                    tuning = Picamera2.load_tuning_file(self._tuning_file)  # type: ignore[attr-defined]
                    picam_kwargs["tuning"] = tuning
                    logger.info("Loaded custom tuning: %s", self._tuning_file)
                except Exception:
                    logger.exception(
                        "Failed to load tuning_file '%s'; falling back to default tuning",
                        self._tuning_file,
                    )
            self._picam = Picamera2(**picam_kwargs)
            cfg_kwargs = {
                "main": {"size": (self.width, self.height), "format": "RGB888"},
            }
            if self._picam_controls:
                cfg_kwargs["controls"] = dict(self._picam_controls)
            video_cfg = self._picam.create_video_configuration(**cfg_kwargs)
            self._picam.configure(video_cfg)
            self._picam.start()
            if self._picam_controls:
                try:
                    self._picam.set_controls(self._picam_controls)
                except Exception:
                    logger.exception("Failed to apply picamera2 controls: %s", self._picam_controls)
            time.sleep(0.5)  # warm-up
            return

        if backend == "stream":
            # Network camera (phone, IP webcam, RTSP). The OpenCV grab path
            # is identical to the local /dev/video0 case below — only the
            # VideoCapture constructor argument changes. CAP_FFMPEG makes
            # the demuxer choice deterministic across machines that have
            # both GStreamer and FFmpeg backends compiled in.
            self._cap = cv2.VideoCapture(self.source, cv2.CAP_FFMPEG)
            if not self._cap.isOpened():
                raise RuntimeError(f"OpenCV could not open stream: {self.source}")
            logger.info("CameraService stream backend opened: %s", self.source)
            return

        # OpenCV local fallback (PC dev or Pi without picamera2)
        self._cap = cv2.VideoCapture(0)
        self._cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.width)
        self._cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.height)
        self._cap.set(cv2.CAP_PROP_FPS, self.target_fps)
        if not self._cap.isOpened():
            raise RuntimeError("OpenCV could not open /dev/video0")
```

- [ ] **Step 4: Run all camera-source tests**

Run: `pytest tests/test_camera_source.py -v`

Expected: 10 passed.

- [ ] **Step 5: Run the full test suite to confirm no regression**

Run: `pytest tests/ -v --ignore=tests/scratch_test.py`

Expected: all previously-passing tests still pass.

- [ ] **Step 6: Commit**

```bash
git add engine/camera.py tests/test_camera_source.py
git commit -m "feat(camera): stream backend for HTTP/RTSP sources

When camera.source is a URL, open it via cv2.VideoCapture(url,
cv2.CAP_FFMPEG). Local /dev/video0 fallback and picamera2 paths
unchanged. Tests cover both new branches and existing one."
```

---

## Task 3: Reconnect logic for stream backend

**Files:**
- Modify: `engine/camera.py` (`__init__` and `_loop`)
- Modify: `tests/test_camera_source.py`

**Context:** When a stream goes silent (Wi-Fi blip, app backgrounded), `cv2.VideoCapture.read()` returns `(False, None)` repeatedly without raising. The existing `_loop` handles **exceptions** but doesn't react to repeated false reads. We add a counter that triggers a reconnect once it crosses a threshold — only on the stream backend.

- [ ] **Step 1: Write the failing test**

Append to `tests/test_camera_source.py`:

```python
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
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_camera_source.py -v -k reconnect`

Expected: FAIL with `ImportError` on `STREAM_RECONNECT_THRESHOLD` plus `AssertionError`s on the call_count expectations — the reconnect logic doesn't exist yet.

- [ ] **Step 3: Add the threshold constant + counter init**

In `engine/camera.py`, add the module-level constant just below the imports (around line 35, near `logger = ...`):

```python
# Number of consecutive empty reads on a stream source before we tear down
# the VideoCapture and reopen it. At fps=5 this is ~6 s of dead air, which
# is long enough to ride out a typical Wi-Fi hiccup but short enough that
# the user notices a stale feed quickly.
STREAM_RECONNECT_THRESHOLD = 30
```

In `CameraService.__init__` (around line 127, after `self._fps: float = 0.0`), add:

```python
        # Stream-backend reconnect: count consecutive failed reads.
        self._stream_fail_count: int = 0
```

- [ ] **Step 4: Add the reconnect helper and wire it into `_grab_one`**

Add a private method below `_grab_one` (i.e. after the current line ~231):

```python
    def _reconnect_stream(self) -> None:
        """Tear down the current stream VideoCapture and reopen it."""
        logger.warning(
            "stream lost (>%d consecutive empty reads), reconnecting to %s",
            STREAM_RECONNECT_THRESHOLD,
            self.source,
        )
        if self._cap is not None:
            try:
                self._cap.release()
            except Exception:
                logger.exception("VideoCapture.release() raised during reconnect")
        self._cap = cv2.VideoCapture(self.source, cv2.CAP_FFMPEG)
        if not self._cap.isOpened():
            logger.error("Reconnect failed; will retry on next failure batch")
        self._stream_fail_count = 0
```

Now modify `_grab_one` (currently `engine/camera.py:214-231`) to track failures on the stream backend. Replace its body with:

```python
    def _grab_one(self) -> Optional[np.ndarray]:
        if self._picam is not None:
            # picamera2 returns RGB; convert to BGR for OpenCV consistency.
            rgb = self._picam.capture_array()
            if self._ccm is not None:
                rgb = self._apply_ccm(rgb)
            return cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
        if self._cap is not None:
            ok, frame = self._cap.read()
            is_stream = _resolve_backend(self.source, _HAS_PICAMERA2) == "stream"
            if not ok or frame is None:
                if is_stream:
                    self._stream_fail_count += 1
                    if self._stream_fail_count > STREAM_RECONNECT_THRESHOLD:
                        self._reconnect_stream()
                return None
            if is_stream:
                self._stream_fail_count = 0
            if self._ccm is not None:
                # OpenCV gives BGR; CCM is RGB-shaped, so we reorder.
                rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                rgb = self._apply_ccm(rgb)
                frame = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
            return frame
        return None
```

- [ ] **Step 5: Run reconnect tests to verify they pass**

Run: `pytest tests/test_camera_source.py -v -k reconnect`

Expected: 3 passed.

- [ ] **Step 6: Run the full camera-source test suite**

Run: `pytest tests/test_camera_source.py -v`

Expected: 13 passed.

- [ ] **Step 7: Commit**

```bash
git add engine/camera.py tests/test_camera_source.py
git commit -m "feat(camera): auto-reconnect on stream backend failure

Track consecutive failed reads on stream sources; once
STREAM_RECONNECT_THRESHOLD (30) is exceeded, release the
VideoCapture and reopen with the configured URL. Only applies to
stream backend — picamera2 and local opencv paths untouched. Tests
cover threshold-exact, counter reset on success, and isolation
from non-stream backends."
```

---

## Task 4: Update `config/system_config.yaml`

**Files:**
- Modify: `config/system_config.yaml:32-33` (the `source:` line and its preceding comment)

**Context:** The YAML comment currently says `auto | picamera2 | opencv`. Widen it to mention URL form, and leave the active value as `auto` so existing Pi installs continue to behave exactly as before. The user's actual switch happens in their local copy after installation.

- [ ] **Step 1: Edit `config/system_config.yaml`**

Replace lines 32-33 (the comment + `source:` line) with:

```yaml
  # Capture source. Accepts:
  #   auto       — picamera2 if available, else OpenCV /dev/video0
  #   picamera2  — Raspberry Pi camera via libcamera/picamera2
  #   opencv     — local USB/CSI camera via OpenCV /dev/video0
  #   <URL>      — HTTP MJPEG or RTSP stream from a phone/IP camera.
  #                Example: "http://10.107.97.1:8080/video" (IP Webcam app
  #                on a Samsung phone acting as Wi-Fi hotspot, where the Pi
  #                connects through the phone — see README §9 for setup).
  #                The picamera2-only fields below (tuning_file, awb_mode,
  #                colour_gains) are silently ignored on stream sources.
  #                The post-capture color_matrix still applies if non-identity.
  source: auto
```

- [ ] **Step 2: Sanity-check YAML still parses**

Run: `python3 -c "import yaml; print(yaml.safe_load(open('config/system_config.yaml'))['camera']['source'])"`

Expected: `auto`

- [ ] **Step 3: Commit**

```bash
git add config/system_config.yaml
git commit -m "docs(config): document URL form for camera.source

camera.source now accepts http://, https://, rtsp:// URLs in
addition to the auto/picamera2/opencv enum. Default value
unchanged."
```

---

## Task 5: README — §1 cross-reference + new §9 section

**Files:**
- Modify: `README.md` (Table of Contents around line 13-37, §1 around line 105-117, insert new §9 between current §8 at line 243 and §Tech Stack at line 260)

**Context:** Add a single sentence in §1 telling the user the phone alternative exists, then add a fully self-contained §9 explaining the setup. Renumber nothing else — current §1-§8 numbering is stable; the new section becomes §9 sitting between "Camera calibration" (§8) and "Tech Stack" (next top-level heading). Update the Table of Contents accordingly.

- [ ] **Step 1: Add Table of Contents entry**

In the TOC inside `## Full Setup Guide` (currently lines 18-26 of README.md), add a new line after `- [8. Camera calibration (optional)](#8-camera-calibration-optional)`:

Replace:

```markdown
  - [7. Train your own models (optional)](#7-train-your-own-models-optional)
  - [8. Camera calibration (optional)](#8-camera-calibration-optional)
- [Tech Stack](#tech-stack)
```

With:

```markdown
  - [7. Train your own models (optional)](#7-train-your-own-models-optional)
  - [8. Camera calibration (optional)](#8-camera-calibration-optional)
  - [9. Use an Android phone as the camera (alternative)](#9-use-an-android-phone-as-the-camera-alternative)
- [Tech Stack](#tech-stack)
```

- [ ] **Step 2: Add cross-reference at the end of §1 hardware list**

Find this passage in `### 1. What you'll need` (around lines 110-115):

```markdown
- Raspberry Pi 4 Model B with **8 GB RAM** (4 GB works but the dashboard is slower)
- Official 27 W USB-C power supply
- microSD card, **32 GB+**, Class 10 / A2
- Raspberry Pi Camera Module 3 + ribbon cable
- Ethernet or Wi-Fi
- Optional: case + fan, light source, printed checkerboard for calibration
```

Replace the camera bullet to read:

```markdown
- Raspberry Pi Camera Module 3 + ribbon cable — *or* a recent Android phone running an MJPEG webcam app (see [§9](#9-use-an-android-phone-as-the-camera-alternative))
```

- [ ] **Step 3: Insert the new §9 section**

In `README.md`, find the line `## Tech Stack` (currently line 260). Immediately **before** it, insert:

````markdown
### 9. Use an Android phone as the camera (alternative)

Instead of the Pi Camera Module, you can stream from an Android phone over Wi-Fi. This is useful when you want better colour fidelity than the NoIR sensor or higher resolution than the Camera Module 3 provides — the phone's ISP does the heavy lifting and the Pi just consumes frames.

**One-time phone setup**

1. Install **IP Webcam** by Pavel Khlebovich from the Play Store (free).
2. Open the app. Optional but recommended: scroll to **Video preferences → Photo resolution** and pick a 720p or 1080p mode (4K MJPEG saturates Wi-Fi).
3. Scroll to the bottom and tap **Start server**.
4. The screen now shows the stream URL, e.g. `http://192.0.2.42:8080`. The Pi will connect to `<that-ip>:8080/video`.

**Connect the Pi to the phone hotspot**

If the phone is also routing the Pi's internet via hotspot, you already have the network. Otherwise, enable the hotspot on the phone and join it from the Pi (`raspi-config` → **System Options → Wireless LAN**).

**Find the phone's IP from the Pi**

The phone (as the hotspot AP) is the Pi's default gateway:

```bash
ip route | awk '/default/ {print $3}'
# example output: 10.107.97.1
```

That IP is what you put in the config below.

**Configure MLAI**

Edit `config/system_config.yaml`, replacing the value of `source`:

```yaml
camera:
  source: "http://10.107.97.1:8080/video"   # <-- substitute your gateway IP
```

Then restart the API service:

```bash
sudo systemctl restart mlai-api
```

Open the dashboard (`http://<pi-ip>:3000`) — the live page now shows the phone view.

**Notes**

- **Re-run camera calibration.** `config/camera_calibration.json` was computed for the Pi camera lens; it is invalid for the phone. Run `python3 scripts/calibrate_camera.py` again with the phone mounted in its final position. Until you do, MLAI falls back to pixel measurements.
- **Camera Tuning sliders.** AWB / red-gain / blue-gain sliders are picamera2-only and are inert when streaming from a phone — the phone's ISP owns colour. The CCM matrix sliders still apply as a post-capture filter.
- **Stream stutters.** Drop the phone-side resolution. 720p MJPEG over a hotspot is comfortably above the project's 3 FPS target.
- **Reconnect.** If the phone goes to sleep or Wi-Fi blips, MLAI auto-reconnects after ~6 s of dead air. No manual intervention.

#### Updating an existing install

If MLAI is already running on the Pi and you want to add this feature, the upgrade is backend-only — the dashboard build is untouched.

```bash
ssh felipe@mlai.local
cd ~/MLAI-Machine-Learning
git pull
sudo systemctl restart mlai-api
```

That's the whole list. Specifically, you do **not** need to:

- `pip install -r requirements.txt` — no new dependencies were added.
- `npm run build` in `web/` — frontend is unchanged.
- `sudo systemctl restart mlai-web` — only `mlai-api` was updated.
- `bash scripts/download_models.sh` — no model changes.

After the restart, edit `config/system_config.yaml` to set `source` to your phone's stream URL (see *Configure MLAI* above) — or leave it as `auto` if you're not switching cameras yet.

---

````

(The trailing `---` separator matches the rest of the README's section style.)

- [ ] **Step 4: Spot-check the rendered structure**

Run: `grep -n "^##\|^###" README.md | head -30`

Expected output now includes `### 9. Use an Android phone as the camera (alternative)` between line ~245 (the current §8) and the line of `## Tech Stack`.

- [ ] **Step 5: Commit**

```bash
git add README.md
git commit -m "docs(readme): add §9 — Android phone as camera (alternative)

Self-contained guide: install IP Webcam, find the hotspot gateway
IP, edit camera.source to the URL, restart mlai-api. Cross-linked
from §1 hardware list and added to the Table of Contents."
```

---

## Task 6: README — Troubleshooting rows

**Files:**
- Modify: `README.md` Troubleshooting table (currently around lines 343-355)

**Context:** Add three new rows to the existing troubleshooting markdown table. The table is two-column `| Symptom | Fix |`. Insert the new rows just before the "Pi reboots randomly" row so the camera-related entries stay grouped.

- [ ] **Step 1: Find the current table block**

Open `README.md`. The Troubleshooting table starts at the line `| Symptom | Fix |` (around line 345). The last row in the camera-related cluster is `| Calibration: no checkerboard found | More light, flatter board, match \`--pattern\` |`.

- [ ] **Step 2: Insert the new rows after the calibration row**

Replace this passage:

```markdown
| Calibration: no checkerboard found | More light, flatter board, match `--pattern` |
| Pi reboots randomly | Power supply too weak — use the official 27 W USB-C PSU |
```

With:

```markdown
| Calibration: no checkerboard found | More light, flatter board, match `--pattern` |
| Live feed black after switching to phone (§9) | Confirm IP Webcam is running, gateway IP matches `ip route`, port 8080 reachable: `curl -I http://<phone-ip>:8080/video` |
| Phone stream stutters / drops | Lower the phone-side resolution to 720p; check Wi-Fi signal strength on the Pi |
| AWB / red-gain / blue-gain sliders inert on phone stream | Those are picamera2 controls; only the CCM matrix slider applies on stream sources |
| Pi reboots randomly | Power supply too weak — use the official 27 W USB-C PSU |
```

- [ ] **Step 3: Verify the table renders**

Run: `grep -A 1 "AWB / red-gain" README.md`

Expected: the new row appears in context.

- [ ] **Step 4: Commit**

```bash
git add README.md
git commit -m "docs(readme): add troubleshooting rows for phone camera stream

Three new rows: black feed (network/IP issue), stutters (drop
resolution), inert AWB sliders (picamera2-only)."
```

---

## Task 7: End-to-end manual smoke test

**Files:** none (user-driven verification, executed after merge / on the Pi)

**Context:** Unit tests cover backend resolution and reconnect logic. Real video over hotspot Wi-Fi can only be validated on the device. This task captures the smoke procedure so the user (or a future engineer) can confirm the change works without re-deriving it.

- [ ] **Step 1: User installs/opens IP Webcam**

On the S24 Ultra, open IP Webcam, set resolution to 1280x720, tap **Start server**. Note the URL printed at the bottom of the app screen.

- [ ] **Step 2: User edits config on the Pi**

```bash
ssh felipe@mlai.local
cd ~/MLAI-Machine-Learning
GW=$(ip route | awk '/default/ {print $3}')
echo "Phone gateway: $GW"
# Manually set camera.source in config/system_config.yaml to "http://$GW:8080/video"
sudo systemctl restart mlai-api
```

- [ ] **Step 3: User runs the in-process probe**

```bash
python3 -m engine.camera
```

Expected: `Got frame: (720, 1280, 3) fps: <something > 0>`. If frame is `None`, check `journalctl -u mlai-api -n 50` and confirm IP Webcam is still running on the phone.

- [ ] **Step 4: User opens the dashboard and confirms ML pipeline works**

From the PC, open `http://mlai.local:3000` (or via SSH tunnel as documented in §6). Navigate to the AGRO live page. Hold a fruit in front of the phone. Confirm:

- The live feed shows what the phone sees (not the Pi camera).
- Detection bounding boxes render on detected fruits.
- WebSocket FPS counter is non-zero.

- [ ] **Step 5: User verifies reconnect**

With the dashboard open, briefly close IP Webcam on the phone (the stream goes black). Wait ~10 s. Reopen IP Webcam. The dashboard feed should resume within ~5-10 s without restarting the Pi service. Confirm in `journalctl -u mlai-api -f` that the line `stream lost (>30 consecutive empty reads), reconnecting` appears, followed by frames flowing again.

- [ ] **Step 6: User commits the local config change (separate, opcional)**

```bash
# Only if the user wants to track their phone URL in their own fork.
# Most users will leave config/system_config.yaml at default `source: auto`
# and patch it locally. Skip this step if so.
git add config/system_config.yaml
git commit -m "chore(config): use phone camera stream as default source"
```

If the smoke test fails at any step, do **not** mark the implementation complete. Investigate (logs, network, app state) before declaring done.

---

## Self-Review Checklist

Run this once after the plan is written. Fix issues inline.

- [x] **Spec coverage:** every section of the spec maps to a task.
  - §3 Approach → Task 1, 2
  - §4 Configuration schema → Task 4 (YAML comment) + Task 1, 2 (parser)
  - §5 Reconnect logic → Task 3
  - §6 Tuning fields on stream → covered by Task 2 design (no extra code, picamera2 branch is skipped naturally) + Task 6 (troubleshooting row about AWB sliders inert)
  - §7 Calibration → Task 5 (note in new §9), no code change
  - §8 Documentation → Task 5 (cross-ref + new section) + Task 6 (troubleshooting)
  - §9 Testing — unit → Tasks 1, 2, 3 (tests embedded). Manual smoke → Task 7.
  - §10 Implementation order → matches Tasks 1-7 sequence.
  - §11 Risks → Task 7 step 3 implicitly verifies FFmpeg backend works (it would fail to open the URL otherwise).

- [x] **No placeholders:** no "TBD", "implement later", "similar to Task N" without code, "add validation". Every code step shows the code.

- [x] **Type / name consistency:**
  - `_resolve_backend(source, has_picamera2) -> str` returning `"stream"|"picamera2"|"opencv"` — used the same way in Task 2 (`_open_backend`) and Task 3 (`_grab_one`).
  - `STREAM_RECONNECT_THRESHOLD` (module-level int) — referenced in tests, `_grab_one`, `_reconnect_stream`, log messages — consistent.
  - `self._stream_fail_count` — initialised in `__init__`, mutated only in `_grab_one` and `_reconnect_stream`.
  - `cv2.CAP_FFMPEG` — used in both `_open_backend` (initial) and `_reconnect_stream` (after failure). Same constant.

- [x] **Spec scope check:** plan stays inside the spec. No Docker, no API/web changes, no new modules.

---

## Done means

- 13 unit tests pass on `pytest tests/test_camera_source.py -v`.
- Pre-existing tests still pass (`pytest tests/ --ignore=tests/scratch_test.py`).
- README and config documented; commits separated by concern (logic, tests, config, docs).
- User has run Task 7 smoke test on their Pi with the S24 Ultra and confirmed live ML inference on phone-streamed frames + auto-reconnect after a deliberate stream interruption.
