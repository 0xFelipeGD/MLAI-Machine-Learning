# Phone Camera as Capture Source — Design

**Date:** 2026-04-29
**Status:** Draft, awaiting user review
**Owner:** Felipe

## 1. Problem

The Raspberry Pi Camera Module 3 currently in use is the **NoIR** variant (no IR-cut filter). On daytime fruit-inspection scenes this produces washed-out colours and lower effective resolution than a modern phone camera. Recent commits (`cc2b953`, `b20bc4e`, `0db6606`, ...) tried to compensate with custom tuning files, manual ColourGains, and a colour-correction matrix, but the underlying sensor is still the wrong tool for the job.

The user owns a Samsung Galaxy S24 Ultra. Its rear camera is dramatically better than the Pi NoIR for visible-light produce inspection: higher resolution, accurate AWB, vastly better low-light. The phone is also already the **Wi-Fi hotspot** the Pi connects to for internet, so the Pi can reach the phone on the hotspot subnet with zero extra network setup.

**Goal:** let MLAI use the S24 Ultra's rear camera as its capture source, without changing anything on the dashboard, the inference engine, the API, or the WebSocket contract — and without removing the existing picamera2 path so the project still works on a stock Pi + Camera Module 3 setup.

## 2. Non-goals

- Replacing the picamera2 backend. It stays as the recommended path for users with a Pi camera.
- Building a custom Android app. The user explicitly asked for "an off-the-shelf app", nothing custom.
- Any image-quality enhancement on the Pi side (denoising, white-balance overrides, etc.). The phone's ISP already does this far better than we could.
- USB tethering / phone-as-UVC-webcam. Wi-Fi only.
- Authentication / TLS for the phone-to-Pi stream. Hotspot is local, single-tenant, low-stakes.
- Multi-camera fusion. One source at a time.

## 3. Approach

The user will install **IP Webcam** by Pavel Khlebovich (or any equivalent app — DroidCam, Larix Broadcaster — that exposes an MJPEG-over-HTTP or RTSP stream at a known URL). The Pi consumes that URL through OpenCV's existing `cv2.VideoCapture`, which natively supports HTTP MJPEG and RTSP via FFmpeg.

The whole change is contained in `engine/camera.py` plus a one-line config schema extension. The `CameraService` class abstracts the backend already; we add a third backend kind, `stream`, that lives next to `picamera2` and `opencv`.

```
┌──────────────────┐   Wi-Fi hotspot    ┌──────────────────┐
│ S24 Ultra        │  192.168.x.x:8080  │ Raspberry Pi 4   │
│ ┌──────────────┐ │ ─────────────────▶ │ ┌──────────────┐ │
│ │ IP Webcam    │ │   MJPEG over HTTP  │ │ CameraService│ │
│ │ (rear cam)   │ │                    │ │  backend=    │ │
│ └──────────────┘ │                    │ │   "stream"   │ │
│                  │                    │ └──────┬───────┘ │
│ (also routes     │                    │        │ BGR     │
│  Pi's internet)  │                    │        ▼ frames  │
└──────────────────┘                    │ ┌──────────────┐ │
                                        │ │ Inference,   │ │
                                        │ │ WebSocket,   │ │
                                        │ │ Dashboard    │ │
                                        │ └──────────────┘ │
                                        └──────────────────┘
```

Why this approach:

- **Minimal blast radius.** No changes to `api/`, `web/`, the inference engine, or the WebSocket protocol. Frames are still numpy BGR HxWx3, still pulled from the same `cam.read()` method.
- **Reversible.** Switching back to the Pi camera is a one-line YAML change.
- **Resilient to Wi-Fi flakiness.** The grab loop already swallows exceptions; we add a stream-specific reconnect path on top of that.
- **No phone-side custom work.** "Install IP Webcam, press start" is the entire mobile setup.

## 4. Configuration schema

`config/system_config.yaml`'s `camera.source` field is widened from an enum to an enum-or-URL string.

**Before**

```yaml
camera:
  source: auto  # auto | picamera2 | opencv
```

**After**

```yaml
camera:
  # auto | picamera2 | opencv | <stream URL>
  # When set to a URL (http://, https://, rtsp://) the Pi consumes the stream
  # through OpenCV. Tuning fields below (tuning_file, awb_mode, colour_gains,
  # color_matrix) are silently ignored for stream sources — the phone's ISP
  # already handles colour science.
  source: "http://192.168.43.1:8080/video"
```

Detection rule (in `_open_backend`):

```python
if isinstance(source, str) and re.match(r'^(https?|rtsp)://', source):
    backend = "stream"
elif source == "auto":
    backend = "picamera2" if _HAS_PICAMERA2 else "opencv"
else:
    backend = source  # "picamera2" | "opencv"
```

The `stream` backend opens `cv2.VideoCapture(url, cv2.CAP_FFMPEG)` and otherwise reuses the existing OpenCV grab path. We pass `CAP_FFMPEG` explicitly so the choice of demuxer is deterministic across machines.

`resolution` and `fps` keep their meaning for `picamera2` and `opencv` backends. For `stream` they are advisory — the actual frame size is whatever the phone app sends. README documents this.

## 5. Reconnect logic

The current `_loop` catches exceptions but doesn't deal with `VideoCapture.read()` returning `False` indefinitely (which is what happens when the stream goes dark while the socket stays open).

Added behaviour, **only for the stream backend**:

- Track consecutive failed reads in a counter `self._stream_fail_count`.
- After `STREAM_RECONNECT_THRESHOLD = 30` consecutive failures (~6 s at 5 fps), close the `VideoCapture`, log `"stream lost, reconnecting to <url>"`, and reopen.
- Reset the counter on the first successful frame.
- No exponential backoff. No max-retry cap. systemd handles a fully dead service.

The picamera2 and opencv-local paths are untouched — they have their own failure characteristics and a reconnect loop there would mask real bugs.

## 6. Tuning fields on a stream source

The existing tuning fields split into two groups based on where they apply in the pipeline:

**picamera2-only (silently no-op for stream sources):**

- `tuning_file` — loaded into `Picamera2.load_tuning_file()`. No equivalent in OpenCV.
- `awb_mode`, `colour_gains` — applied via `Picamera2.set_controls({"AwbEnable": ..., "ColourGains": ...})`. No equivalent in OpenCV.

For stream sources, `_open_backend` skips this block entirely. Live updates via `update_gains()` are guarded by `if self._picam is not None`, so they're already no-ops when `_picam` is `None` — confirmed by reading `engine/camera.py:253` and `:267`. No code change needed there.

**Post-capture, still applies:**

- `color_matrix` (CCM) is a numpy 3×3 multiply applied in `_grab_one()` for both the picamera2 and opencv branches (see `engine/camera.py:218` and `:225`). Stream frames go through the OpenCV branch, so CCM **continues to work** as a post-capture colour filter. Live updates via `update_color_matrix()` also work. Defaults to identity, which short-circuits the multiply.

**`get_controls()` shape on stream:** `awb_auto=True` (since `AwbEnable` is never set in `_picam_controls`), gains=`(1.0, 1.0)` from the fallback in `engine/camera.py:296`, and `color_matrix` reflects the actual configured CCM (or `None` if identity). The dashboard's "Camera Tuning" panel renders without errors; the AWB/gain sliders are visually inert (their `update_gains` calls hit the `_picam is None` guard) while the CCM sliders behave as before.

README's troubleshooting section gets one row: *"AWB / red-gain / blue-gain sliders do nothing on a phone stream — those are picamera2 controls. The CCM (colour matrix) still works."*

## 7. Calibration

`config/camera_calibration.json` stores intrinsics (focal length, principal point, distortion coefficients) for the **specific lens** that was calibrated. The S24 Ultra rear lens is a different optical system — the existing JSON is invalid for it.

The spec **does not** auto-invalidate the file. Instead:

- README §8 (Camera calibration) gets a callout: *"If you change cameras (Pi → phone, or vice versa) you must re-run calibration. Until then, measurements default to pixels."*
- `engine/measurement.py` already falls back to pixel measurements when calibration is missing — no change needed there. (Verified separately during implementation.)

This is the right scope: calibration is a user concern, not something the camera service should auto-detect.

## 8. Documentation

Three documentation surfaces:

1. **README.md, §1 "What you'll need":** add a parenthetical: *"or a recent Android phone running an MJPEG webcam app — see §X for the alternative path."*

2. **README.md, new §X "Use an Android phone as the camera (alternative)":** a self-contained section covering:
   - Install **IP Webcam** from the Play Store on the phone.
   - Open the app, scroll to **Start server** (default port 8080).
   - Make sure the phone's hotspot is enabled and the Pi is connected to it.
   - On the Pi, find the phone's IP: `ip route | awk '/default/ {print $3}'` (the gateway is the phone).
   - Edit `config/system_config.yaml`: set `source: "http://<phone-ip>:8080/video"`.
   - `sudo systemctl restart mlai-api`.
   - Open the dashboard — live feed shows the phone camera.
   - **Calibration:** re-run `scripts/calibrate_camera.py` with the phone mounted in its final position, or accept pixel-only measurements.
   - **Updating an existing install:** explicit sub-block listing the upgrade path for users who already have MLAI running on a Pi. The change is backend-only, so the procedure is `git pull` + `sudo systemctl restart mlai-api`. The sub-block also enumerates what is **not** needed (no `pip install`, no `npm run build`, no `mlai-web` restart, no model re-download) — these are common reflexes from larger updates and need to be explicitly ruled out so users don't waste time.

3. **README.md, Troubleshooting table:** three new rows:
   - "Live feed black after switching to phone" → check IP Webcam is running, IP matches hotspot gateway, port 8080 not blocked.
   - "Stream stutters / drops" → expected on poor Wi-Fi; lower phone-side resolution.
   - "AWB / red-gain / blue-gain sliders inert on phone stream" → those are picamera2 controls; only the CCM matrix still applies as a post-capture filter.

Spec keeps the doc text minimal — exact phrasing is decided during implementation.

## 9. Testing

### Unit (pytest)

New file `tests/test_camera_source.py`. Uses `unittest.mock` to patch `cv2.VideoCapture` and the `Picamera2` import sentinel. Tests:

- `source: "http://1.2.3.4:8080/video"` → `_open_backend` resolves backend to `stream`, calls `cv2.VideoCapture("http://1.2.3.4:8080/video", cv2.CAP_FFMPEG)`.
- `source: "rtsp://1.2.3.4:554/stream"` → same, backend is `stream`.
- `source: "picamera2"` → backend is `picamera2` (skipped on PC where `_HAS_PICAMERA2` is False — guard with a fixture).
- `source: "opencv"` → backend is `opencv`, calls `cv2.VideoCapture(0)`.
- `source: "auto"` → picks based on `_HAS_PICAMERA2` (parametrize both).
- Stream reconnect: when mocked `read()` returns `(False, None)` 30 times in a row, the test observes a `release()` + new `VideoCapture` call.
- `get_controls()` on a stream source returns the benign shape (`awb_auto=True`, gains `(1.0, 1.0)`, `color_matrix=None`) without exception.

### Manual smoke test

Documented in the spec only, not automated:

1. Phone: open IP Webcam, press Start.
2. Pi: edit YAML to the phone URL, restart `mlai-api`.
3. `python -m engine.camera` — should print a frame shape and a non-zero FPS.
4. Open dashboard at `http://<pi-ip>:3000`, confirm AGRO live page shows the phone view.
5. Place a fruit in front of the phone, confirm detection boxes render.

### What we are NOT testing

- Real network failures (mocking is enough).
- Stream image quality (subjective, user-validated).
- Phone-app-specific URL formats (we trust IP Webcam's documented `/video` path; user can override).

## 10. Implementation order

The implementation will be a single PR. Inside the PR, the natural order is:

1. `engine/camera.py`: add URL detection, the `stream` backend, and the reconnect counter.
2. `tests/test_camera_source.py`: cover the matrix above. **Run locally before moving on.**
3. `config/system_config.yaml`: update the comment on the `source` line.
4. `README.md`: add §X, the §1 cross-reference, and the three troubleshooting rows.
5. Manual smoke test on the Pi (user-driven, post-merge).

These are sequential — each step builds on the previous. No parallel-agent dispatching is warranted.

## 11. Risks and open questions

- **OpenCV FFmpeg backend availability.** `cv2.VideoCapture(url, cv2.CAP_FFMPEG)` requires the OpenCV build to be linked against FFmpeg. The system `python3-opencv` apt package on Bookworm 64-bit ships with FFmpeg support; the `opencv-python` PyPI wheels also include FFmpeg. The wizard installs OpenCV one of these ways, so this is not expected to be an issue, but the implementation must verify with `python3 -c "import cv2; print(cv2.getBuildInformation())" | grep -i ffmpeg` on the actual Pi before declaring done. If the build lacks FFmpeg, the fallback is to drop the explicit `CAP_FFMPEG` argument and let OpenCV pick its default backend (which on Linux is usually GStreamer or V4L2 — both can read MJPEG-over-HTTP for our case).
- **MJPEG latency variance.** Depends entirely on Wi-Fi and phone-side encoding. Acceptable for the project's target of `≥3 FPS, <500 ms end-to-end`. If the user finds it too laggy in practice, the next escalation is RTSP via Larix Broadcaster (lower latency, same code path).
- **App choice not validated.** "IP Webcam" is the de-facto default for this use case but the user has not yet installed it. If it doesn't work on the S24 Ultra (One UI 6.x), DroidCam is the obvious fallback. Either way the Pi-side code is identical.
- **Calibration drift.** A phone hand-mounted on a tripod is more likely to shift than a screwed-down Pi camera. Documenting "recalibrate after moving the phone" is the cheapest mitigation.
- **Hotspot subnet.** Samsung's default is `192.168.43.x` but One UI versions vary. README tells the user how to discover the gateway IP rather than hard-coding one.
