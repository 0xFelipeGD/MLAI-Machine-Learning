# MLAI — Machine Learning for Agriculture (Fruit Inspection)

> **Edge AI visual inspection** for agricultural produce, running entirely on a Raspberry Pi 4. One SCADA-style dashboard, one inference pipeline.

[![Python](https://img.shields.io/badge/python-3.11+-blue.svg)](https://www.python.org/)
[![TensorFlow](https://img.shields.io/badge/tensorflow-2.16-orange.svg)](https://www.tensorflow.org/)
[![Next.js](https://img.shields.io/badge/next.js-16-black.svg)](https://nextjs.org/)
[![Raspberry Pi](https://img.shields.io/badge/raspberry%20pi-4%20|%208GB-c51a4a.svg)](https://www.raspberrypi.com/)
[![License](https://img.shields.io/badge/license-MIT-green.svg)](LICENSE)

---

## Table of Contents

- [What is this?](#what-is-this)
- [Features](#features)
- [Quick Start](#quick-start)
- [Full Setup Guide](#full-setup-guide)
  - [1. What you'll need](#1-what-youll-need)
  - [2. Flash Raspberry Pi OS](#2-flash-raspberry-pi-os)
  - [3. First boot & SSH](#3-first-boot--ssh)
  - [4. Clone the repo and run the wizard](#4-clone-the-repo-and-run-the-wizard)
  - [5. Fetch the trained models](#5-fetch-the-trained-models)
  - [6. Open the dashboard](#6-open-the-dashboard)
  - [7. Train your own models (optional)](#7-train-your-own-models-optional)
  - [8. Camera calibration (optional)](#8-camera-calibration-optional)
  - [9. Use an Android phone as the camera (alternative)](#9-use-an-android-phone-as-the-camera-alternative)
- [Tech Stack](#tech-stack)
- [Project Structure](#project-structure)
- [API Reference](#api-reference)
- [Performance](#performance)
- [Troubleshooting](#troubleshooting)
- [Manual Setup (appendix)](#manual-setup-appendix)
- [Glossary](#glossary)
- [Contributing](#contributing)
- [License](#license)
- [Acknowledgments](#acknowledgments)

---

## What is this?

MLAI turns a Raspberry Pi 4 + Camera Module 3 into a fruit-inspection station. It runs a single AGRO pipeline that detects fruits, classifies their quality, and estimates their size — fully offline.

| Module | Purpose | Tech |
|--------|---------|------|
| **AGRO** | Fruit detection, sizing, quality grading | SSD MobileNet V1 + MobileNet V2 |

Everything runs locally — **no cloud, no internet, no telemetry**. Inference happens on the Pi via TFLite / LiteRT.

> **New to the project?** There's a beginner-friendly study pack in [`estudos/`](estudos/) (written in Portuguese) that explains MLAI from three angles: ML concepts, computer-vision pipeline, and full-stack integration.

---

## Features

- Live camera feed with configurable resolution / FPS
- Camera intrinsic calibration via printable checkerboard
- Object segmentation and dimensional measurement (px → mm)
- SQLite history of every inspection
- WebSocket live stream + REST API
- SCADA-style web dashboard accessible from any LAN browser
- Runs as two systemd services that auto-restart on crash (API+engine + web dashboard)
- SSD MobileNet V1 fruit detection (apple, banana, orange — extendable)
- Per-fruit quality classification (good / defective)
- Diameter estimation from contours
- Live size histogram
- Per-fruit detail cards on the live page

---

## Quick Start

Full step-by-step instructions are in [Full Setup Guide](#full-setup-guide). Below is the 5-step summary for an experienced user:

```bash
# 1. On the Pi, clone the repo
git clone https://github.com/0xFelipeGD/MLAI-Machine-Learning.git
cd MLAI-Machine-Learning

# 2. Run the setup wizard (apt + pip + Node.js + build + systemd)
bash scripts/setup_wizard.sh

# 3. Pull trained models from the latest GitHub Release
bash scripts/download_models.sh

# 4. From any browser on your network, open the dashboard
#    (replace with your Pi's IP or use mlai.local)
http://192.168.x.x:3000
```

---

## Full Setup Guide

> **The 5-command path.** A single wizard (`scripts/setup_wizard.sh`) handles every system-level chore for you. If something goes wrong, the [Manual Setup appendix](#manual-setup-appendix) shows what the wizard does, step by step.

```
PC          Flash SD card                  →  §1, §2
Pi          First boot + SSH               →  §3
Pi          Clone repo + run the wizard    →  §4
Pi          Download trained models        →  §5   (mock mode works without them)
PC          SSH tunnel + open browser      →  §6
```

### 1. What you'll need

**Hardware**
- Raspberry Pi 4 Model B with **8 GB RAM** (4 GB works but the dashboard is slower)
- Official 27 W USB-C power supply
- microSD card, **32 GB+**, Class 10 / A2
- Raspberry Pi Camera Module 3 + ribbon cable — *or* a recent Android phone running an MJPEG webcam app (see [§9](#9-use-an-android-phone-as-the-camera-alternative))
- Ethernet or Wi-Fi
- Optional: case + fan, light source, printed checkerboard for calibration

**Software on your PC**
- [Raspberry Pi Imager](https://www.raspberrypi.com/software/)
- Python 3.11+ — only if you plan to train your own models

### 2. Flash Raspberry Pi OS

1. Open **Raspberry Pi Imager** → **Choose Device**: *Raspberry Pi 4*.
2. **Choose OS**: *Raspberry Pi OS (64-bit)*. The 64-bit build is mandatory — TFLite / LiteRT only ships for arm64.
3. **Choose Storage**: select your microSD card.
4. Click the ⚙ (Advanced Options):
   - hostname `mlai`
   - enable SSH (password auth is fine)
   - username **`felipe`** + a password (the guide and all systemd units assume this username; if you change it, edit `systemd/*.service` afterwards)
   - Wi-Fi credentials (or skip if using Ethernet)
   - locale + timezone
5. **Write**. ~5 minutes.
6. Eject, put the card in the Pi, connect the camera ribbon to the **CAM** port (metal contacts face the HDMI ports), plug in Ethernet (if used) and power.

### 3. First boot & SSH

Wait ~30 seconds for first boot, then from your PC:

```bash
ssh felipe@mlai.local
```

If `mlai.local` doesn't resolve, grab the Pi's IP from your router and use that.

Update the system:

```bash
sudo apt update && sudo apt full-upgrade -y && sudo reboot
```

SSH back in after the reboot.

### 4. Clone the repo and run the wizard

```bash
cd ~
git clone https://github.com/0xFelipeGD/MLAI-Machine-Learning.git
cd MLAI-Machine-Learning
bash scripts/setup_wizard.sh
```

The wizard runs 8 steps and prompts before anything destructive:

1. Installs apt packages (detects Bookworm vs Trixie automatically)
2. Installs Node.js 22
3. Stops anything that could conflict (old MLAI services, camera holders, port 8000/3000 listeners)
4. `pip install -r requirements.txt`
5. Downloads trained models from the GitHub Release
6. End-to-end camera probe (`rpicam-hello` + `scripts/test_camera.py`)
7. Builds the Next.js dashboard
8. Installs and starts the two systemd services (api+engine and web)

Pass `--yes` for a fully non-interactive run:

```bash
bash scripts/setup_wizard.sh --yes
```

Re-running is safe — each step checks state first and skips what's already done.

> **Prefer SSH keys for git?** The default clone above uses HTTPS (no setup needed). If you want SSH: `ssh-keygen -t ed25519 -C "felipe@mlai"`, paste `~/.ssh/id_ed25519.pub` into <https://github.com/settings/keys>, then clone with `git clone git@github.com:0xFelipeGD/MLAI-Machine-Learning.git`.

### 5. Fetch the trained models

Models (`.tflite` / `.labels.txt`) aren't in the repo — they're large and user-specific. They live as assets on a tagged [GitHub Release](https://github.com/0xFelipeGD/MLAI-Machine-Learning/releases). The wizard (§4) already fetches them. To pull a specific version manually:

```bash
MLAI_MODELS_TAG=v1.0.0 bash scripts/download_models.sh
sudo systemctl restart mlai-api
```

If the release or any asset is missing, the engine falls back to **mock mode** (fake but plausible predictions) so the camera / API / dashboard still work for development.

### 6. Open the dashboard

The Next.js server on the Pi serves the dashboard on port 3000 and proxies `/api/*` and `/ws/*` internally to the FastAPI on port 8000. The browser only ever sees port 3000 — you never need to expose both.

**Simplest path — direct LAN access:**

```bash
# From any browser on the same network:
http://<pi-ip>:3000
# or
http://mlai.local:3000
```

Find the Pi IP with `hostname -I` on the Pi itself, or from your router's DHCP page.

**SSH tunnel (useful on restrictive networks / VPNs):**

```bash
# On your PC — keep this terminal open while using the dashboard
ssh -L 3000:localhost:3000 felipe@mlai.local
```

Then open <http://localhost:3000>.

> **On the Pi with a monitor attached?** Just launch Chromium at `http://localhost:3000` — same URL works.
>
> **mDNS not resolving `mlai.local`?** Use the IP directly. On Ubuntu PCs without `avahi-daemon`, install it with `sudo apt install avahi-daemon libnss-mdns` to enable `.local` resolution.

### 7. Train your own models (optional)

This happens **on your PC**, not the Pi. See [`training/README.md`](training/README.md) for the full walkthrough. The short version:

```bash
# On your PC
cd MLAI-Machine-Learning
python3 -m venv .venv-train && source .venv-train/bin/activate
pip install -r training/requirements.txt

# AGRO detector — pretrained COCO SSD MobileNet V1 (download, no training)
cd models/agro
curl -L -o coco.zip https://storage.googleapis.com/download.tensorflow.org/models/tflite/coco_ssd_mobilenet_v1_1.0_quant_2018_06_29.zip
unzip coco.zip && mv detect.tflite fruit_detector.tflite && mv labelmap.txt fruit_detector.labels.txt && rm coco.zip
cd ../..

# AGRO quality classifier — transfer learning on the Fruits Fresh/Rotten dataset
python training/agro/reorganise_quality.py
python training/agro/train_quality.py
```

Once you have new `.tflite` files, publish them as a GitHub Release (see [`training/README.md`](training/README.md) §5) and the Pi pulls them with `bash scripts/download_models.sh`.

### 8. Camera calibration (optional)

Calibration teaches the camera the pixel-to-millimetre relationship so MLAI can report real-world measurements.

1. Print this checkerboard at 100 %: <https://github.com/opencv/opencv/blob/4.x/doc/pattern.png>. Measure one square — should be 25 mm. If not, pass `--square <mm>` below.
2. Stick it to a flat board.
3. On the Pi:

   ```bash
   python3 scripts/calibrate_camera.py
   ```

4. Hold the board at different angles/distances. Press **SPACE** when corners are drawn in green — capture at least **15** poses.
5. Press **ENTER** to compute and save. Result lands in `config/camera_calibration.json`.

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

## Tech Stack

| Layer | Tech |
|-------|------|
| OS | Raspberry Pi OS Bookworm 64-bit |
| Camera | libcamera + picamera2 |
| CV / Image | OpenCV 4.10 |
| ML inference | TFLite Runtime (XNNPACK) |
| ML training (PC) | TensorFlow 2.16, TF Model Maker |
| API | FastAPI 0.115 + Pydantic 2 |
| WebSocket | FastAPI native |
| DB | SQLite 3 (WAL) |
| Frontend | Next.js 16 (App Router), React 19 |
| Styling | Tailwind CSS 4 |
| Charts | Recharts 2 |
| Process management | systemd |

---

## Project Structure

```
MLAI-Machine-Learning/
├── engine/             # Inference engine (Python)
│   ├── camera.py
│   ├── preprocessor.py
│   ├── calibration.py
│   ├── measurement.py
│   ├── db.py
│   ├── main.py
│   └── agro/           # AGRO module — fruit detector + grader
├── api/                # FastAPI app (REST + WebSocket)
│   └── routes/         # system, camera, ws, agro
├── web/                # Next.js 16 SCADA dashboard
│   ├── app/            # pages: agro, system, calibration
│   ├── components/
│   │   ├── scada/      # shared widgets
│   │   └── agro/
│   └── lib/            # api client, ws client, types
├── training/           # PC-side ML training scripts
│   └── agro/
├── config/             # YAML configs (system, agro)
├── models/             # .tflite files (git-ignored; fetched from Releases)
├── scripts/            # test_camera, calibrate_camera, benchmark, ...
├── systemd/            # service unit files
├── tests/              # pytest suite
└── estudos/            # Beginner-friendly study material (PT-BR)
```

---

## API Reference

| Endpoint | Method | Purpose |
|----------|--------|---------|
| `/api/system/health` | GET | CPU, RAM, temp, FPS |
| `/api/system/info` | GET | Hostname, Python version |
| `/api/camera/config` | GET | Resolution, FPS, calibration status |
| `/api/agro/status` | GET | Active classes, threshold, last result |
| `/api/agro/history` | GET | Paginated fruit inspection history |
| `/api/agro/stats` | GET | Size histogram, by-class & by-quality counts |
| `/api/agro/config` | POST | Update threshold / fruit classes |
| `/ws/live` | WS | Live frame + result stream |

Interactive Swagger UI: `http://<pi-ip>:8000/docs`.

---

## Performance

Targets on a Raspberry Pi 4 / 8 GB:

| Metric | Target |
|--------|--------|
| End-to-end latency | < 500 ms |
| Live FPS | ≥ 3 |
| RAM (all 3 services) | < 4 GB |
| CPU avg | < 80 % |

Run `python scripts/benchmark.py` on the Pi to measure your own setup.

---

## Troubleshooting

| Symptom | Fix |
|---------|-----|
| Browser shows nothing | `journalctl -u mlai-web -n 50` |
| Live feed black | `journalctl -u mlai-api -n 50`; check camera ribbon |
| Dashboard says "Offline" | `sudo systemctl restart mlai-api` |
| `ModuleNotFoundError: picamera2` | Wizard wasn't run — `python3 -c "import picamera2"` |
| `ai_edge_litert` / `tflite_runtime` missing | Wizard wasn't run — `pip3 install -r requirements.txt --break-system-packages` |
| TFLite model won't load | Wrong arch or corrupted file — `file model.tflite` |
| Calibration: no checkerboard found | More light, flatter board, match `--pattern` |
| Live feed black after switching to phone (§9) | Confirm IP Webcam is running, gateway IP matches `ip route`, port 8080 reachable: `curl -I http://<phone-ip>:8080/video` |
| Phone stream stutters / drops | Lower the phone-side resolution to 720p; check Wi-Fi signal strength on the Pi |
| AWB / red-gain / blue-gain sliders inert on phone stream | Those are picamera2 controls; only the CCM matrix slider applies on stream sources |
| Pi reboots randomly | Power supply too weak — use the official 27 W USB-C PSU |

Full diagnostic dump:

```bash
sudo systemctl status mlai-api mlai-web
journalctl -u mlai-api -u mlai-web -n 100 --no-pager
```

---

## Manual Setup (appendix)

Use this if the wizard fails and you want to run each step by hand. Everything here is what `scripts/setup_wizard.sh` does.

**1. apt packages (Bookworm / Trixie)**

```bash
sudo apt update
sudo apt install -y python3-picamera2 python3-pip python3-venv libopenblas-dev libopenjp2-7 libtiff6 libjpeg-dev rpicam-apps git curl
```

On older Pi OS (Bullseye / Buster), use `libatlas-base-dev`, `libtiff5`, `libcamera-apps` instead. Verify the camera: `rpicam-hello --list-cameras` (or `libcamera-hello --list-cameras` on Bullseye).

**2. Node.js 22**

```bash
curl -fsSL https://deb.nodesource.com/setup_22.x | sudo -E bash -
sudo apt install -y nodejs
```

**3. Stop anything that could conflict** — before re-running setup or installing services, free the camera and ports:

```bash
sudo systemctl stop    mlai-api mlai-web 2>/dev/null || true
sudo systemctl disable mlai-api mlai-web 2>/dev/null || true
# Legacy cleanup — the old mlai-engine.service was merged into mlai-api.service.
sudo systemctl stop    mlai-engine 2>/dev/null || true
sudo systemctl disable mlai-engine 2>/dev/null || true
sudo rm -f /etc/systemd/system/mlai-engine.service
sudo systemctl daemon-reload
sudo fuser -k /dev/video* /dev/media* 2>/dev/null || true
sudo fuser -k 8000/tcp 3000/tcp 2>/dev/null || true
pkill -f "python.*engine" 2>/dev/null || true
pkill -f "uvicorn"        2>/dev/null || true
pkill -f "next-server"    2>/dev/null || true
```

**4. Python requirements**

```bash
pip3 install -r requirements.txt --break-system-packages
```

`--break-system-packages` is required by PEP 668 on Raspberry Pi OS. We install into the system Python on purpose — `picamera2` ships as a system package (`python3-picamera2`) and is not on PyPI, so the engine has to run under the same interpreter.

**5. Models**

Fetches the trained `.tflite` models + label files from a GitHub Release into `models/agro/`. The default tag is baked into the script (`DEFAULT_TAG`); override with an env var or positional arg if you want a different version:

```bash
bash scripts/download_models.sh                       # default tag
MLAI_MODELS_TAG=v1.2.0 bash scripts/download_models.sh
bash scripts/download_models.sh v1.2.0                # same, positional
```

The release is created from your training PC after a successful retrain — see [`training/README.md`](training/README.md) §5 for how to publish one. If the release assets are missing the engine falls back to MOCK MODE, which keeps the camera/API/dashboard usable but returns dummy predictions.

**6. Frontend build**

```bash
cd web
npm install
NEXT_PUBLIC_API_BASE=http://localhost:8000 npm run build
cd ..
```

**7. systemd services**

```bash
sudo cp systemd/*.service /etc/systemd/system/
sudo systemctl daemon-reload
sudo systemctl enable --now mlai-api mlai-web
sudo systemctl status mlai-api mlai-web
```

---

## Glossary

| Term | Plain English |
|------|---------------|
| **Raspberry Pi** | Small, cheap computer the size of a credit card |
| **SSH** | Logging into another computer over the network from your terminal |
| **systemd** | The Linux service manager that keeps MLAI running |
| **journalctl** | Command to view logs from systemd services |
| **Inference** | Running a trained model on new data |
| **TFLite / LiteRT** | Compact ML file format optimised for small devices like the Pi |
| **SSD MobileNet** | Small, fast object detector we use for AGRO |
| **Calibration** | Teaching the camera its lens properties so we can measure things |
| **WebSocket** | A two-way real-time connection between the dashboard and the Pi |
| **NoIR camera** | A Pi camera with the infrared filter removed |

---

## Contributing

PRs welcome — see [`CONTRIBUTING.md`](CONTRIBUTING.md).

---

## License

[MIT](LICENSE)

---

## Acknowledgments

- [Fruits-360](https://www.kaggle.com/datasets/moltean/fruits) — fruit image dataset
- [Fruits Fresh and Rotten](https://www.kaggle.com/datasets/sriramr/fruits-fresh-and-rotten-for-classification) — quality-labelled dataset
- [TensorFlow](https://www.tensorflow.org/) and [TFLite Model Maker](https://www.tensorflow.org/lite/models/modify/model_maker)
- [picamera2](https://github.com/raspberrypi/picamera2) and the libcamera project
