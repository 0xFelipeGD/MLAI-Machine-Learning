# MLAI — Machine Learning for Agriculture (Fruit Inspection)

> **Edge AI visual inspection** for agricultural produce, running entirely on a Raspberry Pi 4. One SCADA-style dashboard, one inference pipeline.

[![Python](https://img.shields.io/badge/python-3.11+-blue.svg)](https://www.python.org/)
[![TensorFlow](https://img.shields.io/badge/tensorflow-2.16-orange.svg)](https://www.tensorflow.org/)
[![Next.js](https://img.shields.io/badge/next.js-16-black.svg)](https://nextjs.org/)
[![Raspberry Pi](https://img.shields.io/badge/raspberry%20pi-4%20|%208GB-c51a4a.svg)](https://www.raspberrypi.com/)
[![License](https://img.shields.io/badge/license-MIT-green.svg)](LICENSE)

---

## What is this?

MLAI turns a Raspberry Pi 4 + Camera Module 3 into a fruit-inspection station. It runs a single AGRO pipeline that detects fruits, classifies their quality, and estimates their size — fully offline.

| Module | Purpose | Tech |
|--------|---------|------|
| **AGRO** | Fruit detection, sizing, quality grading | SSD MobileNet V1 + MobileNet V2 |

Everything runs locally — **no cloud, no internet, no telemetry**. Inference happens on the Pi via TFLite / LiteRT.

---

## Features

- Live camera feed with configurable resolution / FPS
- Camera intrinsic calibration via printable checkerboard
- Object segmentation and dimensional measurement (px → mm)
- SQLite history of every inspection
- WebSocket live stream + REST API
- SCADA-style web dashboard accessible from any LAN browser
- Runs as three systemd services that auto-restart on crash
- SSD MobileNet V1 fruit detection (apple, banana, orange — extendable)
- Per-fruit quality classification (good / defective)
- Diameter estimation from contours
- Live size histogram
- Per-fruit detail cards on the live page

---

## Want to understand the project?

There is a beginner-friendly study pack in [`estudos/`](estudos/) (written in Portuguese) that explains the project from three angles: ML concepts, computer-vision pipeline, and full-stack integration. Start there if ML is new to you.

---

## Quick Start

> Full step-by-step instructions live in [`SetupGuide.md`](SetupGuide.md). Below is the 5-step summary for an experienced user.

```bash
# 1. On the Pi, clone the repo
git clone https://github.com/<you>/MLAI-Machine-Learning.git
cd MLAI-Machine-Learning

# 2. Install Python deps
sudo apt install -y python3-picamera2 python3-pip
pip3 install -r requirements.txt

# 3. Drop your trained models into models/agro/
#    (See training/README.md for how to train them on your PC.)
bash scripts/download_models.sh

# 4. Install systemd services
sudo cp systemd/*.service /etc/systemd/system/
sudo systemctl enable --now mlai-engine mlai-api mlai-web

# 5. Open the dashboard
xdg-open http://$(hostname -I | awk '{print $1}'):3000
```

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
├── models/             # .tflite files (git-ignored)
├── scripts/            # test_camera, calibrate_camera, benchmark, ...
├── systemd/            # service unit files
├── tests/              # pytest suite
└── estudos/            # Beginner-friendly study material (PT-BR)
```

---

## Training Your Own Models

See [`training/README.md`](training/README.md) for the full beginner-friendly walkthrough. Short version:

```bash
python3 -m venv .venv-train
source .venv-train/bin/activate
pip install -r training/requirements.txt
cd training

# AGRO detector — pretrained COCO SSD (no training, just download)
cd ../models/agro
curl -L -o coco.zip https://storage.googleapis.com/download.tensorflow.org/models/tflite/coco_ssd_mobilenet_v1_1.0_quant_2018_06_29.zip
unzip coco.zip && mv detect.tflite fruit_detector.tflite && mv labelmap.txt fruit_detector.labels.txt && rm coco.zip
cd ../../training

# AGRO quality classifier — transfer learning on Kaggle Fruits Fresh/Rotten
python agro/reorganise_quality.py
python agro/train_quality.py
```

Then `scp` the resulting `.tflite` files to the Pi.

---

## API Reference (brief)

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

## Performance (Pi 4, 8 GB)

| Metric | Target |
|--------|--------|
| End-to-end latency | < 500 ms |
| Live FPS | ≥ 3 |
| RAM (all 3 services) | < 4 GB |
| CPU avg | < 80 % |

Run `python scripts/benchmark.py` on the Pi to measure your own setup.

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
