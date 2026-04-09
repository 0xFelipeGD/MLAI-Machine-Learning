# MLAI — Machine Learning for Agriculture and Industry

> **Edge AI visual inspection** for industrial parts and agricultural produce, running entirely on a Raspberry Pi 4. One unified SCADA-style dashboard, two independent inference modules.

[![Python](https://img.shields.io/badge/python-3.11+-blue.svg)](https://www.python.org/)
[![TensorFlow](https://img.shields.io/badge/tensorflow-2.16-orange.svg)](https://www.tensorflow.org/)
[![Next.js](https://img.shields.io/badge/next.js-16-black.svg)](https://nextjs.org/)
[![Raspberry Pi](https://img.shields.io/badge/raspberry%20pi-4%20|%208GB-c51a4a.svg)](https://www.raspberrypi.com/)
[![License](https://img.shields.io/badge/license-MIT-green.svg)](LICENSE)

---

## What is this?

MLAI is a complete, self-contained visual inspection system that turns a Raspberry Pi 4 + Camera Module 3 into a dual-purpose AI quality station. It ships with two ready-to-train modules:

| Module | Purpose | Tech |
|--------|---------|------|
| **INDUST** | Industrial QC — anomaly detection on parts & surfaces | PaDiM (Anomalib) on MVTec AD |
| **AGRO**   | Agriculture — fruit detection, sizing, quality grading | SSD MobileNet V2 + MobileNet V2 |

Both run locally — **no cloud, no internet, no telemetry**. Inference happens on the Pi via TFLite.

---

## Features

### Shared
- Live camera feed with configurable resolution / FPS
- Camera intrinsic calibration via printable checkerboard
- Object segmentation and dimensional measurement (px → mm)
- SQLite history of every inspection
- WebSocket live stream + REST API
- SCADA-style web dashboard accessible from any LAN browser
- Runs as three systemd services that auto-restart on crash

### INDUST
- PaDiM anomaly detection (multiple MVTec categories)
- Anomaly heatmap overlay on the live frame
- PASS / WARN / FAIL verdicts with adjustable threshold
- Per-category thresholds, hot-swappable model categories
- CSV export of inspection history

### AGRO
- SSD MobileNet V2 fruit detection (apple, orange, tomato — extendable)
- Per-fruit quality classification (good / defective / unripe)
- Diameter estimation from contours
- Live size histogram
- Per-fruit detail cards on the live page

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

# 3. Drop your trained models into models/{indust,agro}/
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
| ML training (PC) | TensorFlow 2.16, Anomalib 1.1, TF Model Maker |
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
│   ├── indust/         # INDUST module — PaDiM anomaly detector
│   └── agro/           # AGRO module — fruit detector + grader
├── api/                # FastAPI app (REST + WebSocket)
│   └── routes/         # system, camera, ws, indust, agro
├── web/                # Next.js 16 SCADA dashboard
│   ├── app/            # pages: indust, agro, system, calibration
│   ├── components/
│   │   ├── scada/      # shared widgets
│   │   ├── indust/
│   │   └── agro/
│   └── lib/            # api client, ws client, types
├── training/           # PC-side ML training scripts
│   ├── indust/
│   └── agro/
├── config/             # YAML configs (system, indust, agro)
├── models/             # .tflite files (git-ignored)
├── scripts/            # test_camera, calibrate_camera, benchmark, ...
├── systemd/            # service unit files
├── tests/              # pytest suite
└── INSTRUCTIONS.md     # The full architecture spec this repo implements
```

---

## Training Your Own Models

See [`training/README.md`](training/README.md) for the full beginner-friendly walkthrough. Short version:

```bash
cd training
python3 -m venv ../.venv-train
source ../.venv-train/bin/activate
pip install "tensorflow==2.18.*" opencv-python pillow numpy tqdm requests

# INDUST — anomaly detection on a MVTec category (toothbrush by default)
# Drop the bottle/toothbrush MVTec subset into datasets/mvtec/<category>/ first.
python indust/train_autoencoder.py

# AGRO — pretrained COCO SSD for the detector (no training, just download)
cd ../models/agro
curl -L -o coco.zip https://storage.googleapis.com/download.tensorflow.org/models/tflite/coco_ssd_mobilenet_v1_1.0_quant_2018_06_29.zip
unzip coco.zip && mv detect.tflite fruit_detector.tflite && mv labelmap.txt fruit_detector.labels.txt && rm coco.zip
cd ../../training

# AGRO quality classifier — transfer learning on Kaggle Fruits Fresh/Rotten
# Drop the dataset into datasets/fruits/, then:
python agro/reorganise_quality.py
python agro/train_quality.py
```

Then `scp` the resulting `.tflite` files to the Pi.

---

## API Reference (brief)

| Endpoint | Method | Purpose |
|----------|--------|---------|
| `/api/system/health` | GET | CPU, RAM, temp, FPS, active module |
| `/api/system/module` | POST | Switch INDUST ↔ AGRO |
| `/api/camera/config` | GET | Resolution, FPS, calibration status |
| `/api/indust/status` | GET | Active category, threshold, last result |
| `/api/indust/history` | GET | Paginated inspection history |
| `/api/indust/categories` | GET | Available MVTec categories |
| `/api/indust/config` | POST | Update category / threshold |
| `/api/agro/status` | GET | Active classes, threshold, last result |
| `/api/agro/history` | GET | Paginated fruit inspection history |
| `/api/agro/stats` | GET | Size histogram, by-class & by-quality counts |
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
| Model swap | < 5 s |

Run `python scripts/benchmark.py` on the Pi to measure your own setup.

---

## Contributing

PRs welcome — see [`CONTRIBUTING.md`](CONTRIBUTING.md).

---

## License

[MIT](LICENSE)

---

## Acknowledgments

- [MVTec AD](https://www.mvtec.com/company/research/datasets/mvtec-ad) — industrial defect dataset
- [Fruits-360](https://www.kaggle.com/datasets/moltean/fruits) — fruit image dataset
- [Anomalib](https://github.com/openvinotoolkit/anomalib) — anomaly detection library
- [TensorFlow](https://www.tensorflow.org/) and [TFLite Model Maker](https://www.tensorflow.org/lite/models/modify/model_maker)
- [picamera2](https://github.com/raspberrypi/picamera2) and the libcamera project
