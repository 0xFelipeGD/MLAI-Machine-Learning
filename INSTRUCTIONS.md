# Machine Learning for Agriculture and Industry — Architecture Document

> **Repository:** `machine-learning-for-agriculture-and-industry`
> **Project Name:** Machine Learning for Agriculture and Industry (MLAI)
> **Purpose:** Complete technical blueprint for AI-assisted implementation of an edge-based visual inspection system running on Raspberry Pi 4 Model B (8 GB).
> **Target Audience:** This document is consumed by LLM-based coding agents that will implement each module autonomously. The codebase and all documentation must be **extremely beginner-friendly** — the developer has no prior machine learning experience.

---

## Master Progress Tracker

> Check each box only after the item has been **tested and approved** by a human.

### Overall Milestones

- [ ] Architecture.md reviewed and approved
- [ ] Repository created and initialized
- [ ] Phase 1 — Foundation complete and approved
- [ ] Phase 2 — ML Core complete and approved
- [ ] Phase 3 — Backend complete and approved
- [ ] Phase 4 — Frontend complete and approved
- [ ] Phase 5 — Integration & Deployment complete and approved
- [ ] Full end-to-end system tested on Raspberry Pi 4
- [ ] README.md finalized and approved
- [ ] SetupGuide.md finalized and approved
- [ ] Project ready for public repository

### Feature Checklist — INDUST Module

- [ ] Camera captures frames correctly on Pi 4
- [ ] PaDiM model loads and runs inference on Pi 4
- [ ] Anomaly score is calculated correctly
- [ ] Anomaly heatmap is generated and overlaid on frame
- [ ] Object dimensions are measured (width, height in mm)
- [ ] Calibration reference (px/mm) works accurately (±2mm)
- [ ] Verdict logic works (PASS / FAIL / WARN based on threshold)
- [ ] Results are saved to SQLite database
- [ ] Results are streamed via WebSocket in real-time
- [ ] REST API returns correct INDUST data
- [ ] Frontend INDUST live page renders correctly
- [ ] Frontend INDUST history page works (sort, filter, detail modal)
- [ ] Frontend INDUST settings page updates thresholds
- [ ] Category switching works (bottle → metal_nut → screw)
- [ ] CSV export works from history page
- [ ] Frame + heatmap images are saved to disk
- [ ] Auto-prune cleans old captures

### Feature Checklist — AGRO Module

- [ ] SSD MobileNet V2 model loads and runs on Pi 4
- [ ] Fruit detection works (apple, orange, tomato)
- [ ] Bounding boxes are drawn correctly on live feed
- [ ] Quality classifier runs per detection (good / defective / unripe)
- [ ] Fruit diameter is estimated from contour (mm)
- [ ] Calibration reference works for size estimation
- [ ] Results are saved to SQLite database
- [ ] Results are streamed via WebSocket in real-time
- [ ] REST API returns correct AGRO data
- [ ] Frontend AGRO live page renders correctly
- [ ] Frontend AGRO history page works
- [ ] Frontend AGRO settings page updates thresholds
- [ ] Size histogram displays correctly
- [ ] Fruit cards show crop + class + size + quality
- [ ] Annotated frames are saved to disk

### Feature Checklist — Shared / System

- [ ] Camera capture service starts reliably
- [ ] Camera calibration wizard works (checkerboard detection)
- [ ] Calibration data saves and loads from JSON
- [ ] Module switch (INDUST ↔ AGRO) works in < 5 seconds
- [ ] Dashboard page shows system health (CPU, RAM, Temp)
- [ ] Dashboard shows both module previews
- [ ] WebSocket connection is stable over 1+ hour
- [ ] systemd services start on boot
- [ ] systemd services auto-restart on crash
- [ ] SQLite database is created on first run
- [ ] Disk auto-prune works at threshold
- [ ] Frontend is accessible from LAN browser
- [ ] Frontend looks modern and industrial (not generic AI)

### Documentation Checklist

- [ ] README.md — overview and features
- [ ] README.md — screenshots / placeholders
- [ ] README.md — quick start guide
- [ ] README.md — tech stack table
- [ ] README.md — project structure
- [ ] README.md — contributing section
- [ ] SetupGuide.md — hardware requirements
- [ ] SetupGuide.md — OS installation walkthrough
- [ ] SetupGuide.md — SSH and camera setup
- [ ] SetupGuide.md — Python dependencies
- [ ] SetupGuide.md — Node.js installation
- [ ] SetupGuide.md — Repository cloning
- [ ] SetupGuide.md — Model download
- [ ] SetupGuide.md — Camera test
- [ ] SetupGuide.md — Camera calibration
- [ ] SetupGuide.md — Service startup
- [ ] SetupGuide.md — Dashboard access
- [ ] SetupGuide.md — Training walkthrough (optional)
- [ ] SetupGuide.md — Troubleshooting section
- [ ] SetupGuide.md — Glossary
- [ ] training/README.md — complete ML walkthrough
- [ ] CONTRIBUTING.md — contribution guidelines
- [ ] Architecture.md — kept up to date with changes

### Performance Validation Checklist

- [ ] End-to-end latency < 500 ms (target) / < 1000 ms (hard limit)
- [ ] Live feed FPS ≥ 3 (target) / ≥ 1 (hard limit)
- [ ] Frontend page load < 3 s (target) / < 5 s (hard limit)
- [ ] RAM usage < 4 GB (target) / < 6 GB (hard limit)
- [ ] CPU usage < 80% avg (target) / < 95% sustained (hard limit)
- [ ] SQLite query < 50 ms (target) / < 200 ms (hard limit)
- [ ] Model swap < 5 s (target) / < 10 s (hard limit)
- [ ] No memory leaks after 1-hour continuous run
- [ ] Disk auto-prune triggers correctly at 8 GB

---

## Critical Rules for All Agents

| Rule | Detail |
|------|--------|
| **Documentation First** | Always use **Context7 MCP** to look up framework docs (Next.js 16, TensorFlow, etc.) before writing code. Only fall back to web search if Context7 does not have the docs |
| **UI Skill** | All UI work MUST use the **frontend-design skill** (`/mnt/skills/public/frontend-design/SKILL.md`) for design tokens, components, and styling. The UI must be **modern, polished, industrial** — absolutely no generic AI-looking interfaces |
| **Framework CLIs** | Always use official framework commands (`npx create-next-app`, `conda create`, `anomalib train`, etc.). Only do things manually when no CLI exists |
| **ML Framework** | Use the **TensorFlow ecosystem** exclusively — TensorFlow, Keras, TF Lite, TF Model Maker. Use **Conda** (Miniconda) for environment management on the training machine |
| **Beginner Friendly** | Walk through every ML step like explaining to someone who has never touched machine learning. Extensive inline comments in all training code |
| **Module Separation** | INDUST and AGRO are developed as **separate, independent modules** within the same monorepo. They share only infrastructure code (camera, measurement, API framework, UI shell) |

---

## 1. Project Overview

**MLAI** is a dual-purpose visual inspection platform that runs entirely on a Raspberry Pi 4 Model B (8 GB) equipped with a Raspberry Pi Camera Module 3 NoIR. It provides two independent demonstration modules accessible through a unified SCADA-like web interface built with Next.js 16:

| Module | Codename | Domain | Core Function |
|--------|----------|--------|---------------|
| **Industrial Demo** | `INDUST` | Manufacturing QC | Anomaly detection on surfaces/parts using MVTec AD methodology — detect, localize, and flag defects |
| **Agro Demo** | `AGRO` | Agriculture / Precision Farming | Fruit identification (apple, orange, tomato), size estimation, and quality grading — simulates a rover analyzing produce on the tree |

Both modules share: object detection → dimensional measurement → defect/quality flagging.

---

## 2. Companion Documents

| File | Purpose | Status |
|------|---------|--------|
| `README.md` | Project overview, features, screenshots, quick start, tech stack | - [ ] Created - [ ] Approved |
| `SetupGuide.md` | Step-by-step beginner walkthrough | - [ ] Created - [ ] Approved |
| `Architecture.md` | This document | - [ ] Approved |
| `CONTRIBUTING.md` | How to submit changes | - [ ] Created - [ ] Approved |

---

## 3. Hardware Specification

### 3.1 Compute

| Component | Spec | Verified |
|-----------|------|----------|
| Board | Raspberry Pi 4 Model B | - [ ] |
| SoC | Broadcom BCM2711, quad-core Cortex-A72 @ 1.8 GHz | - [ ] |
| RAM | **8 GB LPDDR4** | - [ ] |
| Storage | 32 GB+ microSD (Class 10 / A2) or USB 3.0 SSD | - [ ] |
| OS | Raspberry Pi OS (64-bit, Bookworm) — **must be 64-bit for TFLite** | - [ ] |
| GPU | VideoCore VI (inference is CPU-bound via XNNPACK) | - [ ] |

### 3.2 Camera

| Component | Spec | Verified |
|-----------|------|----------|
| Module | Raspberry Pi Camera Module 3 NoIR | - [ ] |
| Sensor | Sony IMX708, 12 MP | - [ ] |
| NoIR Note | No IR filter — colors washed under visible light. **This is the camera available** | - [ ] |
| Interface | CSI-2 via `libcamera` stack | - [ ] |
| Resolution | Up to 4608 × 2592; ML uses **640×480** or **1280×720** | - [ ] |

### 3.3 Lighting Considerations (Critical for NoIR)

- **INDUST:** Use controlled white LED ring light or lightbox. Consistent illumination critical.
- **AGRO:** Outdoor/ambient acceptable but colors unreliable. Focus on shape/edge features.

### 3.4 Future Expansion (Not in Scope Yet)

- [ ] VPS with Mosquitto MQTT broker for remote monitoring
- [ ] Desktop/mobile app connecting via MQTT
- [ ] Jetson Nano as alternative compute (TensorRT path)

---

## 4. Software Stack

### 4.1 Core Stack

| Layer | Technology | Version | Installed on Pi | Tested |
|-------|-----------|---------|-----------------|--------|
| OS | Raspberry Pi OS 64-bit | Bookworm | - [ ] | - [ ] |
| ML Training Env | **Miniconda** | Latest | N/A (PC only) | - [ ] |
| ML Framework | **TensorFlow** | 2.16+ | N/A (PC only) | - [ ] |
| ML Inference (Pi) | `tflite-runtime` | 2.16+ | - [ ] | - [ ] |
| Runtime | Python | 3.11+ | - [ ] | - [ ] |
| Camera | `libcamera` + `picamera2` | Latest | - [ ] | - [ ] |
| Image Processing | OpenCV (`opencv-python-headless`) | 4.9+ | - [ ] | - [ ] |
| Frontend | **Next.js** | **16** | - [ ] | - [ ] |
| Node.js | Node.js | 22 LTS | - [ ] | - [ ] |
| API | FastAPI | 0.110+ | - [ ] | - [ ] |
| Process Manager | systemd | — | - [ ] | - [ ] |
| Database | SQLite | 3 | - [ ] | - [ ] |

### 4.2 ML Models

#### INDUST — MVTec Anomaly Detection

| Item | Detail | Status |
|------|--------|--------|
| Dataset | [MVTec AD](https://www.mvtec.com/company/research/datasets/mvtec-ad) — 15 categories | - [ ] Downloaded |
| Model | **PaDiM** via [Anomalib](https://github.com/openvinotoolkit/anomalib) | - [ ] Trained - [ ] Exported TFLite - [ ] Runs on Pi |
| Input | 256×256 | - [ ] Verified |
| Output | Anomaly score (0–1) + heatmap | - [ ] Verified |

#### AGRO — Fruit Detection & Grading

| Item | Detail | Status |
|------|--------|--------|
| Detection Model | **SSD MobileNet V2** via **TF Model Maker** | - [ ] Trained - [ ] Exported TFLite - [ ] Runs on Pi |
| Target Fruits | Apple, Orange, Tomato | - [ ] All 3 detected |
| Detection Dataset | [Fruits-360](https://www.kaggle.com/datasets/moltean/fruits) | - [ ] Downloaded - [ ] Prepared |
| Quality Dataset | Kaggle fruit freshness — fresh vs rotten | - [ ] Downloaded - [ ] Prepared |
| Quality Classifier | MobileNet V2 transfer learning | - [ ] Trained - [ ] Exported TFLite - [ ] Runs on Pi |
| Size Estimation | Classical CV: contour → px area → mm | - [ ] Implemented - [ ] Accurate ±2mm |

### 4.3 Measurement Pipeline (Shared)

```
[Camera Frame] → [Undistort] → [Detect ROI] → [Segment] → [Pixel Dims] → [px→mm] → [Output]
```

- [ ] Camera calibration produces valid intrinsic matrix
- [ ] Undistortion removes barrel/pincushion distortion
- [ ] px-to-mm conversion is accurate with reference object
- [ ] Measurement pipeline tested end-to-end

---

## 5. ML Training Walkthrough (Beginner-Friendly)

> **This section is critical.** The developer has never worked with machine learning.

### 5.1 What is Machine Learning? (60-Second Primer)

- **INDUST:** We show the model thousands of "good" images. It learns what "normal" looks like. Anything different gets flagged as anomalous. This is **anomaly detection**.
- **AGRO:** We show the model labeled images: "this is an apple", "this is an orange". It learns to find and identify fruits. This is **object detection**.

Training happens on your PC (more power). The trained model is exported as a small `.tflite` file that runs on the Pi.

### 5.2 Training Environment Setup (On Your PC)

- [ ] Miniconda installed on PC
- [ ] Conda environment `mlai` created
- [ ] TensorFlow installed and verified
- [ ] GPU detected (if available)

```bash
# Step 1: Install Miniconda
# Download from: https://docs.conda.io/en/latest/miniconda.html

# Step 2: Create a conda environment
conda create -n mlai python=3.11 -y

# Step 3: Activate it (do this every time you open a terminal)
conda activate mlai

# Step 4: Install TensorFlow
pip install tensorflow

# Step 5: Verify
python -c "import tensorflow as tf; print(f'TensorFlow {tf.__version__} OK')"
```

### 5.3 Training INDUST (Anomaly Detection)

- [ ] Anomalib installed
- [ ] MVTec dataset downloaded (~5 GB)
- [ ] PaDiM trained on "bottle" category
- [ ] Model exported to TFLite
- [ ] TFLite file copied to `models/indust/`
- [ ] Repeat for additional categories:
  - [ ] metal_nut
  - [ ] screw
  - [ ] (add more as needed)

```bash
# Install Anomalib
pip install anomalib

# Train PaDiM on "bottle" (downloads MVTec automatically)
anomalib train --model padim --data MVTec --data.category bottle --trainer.max_epochs 1

# Export to TFLite
anomalib export --model padim --export_mode tflite --input_size "[256, 256]"

# Copy to project
cp results/padim/mvtec/bottle/weights/model.tflite models/indust/padim_bottle.tflite
```

> **What happened?** The model learned what "normal" bottles look like. New images get a score: 0.0 = normal, 1.0 = very abnormal, plus a heatmap showing WHERE.

### 5.4 Training AGRO (Fruit Detection)

- [ ] TF Model Maker installed
- [ ] Fruits-360 dataset downloaded
- [ ] Dataset prepared (train/validation split)
- [ ] Fruit detector trained (SSD MobileNet V2)
- [ ] Fruit detector exported to TFLite
- [ ] Quality dataset downloaded
- [ ] Quality classifier trained
- [ ] Quality classifier exported to TFLite
- [ ] All TFLite files copied to `models/agro/`

```bash
# Install TF Model Maker
pip install tflite-model-maker

# Prepare dataset
python training/agro/prepare_dataset.py --source ~/Downloads/fruits-360 --output dataset/agro

# Train fruit detector (transfer learning)
python training/agro/train_detector.py --dataset dataset/agro --output models/agro/fruit_detector.tflite --epochs 50

# Train quality classifier (fresh vs rotten)
python training/agro/train_quality.py --dataset dataset/agro_quality --output models/agro/fruit_quality.tflite --epochs 30
```

> **What happened?** We took MobileNet V2 (already knows how to "see") and specialized it for our 3 fruits via **transfer learning**.

### 5.5 Deploy to Pi

- [ ] INDUST models copied to Pi
- [ ] AGRO models copied to Pi
- [ ] Models load successfully on Pi

```bash
scp models/indust/*.tflite pi@<pi-ip>:~/mlai/models/indust/
scp models/agro/*.tflite pi@<pi-ip>:~/mlai/models/agro/
```

### 5.6 Key Concepts

| Term | Simple Explanation |
|------|-------------------|
| **Model** | A file with learned patterns — like a trained brain |
| **Training** | Showing the model thousands of examples so it learns |
| **Inference** | Using the trained model on new data |
| **TFLite** | Small, fast ML format for devices like Raspberry Pi |
| **Transfer Learning** | Reusing a model trained on millions of images for your task |
| **Anomaly Detection** | Learning "normal" and flagging anything different |
| **Object Detection** | Finding and labeling objects with bounding boxes |
| **Epoch** | One pass through all training images |
| **Conda** | Tool for isolated Python environments |
| **Dataset** | Collection of labeled images for training |

---

## 6. System Architecture

### 6.1 High-Level Diagram

```
┌──────────────────────────────────────────────────────────────┐
│                    RASPBERRY PI 4 (8 GB)                     │
│                                                              │
│  ┌─────────────┐     ┌───────────────────────────────────┐   │
│  │  Camera      │     │  INFERENCE ENGINE (Python)         │   │
│  │  Module 3    │────▶│                                   │   │
│  │  NoIR        │     │  ┌────────┐  ┌────────────────┐   │   │
│  └─────────────┘     │  │Capture │─▶│  PreProcess    │   │   │
│                       │  └────────┘  └───────┬────────┘   │   │
│                       │           ┌──────────┴─────────┐  │   │
│                       │           ▼                    ▼  │   │
│                       │   ┌──────────────┐  ┌───────────┐ │   │
│                       │   │   INDUST     │  │   AGRO    │ │   │
│                       │   │   Module     │  │   Module  │ │   │
│                       │   │ (separate)   │  │ (separate)│ │   │
│                       │   └──────┬───────┘  └─────┬─────┘ │   │
│                       │          └──────┬─────────┘       │   │
│                       │                 ▼                 │   │
│                       │         ┌──────────────┐          │   │
│                       │         │  SQLite DB   │          │   │
│                       │         └──────┬───────┘          │   │
│                       └────────────────┼──────────────────┘   │
│                           ┌────────────┴────────────┐        │
│                           │   FastAPI (port 8000)   │        │
│                           └────────────┬────────────┘        │
│                           ┌────────────┴────────────┐        │
│                           │  Next.js 16 (port 3000) │        │
│                           └─────────────────────────┘        │
└──────────────────────────────────────────────────────────────┘
         │  http://<pi-ip>:3000  │  ssh pi@<pi-ip>
         ▼                       ▼
    ┌──────────┐           ┌──────────┐
    │ Browser  │           │ Terminal │
    └──────────┘           └──────────┘
```

### 6.2 Services

| Service | Name | Port | Role | Running |
|---------|------|------|------|---------|
| Inference Engine | `mlai-engine` | — | Camera, ML inference, measurement | - [ ] |
| API Server | `mlai-api` | 8000 | REST + WebSocket | - [ ] |
| Web Frontend | `mlai-web` | 3000 | SCADA dashboard | - [ ] |

### 6.3 Active Module

Only **one module runs at a time**. Switch via `POST /api/system/module`. Model swap takes ~3–5 seconds.

- [ ] Module switch INDUST → AGRO works
- [ ] Module switch AGRO → INDUST works
- [ ] Switch completes in < 5 seconds

---

## 7. Directory Structure

```
machine-learning-for-agriculture-and-industry/
├── README.md
├── SetupGuide.md
├── Architecture.md
├── CONTRIBUTING.md
├── LICENSE
├── .gitignore
│
├── config/
│   ├── camera_calibration.json
│   ├── system_config.yaml
│   ├── indust/
│   │   └── config.yaml              ← INDUST-only config
│   └── agro/
│       └── config.yaml              ← AGRO-only config
│
├── models/                           ← git-ignored, downloaded via setup
│   ├── indust/
│   │   ├── padim_bottle.tflite
│   │   └── ...
│   └── agro/
│       ├── fruit_detector.tflite
│       └── fruit_quality.tflite
│
├── engine/                           ← SHARED infrastructure
│   ├── __init__.py
│   ├── main.py
│   ├── camera.py
│   ├── preprocessor.py
│   ├── calibration.py
│   ├── measurement.py
│   ├── db.py
│   ├── indust/                       ← INDUST module (self-contained)
│   │   ├── __init__.py
│   │   ├── pipeline.py
│   │   ├── detector.py
│   │   ├── heatmap.py
│   │   └── reporter.py
│   └── agro/                         ← AGRO module (self-contained)
│       ├── __init__.py
│       ├── pipeline.py
│       ├── detector.py
│       ├── classifier.py
│       ├── sizer.py
│       └── reporter.py
│
├── api/
│   ├── __init__.py
│   ├── main.py
│   ├── schemas.py
│   └── routes/
│       ├── system.py
│       ├── camera.py
│       ├── ws.py
│       ├── indust.py                 ← INDUST routes (separate)
│       └── agro.py                   ← AGRO routes (separate)
│
├── web/                              ← npx create-next-app@latest
│   ├── package.json
│   ├── next.config.ts
│   ├── tailwind.config.ts
│   ├── app/
│   │   ├── layout.tsx
│   │   ├── page.tsx
│   │   ├── indust/                   ← INDUST pages (separate)
│   │   │   ├── page.tsx
│   │   │   ├── history/page.tsx
│   │   │   └── settings/page.tsx
│   │   ├── agro/                     ← AGRO pages (separate)
│   │   │   ├── page.tsx
│   │   │   ├── history/page.tsx
│   │   │   └── settings/page.tsx
│   │   └── system/
│   │       ├── page.tsx
│   │       └── calibration/page.tsx
│   ├── components/
│   │   ├── ui/
│   │   ├── scada/
│   │   ├── indust/                   ← INDUST-only components
│   │   └── agro/                     ← AGRO-only components
│   └── lib/
│       ├── api.ts
│       ├── ws.ts
│       └── types.ts
│
├── training/                         ← Runs on PC, NOT Pi
│   ├── README.md
│   ├── environment.yml
│   ├── indust/
│   │   ├── train_padim.py
│   │   └── export_tflite.py
│   └── agro/
│       ├── prepare_dataset.py
│       ├── train_detector.py
│       └── train_quality.py
│
├── scripts/
│   ├── download_models.sh
│   ├── calibrate_camera.py
│   ├── benchmark.py
│   └── test_camera.py
│
├── tests/
│   ├── test_indust_pipeline.py
│   ├── test_agro_pipeline.py
│   ├── test_measurement.py
│   └── test_api.py
│
├── systemd/
│   ├── mlai-engine.service
│   ├── mlai-api.service
│   └── mlai-web.service
│
└── data/
    ├── mlai.db
    └── captures/
```

---

## 8. Data Flow

```
picamera2 capture (640×480 @ ~5 FPS)
    → Undistort (camera_calibration.json)
    → Route to active module:

INDUST:                              AGRO:
  Resize 256×256                       Resize 320×320
  Normalize [0,1]                      Run SSD MobileNet V2 → boxes
  Run PaDiM → score + heatmap         Per detection:
  Segment ROI                            Run quality classifier
  Measure dimensions                     Extract contour → diameter
  Threshold → PASS/FAIL                Aggregate results

    → Result Object → SQLite (persist) + WebSocket (real-time)
```

---

## 9. API Specification

### 9.1 REST Endpoints

| Endpoint | Method | Purpose | Implemented | Tested |
|----------|--------|---------|-------------|--------|
| `/api/system/health` | GET | CPU, RAM, temp, uptime, active module | - [ ] | - [ ] |
| `/api/system/module` | POST | Switch INDUST ↔ AGRO | - [ ] | - [ ] |
| `/api/camera/config` | GET | Resolution, fps, calibrated status | - [ ] | - [ ] |
| `/api/camera/calibrate` | POST | Trigger calibration routine | - [ ] | - [ ] |
| `/api/camera/capture` | POST | Single frame JPEG | - [ ] | - [ ] |
| `/api/indust/status` | GET | Running, category, threshold, last result | - [ ] | - [ ] |
| `/api/indust/history` | GET | Paginated inspection results | - [ ] | - [ ] |
| `/api/indust/history/:id` | GET | Full detail + image + heatmap | - [ ] | - [ ] |
| `/api/indust/config` | POST | Update thresholds, category | - [ ] | - [ ] |
| `/api/indust/categories` | GET | Available MVTec categories | - [ ] | - [ ] |
| `/api/agro/status` | GET | Running, fruit classes, last result | - [ ] | - [ ] |
| `/api/agro/history` | GET | Paginated results | - [ ] | - [ ] |
| `/api/agro/history/:id` | GET | Full detail + annotated image | - [ ] | - [ ] |
| `/api/agro/config` | POST | Update thresholds, fruits | - [ ] | - [ ] |
| `/api/agro/stats` | GET | Size distribution, quality ratios | - [ ] | - [ ] |

### 9.2 WebSocket Protocol

```
ws://<pi-ip>:8000/ws/live

Server → Client:
  { type: "frame", module, frame_b64, fps, timestamp }
  { type: "indust_result", verdict, anomaly_score, heatmap_b64, measurements, defect_type, inference_ms }
  { type: "agro_result", detections: [{class, confidence, bbox, diameter_mm, quality}], total_count, inference_ms }

Client → Server:
  { type: "command", action: "pause" | "resume" | "capture" }
```

- [ ] WebSocket connects from frontend
- [ ] Frames stream at ≥ 3 FPS
- [ ] INDUST results arrive in real-time
- [ ] AGRO results arrive in real-time
- [ ] Pause/resume commands work
- [ ] Capture command saves frame

---

## 10. Frontend Design

### 10.1 Design Language

> **Use frontend-design skill for implementation.** Think factory control room, not chatbot.

- **Background:** Dark (`#0f1117`), subtle grid texture
- **Status colors:** Green `#00e676` (OK), Amber `#ffab00` (warn), Red `#ff1744` (fault), Blue `#2979ff` (info)
- **Typography:** Monospace for values (JetBrains Mono / IBM Plex Mono), sans-serif for labels (Inter)
- **Feel:** Blinking status dots, live FPS counter, gauge animations, timestamps on every reading
- **NOT generic AI:** No gradient blobs, no chatbot bubbles, no purple/blue AI gradients

### 10.2 Pages

| Route | Content | Implemented | Tested |
|-------|---------|-------------|--------|
| `/` | Dashboard — health gauges, module switch, previews | - [ ] | - [ ] |
| `/indust` | Live feed + heatmap, anomaly gauge, verdict, measurements, category, threshold | - [ ] | - [ ] |
| `/indust/history` | Sortable table, detail modal, CSV export | - [ ] | - [ ] |
| `/indust/settings` | Threshold config, category management | - [ ] | - [ ] |
| `/agro` | Live feed + bounding boxes, count, fruit cards, size histogram | - [ ] | - [ ] |
| `/agro/history` | Sortable table, fruit count, avg diameter | - [ ] | - [ ] |
| `/agro/settings` | Threshold config, fruit class management | - [ ] | - [ ] |
| `/system` | CPU/RAM/Temp gauges, service status, disk, network | - [ ] | - [ ] |
| `/system/calibration` | Wizard, checkerboard detection, capture & compute | - [ ] | - [ ] |

### 10.3 Frontend Components

| Component | Location | Purpose | Implemented | Tested |
|-----------|----------|---------|-------------|--------|
| Sidebar | `components/scada/` | Navigation between modules | - [ ] | - [ ] |
| Topbar | `components/scada/` | System status, active module | - [ ] | - [ ] |
| StatusIndicator | `components/scada/` | Green/yellow/red dots | - [ ] | - [ ] |
| LiveFeed | `components/scada/` | WebSocket video canvas | - [ ] | - [ ] |
| GaugeWidget | `components/scada/` | Circular gauge | - [ ] | - [ ] |
| HistoryTable | `components/scada/` | Sortable/filterable table | - [ ] | - [ ] |
| HeatmapOverlay | `components/scada/` | Canvas overlay for anomalies | - [ ] | - [ ] |
| MeasurementCard | `components/scada/` | Dimension readout with units | - [ ] | - [ ] |
| IndustDashboard | `components/indust/` | INDUST live view composition | - [ ] | - [ ] |
| DefectPanel | `components/indust/` | Defect details panel | - [ ] | - [ ] |
| AgroDashboard | `components/agro/` | AGRO live view composition | - [ ] | - [ ] |
| FruitCard | `components/agro/` | Single fruit detection card | - [ ] | - [ ] |
| SizeHistogram | `components/agro/` | Distribution chart | - [ ] | - [ ] |

### 10.4 Tech

Next.js 16 (App Router), React 19, Tailwind CSS 4, shadcn/ui, Recharts, HTML5 Canvas.

---

## 11. Database Schema

```sql
CREATE TABLE system_state (
    key TEXT PRIMARY KEY, value TEXT, updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- INDUST (independent)
CREATE TABLE indust_results (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    category TEXT NOT NULL, anomaly_score REAL NOT NULL, verdict TEXT NOT NULL,
    defect_type TEXT, width_mm REAL, height_mm REAL, area_mm2 REAL,
    threshold_used REAL NOT NULL, inference_ms INTEGER,
    frame_path TEXT, heatmap_path TEXT, notes TEXT
);

-- AGRO (independent)
CREATE TABLE agro_results (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    frame_path TEXT, annotated_frame_path TEXT,
    total_detections INTEGER DEFAULT 0, avg_diameter_mm REAL,
    inference_ms INTEGER, notes TEXT
);

CREATE TABLE agro_detections (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    result_id INTEGER NOT NULL REFERENCES agro_results(id) ON DELETE CASCADE,
    fruit_class TEXT NOT NULL, confidence REAL NOT NULL,
    bbox_x1 INTEGER, bbox_y1 INTEGER, bbox_x2 INTEGER, bbox_y2 INTEGER,
    diameter_mm REAL, quality TEXT, quality_confidence REAL
);
```

- [ ] Schema created on first run
- [ ] INDUST results insert/query works
- [ ] AGRO results insert/query works
- [ ] AGRO detections linked to results correctly
- [ ] Indexes improve query performance

---

## 12. Multi-Agent Development

### 12.1 Hierarchy

```
                    MASTER ORCHESTRATOR
                    (reads Architecture.md)
                           │
     ┌─────────┬──────────┬┴──────────┬──────────┬──────────┐
  AGENT-CV  AGENT-ML  AGENT-BE  AGENT-FE  AGENT-DV  AGENT-QA
  Camera    TFLite    FastAPI   Next.js   systemd   Tests
  OpenCV    Anomalib  SQLite    SCADA     Scripts   Bench
  Calib     Training  REST/WS   Canvas    Setup     Docs
  Measure   TF Maker  IPC       Charts    Deploy    README
```

### 12.2 Agent Rules

| Agent | Scope | Key Rules |
|-------|-------|-----------|
| **AGENT-CV** | `engine/camera.py`, `engine/preprocessor.py`, `engine/calibration.py`, `engine/measurement.py` | Context7 for OpenCV/picamera2 docs |
| **AGENT-ML** | `engine/indust/`, `engine/agro/`, `training/`, `models/` | **Extensive inline comments** — developer is ML beginner. Context7 for TF docs |
| **AGENT-BE** | `api/`, `engine/db.py` | Keep INDUST and AGRO routes in separate files |
| **AGENT-FE** | `web/` | **Context7 for Next.js 16.** **Frontend-design skill for UI.** `npx create-next-app@latest`. No generic AI aesthetic |
| **AGENT-DV** | `systemd/`, `scripts/`, `SetupGuide.md` | SetupGuide **extremely detailed** — assume reader never used terminal |
| **AGENT-QA** | `tests/`, `config/`, `README.md` | README **best-in-class** — badges, screenshots, polished |

### 12.3 Implementation Phases — Detailed Checklist

#### Phase 1 — Foundation (AGENT-CV + AGENT-QA)

- [ ] `engine/camera.py` — picamera2 capture service
- [ ] `engine/preprocessor.py` — resize, normalize, undistort
- [ ] `engine/calibration.py` — checkerboard calibration with OpenCV
- [ ] `engine/measurement.py` — contour analysis, px-to-mm conversion
- [ ] `scripts/test_camera.py` — camera verification script
- [ ] `scripts/calibrate_camera.py` — interactive calibration wizard
- [ ] `config/system_config.yaml` — default config
- [ ] `config/camera_calibration.json` — placeholder/example
- [ ] `config/indust/config.yaml` — INDUST default config
- [ ] `config/agro/config.yaml` — AGRO default config
- [ ] `tests/test_measurement.py` — measurement accuracy tests
- [ ] README.md skeleton created
- [ ] SetupGuide.md skeleton created
- [ ] **Phase 1 reviewed and approved**

#### Phase 2 — ML Core (AGENT-ML)

- [ ] `training/environment.yml` — conda environment definition
- [ ] `training/README.md` — full beginner-friendly ML walkthrough
- [ ] `training/indust/train_padim.py` — PaDiM training script with extensive comments
- [ ] `training/indust/export_tflite.py` — export to TFLite
- [ ] `training/agro/prepare_dataset.py` — download & organize Fruits-360
- [ ] `training/agro/train_detector.py` — SSD MobileNet V2 fine-tuning
- [ ] `training/agro/train_quality.py` — fresh vs rotten classifier
- [ ] `engine/indust/pipeline.py` — INDUST orchestration
- [ ] `engine/indust/detector.py` — Anomalib TFLite wrapper
- [ ] `engine/indust/heatmap.py` — heatmap generation + overlay
- [ ] `engine/indust/reporter.py` — INDUST result formatting
- [ ] `engine/agro/pipeline.py` — AGRO orchestration
- [ ] `engine/agro/detector.py` — SSD MobileNet V2 wrapper
- [ ] `engine/agro/classifier.py` — quality classification wrapper
- [ ] `engine/agro/sizer.py` — fruit size from contours
- [ ] `engine/agro/reporter.py` — AGRO result formatting
- [ ] `scripts/benchmark.py` — inference latency measurement
- [ ] `scripts/download_models.sh` — pre-trained model downloader
- [ ] Models benchmarked on Pi 4, latency within budget
- [ ] `tests/test_indust_pipeline.py` — INDUST tests
- [ ] `tests/test_agro_pipeline.py` — AGRO tests
- [ ] **Phase 2 reviewed and approved**

#### Phase 3 — Backend (AGENT-BE)

- [ ] `engine/db.py` — SQLite ORM + schema creation
- [ ] `api/main.py` — FastAPI app with CORS, lifespan
- [ ] `api/schemas.py` — Pydantic models for all responses
- [ ] `api/routes/system.py` — health, module switch, config
- [ ] `api/routes/camera.py` — config, calibrate, capture
- [ ] `api/routes/ws.py` — WebSocket live streaming
- [ ] `api/routes/indust.py` — INDUST status, history, config (SEPARATE)
- [ ] `api/routes/agro.py` — AGRO status, history, stats (SEPARATE)
- [ ] Engine ↔ API IPC (Unix socket for frames)
- [ ] `engine/main.py` — entry point orchestrating active module
- [ ] `tests/test_api.py` — API endpoint tests
- [ ] **Phase 3 reviewed and approved**

#### Phase 4 — Frontend (AGENT-FE)

- [ ] `npx create-next-app@latest web` scaffold
- [ ] Frontend-design skill consulted for design system
- [ ] Context7 consulted for Next.js 16 docs
- [ ] `app/layout.tsx` — SCADA shell (sidebar + topbar)
- [ ] `components/scada/Sidebar.tsx`
- [ ] `components/scada/Topbar.tsx`
- [ ] `components/scada/StatusIndicator.tsx`
- [ ] `components/scada/LiveFeed.tsx` — WebSocket canvas
- [ ] `components/scada/GaugeWidget.tsx`
- [ ] `components/scada/HistoryTable.tsx`
- [ ] `components/scada/HeatmapOverlay.tsx`
- [ ] `components/scada/MeasurementCard.tsx`
- [ ] `app/page.tsx` — Dashboard
- [ ] `app/indust/page.tsx` — INDUST live view
- [ ] `app/indust/history/page.tsx` — INDUST history
- [ ] `app/indust/settings/page.tsx` — INDUST settings
- [ ] `components/indust/IndustDashboard.tsx`
- [ ] `components/indust/DefectPanel.tsx`
- [ ] `app/agro/page.tsx` — AGRO live view
- [ ] `app/agro/history/page.tsx` — AGRO history
- [ ] `app/agro/settings/page.tsx` — AGRO settings
- [ ] `components/agro/AgroDashboard.tsx`
- [ ] `components/agro/FruitCard.tsx`
- [ ] `components/agro/SizeHistogram.tsx`
- [ ] `app/system/page.tsx` — System health
- [ ] `app/system/calibration/page.tsx` — Calibration wizard
- [ ] `lib/api.ts` — REST fetch helpers
- [ ] `lib/ws.ts` — WebSocket connection manager
- [ ] `lib/types.ts` — TypeScript types matching API schemas
- [ ] UI looks modern and industrial (human approved)
- [ ] UI does NOT look like generic AI
- [ ] **Phase 4 reviewed and approved**

#### Phase 5 — Integration & Deployment (AGENT-DV + AGENT-QA)

- [ ] `systemd/mlai-engine.service` — engine daemon
- [ ] `systemd/mlai-api.service` — API daemon
- [ ] `systemd/mlai-web.service` — Next.js daemon
- [ ] Services start on boot
- [ ] Services auto-restart on crash
- [ ] `SetupGuide.md` — complete beginner walkthrough (all 14 sections)
- [ ] `README.md` — polished, professional, badges, screenshots
- [ ] `CONTRIBUTING.md` — contribution guide
- [ ] End-to-end test: camera → inference → API → frontend
- [ ] Performance benchmarks pass all targets
- [ ] 1-hour stability test (no memory leaks, no crashes)
- [ ] SetupGuide tested by following it from scratch
- [ ] **Phase 5 reviewed and approved**

### 12.4 Agent Report Format

```yaml
agent: AGENT-XX
phase: N
task: "Brief description"
status: DONE | BLOCKED | IN_PROGRESS
files_created: [...]
files_modified: [...]
tests_added: [...]
blockers: [...]
docs_consulted: ["Context7: next.js app-router", "Web: anomalib export"]
```

---

## 13. Performance Budgets

| Metric | Target | Hard Limit | Passed |
|--------|--------|------------|--------|
| End-to-end latency | < 500 ms | < 1000 ms | - [ ] |
| Live feed FPS | ≥ 3 | ≥ 1 | - [ ] |
| Page load | < 3 s | < 5 s | - [ ] |
| RAM (all services) | < 4 GB | < 6 GB | - [ ] |
| CPU (inference) | < 80% | < 95% | - [ ] |
| DB query | < 50 ms | < 200 ms | - [ ] |
| Model swap | < 5 s | < 10 s | - [ ] |
| Disk (30 days) | < 5 GB | Auto-prune 8 GB | - [ ] |
| 1-hour stability | No leaks/crashes | — | - [ ] |

---

## 14. Configuration

### system_config.yaml

```yaml
project:
  name: "Machine Learning for Agriculture and Industry"
  version: "1.0.0"
system:
  hostname: mlai
  default_module: INDUST
camera:
  resolution: [640, 480]
  fps: 5
  exposure_mode: auto
  awb_mode: auto
inference:
  num_threads: 4
  log_level: INFO
storage:
  db_path: data/mlai.db
  capture_dir: data/captures
  max_captures_gb: 5
  prune_after_days: 30
api:
  host: 0.0.0.0
  port: 8000
  cors_origins: ["http://localhost:3000"]
web:
  port: 3000
mqtt:  # Future
  enabled: false
```

### config/indust/config.yaml

```yaml
indust:
  active_category: bottle
  model_dir: models/indust
  default_threshold: 0.5
  categories:
    bottle: { model: padim_bottle.tflite, input_size: [256, 256], threshold: 0.5 }
    metal_nut: { model: padim_metal_nut.tflite, input_size: [256, 256], threshold: 0.45 }
    screw: { model: padim_screw.tflite, input_size: [256, 256], threshold: 0.55 }
  save_frames: true
  save_heatmaps: true
```

### config/agro/config.yaml

```yaml
agro:
  detector_model: models/agro/fruit_detector.tflite
  quality_model: models/agro/fruit_quality.tflite
  detector_input_size: [320, 320]
  quality_input_size: [224, 224]
  detection_threshold: 0.5
  fruit_classes: [apple, orange, tomato]
  calibration:
    reference_diameter_mm: 25.0
  save_frames: true
  save_annotated: true
```

---

## 15. Expected Performance (Pi 4, 8 GB)

| Model | Input | Time | FPS | Verified |
|-------|-------|------|-----|----------|
| PaDiM TFLite (INDUST) | 256×256 | 300–500 ms | 2–3 | - [ ] |
| SSD MobileNet V2 (AGRO det) | 320×320 | 100–150 ms | 7–10 | - [ ] |
| MobileNet V2 classifier (AGRO qual) | 224×224 | 50–80 ms | 12–20 | - [ ] |
| Full AGRO pipeline | — | 200–400 ms | 2.5–5 | - [ ] |

---

## 16. README.md Specification

AGENT-QA must produce a polished, professional README:

- [ ] Project title with emoji icons
- [ ] Badges (Python, TensorFlow, Next.js, Raspberry Pi, License)
- [ ] Overview paragraph
- [ ] Features list (INDUST / AGRO / Shared)
- [ ] Screenshot placeholders
- [ ] Architecture diagram (simplified)
- [ ] Quick Start (5 steps, link to SetupGuide.md)
- [ ] Tech Stack table
- [ ] Project Structure (simplified tree)
- [ ] Training section (brief, link to training/README.md)
- [ ] API Reference (brief, link to Architecture.md §9)
- [ ] Contributing (brief, link to CONTRIBUTING.md)
- [ ] License (MIT)
- [ ] Acknowledgments (MVTec, Fruits-360, TensorFlow, Anomalib)

---

## 17. SetupGuide.md Specification

AGENT-DV must produce an extremely detailed walkthrough:

- [ ] 1. What You'll Need — hardware list
- [ ] 2. Installing Raspberry Pi OS — Raspberry Pi Imager
- [ ] 3. First Boot & SSH — connect, password, enable camera
- [ ] 4. Installing Python Dependencies — every command explained
- [ ] 5. Installing Node.js — nvm or nodesource
- [ ] 6. Cloning the Repository — what git is, git clone
- [ ] 7. Downloading ML Models — download script
- [ ] 8. Camera Test — verify camera
- [ ] 9. Camera Calibration — print checkerboard, run calibration
- [ ] 10. Starting Services — systemctl explained
- [ ] 11. Accessing the Dashboard — browser, finding Pi IP
- [ ] 12. Training Your Own Models (Optional) — conda on PC
- [ ] 13. Troubleshooting — common issues
- [ ] 14. Glossary — every term defined simply

---

## 18. Glossary

| Term | Simple Explanation |
|------|-------------------|
| Module | Operating mode: INDUST or AGRO |
| Verdict | Pass/fail decision |
| Anomaly Score | How abnormal (0=normal, 1=very abnormal) |
| Heatmap | Colored overlay showing WHERE the problem is |
| NoIR | Camera without infrared filter |
| PaDiM | Anomaly detection algorithm |
| SCADA | Industrial monitoring UI style |
| TFLite | TensorFlow Lite — fast ML for small devices |
| px/mm | Pixels per millimeter conversion ratio |
| ROI | Region of Interest — the target area |
| Conda | Python environment manager |
| Transfer Learning | Reusing a pre-trained model for your task |
| Epoch | One pass through all training data |
| Inference | Running a trained model on new data |
| Bounding Box | Rectangle around a detected object |
| SSD MobileNet | Fast object detection model for edge devices |
| MVTec AD | Industrial defect dataset (15 categories) |
| WebSocket | Real-time server→browser connection |
| FastAPI | Python API framework |
| systemd | Linux service manager |

---

## 19. Open Decisions & Risks

| # | Item | Recommendation | Risk | Resolved |
|---|------|---------------|------|----------|
| 1 | Anomaly model: PaDiM vs PatchCore | PaDiM (faster) | PatchCore more accurate but ~2x slower | - [ ] |
| 2 | Fruit detector: SSD MobileNet V2 vs EfficientDet-Lite0 | SSD MobileNet V2 | EfficientDet may be more accurate | - [ ] |
| 3 | IPC method | Unix socket for frames | Shared memory faster but complex | - [ ] |
| 4 | NoIR color accuracy | Rely on shape/edge features | Color-based quality grading may suffer | - [ ] |
| 5 | Concurrent modules | One at a time | Users may want quick comparison | - [ ] |
| 6 | Anomalib TFLite export | Try direct first | May need ONNX intermediate step | - [ ] |

---

*Project: Machine Learning for Agriculture and Industry*