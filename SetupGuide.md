# MLAI Setup Guide — From Bare Pi to Live Dashboard

> This guide assumes **zero prior experience**. If you have never used a Raspberry Pi, never opened a terminal, and have never trained a machine-learning model, this guide is for you. Read each section in order.

---

## 1. What You'll Need

**Hardware**
- Raspberry Pi 4 Model B with **8 GB RAM** (the 4 GB version may work but the dashboard will be slower)
- Official Raspberry Pi 27 W USB-C power supply
- microSD card, **32 GB or larger**, Class 10 / A2 (a USB 3.0 SSD is faster)
- microSD card reader (for your PC)
- Raspberry Pi Camera Module 3 (the **NoIR** version is what we use, but the regular one works too)
- Camera ribbon cable (sized for the Pi 4)
- Ethernet cable **or** Wi-Fi credentials
- Optional but recommended: a small case with a fan, a light source (LED ring or lightbox), a printed checkerboard for calibration

**Software (on your PC)**
- [Raspberry Pi Imager](https://www.raspberrypi.com/software/)
- [Miniconda](https://docs.conda.io/en/latest/miniconda.html) — only if you want to train your own models

---

## 2. Installing Raspberry Pi OS

1. Download the **Raspberry Pi Imager** from <https://www.raspberrypi.com/software/> and install it.
2. Insert your microSD card into your PC.
3. Open the Imager. Click **Choose Device** → *Raspberry Pi 4*.
4. Click **Choose OS** → *Raspberry Pi OS (64-bit)*. **The 64-bit version is mandatory** because TFLite Runtime is only released for arm64.
5. Click **Choose Storage** → select your microSD card.
6. Click the gear icon (⚙) to open Advanced Options:
   - Set hostname to `mlai`
   - Enable SSH (use password authentication for now)
   - Set username `pi` and a password you'll remember
   - Configure your Wi-Fi (or skip if using Ethernet)
   - Set locale and timezone
7. Click **Save** then **Write**. Wait until it finishes (~5 minutes).
8. Eject the card and put it in the Pi.
9. Connect the camera ribbon cable to the **CAM** port on the Pi (the metal contacts face the HDMI ports).
10. Plug in Ethernet (if used) and power.

---

## 3. First Boot & SSH

The Pi takes ~30 seconds to boot the first time. Then from your PC:

```bash
ssh pi@mlai.local
```

If `mlai.local` does not resolve, find the Pi's IP from your router and use that instead. You'll be prompted for the password you set above.

Once connected, update the system:

```bash
sudo apt update
sudo apt full-upgrade -y
sudo reboot
```

Re-connect via SSH after the reboot.

---

## 4. Installing Python & System Dependencies

```bash
sudo apt install -y python3-picamera2 python3-pip python3-venv \
                    libatlas-base-dev libopenjp2-7 libtiff5 libjpeg-dev \
                    git curl
```

Verify the camera is detected:

```bash
libcamera-hello --list-cameras
```

You should see one camera listed.

---

## 5. Installing Node.js (for the dashboard)

We use **Node.js 22 LTS**. The easiest way on Raspberry Pi OS:

```bash
curl -fsSL https://deb.nodesource.com/setup_22.x | sudo -E bash -
sudo apt install -y nodejs
node --version   # should print v22.x
npm --version
```

---

## 6. Cloning the Repository

`git` is a tool for downloading project code. We use it to grab MLAI from GitHub:

```bash
cd ~
git clone https://github.com/<you>/MLAI-Machine-Learning.git
cd MLAI-Machine-Learning
```

Install Python packages:

```bash
pip3 install -r requirements.txt
```

This pulls FastAPI, OpenCV, NumPy, etc. It will take a few minutes.

---

## 7. Downloading ML Models

MLAI does **not** ship pre-trained models — different users want different categories. You have two options:

**Option A — train your own** (best results). See section 12 below or [`training/README.md`](training/README.md).

**Option B — start with mock mode** (no models). Skip ahead — the engine will produce fake but plausible-looking predictions so you can verify the dashboard works.

If you trained your own models, run:

```bash
bash scripts/download_models.sh   # creates the directories
# then scp your .tflite files in:
#   models/indust/padim_bottle.tflite
#   models/agro/fruit_detector.tflite
#   models/agro/fruit_quality.tflite
```

---

## 8. Camera Test

```bash
python3 scripts/test_camera.py
```

You should see frames being captured and a `test.jpg` saved in the current folder. Open it (e.g. `scp` it back to your PC) to confirm the picture looks right.

If this fails:
- Check the ribbon cable is fully seated and the right way around
- Run `libcamera-hello` again
- Make sure the camera is enabled in `sudo raspi-config` → Interface Options → Camera

---

## 9. Camera Calibration

Calibration teaches the camera the relationship between pixels and millimetres so MLAI can give real-world measurements.

1. Print this checkerboard at 100 % scale: <https://github.com/opencv/opencv/blob/4.x/doc/pattern.png> (10 columns × 7 rows by default — that gives 9×6 inner corners). Measure one square with a ruler — it should be 25 mm. If not, pass `--square <mm>` below.
2. Stick it onto a flat board.
3. Run:

   ```bash
   python3 scripts/calibrate_camera.py
   ```

4. Hold the board in front of the camera at different angles and distances. Press **SPACE** when corners are detected (you'll see them drawn in green). Capture at least **15** poses.
5. Press **ENTER** to compute and save calibration. The result lands in `config/camera_calibration.json`.

---

## 10. Starting the Services

Three systemd services run MLAI: `mlai-engine`, `mlai-api`, `mlai-web`.

```bash
sudo cp systemd/*.service /etc/systemd/system/
sudo systemctl daemon-reload
sudo systemctl enable --now mlai-engine mlai-api mlai-web
```

Check they're running:

```bash
sudo systemctl status mlai-engine mlai-api mlai-web
```

If any service is red, view its logs:

```bash
journalctl -u mlai-engine -f
```

(Press Ctrl-C to stop following.)

---

## 11. Accessing the Dashboard

Find your Pi's IP:

```bash
hostname -I
```

From any computer on the same network, open a browser to:

```
http://<that-ip>:3000
```

You should see the MLAI dashboard with live CPU/RAM gauges, a module switcher, and the live camera feed.

---

## 12. Training Your Own Models (Optional)

This step happens **on your PC**, not the Pi. See [`training/README.md`](training/README.md) for the full walkthrough. The short version:

```bash
# On your PC
cd MLAI-Machine-Learning/training
conda env create -f environment.yml
conda activate mlai

# INDUST
python indust/train_padim.py --category bottle --epochs 1
python indust/export_tflite.py --category bottle --output ../models/indust/padim_bottle.tflite

# AGRO
python agro/prepare_dataset.py --source ~/Downloads/fruits-360 --output dataset/agro
python agro/train_detector.py  --dataset dataset/agro --output ../models/agro/fruit_detector.tflite
python agro/train_quality.py   --dataset ~/Downloads/fruit-quality --output ../models/agro/fruit_quality.tflite
```

Copy the resulting `.tflite` files to the Pi:

```bash
scp models/indust/*.tflite pi@mlai.local:~/MLAI-Machine-Learning/models/indust/
scp models/agro/*.tflite   pi@mlai.local:~/MLAI-Machine-Learning/models/agro/
ssh pi@mlai.local "sudo systemctl restart mlai-engine"
```

---

## 13. Troubleshooting

| Symptom | Likely cause | Fix |
|---------|--------------|-----|
| Browser shows nothing | `mlai-web` failed | `journalctl -u mlai-web -n 50` |
| Live feed black | `mlai-engine` failed or camera unplugged | `journalctl -u mlai-engine -n 50` |
| "ModuleNotFoundError: picamera2" | Wrong Python or pip env | `python3 -c "import picamera2"` to verify |
| TFLite model won't load | Wrong arch or corrupted file | Check file size and arch (`file model.tflite`) |
| Frontend can't reach API | CORS / proxy mismatch | Verify `next.config.ts` rewrites and `mlai-api` is up |
| Dashboard says "Offline" | API down | `sudo systemctl restart mlai-api` |
| Calibration: no checkerboard found | Lighting / pattern wrong | More light, hold flatter, ensure correct `--pattern` |
| Pi reboots randomly | Power supply too weak | Use the official 27 W USB-C PSU |

Still stuck? Open an issue with the output of:

```bash
sudo systemctl status mlai-engine mlai-api mlai-web
journalctl -u mlai-engine -u mlai-api -u mlai-web -n 100 --no-pager
```

---

## 14. Glossary

| Term | Plain English |
|------|---------------|
| **Raspberry Pi** | A small, cheap computer the size of a credit card |
| **microSD** | The tiny memory card the Pi boots from |
| **SSH** | A way to log into another computer over the network from your terminal |
| **systemd** | The Linux service manager that starts/stops/restarts MLAI processes |
| **journalctl** | The command to view logs from systemd services |
| **Inference** | The act of running a trained model on new data |
| **TFLite** | A compact ML file format optimised for small devices like the Pi |
| **PaDiM** | An anomaly-detection algorithm we use for INDUST |
| **SSD MobileNet V2** | A small, fast object detector we use for AGRO |
| **MVTec AD** | A famous open dataset of industrial parts with defects |
| **Fruits-360** | A large open dataset of labelled fruit photos |
| **Calibration** | Teaching the camera its lens properties so we can measure things |
| **px/mm** | Pixels per millimetre — the conversion factor calibration produces |
| **WebSocket** | A two-way real-time connection between the dashboard and the Pi |
| **REST API** | The HTTP endpoints the dashboard uses to fetch history and config |
| **SCADA** | A common industrial control-room style of dashboard — what MLAI uses |
| **NoIR camera** | A Pi camera with the infrared filter removed; colours look washed out |
| **Heatmap** | A coloured overlay showing where in the image something is interesting |
