#!/usr/bin/env bash
# scripts/fix-camera-grab.sh — make sure mlai-api wins the camera against pipewire.
#
# On Pi OS Desktop, pipewire / wireplumber claim /dev/video0 as soon as the
# user session logs in. Even after killing them they respawn. This script
# drops a systemd override (ExecStartPre) that kicks any camera holder off
# before mlai-api starts; once picamera2/libcamera opens the device with an
# exclusive lock, pipewire cannot steal it back even if it respawns later.
#
# Run once after first install (or after running the wizard):
#
#     bash scripts/fix-camera-grab.sh

set -euo pipefail

OVERRIDE_DIR=/etc/systemd/system/mlai-api.service.d
sudo mkdir -p "$OVERRIDE_DIR"
sudo tee "$OVERRIDE_DIR/camera-grab.conf" > /dev/null <<'EOF'
[Service]
ExecStartPre=/bin/bash -c 'fuser -k /dev/video* /dev/media* 2>/dev/null; sleep 1; exit 0'
EOF
sudo systemctl daemon-reload
sudo systemctl restart mlai-api
sleep 8

echo
echo "=== mlai-api startup log (engine + camera lines) ==="
journalctl -u mlai-api -b --no-pager | grep -iE "engine|camera|loaded|started" | tail -15

echo
echo "=== /dev/video0 holders (deve ser apenas o python3 do mlai-api) ==="
sudo fuser -v /dev/video0 2>&1 || true
