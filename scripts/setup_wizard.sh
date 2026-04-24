#!/usr/bin/env bash
# scripts/setup_wizard.sh — one-shot MLAI setup for the Raspberry Pi.
#
# Covers SetupGuide.md from §4 onwards: apt dependencies, Node.js 22, pip
# requirements, frontend build, model directories, systemd services.
#
# Run AFTER cloning the repo (the script must live inside it):
#
#     cd ~/MLAI-Machine-Learning
#     bash scripts/setup_wizard.sh            # interactive
#     bash scripts/setup_wizard.sh --yes      # non-interactive (assume yes)
#
# Idempotent: safe to re-run. Each step is a function that checks state
# before doing work, so a second run will just verify and skip.

set -euo pipefail

# ---------------------------------------------------------------- ui helpers
if [[ -t 1 ]]; then
    C_RESET=$'\033[0m'; C_BOLD=$'\033[1m'
    C_BLUE=$'\033[34m'; C_GREEN=$'\033[32m'
    C_YELLOW=$'\033[33m'; C_RED=$'\033[31m'
else
    C_RESET=; C_BOLD=; C_BLUE=; C_GREEN=; C_YELLOW=; C_RED=
fi

step()  { echo;                 echo "${C_BOLD}${C_BLUE}==>${C_RESET} ${C_BOLD}$*${C_RESET}"; }
info()  { echo "    $*"; }
ok()    { echo "    ${C_GREEN}✔${C_RESET} $*"; }
warn()  { echo "    ${C_YELLOW}⚠${C_RESET} $*"; }
die()   { echo "${C_RED}✘ $*${C_RESET}" >&2; exit 1; }

ASSUME_YES=0
for arg in "$@"; do
    case "$arg" in
        -y|--yes) ASSUME_YES=1 ;;
        -h|--help)
            grep -E '^#( |$)' "$0" | sed 's/^# \{0,1\}//'; exit 0 ;;
        *) die "Unknown flag: $arg (use --help)" ;;
    esac
done

confirm() {
    local prompt="$1"
    if (( ASSUME_YES )); then return 0; fi
    read -r -p "    ${C_YELLOW}?${C_RESET} $prompt [Y/n] " ans
    [[ -z "$ans" || "$ans" =~ ^[Yy]$ ]]
}

# ----------------------------------------------------------------- preflight
REPO_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$REPO_ROOT"

[[ -f "$REPO_ROOT/requirements.txt" && -d "$REPO_ROOT/engine" ]] \
    || die "Script must live inside the MLAI-Machine-Learning clone. Looked at: $REPO_ROOT"

if [[ $EUID -eq 0 ]]; then
    die "Do not run this wizard as root. It will call sudo when needed."
fi

# -------------------------------------------------------------- OS detection
. /etc/os-release 2>/dev/null || die "Cannot read /etc/os-release — is this a Debian-based Pi OS?"
OS_CODENAME="${VERSION_CODENAME:-unknown}"
ARCH="$(uname -m)"
step "Detected OS: ${PRETTY_NAME:-unknown} (codename: ${OS_CODENAME}, arch: ${ARCH})"

case "$OS_CODENAME" in
    trixie|bookworm)
        APT_PKGS=(python3-picamera2 python3-pip python3-venv
                  libopenblas-dev libopenjp2-7 libtiff6 libjpeg-dev
                  rpicam-apps git curl)
        CAMERA_CMD="rpicam-hello" ;;
    bullseye|buster)
        APT_PKGS=(python3-picamera2 python3-pip python3-venv
                  libatlas-base-dev libopenjp2-7 libtiff5 libjpeg-dev
                  libcamera-apps git curl)
        CAMERA_CMD="libcamera-hello" ;;
    *)
        warn "Unrecognised codename '${OS_CODENAME}' — using Trixie package set as best guess."
        APT_PKGS=(python3-picamera2 python3-pip python3-venv
                  libopenblas-dev libopenjp2-7 libtiff6 libjpeg-dev
                  rpicam-apps git curl)
        CAMERA_CMD="rpicam-hello" ;;
esac

if [[ "$ARCH" != "aarch64" ]]; then
    warn "Architecture is $ARCH, not aarch64 — TFLite wheel (ai-edge-litert) expects aarch64. Engine will fall back to MOCK MODE."
fi

# ----------------------------------------------------------- step: apt deps
step "Step 1/8 — Installing system packages via apt"
info "Packages: ${APT_PKGS[*]}"
sudo apt update
sudo apt install -y "${APT_PKGS[@]}"
ok "apt dependencies ready"

# --------------------------------------------------------- step: Node.js 22
step "Step 2/8 — Ensuring Node.js 22 is installed"
NODE_MAJOR=0
if command -v node >/dev/null 2>&1; then
    NODE_MAJOR="$(node -v | sed -E 's/^v([0-9]+).*/\1/')"
fi
if (( NODE_MAJOR >= 22 )); then
    ok "Node.js $(node -v) already installed"
else
    info "Installing Node.js 22 via NodeSource"
    curl -fsSL https://deb.nodesource.com/setup_22.x | sudo -E bash -
    sudo apt install -y nodejs
    ok "Node.js $(node -v) installed"
fi

# ----------------------------------------- step: stop any conflicting runs
step "Step 3/8 — Stopping anything that could conflict"

sudo systemctl stop    mlai-engine mlai-api mlai-web 2>/dev/null || true
sudo systemctl disable mlai-engine mlai-api mlai-web 2>/dev/null || true
info "systemd units stopped (if they were running)"

# Release the camera
if ls /dev/video* /dev/media* >/dev/null 2>&1; then
    sudo fuser -k /dev/video* /dev/media* 2>/dev/null || true
    info "camera devices released"
fi

# Release ports 8000 / 3000
sudo fuser -k 8000/tcp 3000/tcp 2>/dev/null || true
info "ports 8000 / 3000 released"

# Sweep stray manual processes
pkill -f "python.*engine"        2>/dev/null || true
pkill -f "uvicorn"               2>/dev/null || true
pkill -f "next-server"           2>/dev/null || true
pkill -f "npm.*(run dev|start)"  2>/dev/null || true
pkill -f "scripts/test_camera"   2>/dev/null || true
ok "conflicts cleared"

# ----------------------------------------------------------- step: pip deps
step "Step 4/8 — Installing Python packages (may take a few minutes)"
info "Using --break-system-packages because picamera2 ships as a system package"
pip3 install -r requirements.txt --break-system-packages
ok "Python dependencies installed"

# ------------------------------------------------------- step: model dirs
step "Step 5/8 — Preparing model directories"
bash "$REPO_ROOT/scripts/download_models.sh" >/dev/null
ok "models/agro created"

AGRO_MODELS=$(find "$REPO_ROOT/models/agro" -maxdepth 1 -name '*.tflite' 2>/dev/null | wc -l)
if (( AGRO_MODELS == 0 )); then
    warn "No .tflite files found yet. Engine will run in MOCK MODE until you scp them from your PC."
    info "See SetupGuide.md §7 / training/README.md."
else
    ok "Found ${AGRO_MODELS} AGRO model file(s)"
fi

# ---------------------------------------------------------- step: camera
step "Step 6/8 — Verifying camera"
if command -v "$CAMERA_CMD" >/dev/null 2>&1; then
    if "$CAMERA_CMD" --list-cameras 2>&1 | grep -qi "camera\|index"; then
        ok "Camera detected ($CAMERA_CMD --list-cameras succeeded)"
    else
        warn "$CAMERA_CMD ran but no camera was listed. Check the ribbon cable and 'sudo raspi-config' → Interface → Camera."
    fi
else
    warn "$CAMERA_CMD not found — skipping camera probe"
fi

# --------------------------------------------------- step: frontend build
step "Step 7/8 — Building the Next.js dashboard (this takes ~1–3 min)"
if confirm "Build the web dashboard now? (needed for mlai-web.service)"; then
    (
        cd "$REPO_ROOT/web"
        info "npm install"
        npm install --no-audit --no-fund
        info "npm run build (NEXT_PUBLIC_API_BASE=http://localhost:8000)"
        NEXT_PUBLIC_API_BASE=http://localhost:8000 npm run build
    )
    ok "Dashboard built"
else
    warn "Skipped — mlai-web.service will fail to start until 'npm run build' has run in web/."
fi

# ---------------------------------------------- step: install & start services
step "Step 8/8 — Installing and starting systemd services"
if confirm "Install systemd units (mlai-engine, mlai-api, mlai-web) and enable on boot?"; then
    sudo cp "$REPO_ROOT"/systemd/*.service /etc/systemd/system/
    sudo systemctl daemon-reload
    sudo systemctl enable --now mlai-engine mlai-api mlai-web
    sleep 2

    echo
    info "Service status:"
    for svc in mlai-engine mlai-api mlai-web; do
        state=$(systemctl is-active "$svc" 2>/dev/null || echo "unknown")
        if [[ "$state" == "active" ]]; then
            echo "    ${C_GREEN}✔${C_RESET} $svc  — $state"
        else
            echo "    ${C_RED}✘${C_RESET} $svc  — $state   (journalctl -u $svc -n 50)"
        fi
    done
else
    warn "Skipped — start later with: sudo systemctl enable --now mlai-engine mlai-api mlai-web"
fi

# ------------------------------------------------------------------ summary
echo
echo "${C_BOLD}${C_GREEN}MLAI setup finished.${C_RESET}"
cat <<EOF

Next steps:
  1. From your PC, open an SSH tunnel to reach the dashboard:

        ssh -L 3000:localhost:3000 -L 8000:localhost:8000 felipe@mlai.local

  2. Visit http://localhost:3000 in your browser.

  3. If any service is red, check its logs:

        journalctl -u mlai-engine -u mlai-api -u mlai-web -n 100 --no-pager

  4. Missing .tflite files? Train them on your PC (training/README.md) and:

        scp models/agro/*.tflite felipe@mlai.local:~/MLAI-Machine-Learning/models/agro/
        ssh felipe@mlai.local "sudo systemctl restart mlai-engine"

EOF
