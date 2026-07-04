#!/usr/bin/env bash
# One-command mesh onboarding: detects this Mac's real capacity, registers it
# as TWO self-hosted GitHub Actions runners (one for GPU/mlx training, one for
# CPU-only jobs), and installs both as launchd services.
#
# Two runners, not one: a single runner processes exactly one job at a time, so
# a CPU job would sit blocked behind a multi-hour training run (and vice versa)
# even though they don't compete for the same resource.
#
# Usage:
#   MAT_REPO=<owner>/mat ./scripts/join_network.sh          # interactive token prompt
#   MAT_REPO=<owner>/mat MAT_TOKEN=<t> ./scripts/join_network.sh
set -euo pipefail

REPO="${MAT_REPO:?set MAT_REPO=<owner>/mat (the GitHub repo runners register against)}"
REPO_URL="https://github.com/$REPO"
RUNNER_VERSION="2.325.0"
RUNNER_TARBALL="actions-runner-osx-arm64-${RUNNER_VERSION}.tar.gz"

if [ "$(uname -s)" != "Darwin" ] || [ "$(uname -m)" != "arm64" ]; then
  echo "[FAIL] this script only supports Apple Silicon Macs (uname -m must be arm64)"; exit 1
fi

# ---- 1. Detect real capacity ----------------------------------------------------------
TOTAL_MEM_BYTES=$(sysctl -n hw.memsize)
TOTAL_MEM_GB=$((TOTAL_MEM_BYTES / 1024 / 1024 / 1024))
CPU_TOTAL=$(sysctl -n hw.ncpu)
GPU_CORES=$(system_profiler SPDisplaysDataType 2>/dev/null \
  | grep "Total Number of Cores" | head -1 | grep -oE '[0-9]+' || echo "unknown")
CHIP=$(sysctl -n machdep.cpu.brand_string 2>/dev/null || echo "unknown")

# Safe MLX training budget: unified memory is shared with the OS and every other
# app, and MLX's peak-mem logging under-reports true Metal pressure (OOMs occur
# well below the logged peak). 60% of total, floored at 8GB.
MEM_BUDGET_GB=$(( TOTAL_MEM_GB * 60 / 100 ))
[ "$MEM_BUDGET_GB" -lt 8 ] && MEM_BUDGET_GB=8

echo "[INFO] Detected: $CHIP, ${CPU_TOTAL} CPU cores, ${GPU_CORES} GPU cores, ${TOTAL_MEM_GB}GB unified memory"
echo "[INFO] Safe MLX training memory budget: ${MEM_BUDGET_GB}GB"

# mat's queue reads this as the machine's budget; the path is a fleet
# contract (see machine.py) -- do not rename it.
cat > "$HOME/.mycelium-runner-config" <<EOF
MYCELIUM_MEM_BUDGET_GB=$MEM_BUDGET_GB
MYCELIUM_TOTAL_MEM_GB=$TOTAL_MEM_GB
MYCELIUM_CPU_CORES=$CPU_TOTAL
MYCELIUM_GPU_CORES=$GPU_CORES
EOF
echo "[OK] wrote $HOME/.mycelium-runner-config"

# ---- 2. Prereqs -------------------------------------------------------------------------
if ! command -v uv >/dev/null 2>&1; then
  echo "[INFO] installing uv (needed by worker.yml)..."
  curl -LsSf https://astral.sh/uv/install.sh | sh
fi

TOKEN="${MAT_TOKEN:-}"
if [ -z "$TOKEN" ]; then
  if command -v gh >/dev/null 2>&1 && gh auth status >/dev/null 2>&1; then
    echo "[INFO] minting a registration token via gh..."
    TOKEN=$(gh api -X POST "repos/$REPO/actions/runners/registration-token" -q .token)
  else
    echo "[FAIL] no token. Set MAT_TOKEN, or run 'gh auth login' first, or get one from"
    echo "       $REPO_URL/settings/actions/runners/new (expires in 1h)"
    exit 1
  fi
fi

# ---- 3. Register the two runner instances ------------------------------------------------
setup_runner() {
  local dir="$1" labels="$2" svc_name="$3"
  mkdir -p "$dir" && cd "$dir"
  if [ ! -f config.sh ]; then
    curl -o runner.tar.gz -L "https://github.com/actions/runner/releases/download/v${RUNNER_VERSION}/${RUNNER_TARBALL}"
    tar xzf runner.tar.gz && rm runner.tar.gz
  fi
  if [ -f .runner ]; then
    echo "[INFO] $dir already registered, skipping config.sh (remove .runner to force re-register)"
  else
    ./config.sh --url "$REPO_URL" --token "$TOKEN" --labels "$labels" \
      --name "$(hostname -s)-${svc_name}" --unattended --replace
  fi
  ./svc.sh install && ./svc.sh start
  cd - >/dev/null
}

# `ready` at registration: availability is this label (`mcm drain`/`undrain` toggles it).
echo "[INFO] registering GPU/mlx runner (labels: mlx,ready,gpu-${TOTAL_MEM_GB}gb)..."
setup_runner "$HOME/actions-runner-mlx" "mlx,ready,gpu-${TOTAL_MEM_GB}gb" "mlx"

echo "[INFO] registering CPU-only runner (labels: cpu,ready,cpu-${CPU_TOTAL}core)..."
setup_runner "$HOME/actions-runner-cpu" "cpu,ready,cpu-${CPU_TOTAL}core" "cpu"

# ---- 4. Keep the machine awake while it's part of the mesh -------------------------------
# A sleeping Mac drops its runner mid-job -- looks like a crash, burns an attempt.
CAFFEINATE_PLIST="$HOME/Library/LaunchAgents/com.mycelium.keepawake.plist"
if [ ! -f "$CAFFEINATE_PLIST" ]; then
  echo "[INFO] installing keep-awake agent (caffeinate -is)..."
  cat > "$CAFFEINATE_PLIST" <<PLIST
<?xml version="1.0" encoding="UTF-8"?>
<!DOCTYPE plist PUBLIC "-//Apple//DTD PLIST 1.0//EN" "http://www.apple.com/DTDs/PropertyList-1.0.dtd">
<plist version="1.0">
<dict>
  <key>Label</key><string>com.mycelium.keepawake</string>
  <key>ProgramArguments</key>
  <array><string>/usr/bin/caffeinate</string><string>-is</string></array>
  <key>RunAtLoad</key><true/>
  <key>KeepAlive</key><true/>
</dict>
</plist>
PLIST
  launchctl unload "$CAFFEINATE_PLIST" 2>/dev/null || true
  launchctl load "$CAFFEINATE_PLIST"
  echo "[OK] keep-awake agent installed and running"
else
  echo "[INFO] keep-awake agent already installed, skipping"
fi

cat <<EOF

[OK] Joined the mesh as two independent runners:
  - GPU/training (worker.yml)  -> ~/actions-runner-mlx  (labels: mlx, ready, gpu-${TOTAL_MEM_GB}gb)
  - CPU-only                   -> ~/actions-runner-cpu  (labels: cpu, ready, cpu-${CPU_TOTAL}core)

Retract this machine any time:
  mcm drain     # removes the 'ready' label (no assignment) + busy file (in-flight gate)
  mcm undrain   # rejoin
EOF
