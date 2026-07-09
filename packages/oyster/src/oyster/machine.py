"""This machine: capacity, identity, and availability.

Availability is two-layered, and the order matters. The runner's `ready`
label gates *assignment*: with it removed, GitHub never hands this machine
a job, so there is no pickup/refuse/retry cycle to burn attempts. The busy
file gates *execution*: it catches only jobs assigned before the label came
off, failing them fast instead of training on a machine someone is using.
Drain toggles both; the label is the mechanism, the file is the backstop.

The ~/.mycelium-* paths are a fleet contract: every onboarded Mac already
has them, so renaming the files orphans the fleet.
"""

import os
import socket
import subprocess
from pathlib import Path

BUSY_FILE = Path.home() / ".mycelium-busy"
CONFIG_FILE = Path.home() / ".mycelium-runner-config"
FALLBACK_MEM_BUDGET_GB = 40.0

_TRAINING_PATTERN = r"mlx_lm|bench\.py|train\.py"


def name() -> str:
    # Short hostname, because runners register as "<hostname -s>-mlx" --
    # this is the join key for the label API.
    return os.environ.get("OYSTER_RUNNER_NAME") or socket.gethostname().split(".")[0]


def mem_budget_gb() -> float:
    env = os.environ.get("OYSTER_MEM_BUDGET_GB")
    if env:
        return float(env)
    if CONFIG_FILE.exists():
        for line in CONFIG_FILE.read_text().splitlines():
            if line.startswith("MYCELIUM_MEM_BUDGET_GB="):
                try:
                    return float(line.split("=", 1)[1])
                except ValueError:
                    break
    return FALLBACK_MEM_BUDGET_GB


def is_busy() -> bool:
    return BUSY_FILE.exists()


def training_process_running() -> bool:
    """A training process already on the GPU means claiming would double-load
    the same unified memory, whatever the labels say."""
    r = subprocess.run(
        ["pgrep", "-f", _TRAINING_PATTERN], capture_output=True, text=True
    )
    return r.returncode == 0 and bool(r.stdout.strip())


def _gh(*args: str) -> subprocess.CompletedProcess:
    return subprocess.run(["gh", *args], capture_output=True, text=True)


def _repo() -> str | None:
    if os.environ.get("OYSTER_REPO"):
        return os.environ["OYSTER_REPO"]
    r = _gh("repo", "view", "--json", "nameWithOwner", "-q", ".nameWithOwner")
    return r.stdout.strip() if r.returncode == 0 and r.stdout.strip() else None


def set_ready(ready: bool) -> bool:
    """Toggle this host's mlx runner's `ready` label -- assignment-level
    availability, no re-registration needed. Returns False when gh, the
    repo, or the runner can't be reached; callers fall back to the busy
    file rather than failing the drain."""
    repo = _repo()
    if repo is None:
        return False
    r = _gh(
        "api",
        f"repos/{repo}/actions/runners",
        "--paginate",
        "--jq",
        f'.runners[] | select((.labels[].name=="mlx") and '
        f'(.name | startswith("{name()}"))) | .id',
    )
    runner_id = (
        r.stdout.strip().splitlines()[0]
        if r.returncode == 0 and r.stdout.strip()
        else None
    )
    if runner_id is None:
        return False
    if ready:
        r = _gh(
            "api",
            "-X",
            "POST",
            f"repos/{repo}/actions/runners/{runner_id}/labels",
            "-f",
            "labels[]=ready",
        )
    else:
        r = _gh(
            "api",
            "-X",
            "DELETE",
            f"repos/{repo}/actions/runners/{runner_id}/labels/ready",
        )
    return r.returncode == 0
