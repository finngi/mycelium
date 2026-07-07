"""Store-over-git: the mcm store lives inside a git checkout, so claims and
results are commits and the queue is transactional without a server.

sync() before deciding, publish() after changing. publish pushes with a
pull-rebase retry; if OUR change conflicts during rebase we lost a claim
race -- abort, drop our commit, tell the caller to re-pick. Outside a git
checkout (tests, local dev) both are no-ops.
"""

import os
import subprocess
import sys

from reishi import store


def _branch() -> str:
    return os.environ.get("OYSTER_BRANCH", "main")


def _git(*args: str, check: bool = True) -> subprocess.CompletedProcess:
    return subprocess.run(
        ["git", "-C", str(store.root()), *args],
        capture_output=True, text=True, check=check,
    )


def _in_repo_with_remote() -> bool:
    try:
        r = _git("rev-parse", "--is-inside-work-tree", check=False)
        if r.returncode != 0 or r.stdout.strip() != "true":
            return False
        return bool(_git("remote", check=False).stdout.strip())
    except FileNotFoundError:
        return False


def sync() -> None:
    if not _in_repo_with_remote():
        return
    r = _git("pull", "--rebase", "--autostash", "origin", _branch(), check=False)
    if r.returncode != 0:
        print(f"[WARN] store sync failed: {r.stderr.strip()}", file=sys.stderr)


def _reset_to_origin() -> None:
    _git("reset", "--hard", f"origin/{_branch()}", check=False)


def publish(message: str) -> bool:
    """Commit the store and push. False means we lost a race: our commit was
    dropped and the store re-synced -- re-read before acting again."""
    if not _in_repo_with_remote():
        return True
    _git("add", "-A", ".")
    if _git("diff", "--cached", "--quiet", check=False).returncode == 0:
        return True
    _git("commit", "-m", message)
    for _ in range(5):
        if _git("push", "origin", f"HEAD:{_branch()}", check=False).returncode == 0:
            return True
        r = _git("pull", "--rebase", "--autostash", "origin", _branch(), check=False)
        if r.returncode != 0:
            _git("rebase", "--abort", check=False)
            _reset_to_origin()
            return False
    print("[WARN] push failed after 5 rebase retries", file=sys.stderr)
    # A push can fail for reasons no amount of rebasing fixes (e.g. a protected
    # branch rejecting a direct commit) -- leaving our unpushed commit in place
    # would make this trial look locally claimed forever, even though no claim
    # ever reached origin. Reset so the next sync sees the true upstream state.
    _reset_to_origin()
    return False
