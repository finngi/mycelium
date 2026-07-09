"""publish()'s job is to make the local store match origin exactly whenever
it can't land a commit there -- otherwise a rejected push leaves a phantom
local claim that nothing upstream agrees with."""

import subprocess

import pytest

from oyster import gitstore


def _git(*args, cwd):
    return subprocess.run(
        ["git", "-C", str(cwd), *args], capture_output=True, text=True, check=True
    )


@pytest.fixture
def repo_with_rejecting_remote(tmp_path, monkeypatch):
    """A clone whose remote refuses every push, like a branch-protected main."""
    origin = tmp_path / "origin.git"
    origin.mkdir()
    _git("init", "--bare", "-b", "main", cwd=origin)

    work = tmp_path / "work"
    _git("clone", str(origin), str(work), cwd=tmp_path)
    # gitstore.publish() commits with no identity flags of its own, so the
    # clone needs one configured locally -- CI runners have no global git
    # identity the way a dev machine typically does
    _git("config", "user.email", "t@t", cwd=work)
    _git("config", "user.name", "t", cwd=work)
    (work / "seed.txt").write_text("seed\n")
    _git("add", "-A", cwd=work)
    _git("commit", "-m", "seed", cwd=work)
    _git("push", "origin", "main", cwd=work)

    # only reject pushes from here on, like a branch-protected main -- the
    # seed commit above needs to land so origin and work start in sync
    hook = origin / "hooks" / "pre-receive"
    hook.write_text("#!/bin/sh\nexit 1\n")
    hook.chmod(0o755)

    monkeypatch.setenv("MCM_STORE", str(work))
    return origin, work


def test_publish_resets_local_state_when_push_is_permanently_rejected(
    repo_with_rejecting_remote,
):
    origin, work = repo_with_rejecting_remote
    origin_head = _git("rev-parse", "HEAD", cwd=origin).stdout.strip()

    (work / "trial.json").write_text('{"status": "running"}\n')
    ok = gitstore.publish("claim trial-1")

    assert ok is False
    assert _git("rev-parse", "HEAD", cwd=work).stdout.strip() == origin_head
    assert not (work / "trial.json").exists()
