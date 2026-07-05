"""Leak contract: the structural train/eval checks mcm enforces natively."""

from mcm.primitives.dataset import Dataset, leaks


def _ds(name, **kw):
    return Dataset(name=name, uri=f"gs://b/{name}", **kw)


def test_clean_run_has_no_leaks():
    train = [_ds("orgs-train-040726")]
    evals = [_ds("orgs-eval-040726", eval_only=True)]
    assert leaks(train, evals) == []


def test_eval_only_dataset_rejected_as_training_input():
    ds = _ds("orgs-eval-040726", eval_only=True)
    problems = leaks([ds], [])
    assert any("eval_only" in p for p in problems)


def test_same_dataset_in_train_and_eval_flagged():
    ds = _ds("orgs-040726")
    problems = leaks([ds], [ds])
    assert any("both a training input and an eval set" in p for p in problems)


def test_disjoint_from_violation_flagged():
    train = [_ds("orgs-train-040726", disjoint_from=("orgs-eval-040726",))]
    evals = [_ds("orgs-eval-040726", eval_only=True)]
    problems = leaks(train, evals)
    assert any("disjoint_from" in p for p in problems)


def test_disjoint_from_satisfied_when_eval_absent():
    train = [_ds("orgs-train-040726", disjoint_from=("orgs-eval-040726",))]
    evals = [_ds("other-eval-040726", eval_only=True)]
    assert leaks(train, evals) == []


def test_revision_roundtrips_and_old_manifests_still_load():
    ds = _ds("orgs-040726", revision="sha256:abc")
    assert Dataset.from_manifest(ds.to_manifest()).revision == "sha256:abc"
    # A manifest written before `revision` existed must still load.
    legacy = {"name": "old-010125", "uri": "gs://b/old"}
    assert Dataset.from_manifest(legacy).revision == ""
