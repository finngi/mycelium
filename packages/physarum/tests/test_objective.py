"""objective.py is the only place Optuna's suggested scalars become a real
mcm Trial -- these tests pin that translation and the resulting Trial state,
without needing Optuna's sampler machinery or a real producer."""

import optuna
import pytest

from reishi.primitives import trial as trial_store
from reishi.primitives.recipe import Recipe

from physarum.objective import build_recipe, make_trial_fn, resolve_metric, suggest
from physarum.primitives.sweep import Sweep

TEMPLATE = Recipe(
    name="placeholder",
    task="extract-fixture",
    train_dataset="extract-v3",
    base_model="mlx-community/Qwen2.5-7B-Instruct-4bit",
    runtime="mlx",
    prompt="parse: {name}",
    hparams={"iters": 500},
).to_manifest()

SEARCH_SPACE = {
    "hparams.lr": {"type": "loguniform", "low": 1e-6, "high": 1e-4},
    "hparams.rank": {"type": "categorical", "choices": [4, 8, 16, 32]},
}


@pytest.fixture(autouse=True)
def store(tmp_path, monkeypatch):
    monkeypatch.setenv("MCM_STORE", str(tmp_path))


def _study():
    return optuna.create_study(sampler=optuna.samplers.RandomSampler(seed=0))


def test_suggest_covers_every_param_type():
    study = _study()
    ot = study.ask()
    suggested = suggest(ot, SEARCH_SPACE)
    assert set(suggested) == {"hparams.lr", "hparams.rank"}
    assert 1e-6 <= suggested["hparams.lr"] <= 1e-4
    assert suggested["hparams.rank"] in (4, 8, 16, 32)


class _FakeSuggester:
    """A non-Optuna stand-in: proves suggest() only needs this structural surface."""

    number = 0

    def suggest_float(self, name, low, high, *, log=False, step=None):
        return low

    def suggest_int(self, name, low, high, *, step=1):
        return low

    def suggest_categorical(self, name, choices):
        return choices[0]

    def set_user_attr(self, key, value):
        pass

    def report(self, value, step):
        pass

    def should_prune(self):
        return False


def test_suggest_accepts_any_backend_offering_the_suggester_surface():
    suggested = suggest(_FakeSuggester(), SEARCH_SPACE)
    assert suggested == {"hparams.lr": 1e-6, "hparams.rank": 4}


def test_build_recipe_merges_suggestions_over_template_defaults():
    recipe = build_recipe(
        TEMPLATE, {"hparams.lr": 5e-5, "hparams.rank": 16}, "my-sweep", trial_number=3
    )
    assert recipe.name == "my-sweep-t3"
    assert recipe.n_seeds == 1
    assert recipe.hparams == {"iters": 500, "lr": 5e-5, "rank": 16}


def test_make_trial_fn_saves_a_done_trial_and_returns_its_metric():
    sweep = Sweep(
        name="my-sweep",
        template=TEMPLATE,
        search_space=SEARCH_SPACE,
        goal={"metric": "f1", "direction": "maximize"},
    )

    seen_manifests = []

    def fake_producer(manifest):
        seen_manifests.append(manifest)
        return {"metrics": {"f1": 0.87}, "artifacts": {"weights": "/tmp/adapter"}}

    study = _study()
    study.optimize(make_trial_fn(sweep, fake_producer), n_trials=1)

    assert len(seen_manifests) == 1
    assert seen_manifests[0]["spec"]["runtime"] == "mlx"

    trial_id = study.best_trial.user_attrs["mcm_trial_id"]
    saved = trial_store.load(trial_id)
    assert saved.status == "done"
    assert saved.metrics == {"f1": 0.87}
    assert study.best_value == 0.87


def test_make_trial_fn_reports_unknown_metric_clearly():
    sweep = Sweep(
        name="my-sweep",
        template=TEMPLATE,
        search_space=SEARCH_SPACE,
        goal={"metric": "field_f1", "direction": "maximize"},
    )

    def fake_producer(manifest):
        return {"metrics": {"f1": 0.87}, "artifacts": {}}

    with pytest.raises(KeyError, match="field_f1.*available: f1"):
        make_trial_fn(sweep, fake_producer)(_FakeSuggester())


def test_make_trial_fn_marks_trial_pruned_and_excludes_it_from_best():
    sweep = Sweep(
        name="my-sweep",
        template=TEMPLATE,
        search_space=SEARCH_SPACE,
        goal={"metric": "f1", "direction": "minimize"},
    )
    # ThresholdPruner prunes deterministically off a single reported value --
    # no warmup/history needed like Median/Percentile pruners, so it pins the
    # report(step=0)/should_prune() wiring without depending on sampler order.
    study = optuna.create_study(
        direction="minimize",
        sampler=optuna.samplers.RandomSampler(seed=0),
        pruner=optuna.pruners.ThresholdPruner(upper=0.5),
    )
    values = iter([0.9, 0.1])

    def fake_producer(manifest):
        return {"metrics": {"f1": next(values)}, "artifacts": {}}

    study.optimize(make_trial_fn(sweep, fake_producer), n_trials=2, catch=(Exception,))

    assert [t.state for t in study.trials] == [
        optuna.trial.TrialState.PRUNED,
        optuna.trial.TrialState.COMPLETE,
    ]

    pruned_trial_id = study.trials[0].user_attrs["mcm_trial_id"]
    assert trial_store.load(pruned_trial_id).status == "pruned"

    assert study.best_value == 0.1  # the pruned trial's 0.9 never competes for best


def test_make_trial_fn_marks_trial_failed_and_reraises_on_producer_crash():
    # a raising producer must not leave the Trial stuck at "planned" forever
    # (indistinguishable from still running) -- and must still propagate, so
    # the search backend's own per-trial failure handling can take over.
    sweep = Sweep(
        name="my-sweep",
        template=TEMPLATE,
        search_space=SEARCH_SPACE,
        goal={"metric": "f1", "direction": "maximize"},
    )

    def crashing_producer(manifest):
        raise RuntimeError("boom")

    with pytest.raises(RuntimeError, match="boom"):
        make_trial_fn(sweep, crashing_producer)(_FakeSuggester())

    [saved] = trial_store.load_all()
    assert saved.status == "failed"
    assert "boom" in saved.execution["last_error"]


def _namespaces(metrics=None, observables=None):
    return {"metrics": metrics or {}, "observables": observables or {}}


def test_resolve_metric_plain_key_is_a_metrics_lookup():
    assert resolve_metric(_namespaces(metrics={"f1": 0.5}), "f1") == 0.5


def test_resolve_metric_dotted_metrics_path():
    ns = _namespaces(metrics={"field_f1": 0.9})
    assert resolve_metric(ns, "metrics.field_f1") == 0.9


def test_resolve_metric_dotted_observables_path():
    ns = _namespaces(observables={"wall_time_s": 12.0})
    assert resolve_metric(ns, "observables.wall_time_s") == 12.0


def test_resolve_metric_slash_key_is_plain_not_a_path():
    # 'val/field_f1' is record_scoring's split-prefix convention -- the '/' is
    # not a path separator, so it stays a plain lookup in metrics.
    ns = _namespaces(metrics={"val/field_f1": 0.75})
    assert resolve_metric(ns, "val/field_f1") == 0.75


def test_resolve_metric_missing_dotted_path_raises_with_available_keys():
    ns = _namespaces(observables={"wall_time_s": 1.0})
    with pytest.raises(KeyError, match="cost_usd.*available: wall_time_s"):
        resolve_metric(ns, "observables.cost_usd")


SWEEP_WITH_CONSTRAINT_KWARGS = dict(
    name="my-sweep",
    template=TEMPLATE,
    search_space=SEARCH_SPACE,
    goal={"metric": "f1", "direction": "maximize"},
)


def test_make_trial_fn_excludes_infeasible_trial_from_best_even_with_best_raw_value():
    sweep = Sweep(
        **SWEEP_WITH_CONSTRAINT_KWARGS,
        constraints=[{"metric": "cost", "max": 5.0}],
    )
    # Trial 0 has the best raw f1 but blows the cost budget; trial 1 is
    # worse on f1 but feasible, and must win.
    values = iter([(0.99, 10.0), (0.5, 2.0)])

    def fake_producer(manifest):
        f1, cost = next(values)
        return {"metrics": {"f1": f1, "cost": cost}, "artifacts": {}}

    # direction must match sweep.goal's "maximize", or study.best_trial
    # picks against the wrong sense and this test would pass by accident.
    study = optuna.create_study(
        direction="maximize", sampler=optuna.samplers.RandomSampler(seed=0)
    )
    study.optimize(make_trial_fn(sweep, fake_producer), n_trials=2)

    assert [t.user_attrs["mcm_feasible"] for t in study.trials] == [False, True]
    best_trial_id = study.best_trial.user_attrs["mcm_trial_id"]
    assert trial_store.load(best_trial_id).metrics["f1"] == 0.5
    # The infeasible trial's true value is still on its own trial record.
    infeasible_trial_id = study.trials[0].user_attrs["mcm_trial_id"]
    assert trial_store.load(infeasible_trial_id).metrics["f1"] == 0.99


def test_make_trial_fn_min_constraint_direction():
    sweep = Sweep(
        **SWEEP_WITH_CONSTRAINT_KWARGS,
        constraints=[{"metric": "recall", "min": 0.6}],
    )

    def fake_producer(manifest):
        return {"metrics": {"f1": 0.8, "recall": 0.5}, "artifacts": {}}

    study = _study()
    study.optimize(make_trial_fn(sweep, fake_producer), n_trials=1)

    assert study.trials[0].user_attrs["mcm_feasible"] is False


def test_make_trial_fn_missing_constrained_metric_counts_infeasible():
    sweep = Sweep(
        **SWEEP_WITH_CONSTRAINT_KWARGS,
        constraints=[{"metric": "observables.wall_time_s", "max": 60.0}],
    )

    def fake_producer(manifest):
        return {"metrics": {"f1": 0.8}, "artifacts": {}}  # no observables at all

    study = _study()
    study.optimize(make_trial_fn(sweep, fake_producer), n_trials=1)

    assert study.trials[0].user_attrs["mcm_feasible"] is False


def test_make_trial_fn_carries_observables_onto_the_trial():
    sweep = Sweep(**SWEEP_WITH_CONSTRAINT_KWARGS)

    def fake_producer(manifest):
        return {
            "metrics": {"f1": 0.8},
            "observables": {"wall_time_s": 42.0},
            "artifacts": {},
        }

    study = _study()
    study.optimize(make_trial_fn(sweep, fake_producer), n_trials=1)

    trial_id = study.best_trial.user_attrs["mcm_trial_id"]
    assert trial_store.load(trial_id).observables == {"wall_time_s": 42.0}
