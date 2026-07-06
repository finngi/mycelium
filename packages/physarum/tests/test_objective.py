"""objective.py is the only place Optuna's suggested scalars become a real
mcm Trial -- these tests pin that translation and the resulting Trial state,
without needing Optuna's sampler machinery or a real trainer."""

import optuna
import pytest

import reishi.tasks  # noqa: F401  (populate the task registry)
from reishi.primitives import trial as trial_store
from reishi.primitives.recipe import Recipe

from physarum.objective import build_recipe, make_objective, suggest
from physarum.primitives.sweep import Sweep

TEMPLATE = Recipe(
    name="placeholder",
    task="nameparse",
    dataset="nameparse-v3",
    base_model="mlx-community/Qwen2.5-7B-Instruct-4bit",
    accelerator="mlx",
    prompt="parse: {name}",
    trainer={"iters": 500},
).to_manifest()

SEARCH_SPACE = {
    "trainer.lr": {"type": "loguniform", "low": 1e-6, "high": 1e-4},
    "trainer.rank": {"type": "categorical", "choices": [4, 8, 16, 32]},
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
    assert set(suggested) == {"trainer.lr", "trainer.rank"}
    assert 1e-6 <= suggested["trainer.lr"] <= 1e-4
    assert suggested["trainer.rank"] in (4, 8, 16, 32)


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


def test_suggest_accepts_any_backend_offering_the_suggester_surface():
    suggested = suggest(_FakeSuggester(), SEARCH_SPACE)
    assert suggested == {"trainer.lr": 1e-6, "trainer.rank": 4}


def test_build_recipe_merges_suggestions_over_template_defaults():
    recipe = build_recipe(TEMPLATE, {"trainer.lr": 5e-5, "trainer.rank": 16}, "my-sweep", trial_number=3)
    assert recipe.name == "my-sweep-t3"
    assert recipe.seeds == 1
    assert recipe.trainer == {"iters": 500, "lr": 5e-5, "rank": 16}


def test_make_objective_saves_a_done_trial_and_returns_its_metric():
    sweep = Sweep(
        name="my-sweep",
        template=TEMPLATE,
        search_space=SEARCH_SPACE,
        objective={"metric": "f1", "direction": "maximize"},
    )

    seen_manifests = []

    def fake_trainer(manifest):
        seen_manifests.append(manifest)
        return {"metrics": {"f1": 0.87}, "artifacts": {"weights": "/tmp/adapter"}}

    study = _study()
    study.optimize(make_objective(sweep, fake_trainer), n_trials=1)

    assert len(seen_manifests) == 1
    assert seen_manifests[0]["spec"]["accelerator"] == "mlx"

    trial_id = study.best_trial.user_attrs["mcm_trial_id"]
    saved = trial_store.load(trial_id)
    assert saved.status == "done"
    assert saved.metrics == {"f1": 0.87}
    assert study.best_value == 0.87


def test_make_objective_reports_unknown_metric_clearly():
    sweep = Sweep(
        name="my-sweep",
        template=TEMPLATE,
        search_space=SEARCH_SPACE,
        objective={"metric": "field_f1", "direction": "maximize"},
    )

    def fake_trainer(manifest):
        return {"metrics": {"f1": 0.87}, "artifacts": {}}

    with pytest.raises(KeyError, match="field_f1.*available: f1"):
        make_objective(sweep, fake_trainer)(_FakeSuggester())


def test_make_objective_marks_trial_failed_and_reraises_on_trainer_crash():
    # a raising trainer must not leave the Trial stuck at "planned" forever
    # (indistinguishable from still running) -- and must still propagate, so
    # the search backend's own per-trial failure handling can take over.
    sweep = Sweep(
        name="my-sweep",
        template=TEMPLATE,
        search_space=SEARCH_SPACE,
        objective={"metric": "f1", "direction": "maximize"},
    )

    def crashing_trainer(manifest):
        raise RuntimeError("boom")

    with pytest.raises(RuntimeError, match="boom"):
        make_objective(sweep, crashing_trainer)(_FakeSuggester())

    [saved] = trial_store.load_all()
    assert saved.status == "failed"
    assert "boom" in saved.execution["last_error"]
