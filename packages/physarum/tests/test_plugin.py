"""physarum's face inside the mcm CLI: vocabulary registers cleanly and
`optimize` doesn't collide with `run` (already home to `recipe`)."""

import pytest

from reishi.cli import grammar
from reishi.cli.grammar import canonicalize

from physarum import mcm_plugin


@pytest.fixture(autouse=True)
def sweep_grammar():
    domains, plurals, verbs = (
        grammar.DOMAINS,
        dict(grammar._PLURALS),
        dict(grammar.VERBS),
    )
    grammar.extend(mcm_plugin.DOMAINS, mcm_plugin.VERBS)
    yield
    grammar.DOMAINS = domains
    grammar._PLURALS.clear(), grammar._PLURALS.update(plurals)
    grammar.VERBS.clear(), grammar.VERBS.update(verbs)


def test_sweep_vocabulary_canonicalizes():
    cmd = canonicalize(["sweep", "optimize", "sweep.yaml"])
    assert (cmd.domain, cmd.action, cmd.objects) == (
        "sweep",
        "optimize",
        ["sweep.yaml"],
    )
    assert canonicalize(["sweeps"]).action == "list"


def test_every_verb_has_a_handler():
    for v in mcm_plugin.VERBS:
        assert ("sweep", v.name) in mcm_plugin.HANDLERS


def test_resolve_producer_rejects_unwired_runtime():
    with pytest.raises(ValueError, match="l4"):
        mcm_plugin._resolve_producer("l4")


def test_resolve_sampler_rejects_unknown_backend():
    with pytest.raises(ValueError, match="ouija"):
        mcm_plugin._resolve_sampler("ouija", {})


def test_resolve_sampler_grid_covers_every_categorical_combination():
    import optuna

    search_space = {
        "hparams.a": {"type": "categorical", "choices": [True, False]},
        "hparams.b": {"type": "categorical", "choices": [True, False]},
    }
    sampler = mcm_plugin._resolve_sampler("grid", search_space)
    assert isinstance(sampler, optuna.samplers.GridSampler)

    def objective(t):
        t.suggest_categorical("hparams.a", [True, False])
        t.suggest_categorical("hparams.b", [True, False])
        return 0.0

    study = optuna.create_study(sampler=sampler)
    study.optimize(objective, n_trials=10)
    seen = {(t.params["hparams.a"], t.params["hparams.b"]) for t in study.trials}
    assert seen == {(True, True), (True, False), (False, True), (False, False)}


def test_grid_search_space_rejects_unbounded_float_param():
    with pytest.raises(ValueError, match="learning_rate"):
        mcm_plugin._grid_search_space(
            {"hparams.learning_rate": {"type": "float", "low": 0.0, "high": 1.0}}
        )


def test_grid_search_space_enumerates_stepped_int_param():
    grid = mcm_plugin._grid_search_space(
        {"hparams.n": {"type": "int", "low": 1, "high": 5, "step": 2}}
    )
    assert grid == {"hparams.n": [1, 3, 5]}


def test_flag_value_accepts_space_and_equals_forms():
    assert mcm_plugin._flag_value(["--port", "9999"], "--port") == "9999"
    assert mcm_plugin._flag_value(["--port=9999"], "--port") == "9999"
    assert mcm_plugin._flag_value([], "--port") is None


def test_sweep_watch_rejects_non_numeric_port():
    from reishi.cli.grammar import Command

    cmd = Command(
        domain="sweep", action="watch", objects=["my-sweep"], flags=["--port", "nope"]
    )
    assert mcm_plugin.sweep_watch(cmd) == 1


def test_progress_callback_denominator_is_sweep_total_not_trials_so_far(capsys):
    import optuna

    optuna.logging.set_verbosity(optuna.logging.WARNING)
    study = optuna.create_study(direction="maximize")
    # n_trials=3 < total=5: len(study.trials) would equal trial.number + 1 (1, 2, 3)
    # at callback time regardless -- only reading `total` catches the old "K/K" bug
    study.optimize(
        lambda t: t.suggest_float("x", 0, 1),
        n_trials=3,
        callbacks=[mcm_plugin._make_progress_callback(total=5)],
    )

    lines = [
        line
        for line in capsys.readouterr().err.splitlines()
        if line.startswith("[INFO] trial")
    ]
    assert [line.split()[2] for line in lines] == ["1/5", "2/5", "3/5"]
