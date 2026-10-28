"""On-policy signs, terminal precedence, final-observation bootstrap and resume."""

import json
from pathlib import Path

import pytest

np = pytest.importorskip("numpy")

from hnef.engine import GameState, Outcome, Player, initial_state  # noqa: E402
from hnef.learning.copenhagen import CODEC, RULESET  # noqa: E402
from hnef.learning.onpolicy import (  # noqa: E402
    ALGORITHM,
    collect,
    evaluate,
    features,
    gradients,
    starts,
    target,
    trace_state,
    train,
    update,
)
from hnef.learning.plumbing import LinearModel, contract, load  # noqa: E402

DATA = Path(__file__).parent / "fixtures/copenhagen-training-plumbing-v1/cases.json"
HELDOUT = DATA.parents[1] / "copenhagen-heldout-v1/cases.json"


@pytest.fixture(autouse=True)
def cpu_caps(monkeypatch):
    for key in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS"):
        monkeypatch.setenv(key, "1")


def test_terminal_targets_override_bootstrap_and_preserve_perspective():
    win = Outcome(Player.ATTACKERS, "king_captured")
    assert target(win, Player.ATTACKERS, -0.8, 0.99) == 1
    assert target(win, Player.DEFENDERS, 0.8, 0.99) == -1
    assert target(Outcome(None, "adjudicated_draw"), Player.DEFENDERS, 0.8, 0.99) == 0
    assert target(None, Player.DEFENDERS, -0.8, 0.5) == -0.4


def test_cutoff_bootstraps_final_same_player_position_not_reset_or_draw():
    model = LinearModel()
    model.critic[:] = np.random.default_rng(123).normal(0, 0.02, len(model.critic))
    initial = initial_state(RULESET)
    samples, trace = collect(model, initial, np.random.default_rng(2), 3, 1, 0.99)
    final = trace_state(trace)
    assert final.ply == 2 and final.player is initial.player
    assert final.key != initial.key
    assert trace["truncated"] and not trace["terminated"] and trace["reward"] is None
    nx, nm = features(final)
    expected = float(model.forward(nx[None], nm[None])[1][0])
    ix, im = features(initial)
    reset_value = float(model.forward(ix[None], im[None])[1][0])
    assert expected != pytest.approx(reset_value)
    assert trace["bootstrap_value"] == pytest.approx(expected)
    assert samples[-1][-1] == pytest.approx(0.99 * expected)
    with pytest.raises(ValueError, match="Trace result"):
        trace_state({**trace, "reward": 0})


def test_terminal_on_opponent_reply_has_negative_reward(monkeypatch):
    import hnef.learning.onpolicy as module
    from hnef.engine import Move

    board = [list(row) for row in starts(DATA)[0][2].board]
    initial = GameState(tuple("".join(row) for row in board), Player.DEFENDERS, ruleset=RULESET)
    own_action = CODEC.encode(Move((6, 4), (6, 5)))

    class FixedModel:
        def forward(self, x, mask):
            probs = np.zeros_like(mask, dtype=float)
            assert mask[0, own_action]
            probs[0, own_action] = 1
            return probs, np.array([0.7])

    class FixedReply:
        def __init__(self, seed):
            pass

        def choose_move(self, state):
            return Move((7, 8), (3, 8))

    monkeypatch.setattr(module, "RandomPolicy", FixedReply)
    samples, trace = collect(FixedModel(), initial, np.random.default_rng(1), 3, 1, 0.99)
    assert trace["terminated"] and not trace["truncated"]
    assert trace["reason"] == "king_captured" and trace["reward"] == -1
    assert trace["bootstrap_value"] == 0 and samples[0][-1] == -1
    assert trace_state(trace).outcome.winner is Player.ATTACKERS


def test_negative_advantage_reduces_selected_probability_and_gradient_matches_difference():
    model = LinearModel()
    x, mask = features(initial_state(RULESET))
    action = int(np.flatnonzero(mask)[0])
    sample = [(x, mask, action, 0.0, -1.0)]
    ag, cg, metrics = gradients(model, sample)
    before = model.forward(x[None], mask[None])[0][0, action]
    # Frozen target/baseline actor surrogate derivative at the bias weight.
    epsilon = 1e-6
    model.actor[-1, action] = epsilon
    plus = np.log(model.forward(x[None], mask[None])[0][0, action])
    model.actor[-1, action] = -epsilon
    minus = np.log(model.forward(x[None], mask[None])[0][0, action])
    assert ag[-1, action] == pytest.approx((plus - minus) / (2 * epsilon), rel=1e-6)
    assert cg[-1] == pytest.approx(1.0) and np.isfinite(metrics["loss"])
    model.actor[-1, action] = 0
    update(model, sample, 0.05)
    after = model.forward(x[None], mask[None])[0][0, action]
    assert after < before
    assert np.all(model.forward(x[None], mask[None])[0][0, ~mask] == 0)


def test_resume_and_heldout_evaluation_do_not_consume_training_rng(tmp_path):
    whole = train(tmp_path / "whole", DATA, episodes=8, horizon=4)
    first = train(tmp_path / "first", DATA, episodes=4, horizon=4)
    resumed = train(
        tmp_path / "resumed",
        DATA,
        episodes=4,
        horizon=4,
        resume=tmp_path / "first" / first["checkpoint"],
    )
    expected = {**contract(starts(DATA)[1]), "schema": 2, "algorithm": ALGORITHM}
    path = tmp_path / "whole" / whole["checkpoint"]
    a, am, ar = load(path, expected)
    b, bm, br = load(tmp_path / "resumed" / resumed["checkpoint"], expected)
    assert np.any(a.actor != 0) and np.any(a.critic != 0)
    assert np.array_equal(a.actor, b.actor) and np.array_equal(a.critic, b.critic)
    assert am == bm and ar.bit_generator.state == br.bit_generator.state
    report = evaluate(path, DATA, HELDOUT, tmp_path / "eval", seeds=(100000,))
    assert len(report["records"]) == 12
    assert len({record["trace"] for record in report["records"]}) == 12
    saved = {p.name: p.read_bytes() for p in (tmp_path / "eval").iterdir()}
    with pytest.raises(FileExistsError):
        evaluate(path, DATA, HELDOUT, tmp_path / "eval", seeds=(100000,))
    assert saved == {p.name: p.read_bytes() for p in (tmp_path / "eval").iterdir()}
    for record in report["records"]:
        trace = json.loads((tmp_path / "eval" / record["trace"]).read_text())
        end = trace_state(trace)
        assert (end.outcome is None) == record["truncated"]
    _, after, _ = load(path, expected)
    assert am == after
    with pytest.raises(ValueError, match="contract mismatch"):
        load(path, contract(starts(DATA)[1]))  # Supervised schema cannot resume TD.
    with pytest.raises(ValueError, match="symmetry overlap"):
        evaluate(path, DATA, DATA, tmp_path / "leak")
    with pytest.raises(ValueError, match="contract mismatch"):
        train(tmp_path / "bad-resume", DATA, episodes=1, horizon=3, resume=path)


def test_own_terminal_at_horizon_does_not_request_opponent_reply(monkeypatch):
    import hnef.learning.onpolicy as module
    from hnef.engine import Move

    initial = starts(DATA)[0][0]
    winning_action = CODEC.encode(Move((0, 6), (0, 10)))

    class FixedModel:
        def forward(self, x, mask):
            probs = np.zeros_like(mask, dtype=float)
            probs[0, winning_action] = 1
            return probs, np.array([-0.9])

    class NoReply:
        def __init__(self, seed):
            pass

        def choose_move(self, state):
            raise AssertionError("Opponent must not move after a win")

    monkeypatch.setattr(module, "RandomPolicy", NoReply)
    samples, trace = collect(FixedModel(), initial, np.random.default_rng(1), 3, 1, 0.99)
    assert trace["terminated"] and not trace["truncated"]
    assert trace["reward"] == 1 and trace["bootstrap_value"] == 0
    assert samples[0][-1] == 1 and len(trace["actions"]) == 1


@pytest.mark.parametrize("seeds", [(100000, 100000), [100000, 100001, 100000]])
def test_duplicate_evaluation_seeds_fail_before_loading_or_writing(tmp_path, seeds):
    output = tmp_path / "evaluation"
    missing = tmp_path / "missing"
    with pytest.raises(ValueError, match="Evaluation seeds must be unique"):
        evaluate(missing, missing, missing, output, seeds=seeds)
    assert not output.exists()


def test_reference_and_side_schedule_are_bound_to_resume(tmp_path):
    settings = dict(reference_id="heuristic-v1", both_sides=True, horizon=2)
    whole = train(tmp_path / "whole", DATA, episodes=4, **settings)
    part = train(tmp_path / "part", DATA, episodes=2, **settings)
    checkpoint = tmp_path / "part" / part["checkpoint"]
    resumed = train(tmp_path / "resumed", DATA, episodes=2, resume=checkpoint, **settings)
    expected = {**contract(starts(DATA)[1]), "schema": 2, "algorithm": ALGORITHM}
    a, am, ar = load(tmp_path / "whole" / whole["checkpoint"], expected)
    b, bm, br = load(tmp_path / "resumed" / resumed["checkpoint"], expected)
    assert np.array_equal(a.actor, b.actor) and np.array_equal(a.critic, b.critic)
    assert am == bm and ar.bit_generator.state == br.bit_generator.state
    assert [r["learner"] for r in whole["records"]] == ["attackers", "defenders"] * 2
    for r in whole["records"]:
        trace = json.loads((tmp_path / "whole" / r["trace"]).read_text())
        assert trace["opponent"] == "heuristic-v1"
        trace_state(trace)
    for changed in ({"reference_id": "random-v1"}, {"both_sides": False}):
        with pytest.raises(ValueError, match="contract mismatch"):
            train(tmp_path / "bad", DATA, episodes=1, resume=checkpoint, **(settings | changed))
        assert not (tmp_path / "bad").exists()


def test_multi_opponent_evaluation_pairs_streams_without_mutating_checkpoint(tmp_path):
    report = train(tmp_path / "train", DATA, episodes=2, horizon=1)
    checkpoint = tmp_path / "train" / report["checkpoint"]
    original = checkpoint.read_bytes()
    refs = ("random-v1", "heuristic-v1", "mcts-v1-i4-r4")
    result = evaluate(
        checkpoint, DATA, HELDOUT, tmp_path / "eval", seeds=(100003,), references=refs, horizon=2
    )
    assert result["horizon"] == 2 and result["training_config"]["horizon"] == 1
    assert len(result["records"]) == 36
    assert len({r["trace"] for r in result["records"]}) == 36
    assert checkpoint.read_bytes() == original
    for record in result["records"]:
        peer = next(
            r
            for r in result["records"]
            if r["model"] != record["model"]
            and (r["opponent"], r["start_index"], r["seed"])
            == (record["opponent"], record["start_index"], record["seed"])
        )
        assert peer["learner_seed"] == record["learner_seed"]
        assert peer["opponent_seed"] == record["opponent_seed"]
        trace = json.loads((tmp_path / "eval" / record["trace"]).read_text())
        assert trace["opponent"] == record["opponent"]
        trace_state(trace)
    assert len(result["paired"]) == 6


@pytest.mark.parametrize(
    "kwargs",
    [
        {"references": ("heuristic-v1", "heuristic-v1")},
        {"references": ()},
        {"references": ("mcts-v1-i64-r48",)},
        {"horizon": 33},
        {"horizon": True},
        {"horizon": 0},
    ],
)
def test_invalid_evaluation_budget_or_reference_fails_before_output(tmp_path, kwargs):
    missing = tmp_path / "missing"
    with pytest.raises(ValueError):
        evaluate(missing, missing, missing, tmp_path / "out", **kwargs)
    assert not (tmp_path / "out").exists()


def test_censored_pair_bounds_do_not_invent_draws_or_drop_losses():
    from hnef.learning.onpolicy import evaluation_summary

    records = [
        {
            "model": model,
            "opponent": "heuristic-v1",
            "side": "attackers",
            "start_index": i,
            "seed": 100000,
            "reward": reward,
            "reason": "king_captured" if reward is not None else None,
        }
        for model, rewards in (("initialized", [None, 1, -1, 0]), ("candidate", [1, None, -1, 0]))
        for i, reward in enumerate(rewards)
    ]
    summary, paired = evaluation_summary(records)
    assert summary[0]["results"] == {"win": 1, "loss": 1, "draw": 1, "unfinished": 1}
    assert summary[0]["cutoff_rate"] == 0.25
    assert summary[0]["win_rate_bounds"] == [0.25, 0.5]
    assert paired[0]["win_rate_delta_bounds"] == [-0.25, 0.25]
    with pytest.raises(ValueError, match="Unpaired"):
        evaluation_summary(records[:-1])
    with pytest.raises(ValueError, match="Duplicate"):
        evaluation_summary(records + records[:1])
