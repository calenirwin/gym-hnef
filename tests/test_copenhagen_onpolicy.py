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
