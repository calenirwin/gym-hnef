"""Synthetic tactics: exact witnesses, observed gradients, and disjoint splits."""

import json
from pathlib import Path

import pytest

np = pytest.importorskip("numpy")

from hnef.engine import GameState, Move, Player, apply_move, legal_moves  # noqa: E402
from hnef.learning.copenhagen import RULESET  # noqa: E402
from hnef.learning.onpolicy import (  # noqa: E402
    ALGORITHM,
    starts,
    symmetric_keys,
    trace_state,
    train,
    training_reference,
    update_coverage,
)
from hnef.learning.plumbing import contract, load  # noqa: E402

FIXTURES = Path(__file__).parent / "fixtures"
TRAIN = FIXTURES / "copenhagen-training-tactics-v1/cases.json"
BROAD = FIXTURES / "copenhagen-training-broad-v1/cases.json"
VALIDATION = FIXTURES / "copenhagen-validation-tactics-v1/cases.json"


@pytest.fixture(autouse=True)
def cpu_caps(monkeypatch):
    for key in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS"):
        monkeypatch.setenv(key, "1")


@pytest.mark.parametrize(
    "path", [TRAIN, BROAD, VALIDATION], ids=["training", "broad", "validation"]
)
def test_tactical_witnesses_are_legal_terminal_sequences_not_training_labels(path):
    cases = json.loads(path.read_text())["cases"]
    assert len({c["id"] for c in cases}) == len(cases) == 8
    assert {(c["player"], c["expected_reward"]) for c in cases} == {
        (side.value, reward) for side in Player for reward in (-1, 1)
    }
    for case in cases:
        assert case["source"] == "https://aagenielsen.dk/copenhagen_rules.php"
        assert case["ruleset"] == RULESET.value and case["rule"]
        state = GameState(tuple(case["board"]), Player(case["player"]), ruleset=RULESET)
        legal = legal_moves(state)
        assert len(legal) == case["initial_legal_moves"] >= 2
        winning = 0
        for move in legal:
            successor = apply_move(state, move)
            winning += successor.outcome is not None and successor.outcome.winner is state.player
        assert winning == case["initial_winning_moves"]
        for coordinates in case["witness"]:
            assert state.outcome is None
            move = Move(*coordinates)
            assert move in legal_moves(state)
            state = apply_move(state, move)
        assert state.outcome.reason == case["expected_reason"]
        assert state.outcome.reward(Player(case["player"])) == case["expected_reward"]


def test_new_validation_is_disjoint_from_all_training_and_previous_diagnostics():
    forbidden = set()
    for path in (
        TRAIN,
        BROAD,
        FIXTURES / "copenhagen-training-plumbing-v1/cases.json",
        FIXTURES / "copenhagen-heldout-v1/cases.json",
    ):
        for state in starts(path)[0]:
            forbidden.update(symmetric_keys(state.board))
    assert all(state.board not in forbidden for state in starts(VALIDATION)[0])
    # New training also cannot reuse old public held-out positions.
    old_heldout = {
        key
        for state in starts(FIXTURES / "copenhagen-heldout-v1/cases.json")[0]
        for key in symmetric_keys(state.board)
    }
    assert all(state.board not in old_heldout for state in starts(TRAIN)[0] + starts(BROAD)[0])


def test_training_does_not_consume_witness_annotations(tmp_path):
    original = json.loads(TRAIN.read_text())
    for case in original["cases"]:
        for field in ("witness", "expected_reward", "expected_reason", "initial_winning_moves"):
            case[field] = "not a training target"
    path = tmp_path / "without-labels.json"
    path.write_text(json.dumps(original))
    assert starts(path)[0] == starts(TRAIN)[0]


def test_mixed_schedule_crosses_every_start_and_both_sides():
    positions = starts(TRAIN)[0]
    assignments = [
        (
            index % len(positions),
            training_reference("mixed-random-heuristic-v1", index, len(positions)),
        )
        for index in range(16)
    ]
    assert set(assignments) == {(i, ref) for i in range(8) for ref in ("random-v1", "heuristic-v1")}
    for ref in ("random-v1", "heuristic-v1"):
        assert [positions[i].player for i, opponent in assignments if opponent == ref].count(
            Player.ATTACKERS
        ) == 4
        assert [positions[i].player for i, opponent in assignments if opponent == ref].count(
            Player.DEFENDERS
        ) == 4


def test_mixed_resume_crosses_reference_boundary_and_reports_actual_opponent(tmp_path):
    config = dict(reference_id="mixed-random-heuristic-v1", horizon=1, seed=17)
    whole = train(tmp_path / "whole", TRAIN, episodes=10, **config)
    part = train(tmp_path / "part", TRAIN, episodes=7, **config)
    resumed = train(
        tmp_path / "resume",
        TRAIN,
        episodes=3,
        resume=tmp_path / "part" / part["checkpoint"],
        **config,
    )
    expected = {**contract(starts(TRAIN)[1]), "schema": 2, "algorithm": ALGORITHM}
    a, am, ar = load(tmp_path / "whole" / whole["checkpoint"], expected)
    b, bm, br = load(tmp_path / "resume" / resumed["checkpoint"], expected)
    assert np.array_equal(a.actor, b.actor) and np.array_equal(a.critic, b.critic)
    assert am == bm and ar.bit_generator.state == br.bit_generator.state
    assert [r["opponent"] for r in whole["records"]] == ["random-v1"] * 8 + ["heuristic-v1"] * 2
    exposure = {(r["opponent"], r["side"]): r["episodes"] for r in whole["reference_side_coverage"]}
    assert exposure == {
        (ref, side.value): count
        for ref, count in (("random-v1", 4), ("heuristic-v1", 1))
        for side in Player
    }
    for row in whole["records"]:
        trace = json.loads((tmp_path / "whole" / row["trace"]).read_text())
        assert trace["opponent"] == row["opponent"]
        trace_state(trace)
    with pytest.raises(ValueError, match="contract mismatch"):
        train(
            tmp_path / "changed",
            TRAIN,
            episodes=1,
            resume=tmp_path / "part" / part["checkpoint"],
            **(config | {"reference_id": "heuristic-v1"}),
        )


def test_coverage_counts_observed_actor_and_critic_updates_not_unfinished_labels():
    rows = [
        {
            "learner": "attackers",
            "reward": reward,
            "actor_gradient_norm": actor,
            "critic_gradient_norm": critic,
        }
        for reward, actor, critic in [(1, 2, 3), (-1, 0, 2), (None, 3, 4), (0, 1, 1)]
    ]
    assert update_coverage(rows) == [
        {
            "side": "attackers",
            "result": "win",
            "episodes": 1,
            "nonzero_actor_updates": 1,
            "nonzero_critic_updates": 1,
        },
        {
            "side": "attackers",
            "result": "loss",
            "episodes": 1,
            "nonzero_actor_updates": 0,
            "nonzero_critic_updates": 1,
        },
        {
            "side": "defenders",
            "result": "win",
            "episodes": 0,
            "nonzero_actor_updates": 0,
            "nonzero_critic_updates": 0,
        },
        {
            "side": "defenders",
            "result": "loss",
            "episodes": 0,
            "nonzero_actor_updates": 0,
            "nonzero_critic_updates": 0,
        },
    ]
