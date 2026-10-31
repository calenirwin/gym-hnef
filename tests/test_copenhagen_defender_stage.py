"""Intermediate defender stage: exact tactics, separation and resumed scheduling."""

import json
from collections import Counter
from pathlib import Path

import pytest

np = pytest.importorskip("numpy")

from hnef.engine import GameState, Move, Player, apply_move, legal_moves  # noqa: E402
from hnef.learning.copenhagen import RULESET  # noqa: E402
from hnef.learning.curves import training_positions  # noqa: E402
from hnef.learning.onpolicy import (  # noqa: E402
    ALGORITHM,
    curriculum_choice,
    curriculum_groups,
    starts,
    symmetric_keys,
    train,
    training_reference,
)
from hnef.learning.plumbing import contract, load  # noqa: E402

ROOT = Path(__file__).parent / "fixtures"
DATA = ROOT / "copenhagen-training-defender-staged-v1/cases.json"
SCHEDULE = "constrained16-intermediate16-broad32-v1"


@pytest.fixture(autouse=True)
def cpu_caps(monkeypatch):
    for key in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS"):
        monkeypatch.setenv(key, "1")


def test_intermediate_changes_only_defender_escape_branching_and_preserves_witnesses():
    cases = json.loads(DATA.read_text())["cases"]
    old = json.loads((ROOT / "copenhagen-training-staged-v1/cases.json").read_text())["cases"]
    assert [c for c in cases if c["stage"] != "intermediate"] == old
    parents = {c["id"]: c for c in old}
    assert len({c["id"] for c in cases}) == 24
    for case in cases:
        state = GameState(tuple(case["board"]), Player(case["player"]), ruleset=RULESET)
        legal = legal_moves(state)
        assert len(legal) == case["initial_legal_moves"]
        winners = sum(
            (end := apply_move(state, move)).outcome is not None
            and end.outcome.winner is state.player
            for move in legal
        )
        assert winners == case["initial_winning_moves"]
        for coords in case["witness"]:
            assert state.outcome is None
            state = apply_move(state, Move(*coords))
        assert state.outcome.reason == case["expected_reason"]
        assert state.outcome.reward(Player(case["player"])) == case["expected_reward"]
        if case["stage"] == "intermediate":
            parent = parents[case["intermediate_parent_id"]]
            if "added_defender" in case:
                r, c = case["added_defender"]
                assert case["player"] == "defenders" and winners == 1
                assert {
                    (row, col)
                    for row in range(11)
                    for col in range(11)
                    if case["board"][row][col] != parent["board"][row][col]
                } == {(r, c)}
                assert case["board"][r][c] == "D"
                assert (
                    parent["initial_legal_moves"]
                    < len(legal)
                    < parents["broad-" + parent["id"]]["initial_legal_moves"]
                )
            else:
                assert case["board"] == parent["board"]


def test_new_intermediate_boards_do_not_reuse_viewed_diagnostics():
    forbidden = {
        key
        for name in ("copenhagen-heldout-v1", "copenhagen-validation-tactics-v1")
        for state in starts(ROOT / name / "cases.json")[0]
        for key in symmetric_keys(state.board)
    }
    assert all(state.board not in forbidden for state in starts(DATA)[0])
    assert len(training_positions(DATA, "intermediate")[1]) == 8


@pytest.mark.parametrize("curriculum", [SCHEDULE, "constrained32-broad32-v1"])
def test_all_stages_cross_opponents_and_sides(curriculum):
    positions, _ = starts(DATA)
    groups = curriculum_groups(DATA, positions, curriculum, False)
    exposure = Counter()
    for step in range(64):
        stage, index, size = curriculum_choice(curriculum, groups, step)
        if curriculum == SCHEDULE:
            expected = "constrained" if step < 16 else "intermediate" if step < 32 else "broad"
        else:
            expected = "constrained" if step < 32 else "broad"
        assert stage == expected and size == 8
        exposure[(stage, index, training_reference("mixed-random-heuristic-v1", step, size))] += 1
    for stage in ("constrained", "intermediate", "broad"):
        subset = {k: v for k, v in exposure.items() if k[0] == stage}
        if stage == "intermediate" and curriculum != SCHEDULE:
            assert not subset
            continue
        assert len(subset) == 16  # Each of eight starts meets each reference.
        assert set(subset.values()) == ({1} if curriculum == SCHEDULE and stage != "broad" else {2})


def test_exact_resume_across_both_boundaries_and_matched_initial_phase(tmp_path):
    settings = dict(curriculum=SCHEDULE, reference_id="mixed-random-heuristic-v1", horizon=1)
    a = train(tmp_path / "a", DATA, episodes=32, **settings)
    af = train(
        tmp_path / "af", DATA, episodes=2, resume=tmp_path / "a" / a["checkpoint"], **settings
    )
    b = train(tmp_path / "b", DATA, episodes=15, **settings)
    b2 = train(
        tmp_path / "b2", DATA, episodes=17, resume=tmp_path / "b" / b["checkpoint"], **settings
    )
    bf = train(
        tmp_path / "bf", DATA, episodes=2, resume=tmp_path / "b2" / b2["checkpoint"], **settings
    )
    expected = {**contract(starts(DATA)[1]), "schema": 2, "algorithm": ALGORITHM}
    am, ac, ar = load(tmp_path / "af" / af["checkpoint"], expected)
    bm, bc, br = load(tmp_path / "bf" / bf["checkpoint"], expected)
    assert np.array_equal(am.actor, bm.actor) and np.array_equal(am.critic, bm.critic)
    assert ac == bc and ar.bit_generator.state == br.bit_generator.state
    assert [r["stage"] for r in b2["records"]] == ["constrained"] + ["intermediate"] * 16
    assert [r["stage"] for r in bf["records"]] == ["broad"] * 2
    control = train(
        tmp_path / "control",
        DATA,
        episodes=16,
        **(settings | {"curriculum": "constrained32-broad32-v1"}),
    )
    cm, _, cr = load(tmp_path / "control" / control["checkpoint"], expected)
    initial, _, ir = load(tmp_path / "a/checkpoint-0016.npz", expected)
    assert np.array_equal(cm.actor, initial.actor) and np.array_equal(cm.critic, initial.critic)
    assert cr.bit_generator.state == ir.bit_generator.state
    with pytest.raises(ValueError, match="contract mismatch"):
        train(
            tmp_path / "changed",
            DATA,
            episodes=1,
            resume=tmp_path / "a" / a["checkpoint"],
            **(settings | {"curriculum": "constrained32-broad32-v1"}),
        )


def test_three_stage_contract_and_explicit_dataset_cap(tmp_path):
    with pytest.raises(ValueError, match="explicit training-only"):
        train(
            tmp_path / "bad",
            ROOT / "copenhagen-training-staged-v1/cases.json",
            episodes=1,
            curriculum=SCHEDULE,
        )
    raw = json.loads(DATA.read_text())
    raw["cases"].append(raw["cases"][0])
    path = tmp_path / "too-many.json"
    path.write_text(json.dumps(raw))
    with pytest.raises(ValueError, match="1..24"):
        starts(path)
    assert not (tmp_path / "bad").exists()
