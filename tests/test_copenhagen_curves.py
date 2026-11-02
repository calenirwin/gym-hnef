"""Training curves measure exact policy mass without consuming evaluation labels."""

import hashlib
import json
from pathlib import Path

import pytest

np = pytest.importorskip("numpy")

from hnef.learning.curves import curves, measure, training_positions  # noqa: E402
from hnef.learning.onpolicy import train  # noqa: E402
from hnef.learning.plumbing import LinearModel  # noqa: E402

FIXTURES = Path(__file__).parent / "fixtures"
DATA = FIXTURES / "copenhagen-training-broad-v1/cases.json"


@pytest.fixture(autouse=True)
def cpu_caps(monkeypatch):
    for key in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS"):
        monkeypatch.setenv(key, "1")


def test_broader_branching_is_a_controlled_training_only_perturbation():
    narrow = json.loads((FIXTURES / "copenhagen-training-tactics-v1/cases.json").read_text())
    parents = {c["id"]: c for c in narrow["cases"]}
    for case in json.loads(DATA.read_text())["cases"]:
        parent = parents[case["parent_id"]]
        assert case["split"] == "training"
        assert case["initial_legal_moves"] >= 17
        assert case["initial_legal_moves"] > parent["initial_legal_moves"]
        changed = {
            (r, c)
            for r in range(11)
            for c in range(11)
            if case["board"][r][c] != parent["board"][r][c]
        }
        assert changed == {(8, 8), (8, 2)}
        assert case["board"][8][8] == "A" and case["board"][8][2] == "D"


def test_uniform_and_biased_probability_match_analytic_result():
    ids, states, winning, _ = training_positions(DATA)
    model = LinearModel()
    rows, summary = measure(model, ids, states, winning)
    assert [r["opportunity_positions"] for r in summary] == [2, 2]
    for row in rows:
        if row["immediate_winning_actions"]:
            assert row["immediate_win_probability"] == pytest.approx(1 / row["legal_actions"])
        else:
            assert row["immediate_win_probability"] is None
    action = next(iter(winning[0]))
    model.actor[-1, action] = 2
    changed, _ = measure(model, ids, states, winning)
    n = rows[0]["legal_actions"]
    assert changed[0]["immediate_win_probability"] == pytest.approx(np.exp(2) / (n - 1 + np.exp(2)))


def test_curve_contract_order_and_read_only_checkpoint_behavior(tmp_path):
    output = tmp_path / "train"
    train(output, DATA, episodes=2, horizon=1)
    checkpoints = [output / f"checkpoint-{i:04}.npz" for i in range(3)]
    before = [hashlib.sha256(p.read_bytes()).hexdigest() for p in checkpoints]
    report = curves(DATA, checkpoints, tmp_path / "curve")
    assert [r["step"] for r in report["records"]] == [0, 1, 2]
    assert [hashlib.sha256(p.read_bytes()).hexdigest() for p in checkpoints] == before
    assert [r["checkpoint_sha256"] for r in report["records"]] == before
    saved = (tmp_path / "curve/report.json").read_bytes()
    with pytest.raises(FileExistsError):
        curves(DATA, checkpoints, tmp_path / "curve")
    assert (tmp_path / "curve/report.json").read_bytes() == saved
    with pytest.raises(ValueError, match="strictly increase"):
        curves(DATA, checkpoints[::-1], tmp_path / "reversed")
    assert not (tmp_path / "reversed").exists()
    train(tmp_path / "other-seed", DATA, episodes=1, horizon=1, seed=23)
    with pytest.raises(ValueError, match="seed or training configuration"):
        curves(
            DATA,
            [checkpoints[0], tmp_path / "other-seed/checkpoint-0001.npz"],
            tmp_path / "mismatch",
        )
    assert not (tmp_path / "mismatch").exists()


def test_validation_split_and_duplicate_paths_are_rejected_before_output(tmp_path):
    missing = tmp_path / "missing"
    with pytest.raises(ValueError, match="training-only"):
        curves(
            FIXTURES / "copenhagen-validation-tactics-v1/cases.json",
            [missing, tmp_path / "other"],
            tmp_path / "out",
        )
    with pytest.raises(ValueError, match="unique checkpoint"):
        curves(DATA, [missing, missing], tmp_path / "out")
    assert not (tmp_path / "out").exists()


def test_witness_annotations_cannot_supply_curve_targets(tmp_path):
    original = training_positions(DATA)
    raw = json.loads(DATA.read_text())
    for c in raw["cases"]:
        c["witness"] = []
        c["expected_reward"] = 999
        c["initial_winning_moves"] = 999
    path = tmp_path / "untrusted-labels.json"
    path.write_text(json.dumps(raw))
    assert training_positions(path)[:3] == original[:3]
