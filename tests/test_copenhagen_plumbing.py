"""Synthetic training plumbing; no held-out positions are used for optimization."""

import json
from pathlib import Path

import pytest

np = pytest.importorskip("numpy")

from hnef.learning.plumbing import LinearModel, contract, dataset, load, run  # noqa: E402

DATA = Path(__file__).parent / "fixtures/copenhagen-training-plumbing-v1/cases.json"


@pytest.fixture(autouse=True)
def cpu_caps(monkeypatch):
    for key in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS"):
        monkeypatch.setenv(key, "1")


def test_synthetic_positions_are_not_heldout_symmetries():
    heldout = json.loads((DATA.parents[1] / "copenhagen-heldout-v1/cases.json").read_text())[
        "cases"
    ]
    forbidden = set()
    for item in heldout:
        if item["board"] is None:
            continue
        board = np.array([list(row) for row in item["board"]])
        for k in range(4):
            rotated = np.rot90(board, k)
            forbidden.add(tuple("".join(row) for row in rotated))
            forbidden.add(tuple("".join(row) for row in np.fliplr(rotated)))
    for item in json.loads(DATA.read_text())["cases"]:
        assert tuple(item["board"]) not in forbidden


def test_optimizer_masking_and_finite_gradient_checks():
    (x, mask, actions, targets), _ = dataset(DATA)
    model = LinearModel()
    probs, _ = model.forward(x, mask)
    assert np.all(probs[~mask] == 0)
    assert np.allclose(probs.sum(axis=1), 1)
    metrics = model.update(x, mask, actions, targets, 0.05)
    assert np.isfinite(metrics["loss"]) and metrics["gradient_norm"] > 0
    assert np.any(model.actor != 0) and np.any(model.critic != 0)
    with pytest.raises(ValueError, match="empty legal mask"):
        model.forward(x, np.zeros_like(mask))
    bad_actions = np.array([np.flatnonzero(~row)[0] for row in mask])
    with pytest.raises(ValueError, match="Illegal target"):
        model.update(x, mask, bad_actions, targets, 0.05)
    with pytest.raises(ValueError, match="Nonfinite loss or gradient"):
        model.update(x, mask, actions, targets * np.nan, 0.05)
    model.actor[0, 0] = np.nan
    with pytest.raises(ValueError, match="Nonfinite parameters"):
        model.forward(x, mask)


def test_checkpoint_resume_matches_uninterrupted_training(tmp_path):
    whole = run(tmp_path / "whole", DATA, steps=4)
    part = run(tmp_path / "part", DATA, steps=2)
    resumed = run(
        tmp_path / "resumed", DATA, steps=2, resume=tmp_path / "part" / part["checkpoint"]
    )
    _, digest = dataset(DATA)
    a, am, ar = load(tmp_path / "whole" / whole["checkpoint"], contract(digest))
    b, bm, br = load(tmp_path / "resumed" / resumed["checkpoint"], contract(digest))
    assert np.array_equal(a.actor, b.actor) and np.array_equal(a.critic, b.critic)
    assert am == bm and ar.bit_generator.state == br.bit_generator.state
    assert whole["final_dataset_loss"] < whole["initial_dataset_loss"]
    assert resumed["total_steps"] == 4 and whole["validation_failures"] == []
    with pytest.raises(ValueError, match="contract mismatch"):
        load(tmp_path / "whole" / whole["checkpoint"], {**contract(digest), "features": "mini"})
    with pytest.raises(FileExistsError):
        run(tmp_path / "whole", DATA)


def test_budget_and_thread_guards(tmp_path, monkeypatch):
    with pytest.raises(ValueError, match="1..8"):
        run(tmp_path / "invalid", DATA, steps=9)
    monkeypatch.setenv("OMP_NUM_THREADS", "2")
    with pytest.raises(ValueError, match="thread caps"):
        run(tmp_path / "invalid", DATA)
    assert not (tmp_path / "invalid").exists()
