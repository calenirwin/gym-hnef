"""Refresh timing changes while decision weights, clip and total exposure match."""

import json
from pathlib import Path

import pytest

np = pytest.importorskip("numpy")
from hnef.learning import batches  # noqa: E402
from hnef.learning.plumbing import LinearModel, load  # noqa: E402

DATA = Path(__file__).parent / "fixtures/copenhagen-training-defender-staged-v1/cases.json"


@pytest.fixture(autouse=True)
def caps(monkeypatch):
    for key in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS"):
        monkeypatch.setenv(key, "1")


@pytest.mark.parametrize("mode", batches.NORMALIZED_MODES)
def test_exposure_clipping_weights_and_refresh(mode, monkeypatch):
    _, states, _, _ = batches.setup(DATA, mode, 53)
    model = LinearModel()
    model.critic[-1] = 0.2
    original_actor, original_critic = model.actor.copy(), model.critic.copy()
    snapshots = []
    parts = []
    collect = batches.collect
    gradient = batches.gradients

    def capture_collect(*args):
        snapshots.append((model.actor.copy(), model.critic.copy()))
        return collect(*args)

    def capture_gradient(m, samples):
        a, c, metrics = gradient(m, samples)
        scale = min(1.0, 1 / max(metrics["gradient_norm"], 1e-12))
        parts.append((a.copy() * scale * len(samples) / 8, c.copy() * scale * len(samples) / 8))
        return a, c, metrics

    monkeypatch.setattr(batches, "collect", capture_collect)
    monkeypatch.setattr(batches, "gradients", capture_gradient)
    record = batches.cell(model, states[0], 53, 0, mode)
    assert sum(r["decisions"] for r in record["rollouts"]) == 8
    assert sum(r["decision_weight"] for r in record["rollouts"]) == 1
    assert record["learning_rate_exposure"] == pytest.approx(0.05, abs=1e-15)
    assert all(
        r["effective_learning_rate"] == 0.05 * r["decisions"] / 8 for r in record["rollouts"]
    )
    assert all(0 < r["clip_scale"] <= 1 for r in record["rollouts"])
    assert np.allclose(
        model.actor, original_actor - 0.05 * sum(a for a, _ in parts), rtol=0, atol=1e-15
    )
    assert np.allclose(
        model.critic, original_critic - 0.05 * sum(c for _, c in parts), rtol=0, atol=1e-15
    )
    assert record["actor_delta_norm"] > 0 and record["critic_delta_norm"] > 0
    if mode == "cell-normalized-v1":
        assert all(
            np.array_equal(a, original_actor) and np.array_equal(c, original_critic)
            for a, c in snapshots
        )
        assert record["updates"] == 1
    else:
        assert any(not np.array_equal(a, original_actor) for a, _ in snapshots[1:])
        assert record["updates"] == len(record["rollouts"])


@pytest.mark.parametrize("mode", batches.NORMALIZED_MODES)
def test_exact_resume_baselines_and_normalization_contract(tmp_path, mode):
    cfg = dict(mode=mode, seed=53)
    batches.train(tmp_path / "full", DATA, cells=3, **cfg)
    batches.train(tmp_path / "one", DATA, cells=1, **cfg)
    checkpoint = tmp_path / "one/checkpoint-0001.npz"
    batches.train(tmp_path / "resume", DATA, cells=2, resume=checkpoint, **cfg)
    a, am, ar = load(tmp_path / "full/checkpoint-0003.npz", {})
    b, bm, br = load(tmp_path / "resume/checkpoint-0003.npz", {})
    assert am["algorithm"] == batches.NORMALIZED_ALGORITHM
    assert am["gradient_weighting"] == "decision-quota-v1"
    assert am["clipping"] == "per-rollout-joint-l2-cap1-v1"
    assert am["cell_learning_rate_exposure"] == 0.05
    assert np.array_equal(a.actor, b.actor) and np.array_equal(a.critic, b.critic)
    assert am == bm and ar.bit_generator.state == br.bit_generator.state
    assert batches.verify(tmp_path / "resume", DATA)["decisions"] == 16
    other = next(m for m in batches.NORMALIZED_MODES if m != mode)
    with pytest.raises(ValueError, match="contract"):
        batches.train(tmp_path / "wrong", DATA, cells=1, mode=other, seed=53, resume=checkpoint)
    path = tmp_path / "resume/report.json"
    report = json.loads(path.read_text())
    report["initial_summary"][0]["mean_immediate_win_probability"] += 0.1
    path.write_text(json.dumps(report))
    with pytest.raises(ValueError, match="Initial baseline"):
        batches.verify(tmp_path / "resume", DATA)
