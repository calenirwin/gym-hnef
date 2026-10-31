"""Fixed decision budgets, frozen collection and reproducible cell boundaries."""

import json
from pathlib import Path

import pytest

np = pytest.importorskip("numpy")

from hnef.learning import batches  # noqa: E402
from hnef.learning.onpolicy import gradients, starts  # noqa: E402
from hnef.learning.plumbing import LinearModel, load  # noqa: E402

DATA = Path(__file__).parent / "fixtures/copenhagen-training-defender-staged-v1/cases.json"


@pytest.fixture(autouse=True)
def caps(monkeypatch):
    for key in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS"):
        monkeypatch.setenv(key, "1")


@pytest.mark.parametrize("mode", batches.MODES)
def test_cell_budget_streams_and_freeze(mode, monkeypatch):
    state = starts(DATA)[0][16]
    model = LinearModel()
    model.critic[-1] = 0.2
    snapshots = []
    collected = []
    real_collect = batches.collect

    def capture(*args):
        snapshots.append((model.actor.copy(), model.critic.copy()))
        samples, trace = real_collect(*args)
        collected.append(samples)
        return samples, trace

    monkeypatch.setattr(batches, "collect", capture)
    row = batches.cell(model, state, 53, 0, mode)
    assert sum(r["decisions"] for r in row["rollouts"]) == 8
    assert all(1 <= r["decisions"] <= 4 for r in row["rollouts"])
    assert len({r["learner_seed"] for r in row["rollouts"]}) == len(row["rollouts"])
    assert len({r["reference_seed"] for r in row["rollouts"]}) == len(row["rollouts"])
    assert row["actor_delta_norm"] > 0 and row["critic_delta_norm"] > 0
    if mode == "cell-batch-v1":
        assert all(
            np.array_equal(a, snapshots[0][0]) and np.array_equal(c, snapshots[0][1])
            for a, c in snapshots
        )
        base = LinearModel()
        base.critic[-1] = 0.2
        # Verify unequal rollout gradients are weighted by decision count.
        per = [gradients(base, s) for s in collected]
        actor, critic, metrics = gradients(base, [s for rollout in collected for s in rollout])
        assert np.allclose(
            actor, sum(g[0] * len(s) / 8 for g, s in zip(per, collected, strict=True))
        )
        assert np.allclose(
            critic, sum(g[1] * len(s) / 8 for g, s in zip(per, collected, strict=True))
        )
        scale = min(1.0, 1 / max(metrics["gradient_norm"], 1e-12))
        assert np.allclose(model.actor, -0.05 * scale * actor)
        assert np.allclose(model.critic, base.critic - 0.05 * scale * critic)
        assert row["updates"] == 1
    else:
        assert row["updates"] == len(row["rollouts"])
        assert any(not np.array_equal(a, snapshots[0][0]) for a, _ in snapshots[1:])


@pytest.mark.parametrize("mode", batches.MODES)
def test_resume_replay_and_contract_rejection(tmp_path, mode):
    config = dict(mode=mode, seed=53)
    batches.train(tmp_path / "full", DATA, cells=3, **config)
    batches.train(tmp_path / "one", DATA, cells=1, **config)
    checkpoint = tmp_path / "one/checkpoint-0001.npz"
    batches.train(tmp_path / "resume", DATA, cells=2, resume=checkpoint, **config)
    a, am, ar = load(tmp_path / "full/checkpoint-0003.npz", {})
    b, bm, br = load(tmp_path / "resume/checkpoint-0003.npz", {})
    assert np.array_equal(a.actor, b.actor) and np.array_equal(a.critic, b.critic)
    assert am == bm and ar.bit_generator.state == br.bit_generator.state
    files = {p.name: p.read_bytes() for p in (tmp_path / "resume").iterdir()}
    assert batches.verify(tmp_path / "resume", DATA)["decisions"] == 16
    assert files == {p.name: p.read_bytes() for p in (tmp_path / "resume").iterdir()}
    with pytest.raises(ValueError, match="contract"):
        batches.train(tmp_path / "bad-seed", DATA, cells=1, mode=mode, seed=67, resume=checkpoint)
    other = next(m for m in batches.MODES if m != mode)
    with pytest.raises(ValueError, match="contract"):
        batches.train(tmp_path / "bad-mode", DATA, cells=1, mode=other, seed=53, resume=checkpoint)
    path = tmp_path / "resume/report.json"
    report = json.loads(path.read_text())
    report["records"][0]["rollouts"][0]["trace"]["final_target"] += 1
    path.write_text(json.dumps(report))
    with pytest.raises(ValueError, match="replay mismatch"):
        batches.verify(tmp_path / "resume", DATA)


def test_assignment_crosses_all_starts_with_each_reference_and_caps(tmp_path):
    ids, states, _, _ = batches.setup(DATA, "cell-batch-v1", 53)
    exposure = {(ids[i % 8], "random-v1" if i < 8 else "heuristic-v1") for i in range(16)}
    assert len(exposure) == 16 and len(states) == 8
    with pytest.raises(ValueError, match="1..8"):
        batches.train(tmp_path / "too-many", DATA, cells=9, mode="cell-batch-v1", seed=53)
    # Exercise the reference transition using a short independently seeded cell.
    assert (
        batches.cell(LinearModel(), states[0], 53, 8, "cell-batch-v1")["reference"]
        == "heuristic-v1"
    )
    with pytest.raises(ValueError, match="mode/index"):
        batches.cell(LinearModel(), states[0], 53, 16, "cell-batch-v1")
