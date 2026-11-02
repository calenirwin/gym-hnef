"""Versioned curve budgets preserve old caps and cycle all start/reference pairs."""

import json
from pathlib import Path

import pytest

np = pytest.importorskip("numpy")
from hnef.learning import batches  # noqa: E402
from hnef.learning.plumbing import LinearModel, load, save  # noqa: E402

DATA = Path(__file__).parent / "fixtures/copenhagen-training-defender-staged-v1/cases.json"


@pytest.fixture(autouse=True)
def caps(monkeypatch):
    for key in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS"):
        monkeypatch.setenv(key, "1")


def test_profile_caps_and_crossing_cycle_resume(tmp_path):
    mode, seed, profile = "cell-normalized-v1", 137, batches.CURVE_PROFILE
    _, _, _, cfg = batches.setup(DATA, mode, seed, profile)
    assert cfg["total_cells"] == 64
    assert batches.setup(DATA, mode, seed)[3]["total_cells"] == 16
    with pytest.raises(ValueError, match="fixed"):
        batches.setup(DATA, "episode-normalized-v1", seed, profile)
    # Synthetic checkpoint at the old boundary exercises the extended schedule.
    checkpoint = tmp_path / "boundary.npz"
    save(
        checkpoint,
        LinearModel(),
        {**cfg, "step": 15, "optimizer_steps": 15},
        np.random.default_rng(seed),
    )
    kw = dict(mode=mode, seed=seed, profile=profile)
    full = batches.train(tmp_path / "full", DATA, cells=2, resume=checkpoint, **kw)
    assert [r["reference"] for r in full["records"]] == ["heuristic-v1", "random-v1"]
    batches.train(tmp_path / "one", DATA, cells=1, resume=checkpoint, **kw)
    batches.train(
        tmp_path / "two", DATA, cells=1, resume=tmp_path / "one/checkpoint-0016.npz", **kw
    )
    a, am, ar = load(tmp_path / "full/checkpoint-0017.npz", cfg)
    b, bm, br = load(tmp_path / "two/checkpoint-0017.npz", cfg)
    assert np.array_equal(a.actor, b.actor) and np.array_equal(a.critic, b.critic)
    assert am == bm and ar.bit_generator.state == br.bit_generator.state
    assert batches.verify(tmp_path / "full", DATA)["decisions"] == 16
    with pytest.raises(ValueError, match="contract"):
        batches.train(tmp_path / "wrong", DATA, mode=mode, seed=seed, cells=1, resume=checkpoint)
    save(tmp_path / "end.npz", b, {**bm, "step": 64}, br)
    with pytest.raises(ValueError, match="cap is 64"):
        batches.train(tmp_path / "overflow", DATA, cells=1, resume=tmp_path / "end.npz", **kw)
    path = tmp_path / "full/report.json"
    report = json.loads(path.read_text())
    report["initial_summary"][0]["mean_immediate_win_probability"] += 0.1
    path.write_text(json.dumps(report))
    with pytest.raises(ValueError, match="Initial baseline"):
        batches.verify(tmp_path / "full", DATA)


def test_all_curve_cycles_cover_both_references(monkeypatch):
    _, states, _, _ = batches.setup(DATA, "cell-normalized-v1", 137, batches.CURVE_PROFILE)

    # Capture actual collector reference assignment without doing optimizer work.
    def capture(*args):
        raise RuntimeError(args[6])

    monkeypatch.setattr(batches, "collect", capture)
    for index in range(64):
        reference = "random-v1" if index % 16 < 8 else "heuristic-v1"
        with pytest.raises(RuntimeError, match=reference):
            batches.cell(
                LinearModel(),
                states[index % 8],
                137,
                index,
                "cell-normalized-v1",
                batches.CURVE_PROFILE,
            )
    with pytest.raises(ValueError, match="index"):
        batches.cell(LinearModel(), states[0], 137, 64, "cell-normalized-v1", batches.CURVE_PROFILE)
