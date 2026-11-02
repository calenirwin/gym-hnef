"""Controlled local effects must not masquerade as newly trained policies."""

import importlib.util
import json
from pathlib import Path

import pytest

np = pytest.importorskip("numpy")
from hnef.learning import batches  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location(
    "factorial", ROOT / "tools/compare_copenhagen_gradients.py"
)
factorial = importlib.util.module_from_spec(spec)
spec.loader.exec_module(factorial)
DATA = ROOT / "tests/fixtures/copenhagen-training-defender-staged-v1/cases.json"


@pytest.fixture(autouse=True)
def caps(monkeypatch):
    for key in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS"):
        monkeypatch.setenv(key, "1")


def test_weighting_and_clip_placement_are_independent_factors():
    parts = [(np.array([4.0, 0.0]), np.zeros(1)), (np.array([0.0, 2.0]), np.zeros(1))]
    expected = {
        ("decision", "aggregate"): np.array([1.0, 1.5]) / np.hypot(1, 1.5),
        ("episode", "aggregate"): np.array([2.0, 1.0]) / np.sqrt(5),
        ("decision", "per-rollout"): np.array([0.25, 0.75]),
        ("episode", "per-rollout"): np.array([0.5, 0.5]),
    }
    for arm, value in expected.items():
        a, c = factorial.direction(parts, [1, 3], *arm)
        assert np.allclose(a, value) and np.array_equal(c, np.zeros(1))
    for clipping in ("aggregate", "per-rollout"):
        a, _ = factorial.direction(parts, [2, 2], "decision", clipping)
        b, _ = factorial.direction(parts, [2, 2], "episode", clipping)
        assert np.array_equal(a, b)
    with pytest.raises(ValueError, match="counts"):
        factorial.direction(parts, [0, 3], "decision", "aggregate")
    with pytest.raises(ValueError, match="Nonfinite"):
        factorial.clipped(np.array([np.nan]), np.zeros(1))


def test_local_comparison_preserves_inputs_and_strict_baseline(tmp_path):
    run = tmp_path / "run"
    batches.train(run, DATA, mode="cell-batch-v1", seed=53, cells=2)
    before = {p.name: p.read_bytes() for p in run.iterdir()}
    report = factorial.compare(run, DATA, tmp_path / "out")
    assert before == {p.name: p.read_bytes() for p in run.iterdir()}
    assert len(report["records"]) == 2
    assert all(len(r["arms"]) == 4 and r["split_step_max_error"] < 1e-12 for r in report["records"])
    assert not list((tmp_path / "out").glob("*.npz"))
    raw = json.loads((run / "report.json").read_text())
    raw["initial_summary"][0]["mean_immediate_win_probability"] += 0.1
    (run / "report.json").write_text(json.dumps(raw))
    with pytest.raises(ValueError, match="Initial baseline"):
        factorial.compare(run, DATA, tmp_path / "tampered")
    assert not (tmp_path / "tampered").exists()


def test_compatibility_rejects_any_training_code_change(tmp_path):
    altered = tmp_path / "batches.py"
    altered.write_text(
        factorial.VERIFIER.read_text().replace("DECISIONS_PER_CELL = 8", "DECISIONS_PER_CELL = 9")
    )
    with pytest.raises(ValueError, match="identical training code"):
        factorial.strict_verify(tmp_path, DATA, verifier=altered)
