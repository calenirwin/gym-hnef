"""Fixed panel identity, exact boundaries, and error attribution."""

import importlib.util
import json
from pathlib import Path

import pytest

np = pytest.importorskip("numpy")
from hnef.learning import batches  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "tests/fixtures/copenhagen-training-defender-staged-v1/cases.json"
spec = importlib.util.spec_from_file_location("panel", ROOT / "tools/diagnose_copenhagen_panel.py")
panel = importlib.util.module_from_spec(spec)
spec.loader.exec_module(panel)


@pytest.fixture(autouse=True)
def caps(monkeypatch):
    for key in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS"):
        monkeypatch.setenv(key, "1")


def test_fixed_targets_can_separate_improvement_and_forgetting():
    x = np.array([[1.0], [-1.0]])
    y = np.array([1.0, 1.0])
    a, b, d = panel.effects(x, y, np.zeros(1), np.array([0.1]))
    assert d[0] < 0 < d[1]
    assert np.array_equal(b - a, d)
    _, _, reverse = panel.effects(x, y, np.array([0.1]), np.zeros(1))
    assert np.allclose(d + reverse, 0, atol=1e-15)


def test_panel_replays_preserves_inputs_and_rejects_tampering(tmp_path):
    runs = []
    for phase in range(3):
        run = tmp_path / str(phase)
        batches.train(
            run,
            DATA,
            mode="cell-normalized-v1",
            seed=137,
            profile=batches.CURVE_PROFILE,
            cells=8,
            resume=None if not phase else runs[-1] / f"checkpoint-{phase * 8:04}.npz",
        )
        runs.append(run)
    before = {str(p): p.read_bytes() for run in runs for p in run.iterdir()}
    result = panel.diagnose(runs, DATA, tmp_path / "out", attribute_outcomes=True)
    assert before == {str(p): p.read_bytes() for run in runs for p in run.iterdir()}
    assert len(result["records"]) == 8
    assert result["schema"] == 2
    for row in result["records"]:
        for group in row["groups"]:
            assert sum(group["outcomes"].values()) == pytest.approx(group["change"], abs=1e-12)
    assert all(r["cell"] < 16 and r["target"] in (-1, 0, 1) for r in result["panel"])
    assert not list((tmp_path / "out").glob("*.npz"))
    # Panel definition must not change with later phases or error results.
    assert result["panel_definition"] == "all-first16-terminal-samples-v1"
    with pytest.raises(ValueError, match="unique"):
        panel.diagnose([runs[0]] * 3, DATA, tmp_path / "dup")
    with pytest.raises(ValueError, match="consecutive"):
        panel.diagnose([runs[1], runs[0], runs[2]], DATA, tmp_path / "order")
    path = runs[0] / "report.json"
    raw = json.loads(path.read_text())
    raw["initial_summary"][0]["mean_immediate_win_probability"] += 0.1
    path.write_text(json.dumps(raw))
    with pytest.raises(ValueError, match="Initial baseline"):
        panel.diagnose(runs, DATA, tmp_path / "bad")


@pytest.mark.parametrize("before,total", [(0.0, 0.2), (0.3, 0.0), (0.3, 1e-12), (30.0, -60.0)])
def test_outcome_allocation_exact_order_independent_and_cancelling(before, total):
    x = np.array([[1.0], [-0.5]])
    y = np.array([1.0, -1.0])
    parts = {"loss": np.array([0.4]), "win": np.array([total - 0.4])}
    result = panel.allocate(x, y, np.array([before]), parts)
    reverse = panel.allocate(x, y, np.array([before]), dict(reversed(list(parts.items()))))
    _, _, actual = panel.effects(x, y, np.array([before]), np.array([before + total]))
    assert np.allclose(sum(result.values()), actual, rtol=0, atol=1e-12)
    assert all(np.array_equal(result[k], reverse[k]) for k in result)
    if total == 0:
        assert np.any(result["loss"] != 0)
        assert np.allclose(result["loss"], -result["win"], rtol=0, atol=1e-15)
