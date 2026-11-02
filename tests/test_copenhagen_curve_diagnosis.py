"""Analytical projections, terminal stratification and immutable exact replay."""

import importlib.util
import json
from pathlib import Path

import pytest

np = pytest.importorskip("numpy")
from hnef.learning import batches  # noqa: E402
from hnef.learning.onpolicy import features  # noqa: E402
from hnef.learning.plumbing import LinearModel  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "tests/fixtures/copenhagen-training-defender-staged-v1/cases.json"
spec = importlib.util.spec_from_file_location(
    "diagnosis", ROOT / "tools/diagnose_copenhagen_curve.py"
)
diagnosis = importlib.util.module_from_spec(spec)
spec.loader.exec_module(diagnosis)


@pytest.fixture(autouse=True)
def caps(monkeypatch):
    for key in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS"):
        monkeypatch.setenv(key, "1")


def test_projection_matches_finite_difference_and_ignores_illegal_logits():
    _, states, winning, _ = batches.setup(DATA, "cell-normalized-v1", 137, batches.CURVE_PROFILE)
    i = next(i for i, wins in enumerate(winning) if wins)
    state, wins = states[i], winning[i]
    model = LinearModel()
    delta = np.random.default_rng(3).normal(size=model.actor.shape)
    x, mask = features(state)

    def mass(scale):
        model.actor = scale * delta
        return model.forward(x[None], mask[None])[0][0, list(wins)].sum()

    finite = (mass(1e-6) - mass(-1e-6)) / 2e-6
    model.actor.fill(0)
    assert diagnosis.projection(model, state, wins, delta) == pytest.approx(finite, abs=1e-9)
    delta[:, mask] = 0
    assert diagnosis.projection(model, state, wins, delta) == 0


def test_residuals_distinguish_terminal_from_intermediate_and_cutoff():
    samples = [(None, None, 0, 0.2, 0.3), (None, None, 0, 0.4, -1.0)]
    rows = diagnosis.residuals(samples, dict(terminated=True, reward=-1))
    assert [r["kind"] for r in rows] == ["bootstrap-intermediate", "terminal--1"]
    assert rows[1]["residual"] == -1.4
    assert diagnosis.residuals(samples, dict(terminated=False))[1]["kind"] == "bootstrap-cutoff"


def test_saved_diagnosis_is_immutable_and_rejects_baseline_tampering(tmp_path):
    run = tmp_path / "run"
    batches.train(
        run, DATA, mode="cell-normalized-v1", seed=137, profile=batches.CURVE_PROFILE, cells=2
    )
    before = {p.name: p.read_bytes() for p in run.iterdir()}
    report = diagnosis.diagnose(run, DATA, tmp_path / "out")
    assert before == {p.name: p.read_bytes() for p in run.iterdir()}
    assert len(report["records"]) == 2
    assert sum(len(t["residuals"]) for r in report["records"] for t in r["rollouts"]) == 16
    assert all(len(r["effects"]) == 4 for r in report["records"])
    assert not list((tmp_path / "out").glob("*.npz"))
    raw = json.loads((run / "report.json").read_text())
    raw["initial_summary"][0]["mean_immediate_win_probability"] += 0.1
    (run / "report.json").write_text(json.dumps(raw))
    with pytest.raises(ValueError, match="Initial baseline"):
        diagnosis.diagnose(run, DATA, tmp_path / "bad")
