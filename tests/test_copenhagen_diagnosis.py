"""Read-only evidence replay must reproduce updates and reject tampering."""

import importlib.util
import json
from pathlib import Path

import pytest

pytest.importorskip("numpy")
from hnef.learning.onpolicy import train  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location("diagnosis", ROOT / "tools/diagnose_copenhagen.py")
diagnostic = importlib.util.module_from_spec(spec)
spec.loader.exec_module(diagnostic)
DATA = ROOT / "tests/fixtures/copenhagen-training-defender-staged-v1/cases.json"


@pytest.fixture(autouse=True)
def caps(monkeypatch):
    for name in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS"):
        monkeypatch.setenv(name, "1")


def test_clipping_is_separate_from_gradient_cancellation():
    assert diagnostic.clipping(0.2, 0.3) == 1
    assert diagnostic.clipping(3, 4) == pytest.approx(0.6)
    assert diagnostic.clipping(0, 2) == 0.5


def test_exact_replay_and_tampered_trace_rejected(tmp_path):
    run = tmp_path / "train"
    train(
        run,
        DATA,
        seed=17,
        episodes=4,
        horizon=2,
        reference_id="mixed-random-heuristic-v1",
        curriculum="constrained16-intermediate16-broad32-v1",
    )
    files = {p.name: p.read_bytes() for p in run.iterdir()}
    result = diagnostic.diagnose(DATA, run, tmp_path / "diagnosis")
    assert len(result["records"]) == 4
    assert all(0 < r["actor_clip_ratio"] <= 1 for r in result["records"])
    assert any(s["selected_win"] for r in result["records"] for s in r["samples"])
    assert all(
        s["opportunity"] for r in result["records"] for s in r["samples"] if s["selected_win"]
    )
    assert files == {p.name: p.read_bytes() for p in run.iterdir()}
    path = run / "trace-0001.json"
    trace = json.loads(path.read_text())
    trace["final_target"] += 1
    path.write_text(json.dumps(trace))
    with pytest.raises(ValueError, match="Exact collection replay"):
        diagnostic.diagnose(DATA, run, tmp_path / "bad")
