"""Read-only evidence replay must reproduce updates and reject tampering."""

import importlib.util
import json
from pathlib import Path

import pytest

np = pytest.importorskip("numpy")
from hnef.learning.onpolicy import collect, starts, train  # noqa: E402

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


@pytest.mark.parametrize("behavior", ["policy-v1", "immediate-win-prior4-v1"])
def test_exact_replay_and_tampered_trace_rejected(tmp_path, behavior):
    run = tmp_path / "train"
    train(
        run,
        DATA,
        behavior=behavior,
        seed=17,
        episodes=4,
        horizon=2,
        reference_id="mixed-random-heuristic-v1",
        curriculum="constrained16-intermediate16-broad32-v1",
    )
    files = {p.name: p.read_bytes() for p in run.iterdir()}
    result = diagnostic.diagnose(DATA, run, tmp_path / "diagnosis")
    assert len(result["records"]) == 4
    assert result["schema"] == 2
    assert sum(r["decisions"] for r in result["visitation"]) == sum(
        len(r["samples"]) for r in result["records"]
    )
    assert sum(r["observed_wins"] for r in result["visitation"]) == sum(
        s["selected_win"] for r in result["records"] for s in r["samples"]
    )
    for record in result["records"]:
        for side, delta in record["broad_probability_delta"].items():
            positions = [p["delta"] for p in record["probe_positions"] if p["side"] == side]
            assert delta == pytest.approx(sum(positions) / len(positions), abs=1e-15)
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


def test_exact_visited_probabilities_and_zero_opportunities():
    from hnef.learning.plumbing import LinearModel

    model = LinearModel()
    states = starts(DATA)[0]
    for state in (states[16], states[18]):
        samples, trace = collect(
            model, state, np.random.default_rng(17), 23, 2, 0.99, behavior="immediate-win-prior4-v1"
        )
        rows = diagnostic.visits(state, trace, samples, model)
        assert rows[0]["state_key"] == state.key
        for row in rows:
            n, w = row["legal_actions"], row["winning_actions"]
            assert row["raw_win_probability"] == pytest.approx(w / n)
            assert row["behavior_win_probability"] == pytest.approx(4 * w / (n + 3 * w))
            assert row["selected_probability"] == pytest.approx(
                (4 if row["selected_win"] else 1) / (n + 3 * w)
            )
            assert row["opportunity"] == (w > 0)
            if w == 0:
                assert row["behavior_win_probability"] == 0


def test_summary_keeps_origins_and_continuations_separate():
    sample = dict(
        decision=0,
        opportunity=True,
        selected_win=False,
        behavior_win_probability=0.2,
        raw_win_probability=0.05,
    )
    records = [
        dict(
            start_id="a",
            learner="attackers",
            samples=[sample, {**sample, "decision": 1, "selected_win": True}],
        ),
        dict(start_id="b", learner="attackers", samples=[sample]),
    ]
    rows = diagnostic.visitation_summary(records)
    assert len(rows) == 3
    assert sum(r["conditional_expected_wins"] for r in rows) == pytest.approx(0.6)
    assert sum(r["conditional_variance_sum"] for r in rows) == pytest.approx(0.48)
    assert sum(r["observed_minus_expected"] for r in rows) == pytest.approx(0.4)
