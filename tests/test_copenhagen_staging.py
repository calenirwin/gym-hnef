"""Fixed staged curricula preserve contracts and balanced reference assignments."""

import json
from collections import Counter
from pathlib import Path

import pytest

np = pytest.importorskip("numpy")

from hnef.learning.curves import curves, training_positions  # noqa: E402
from hnef.learning.onpolicy import (  # noqa: E402
    ALGORITHM,
    curriculum_choice,
    curriculum_groups,
    starts,
    train,
    training_reference,
)
from hnef.learning.plumbing import contract, load  # noqa: E402

ROOT = Path(__file__).parent / "fixtures"
DATA = ROOT / "copenhagen-training-staged-v1/cases.json"


@pytest.fixture(autouse=True)
def cpu_caps(monkeypatch):
    for key in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS"):
        monkeypatch.setenv(key, "1")


def test_staged_manifest_preserves_both_verified_parent_corpora():
    cases = json.loads(DATA.read_text())["cases"]
    assert len({c["id"] for c in cases}) == 16
    for stage, source in (
        ("constrained", "copenhagen-training-tactics-v1"),
        ("broad", "copenhagen-training-broad-v1"),
    ):
        parent = json.loads((ROOT / source / "cases.json").read_text())["cases"]
        subset = [
            {k: v for k, v in c.items() if k not in ("stage", "stage_source")}
            for c in cases
            if c["stage"] == stage
        ]
        assert subset == parent


@pytest.mark.parametrize("curriculum", ["constrained32-broad32-v1", "broad64-v1"])
def test_phase_reference_and_side_cross_product(curriculum):
    positions, _ = starts(DATA)
    groups = curriculum_groups(DATA, positions, curriculum, False)
    exposure = Counter()
    for step in range(64):
        stage, index, size = curriculum_choice(curriculum, groups, step)
        reference = training_reference("mixed-random-heuristic-v1", step, size)
        assert size == 8
        expected = "constrained" if curriculum.startswith("constrained") and step < 32 else "broad"
        assert stage == expected
        exposure[(stage, index, positions[index].player.value, reference)] += 1
    assert len(exposure) == (32 if curriculum.startswith("constrained") else 16)
    assert set(exposure.values()) == ({2} if curriculum.startswith("constrained") else {4})


def test_resume_across_transition_is_exact_and_cannot_switch_curriculum(tmp_path):
    config = dict(
        curriculum="constrained32-broad32-v1",
        reference_id="mixed-random-heuristic-v1",
        horizon=1,
        seed=17,
    )
    # Both paths end at step 34; one resume is on the boundary, one just before it.
    a = train(tmp_path / "a", DATA, episodes=32, **config)
    b = train(tmp_path / "b", DATA, episodes=31, **config)
    ar = train(tmp_path / "ar", DATA, episodes=2, resume=tmp_path / "a" / a["checkpoint"], **config)
    br = train(tmp_path / "br", DATA, episodes=3, resume=tmp_path / "b" / b["checkpoint"], **config)
    expected = {**contract(starts(DATA)[1]), "schema": 2, "algorithm": ALGORITHM}
    am, ac, ag = load(tmp_path / "ar" / ar["checkpoint"], expected)
    bm, bc, bg = load(tmp_path / "br" / br["checkpoint"], expected)
    assert np.array_equal(am.actor, bm.actor) and np.array_equal(am.critic, bm.critic)
    assert np.any(am.actor != 0) and np.any(am.critic != 0)
    assert ac == bc and ag.bit_generator.state == bg.bit_generator.state
    assert [r["stage"] for r in br["records"]] == ["constrained", "broad", "broad"]
    assert [r["start_index"] for r in br["records"]] == [7, 8, 9]
    with pytest.raises(ValueError, match="contract mismatch"):
        train(
            tmp_path / "wrong",
            DATA,
            episodes=1,
            resume=tmp_path / "a" / a["checkpoint"],
            **(config | {"curriculum": "broad64-v1"}),
        )
    assert not (tmp_path / "wrong").exists()
    report = curves(
        DATA,
        [tmp_path / "a/checkpoint-0000.npz", tmp_path / "ar" / ar["checkpoint"]],
        tmp_path / "curve",
        stage="broad",
    )
    assert report["diagnostic_stage"] == "broad"
    assert len(report["records"][0]["positions"]) == 8
    assert report["contract"]["dataset_sha256"] == starts(DATA)[1]
    assert all(r["id"].startswith("broad-") for r in report["records"][0]["positions"])


def test_invalid_staging_inputs_fail_before_output(tmp_path):
    with pytest.raises(ValueError, match="already contain both"):
        train(
            tmp_path / "out",
            DATA,
            episodes=1,
            curriculum="constrained32-broad32-v1",
            both_sides=True,
        )
    with pytest.raises(ValueError, match="16 explicit"):
        train(
            tmp_path / "out",
            ROOT / "copenhagen-training-broad-v1/cases.json",
            episodes=1,
            curriculum="broad64-v1",
        )
    raw = json.loads(DATA.read_text())
    raw["cases"][0]["split"] = "validation"
    altered = tmp_path / "altered.json"
    altered.write_text(json.dumps(raw))
    with pytest.raises(ValueError, match="training-only"):
        train(tmp_path / "out", altered, episodes=1, curriculum="constrained32-broad32-v1")
    with pytest.raises(ValueError, match="Unknown diagnostic"):
        training_positions(DATA, "heldout")
    assert not (tmp_path / "out").exists()
