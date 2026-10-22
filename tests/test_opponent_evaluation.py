"""Evaluation provenance and repeatability, without mistaking cutoffs for draws."""

import hashlib

import pytest

from hnef.engine import Ruleset
from hnef.evaluation import smoke
from hnef.opponents import OpponentSpec, opponent
from hnef.replay import Replay


def test_registry_rejects_unknown_and_incompatible_opponents():
    with pytest.raises(ValueError, match="Unknown"):
        opponent("ppo-planned")
    spec = OpponentSpec("mini-only", "random", (Ruleset.MINI,))
    with pytest.raises(ValueError, match="does not support"):
        spec.create(Ruleset.COPENHAGEN_V2, 0)


def test_smoke_pairs_sides_preserves_replays_and_is_repeatable(tmp_path):
    reports = [smoke(tmp_path / str(i), seeds=(7,), max_plies=2) for i in range(2)]
    for i, report in enumerate(reports):
        assert len(report["games"]) == 6
        assert len({(g["attackers"], g["defenders"]) for g in report["games"]}) == 6
        for game in report["games"]:
            data = (tmp_path / str(i) / game["replay"]).read_bytes()
            assert hashlib.sha256(data).hexdigest() == game["replay_sha256"]
            state = Replay.from_json(data.decode()).verify()
            assert state.ruleset is Ruleset.COPENHAGEN_V2
            assert state.outcome is None
            assert game["truncated"] and game["outcome"] is None
            assert sum(map(len, game["decision_ms"])) == 2
    assert [g["replay_sha256"] for g in reports[0]["games"]] == [
        g["replay_sha256"] for g in reports[1]["games"]
    ]
    with pytest.raises(FileExistsError):
        smoke(tmp_path / "0")


@pytest.mark.parametrize("seeds", [(), (1, 1), (-1,), (True,)])
def test_invalid_smoke_seed_sets_do_not_create_output(tmp_path, seeds):
    output = tmp_path / "invalid"
    with pytest.raises(ValueError):
        smoke(output, seeds=seeds)
    assert not output.exists()
