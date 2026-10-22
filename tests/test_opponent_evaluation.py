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


def test_streams_are_identity_paired_across_seats_and_matchups(tmp_path):
    from hnef.evaluation import policy_seed

    report = smoke(tmp_path / "pairing", seeds=(19,), max_plies=1)
    for game in report["games"]:
        for side in ("attackers", "defenders"):
            assert game["policy_seeds"][side] == policy_seed(19, game[side])
    assert policy_seed(19, "random-v1") != policy_seed(20, "random-v1")
    assert policy_seed(19, "random-v1") != policy_seed(19, "heuristic-v1")


def test_summary_keeps_draws_cutoffs_and_seat_denominators_separate():
    from hnef.evaluation import summarize

    games = [
        {"attackers": "a", "defenders": "b", "outcome": outcome, "decision_ms": [[1, 3], []]}
        for outcome in (
            None,
            {"winner": None, "reason": "adjudicated_draw"},
            {"winner": "attackers", "reason": "king_captured"},
            {"winner": "defenders", "reason": "king_escaped"},
        )
    ]
    a, b = summarize(games)
    assert a["results"] == {"win": 1, "loss": 1, "draw": 1, "unfinished": 1}
    assert a["cutoff_rate"] == 0.25
    assert a["terminal_reasons"] == {"adjudicated_draw": 1, "king_captured": 1, "king_escaped": 1}
    assert a["latency_ms"] == {"decisions": 8, "mean": 2, "p50": 1, "p95": 3, "max": 3}
    assert b["side"] == "defenders" and b["games"] == 4
    assert b["latency_ms"]["p95"] is None
    assert summarize([]) == []


def test_registry_profiles_are_an_explicit_supported_set():
    from hnef.opponents import OPPONENTS

    for spec in OPPONENTS:
        assert spec.rulesets == (Ruleset.MINI, Ruleset.COPENHAGEN, Ruleset.COPENHAGEN_V2)
