"""Training contract checks: legality, history, perspective and cutoff exclusion."""

import json

import pytest

from hnef.engine import Outcome, Player, Ruleset, apply_move, initial_state
from hnef.learning.copenhagen import CODEC, collect, collect_episode, encode, terminal_target
from hnef.replay import Replay


def test_collection_features_match_replayed_states_and_masks():
    data, replay = collect_episode(3, learner=Player.DEFENDERS, max_plies=8)
    state = initial_state(Ruleset.COPENHAGEN_V2)
    samples = iter(data["samples"])
    for action in replay.actions:
        if state.player is Player.DEFENDERS:
            sample = next(samples)
            assert sample["state_key"] == state.key
            assert sample["observation"] == encode(state)
            assert sample["action"] == action
            assert sample["observation"]["legal_mask"][action]
            assert sample["terminal_target"] is None
        state = apply_move(state, CODEC.decode(action))
    assert next(samples, None) is None
    assert data["truncated"] and data["streams"]["learner"] != data["streams"]["reference"]
    assert len(encode(state)["legal_mask"]) == 2420
    assert len(encode(state)["history"]) == 9
    with pytest.raises(ValueError):
        encode(initial_state())


def test_actual_terminal_rewards_use_learner_perspective_and_censor_cutoffs():
    assert terminal_target(Outcome(Player.ATTACKERS, "king_captured"), Player.ATTACKERS) == 1
    assert terminal_target(Outcome(Player.ATTACKERS, "king_captured"), Player.DEFENDERS) == -1
    assert terminal_target(Outcome(None, "adjudicated_draw"), Player.ATTACKERS) == 0
    assert terminal_target(None, Player.ATTACKERS) is None


def test_bounded_collection_is_reproducible_and_retains_verifiable_replays(tmp_path):
    first = collect(tmp_path / "one", episodes=2, max_plies=4)
    second = collect(tmp_path / "two", episodes=2, max_plies=4)
    assert first == second
    assert [e["learner"] for e in first["episodes"]] == ["attackers", "defenders"]
    for entry in first["episodes"]:
        replay = Replay.from_json((tmp_path / "one" / entry["replay"]).read_text())
        assert replay.verify().ply == 4
        assert entry["labeled_samples"] == 0
        assert json.loads((tmp_path / "one" / entry["data"]).read_text())["samples"]
    with pytest.raises(FileExistsError):
        collect(tmp_path / "one")


@pytest.mark.parametrize("kwargs", [{"episodes": 17}, {"max_plies": 129}, {"seed": 99999}])
def test_budget_and_evaluation_seed_guards(tmp_path, kwargs):
    with pytest.raises(ValueError):
        collect(tmp_path / "invalid", **kwargs)
    assert not (tmp_path / "invalid").exists()
