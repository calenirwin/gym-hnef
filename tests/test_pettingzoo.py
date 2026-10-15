"""The optional RL extra is required to run the adapter suite (enabled in CI)."""

from pathlib import Path

import pytest

np = pytest.importorskip("numpy")
pytest.importorskip("pettingzoo")

from pettingzoo.test import api_test, seed_test  # noqa: E402

from hnef.engine import Move, Player, Ruleset, apply_move, initial_state, legal_moves  # noqa: E402
from hnef.envs.pettingzoo import env, raw_env  # noqa: E402
from hnef.replay import Replay, random_game  # noqa: E402


@pytest.mark.parametrize("ruleset", list(Ruleset))
def test_official_aec_compliance(ruleset):
    api_test(env(ruleset=ruleset, max_plies=60), num_cycles=100)
    seed_test(lambda: env(ruleset=ruleset, max_plies=60), num_cycles=100)


@pytest.mark.parametrize("ruleset", list(Ruleset))
def test_observation_planes_masks_and_history(ruleset):
    game = raw_env(ruleset=ruleset, max_plies=40)
    game.reset(seed=12)
    state = initial_state(ruleset)
    actions = []
    for _ in range(40):
        assert game.game_state == state
        for agent in game.possible_agents:
            obs = game.observe(agent)
            assert game.observation_space(agent).contains(obs)
            board = np.array([list(row) for row in state.board])
            for index, piece in enumerate("ADK"):
                np.testing.assert_array_equal(obs["observation"][index], board == piece)
            assert obs["observation"][3].sum() == 1
            assert obs["observation"][4].sum() == (4 if ruleset.is_copenhagen else 0)
            assert np.all(obs["observation"][5] == (state.player == Player.DEFENDERS))
            assert np.all(obs["observation"][6] == (agent == "defenders"))
            assert list(obs["action_history"][: state.ply]) == actions
            assert np.all(obs["action_history"][state.ply :] == -1)
            legal_ids = {game.codec.encode(move) for move in legal_moves(state)}
            expected = legal_ids if agent == state.player.value else set()
            assert set(np.flatnonzero(obs["action_mask"])) == expected
        action = game.action_space(game.agent_selection).sample(
            game.observe(game.agent_selection)["action_mask"]
        )
        state = apply_move(state, game.codec.decode(int(action)))
        game.step(action)
        actions.append(int(action))
        if state.outcome:
            break
    reconstructed = initial_state(ruleset)
    for action in actions:
        reconstructed = apply_move(reconstructed, game.codec.decode(action))
    assert reconstructed == game.game_state
    assert game.infos["attackers"]["experimental"] == ruleset.is_copenhagen


@pytest.mark.parametrize("seed", [0, 9, 42])
def test_terminal_rewards_delivered_once_to_each_player(seed):
    replay = random_game(seed, 500)
    outcome = replay.verify().outcome
    assert outcome is not None
    # A terminal move at the collection limit must win over truncation.
    game = env(max_plies=len(replay.actions))
    game.reset(seed=seed)
    for action in replay.actions:
        assert game.last()[1:4] == (0, False, False)
        game.step(action)
    seen = {}
    for agent in game.agent_iter():
        obs, reward, terminated, truncated, info = game.last()
        assert terminated and not truncated
        assert not obs["action_mask"].any()
        assert info["reason"] == outcome.reason
        assert reward == outcome.reward(Player(agent))
        seen[agent] = reward
        game.step(None)
    assert len(seen) == 2
    assert sum(seen.values()) == 0
    assert game.agents == []
    game.reset(seed=seed)
    assert game.last()[1:4] == (0, False, False)
    assert game.unwrapped.game_state == initial_state()


def test_frozen_replay_matches_every_aec_transition():
    replay = Replay.from_json((Path(__file__).parent / "fixtures/mini-v1-seed42.json").read_text())
    game = env()
    game.reset()
    assert game.unwrapped.game_state.key == replay.state_keys[0]
    for action, key in zip(replay.actions, replay.state_keys[1:], strict=True):
        game.step(action)
        assert game.unwrapped.game_state.key == key


@pytest.mark.parametrize("ruleset", list(Ruleset))
def test_repetition_rewards_and_terminal_history(ruleset):
    game = raw_env(ruleset=ruleset)
    game.reset()
    if ruleset == Ruleset.MINI:
        moves = [((0, 1), (0, 2)), ((1, 2), (1, 1))]
    else:
        moves = [((0, 3), (1, 3)), ((3, 5), (3, 4))]
    cycle = [Move(a, b) for a, b in moves] + [Move(b, a) for a, b in moves]
    for move in cycle * 2:
        game.step(game.codec.encode(move))
    assert all(game.terminations.values())
    assert game.game_state.outcome.reason == (
        "defender_repetition" if ruleset.is_copenhagen else "repetition"
    )
    assert game.rewards == (
        {"attackers": 1, "defenders": -1}
        if ruleset.is_copenhagen
        else {"attackers": 0, "defenders": 0}
    )


@pytest.mark.parametrize("max_plies", [1, 2, 3])
def test_time_limit_is_not_a_draw(max_plies):
    game = raw_env(max_plies=max_plies)
    game.reset()
    for _ in range(max_plies):
        game.step(int(np.flatnonzero(game.last()[0]["action_mask"])[0]))
    assert game.game_state.outcome is None
    for _agent in game.agent_iter():
        obs, reward, terminated, truncated, info = game.last()
        assert not terminated and truncated and reward == 0
        assert info["reason"] is None and info["time_limit_reached"]
        assert not obs["action_mask"].any()
        with pytest.raises(ValueError):
            game.step(0)
        game.step(None)


@pytest.mark.parametrize("action", [None, True, np.bool_(False), 1.0, "1", -1, 200, 0])
def test_invalid_action_leaves_state_untouched(action):
    game = raw_env()
    game.reset()
    before = game.game_state
    with pytest.raises(ValueError):
        game.step(action)
    assert game.game_state is before
    assert game.agent_selection == "attackers"
    assert game.rewards == {"attackers": 0, "defenders": 0}
    assert np.all(game.observe("attackers")["action_history"] == -1)


def test_observations_cannot_mutate_environment():
    game = raw_env()
    game.reset()
    original = game.observe("attackers")
    for value in game.observe("attackers").values():
        value[:] = 0
    for key, value in game.observe("attackers").items():
        np.testing.assert_array_equal(value, original[key])


def test_same_seed_reproduces_masked_sampled_episode():
    game = raw_env()

    def run():
        game.reset(seed=127)
        keys = []
        for agent in game.agent_iter():
            obs, _, terminated, truncated, _ = game.last()
            action = (
                None
                if terminated or truncated
                else game.action_space(agent).sample(obs["action_mask"])
            )
            game.step(action)
            keys.append(game.game_state.key)
        return keys

    assert run() == run()


@pytest.mark.parametrize(
    "kwargs",
    [
        {"max_plies": 0},
        {"max_plies": True},
        {"max_plies": 1.5},
        {"ruleset": "full-copenhagen"},
        {"render_mode": "rgb_array"},
    ],
)
def test_invalid_configuration(kwargs):
    with pytest.raises(ValueError):
        raw_env(**kwargs)


def test_render_and_call_order(capsys):
    game = env(render_mode="ansi")
    with pytest.raises(AssertionError):
        game.step(0)
    game.reset()
    text = game.render()
    assert "mini-v1 | ply 0 | attackers to move" in text
    assert ". A . A ." in text
    game.close()
    human = raw_env(render_mode="human")
    human.reset()
    assert capsys.readouterr().out.rstrip() == text
    with pytest.raises(KeyError):
        human.observe("unknown")
