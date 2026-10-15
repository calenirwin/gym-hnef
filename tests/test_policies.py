import json
import math
from dataclasses import replace

import pytest

from hnef.cli import main
from hnef.engine import (
    GameState,
    Move,
    Outcome,
    Player,
    Ruleset,
    apply_move,
    initial_state,
    legal_moves,
)
from hnef.matches import play_game, policy_game
from hnef.policies import HeuristicPolicy, MCTSPolicy, RandomPolicy, _Node, evaluate, make_policy
from hnef.replay import Replay, random_game

A, D = Player.ATTACKERS, Player.DEFENDERS


@pytest.mark.parametrize("ruleset", list(Ruleset))
@pytest.mark.parametrize("name", ["random", "heuristic", "mcts"])
def test_seeded_policies_are_legal_deterministic_and_do_not_mutate(ruleset, name):
    state = initial_state(ruleset)
    before = state.key
    first = make_policy(name, 12, iterations=4, rollout_plies=2).choose_move(state)
    second = make_policy(name, 12, iterations=4, rollout_plies=2).choose_move(state)
    assert first == second
    assert first in legal_moves(state)
    assert state.key == before


@pytest.mark.parametrize("player", [A, D])
@pytest.mark.parametrize("name", ["heuristic", "mcts"])
def test_take_immediate_escape_or_king_capture(player, name):
    state = (
        GameState(("A....", "..K..", ".....", ".....", "....D"), D)
        if player is D
        else GameState(("..A..", ".AK..", ".....", "...A.", "....D"), A)
    )
    policy = make_policy(name, 4, iterations=len(legal_moves(state)), rollout_plies=0)
    successor = apply_move(state, policy.choose_move(state))
    assert successor.outcome == Outcome(player, "king_escaped" if player is D else "king_captured")


@pytest.mark.parametrize("player", [A, D])
def test_uct_selection_models_opponent_instead_of_cooperation(player):
    # Same root-value estimates at either player's turn: opponent must pick low.
    state = GameState(("A....", ".....", "..K..", ".....", "....D"), player)
    parent = _Node(state, visits=20)
    high = _Node(state, visits=10, value_sum=8)
    low = _Node(state, visits=10, value_sum=-8)
    parent.children = {Move((0, 0), (0, 1)): high, Move((0, 0), (0, 2)): low}
    selected = MCTSPolicy(exploration=0)._select(parent, A)
    assert selected is (high if player is A else low)


def test_search_budget_values_and_cutoffs():
    state = initial_state()
    result = MCTSPolicy(7, iterations=9, rollout_plies=0).search(state)
    assert result.iterations == 9
    assert sum(child.visits for child in result.children) == 9
    assert result.rollout_cutoffs == 9
    assert all(child.mean_value == 0 for child in result.children)
    assert len(result.children) == 9
    assert state.outcome is None
    assert result == MCTSPolicy(7, iterations=9, rollout_plies=0).search(state)


def test_search_revisits_children_and_selects_by_visits():
    state = initial_state()
    result = MCTSPolicy(21, iterations=100, rollout_plies=12).search(state)
    assert sum(child.visits for child in result.children) == 100
    assert {child.move for child in result.children} == set(legal_moves(state))
    assert all(-1 <= child.mean_value <= 1 for child in result.children)
    selected = next(child for child in result.children if child.move == result.move)
    assert selected.visits == max(child.visits for child in result.children)


@pytest.mark.parametrize("ruleset", list(Ruleset))
def test_mcts_uses_real_repetition_history_and_root_reward(ruleset):
    state = initial_state(ruleset)
    moves = (
        [Move((0, 1), (0, 2)), Move((1, 2), (1, 1))]
        if ruleset is Ruleset.MINI
        else [Move((0, 3), (1, 3)), Move((3, 5), (3, 4))]
    )
    cycle = moves + [Move(move.destination, move.source) for move in moves]
    for move in (cycle * 2)[:-1]:
        state = apply_move(state, move)
    assert state.player is D
    result = MCTSPolicy(iterations=len(legal_moves(state)), rollout_plies=0).search(state)
    repeated = next(child for child in result.children if child.move == cycle[-1])
    assert repeated.mean_value == (-1 if ruleset.is_copenhagen else 0)
    assert result.rollout_cutoffs < result.iterations
    assert state.outcome is None and state.ply == 7


def test_rollout_checks_outcome_on_final_allowed_ply():
    state = GameState(("A....", "..K..", ".....", ".....", "....D"), D)
    policy = MCTSPolicy(rollout_plies=1)

    # Force a legal escape on the last rollout move without changing engine logic.
    class EscapeRNG:
        def choice(self, moves):
            move = Move((1, 2), (0, 2))
            assert move in moves
            return move

    policy.rng = EscapeRNG()
    assert policy._rollout(state, D) == (1.0, False)
    assert policy._rollout(state, A) == (-1.0, False)


@pytest.mark.parametrize("ruleset", list(Ruleset))
def test_evaluation_is_zero_sum_bounded_and_prefers_fewer_enemies(ruleset):
    state = initial_state(ruleset)
    assert -1 < evaluate(state, A) < 1
    assert evaluate(state, A) == -evaluate(state, D)
    rows = list(state.board)
    rows[0] = rows[0].replace("A", ".", 1)
    fewer_attackers = GameState(tuple(rows), ruleset=ruleset)
    assert evaluate(fewer_attackers, D) > evaluate(state, D)
    rows = list(state.board)
    for index, row in enumerate(rows):
        if "D" in row:
            rows[index] = row.replace("D", ".", 1)
            break
    fewer_defenders = GameState(tuple(rows), ruleset=ruleset)
    assert evaluate(fewer_defenders, A) > evaluate(state, A)
    draw = replace(state, outcome=Outcome(None, "repetition"))
    assert evaluate(draw, A) == evaluate(draw, D) == 0


@pytest.mark.parametrize("factory", [RandomPolicy, HeuristicPolicy, MCTSPolicy])
def test_policies_reject_finished_and_immobile_states(factory):
    finished = random_game(42).verify()
    immobile = GameState((".....", ".....", "..K..", ".....", "....."), A)
    for state in (finished, immobile):
        with pytest.raises(ValueError, match="terminal or immobile"):
            factory().choose_move(state)


@pytest.mark.parametrize(
    "kwargs",
    [
        {"iterations": 0},
        {"iterations": True},
        {"iterations": 1.5},
        {"rollout_plies": -1},
        {"rollout_plies": True},
        {"rollout_plies": 1.5},
        {"exploration": -1},
        {"exploration": math.nan},
        {"exploration": math.inf},
        {"exploration": True},
        {"exploration": "2"},
    ],
)
def test_invalid_search_configuration(kwargs):
    with pytest.raises(ValueError):
        MCTSPolicy(**kwargs)


@pytest.mark.parametrize("ruleset", list(Ruleset))
def test_policy_games_verify_and_legacy_random_sequences_survive(ruleset):
    assert policy_game(42, 30, ruleset=ruleset) == random_game(42, 30, ruleset=ruleset)
    kwargs = dict(
        ruleset=ruleset, attackers="mcts", defenders="heuristic", iterations=3, rollout_plies=2
    )
    first = policy_game(42, 3, **kwargs)
    assert first == policy_game(42, 3, **kwargs)
    assert Replay.from_json(first.to_json()) == first
    assert first.verify().ply <= 3


def test_match_rejects_illegal_policy_moves_and_invalid_settings():
    class IllegalPolicy:
        def choose_move(self, state):
            return Move((0, 0), (0, 1))

    with pytest.raises(ValueError, match="Illegal move"):
        play_game(IllegalPolicy(), RandomPolicy())
    with pytest.raises(ValueError, match="Unknown policy"):
        policy_game(0, attackers="unknown")
    with pytest.raises(ValueError, match="max_plies"):
        play_game(RandomPolicy(), RandomPolicy(), max_plies=0)


def test_cli_policy_game_and_reproducible_report(tmp_path, capsys):
    replay_path, report_path = tmp_path / "game.json", tmp_path / "report.json"
    flags = [
        "--attackers",
        "mcts",
        "--defenders",
        "heuristic",
        "--iterations",
        "3",
        "--rollout-plies",
        "2",
        "--max-plies",
        "3",
        "--seed",
        "42",
    ]
    main(["play", *flags, "--output", str(replay_path)])
    expected = capsys.readouterr().out
    main(["replay", str(replay_path)])
    assert capsys.readouterr().out == expected
    main(["benchmark", *flags, "--games", "2", "--output", str(report_path)])
    report = json.loads(capsys.readouterr().out)
    assert report == json.loads(report_path.read_text())
    assert sum(report["results"].values()) == report["games"] == 2
    assert report["config"] == {
        "policy_version": 1,
        "ruleset": "mini-v1",
        "attackers": "mcts",
        "defenders": "heuristic",
        "iterations": 3,
        "rollout_plies": 2,
        "seed": 42,
        "max_plies": 3,
    }
    main(["benchmark", *flags, "--games", "2"])
    assert json.loads(capsys.readouterr().out) == report


@pytest.mark.parametrize(
    "flags", [["--iterations", "0"], ["--rollout-plies", "-1"], ["--attackers", "unknown"]]
)
def test_cli_rejects_bad_policy_options(flags):
    with pytest.raises(SystemExit) as error:
        main(["play", *flags])
    assert error.value.code == 2
