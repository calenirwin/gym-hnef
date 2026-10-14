from dataclasses import FrozenInstanceError, replace

import pytest

from hnef.engine import GameState, Move, Outcome, Player, apply_move, initial_state, legal_moves
from hnef.engine.state import position_key

A, D = Player.ATTACKERS, Player.DEFENDERS


def position(*rows, player=A):
    return GameState(rows, player)


def test_original_mini_opening_and_immutable_state():
    state = initial_state()
    assert state.board == (".A.A.", "A.D.A", ".DKD.", "A.D.A", ".A.A.")
    assert "".join(state.board).count("A") == 8
    assert "".join(state.board).count("D") == 4
    with pytest.raises(FrozenInstanceError):
        state.ply = 2
    mutable_rows = list(state.board)
    copied = GameState(mutable_rows)
    mutable_rows[0] = "....."
    assert copied == state


def test_move_lists_are_copied_to_tuples():
    source = [0, 1]
    move = Move(source, [0, 0])
    source[0] = 4
    assert move.source == (0, 1)


@pytest.mark.parametrize("square", [(0,), (0, 1, 2), (0, True), (0, 1.0)])
def test_invalid_coordinates(square):
    with pytest.raises(ValueError):
        Move(square, (1, 1))


@pytest.mark.parametrize(
    "rows",
    [
        (),
        (".....",) * 5,
        ("..K..",) * 5,
        (".....", ".....", "..A..", "..K..", "....."),
        (".....", ".....", "..K..", "..X..", "....."),
    ],
)
def test_invalid_boards(rows):
    with pytest.raises(ValueError):
        GameState(rows)


def test_straight_movement_stops_at_occupied_square():
    state = position(".....", ".A.D.", "..K..", ".....", ".....")
    destinations = {move.destination for move in legal_moves(state) if move.source == (1, 1)}
    assert destinations == {(0, 1), (2, 1), (3, 1), (4, 1), (1, 0), (1, 2)}


def test_nonking_can_cross_but_not_stop_on_empty_throne():
    state = position(".....", "...K.", "A....", ".....", ".....")
    moves = legal_moves(state)
    assert Move((2, 0), (2, 2)) not in moves
    assert Move((2, 0), (2, 3)) in moves
    assert Move((2, 0), (2, 4)) in moves


def test_king_can_return_to_throne():
    state = position("A....", "..K..", ".....", ".....", ".....", player=D)
    assert Move((1, 2), (2, 2)) in legal_moves(state)


@pytest.mark.parametrize(
    "move",
    [
        Move((0, 1), (1, 2)),
        Move((2, 2), (2, 1)),
        Move((0, 1), (-1, 1)),
        Move((0, 1), (0, 1)),
        Move((0, 1), (0, 3)),
        Move((0, 0), (0, 2)),
    ],
)
def test_illegal_moves_leave_state_unchanged(move):
    state = initial_state()
    before = state.key
    with pytest.raises(ValueError):
        apply_move(state, move)
    assert state.key == before


def test_apply_is_deterministic_and_does_not_mutate_parent():
    state = initial_state()
    move = legal_moves(state)[0]
    successor = apply_move(state, move)
    assert successor == apply_move(state, move)
    assert state == initial_state()
    assert successor.ply == 1 and successor.player == D
    assert successor.history[:-1] == state.history


def test_simultaneous_captures():
    state = position("..A..", "AD.DA", "..K..", ".....", ".....")
    successor = apply_move(state, Move((0, 2), (1, 2)))
    assert successor.board[1] == "A.A.A"


@pytest.mark.parametrize(
    "player,rows,move,victim",
    [
        (A, ("..A..", "..D..", ".....", "...K.", "....."), Move((0, 2), (0, 1)), (1, 2)),
        (A, ("A....", "..D..", ".....", "...K.", "....."), Move((0, 0), (0, 2)), (1, 2)),
        (D, ("D....", "..A..", "..K..", ".....", "....."), Move((0, 0), (0, 2)), (1, 2)),
    ],
)
def test_throne_capture_requires_move_against_victim(player, rows, move, victim):
    state = GameState(rows, player)
    successor = apply_move(state, move)
    expected = "D" if move.destination == (0, 1) else "."
    assert successor.at(victim) == expected


def test_occupied_throne_protects_defender():
    state = position("A....", "..D..", "..K..", ".....", ".....")
    successor = apply_move(state, Move((0, 0), (0, 2)))
    assert successor.at((1, 2)) == "D"


def test_moving_between_enemies_does_not_capture_self():
    state = position(".....", "A.A..", ".....", ".D.K.", ".....", player=D)
    successor = apply_move(state, Move((3, 1), (1, 1)))
    assert successor.at((1, 1)) == "D"


@pytest.mark.parametrize(
    "rows,move",
    [
        ((".....", "..A..", ".AKA.", ".....", "..A.."), Move((4, 2), (3, 2))),
        (("A....", ".AKA.", ".....", ".....", "....."), Move((0, 0), (0, 2))),
        ((".....", "AK...", ".....", ".....", "..A.."), Move((4, 2), (1, 2))),
    ],
)
def test_king_capture_on_next_to_and_away_from_throne(rows, move):
    state = GameState(rows)
    successor = apply_move(state, move)
    assert successor.outcome == Outcome(A, "king_captured")
    assert not any("K" in row for row in successor.board)
    assert successor.player == D
    assert successor.outcome.reward(A) == 1
    assert successor.outcome.reward(D) == -1
    assert state.outcome is None


def test_king_next_to_throne_survives_two_sided_sandwich():
    state = position(".....", ".AK..", ".....", ".....", "...A.")
    successor = apply_move(state, Move((4, 3), (1, 3)))
    assert successor.at((1, 2)) == "K"
    assert successor.outcome is None


def test_king_can_participate_in_capture():
    state = position(".....", ".DA..", ".....", "...K.", ".....", player=D)
    successor = apply_move(state, Move((3, 3), (1, 3)))
    assert successor.at((1, 2)) == "."


@pytest.mark.parametrize("destination", [(0, 2), (4, 2), (2, 0), (2, 4)])
def test_king_escapes_to_any_edge_and_reward_uses_winner(destination):
    state = position("A....", ".....", "..K..", ".....", ".....", player=D)
    successor = apply_move(state, Move((2, 2), destination))
    assert successor.outcome == Outcome(D, "king_escaped")
    assert successor.outcome.reward(D) == 1
    assert successor.player == A
    assert legal_moves(successor) == ()
    with pytest.raises(ValueError):
        apply_move(successor, Move((0, 0), (0, 1)))


def test_player_with_no_moves_loses():
    state = position("A....", "DA...", "..K..", ".....", ".....", player=D)
    successor = apply_move(state, Move((2, 2), (1, 2)))
    # The defender captures the last movable attacker against its own defender.
    assert successor.at((1, 1)) == "."
    # Remaining corner attacker can still move, so this is not a terminal position.
    assert successor.outcome is None
    state = position(".....", "DA...", "..K..", ".....", ".....", player=D)
    successor = apply_move(state, Move((2, 2), (1, 2)))
    assert successor.outcome == Outcome(D, "no_legal_moves")


def test_threefold_repetition_includes_initial_position_and_turn():
    state = position("A....", ".....", "..K..", ".....", "....D")
    cycle = [Move((0, 0), (0, 1)), Move((4, 4), (4, 3)), Move((0, 1), (0, 0)), Move((4, 3), (4, 4))]
    for move in cycle:
        state = apply_move(state, move)
    assert state.outcome is None
    for move in cycle:
        state = apply_move(state, move)
    assert state.outcome == Outcome(None, "repetition")
    assert state.outcome.reward(A) == state.outcome.reward(D) == 0


def test_state_identity_includes_player_and_history():
    state = initial_state()
    other_turn = replace(state, player=D, history=())
    assert state.key != other_turn.key
    assert position_key(state.board, A) != position_key(state.board, D)
    other_history = replace(state, history=(state.history[0], state.history[0]))
    assert other_history.key != state.key
