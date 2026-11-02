"""Rule fixtures for the explicitly incomplete Copenhagen implementation."""

import json
from dataclasses import replace
from pathlib import Path

import pytest

from hnef.actions import ActionCodec
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
from hnef.engine.copenhagen import defenders_encircled
from hnef.replay import Replay, random_game

A, D = Player.ATTACKERS, Player.DEFENDERS
COPENHAGEN = Ruleset.COPENHAGEN


@pytest.fixture(autouse=True, params=[Ruleset.COPENHAGEN, Ruleset.COPENHAGEN_V2])
def copenhagen_profile(request, monkeypatch):
    """Every ordinary-rule fixture must also exercise the current profile."""
    monkeypatch.setattr(__import__(__name__), "COPENHAGEN", request.param)


def position(pieces, player=A):
    pieces = dict(pieces)
    if "K" not in pieces.values():
        pieces[(5, 5)] = "K"
    board = [["."] * 11 for _ in range(11)]
    for (r, c), piece in pieces.items():
        board[r][c] = piece
    return GameState(tuple("".join(row) for row in board), player, ruleset=COPENHAGEN)


def rotate(square, turns):
    for _ in range(turns):
        square = square[1], 10 - square[0]
    return square


def rotated_position(pieces, turns, player=A):
    return position({rotate(square, turns): piece for square, piece in pieces.items()}, player)


def rotated_move(source, destination, turns):
    return Move(rotate(source, turns), rotate(destination, turns))


def test_copenhagen_opening():
    state = initial_state(COPENHAGEN)
    pieces = "".join(state.board)
    assert len(state.board) == 11
    assert pieces.count("A") == 24
    assert pieces.count("D") == 12
    assert pieces.count("K") == 1
    assert state.at((5, 5)) == "K"
    assert state.player is A
    assert state.ruleset is COPENHAGEN
    assert all(state.at(square) == "." for square in COPENHAGEN.corners)
    codec = ActionCodec(11)
    mask = codec.mask(state)
    assert len(mask) == 2420
    assert {codec.decode(i) for i, valid in enumerate(mask) if valid} == set(legal_moves(state))


@pytest.mark.parametrize("turns", range(4))
def test_king_wins_at_each_corner_but_not_an_ordinary_edge(turns):
    state = rotated_position({(2, 3): "K", (7, 7): "A"}, turns, D)
    edge = apply_move(state, rotated_move((2, 3), (0, 3), turns))
    assert edge.outcome is None
    edge = apply_move(edge, rotated_move((7, 7), (7, 8), turns))
    escaped = apply_move(edge, rotated_move((0, 3), (0, 0), turns))
    assert escaped.outcome == Outcome(D, "king_escaped")
    assert not legal_moves(escaped)
    assert escaped.outcome.reward(D) == 1


@pytest.mark.parametrize("piece,player", [("A", A), ("D", D)])
@pytest.mark.parametrize("turns", range(4))
def test_soldiers_cannot_land_on_corners(piece, player, turns):
    state = rotated_position({(0, 2): piece}, turns, player)
    assert rotated_move((0, 2), (0, 0), turns) not in legal_moves(state)
    with pytest.raises(ValueError, match="restricted"):
        rotated_position({(0, 0): piece}, turns, player)


def test_throne_can_be_crossed_but_only_occupied_by_king():
    state = position({(4, 4): "K", (5, 1): "A"})
    assert Move((5, 1), (5, 5)) not in legal_moves(state)
    assert Move((5, 1), (5, 8)) in legal_moves(state)
    state = position({(5, 4): "K", (8, 8): "A"}, D)
    assert Move((5, 4), (5, 5)) in legal_moves(state)


@pytest.mark.parametrize("player,victim,mover", [(A, "D", "A"), (D, "A", "D")])
@pytest.mark.parametrize("turns", range(4))
def test_corner_supports_capture_for_either_side(player, victim, mover, turns):
    state = rotated_position({(0, 1): victim, (2, 2): mover}, turns, player)
    successor = apply_move(state, rotated_move((2, 2), (0, 2), turns))
    assert successor.at(rotate((0, 1), turns)) == "."


@pytest.mark.parametrize("king", [(5, 5), (4, 5), (3, 3)])
def test_king_requires_four_sides_or_three_next_to_throne(king):
    r, c = king
    pieces = {king: "K", (r - 1, c): "A", (r, c - 1): "A", (r, c + 1): "A"}
    # Complete from below unless that side is the throne.
    if (r + 1, c) == COPENHAGEN.throne:
        del pieces[(r - 1, c)]
        source, destination = (r - 2, c), (r - 1, c)
    else:
        source, destination = (r + 2, c), (r + 1, c)
    pieces[source] = "A"
    successor = apply_move(position(pieces), Move(source, destination))
    assert successor.outcome == Outcome(A, "king_captured")
    assert not any("K" in row for row in successor.board)


@pytest.mark.parametrize("turns", range(4))
def test_two_sided_sandwich_does_not_capture_copenhagen_king(turns):
    state = rotated_position({(3, 3): "K", (3, 2): "A", (6, 4): "A"}, turns)
    successor = apply_move(state, rotated_move((6, 4), (3, 4), turns))
    assert successor.at(rotate((3, 3), turns)) == "K"
    assert successor.outcome is None


@pytest.mark.parametrize("turns", range(4))
def test_king_is_immune_to_capture_on_an_edge(turns):
    state = rotated_position(
        {(0, 3): "K", (0, 2): "A", (0, 4): "A", (3, 3): "A", (8, 8): "D"}, turns
    )
    successor = apply_move(state, rotated_move((3, 3), (1, 3), turns))
    assert successor.at(rotate((0, 3), turns)) == "K"
    assert successor.outcome is None


@pytest.mark.parametrize("turns", range(4))
@pytest.mark.parametrize("closing_column", [1, 5])
def test_shieldwall_captures_from_either_end_on_each_edge(turns, closing_column):
    pieces = {(0, c): "D" for c in (2, 3, 4)}
    pieces.update({(1, c): "A" for c in (2, 3, 4)})
    pieces[(0, 6 - closing_column)] = "A"
    pieces[(2, closing_column)] = "A"
    state = rotated_position(pieces, turns)
    successor = apply_move(state, rotated_move((2, closing_column), (0, closing_column), turns))
    assert all(successor.at(rotate((0, c), turns)) == "." for c in (2, 3, 4))
    assert sum(row.count("D") for row in state.board) == 3  # Parent remains untouched.


@pytest.mark.parametrize("turns", range(4))
def test_shieldwall_corner_can_replace_far_bracket(turns):
    pieces = {(0, 1): "D", (0, 2): "D", (1, 1): "A", (1, 2): "A", (2, 3): "A"}
    successor = apply_move(rotated_position(pieces, turns), rotated_move((2, 3), (0, 3), turns))
    assert successor.at(rotate((0, 1), turns)) == "."
    assert successor.at(rotate((0, 2), turns)) == "."


@pytest.mark.parametrize("turns", range(4))
def test_king_survives_in_a_captured_shieldwall(turns):
    pieces = {(0, 2): "K", (0, 3): "D", (0, 4): "D", (0, 1): "A", (2, 5): "A"}
    pieces.update({(1, c): "A" for c in (2, 3, 4)})
    successor = apply_move(rotated_position(pieces, turns), rotated_move((2, 5), (0, 5), turns))
    assert successor.at(rotate((0, 2), turns)) == "K"
    assert all(successor.at(rotate((0, c), turns)) == "." for c in (3, 4))
    assert successor.outcome is None


@pytest.mark.parametrize("king_square", [(1, 3), (2, 5)])
def test_king_can_back_or_close_a_defender_shieldwall(king_square):
    pieces = {(0, c): "A" for c in (2, 3, 4)}
    pieces.update({(1, c): "D" for c in (2, 3, 4)})
    pieces.update({(0, 1): "D", (2, 5): "D", king_square: "K"})
    successor = apply_move(position(pieces, D), Move((2, 5), (0, 5)))
    assert all(successor.at((0, c)) == "." for c in (2, 3, 4))


@pytest.mark.parametrize("missing", [(1, 3), (0, 3), (0, 1)])
def test_incomplete_shieldwall_does_not_capture(missing):
    pieces = {(0, c): "D" for c in (2, 3, 4)}
    pieces.update({(1, c): "A" for c in (2, 3, 4)})
    pieces.update({(0, 1): "A", (2, 5): "A"})
    del pieces[missing]
    successor = apply_move(position(pieces), Move((2, 5), (0, 5)))
    assert successor.at((0, 2)) == "D"
    assert successor.at((0, 4)) == "D"


def test_backing_move_does_not_trigger_shieldwall():
    pieces = {(0, c): "D" for c in (2, 3, 4)}
    pieces.update({(1, 2): "A", (1, 4): "A", (0, 1): "A", (0, 5): "A", (3, 3): "A"})
    successor = apply_move(position(pieces), Move((3, 3), (1, 3)))
    assert all(successor.at((0, c)) == "D" for c in (2, 3, 4))


def test_unrelated_move_does_not_retroactively_capture_a_shieldwall():
    pieces = {(0, c): "D" for c in (2, 3, 4)}
    pieces.update({(1, c): "A" for c in (2, 3, 4)})
    pieces.update({(0, 1): "A", (0, 5): "A", (7, 7): "A"})
    successor = apply_move(position(pieces), Move((7, 7), (7, 8)))
    assert all(successor.at((0, c)) == "D" for c in (2, 3, 4))


def test_trapped_edge_king_loses_by_immobility_not_capture():
    state = position({(0, 3): "K", (0, 2): "A", (0, 4): "A", (3, 3): "A"})
    successor = apply_move(state, Move((3, 3), (1, 3)))
    assert successor.at((0, 3)) == "K"
    assert successor.outcome == Outcome(A, "no_legal_moves")


def test_ordinary_and_shieldwall_captures_are_simultaneous():
    pieces = {(0, c): "D" for c in (2, 3, 4)}
    pieces.update({(1, c): "A" for c in (2, 3, 4)})
    pieces.update({(0, 1): "A", (0, 8): "A", (1, 5): "D", (2, 5): "A"})
    successor = apply_move(position(pieces), Move((0, 8), (0, 5)))
    assert all(successor.at((0, c)) == "." for c in (2, 3, 4))
    assert successor.at((1, 5)) == "."


def ring():
    pieces = {(r, c): "A" for r in range(3, 8) for c in range(3, 8) if r in (3, 7) or c in (3, 7)}
    pieces[(4, 4)] = "D"
    return pieces


def test_closed_attacker_ring_wins():
    pieces = ring()
    del pieces[(3, 5)]
    pieces[(1, 5)] = "A"
    state = position(pieces)
    assert not defenders_encircled(state.board)
    successor = apply_move(state, Move((1, 5), (3, 5)))
    assert successor.outcome == Outcome(A, "encircled")


@pytest.mark.parametrize("change", ["gap", "defender_in_wall", "defender_outside", "edge_king"])
def test_partial_enclosure_does_not_win(change):
    pieces = ring()
    if change == "gap":
        del pieces[(3, 5)]
    elif change == "defender_in_wall":
        pieces[(3, 5)] = "D"
    elif change == "defender_outside":
        pieces[(1, 1)] = "D"
    else:
        pieces[(0, 5)] = "K"
    assert not defenders_encircled(position(pieces).board)


def test_separate_enclosed_groups_are_all_checked():
    pieces = ring()
    pieces.update({(r, c): "A" for r, c in [(0, 1), (1, 0), (1, 2), (2, 1)]})
    pieces[(1, 1)] = "D"
    assert defenders_encircled(position(pieces).board)
    del pieces[(1, 2)]
    assert not defenders_encircled(position(pieces).board)


@pytest.mark.parametrize("player", [A, D])
def test_third_repetition_loses_for_defenders_regardless_of_mover(player):
    state = position({(1, 1): "A", (9, 9): "D"}, player)
    attacker_moves = [Move((1, 1), (1, 2)), Move((1, 2), (1, 1))]
    defender_moves = [Move((9, 9), (9, 8)), Move((9, 8), (9, 9))]
    first, second = (
        (attacker_moves, defender_moves) if player is A else (defender_moves, attacker_moves)
    )
    cycle = [first[0], second[0], first[1], second[1]]
    for move in cycle:
        state = apply_move(state, move)
    assert state.outcome is None
    for move in cycle:
        state = apply_move(state, move)
    assert state.outcome == Outcome(A, "defender_repetition")


def test_no_legal_moves_loses_after_last_attacker_is_captured():
    state = position({(3, 2): "D", (3, 3): "A", (6, 4): "D"}, D)
    successor = apply_move(state, Move((6, 4), (3, 4)))
    assert successor.outcome == Outcome(D, "no_legal_moves")


def test_mini_replay_written_before_ruleset_refactor_still_verifies():
    text = (Path(__file__).parent / "fixtures" / "mini-v1-seed42.json").read_text()
    record = Replay.from_json(text)
    assert record == random_game(42)
    assert record.to_json() == text


def test_copenhagen_replay_roundtrip_and_ruleset_mismatch():
    record = random_game(42, 30, ruleset=COPENHAGEN)
    assert record == random_game(42, 30, ruleset=COPENHAGEN)
    assert Replay.from_json(record.to_json()) == record
    assert record.verify().ruleset is COPENHAGEN
    data = json.loads(record.to_json())
    data["ruleset"] = Ruleset.MINI.value
    with pytest.raises(ValueError):
        Replay.from_json(json.dumps(data))


def test_ruleset_and_board_size_must_match():
    with pytest.raises(ValueError, match="requires"):
        replace(initial_state(), ruleset=COPENHAGEN, history=())
    with pytest.raises(ValueError):
        initial_state("copenhagen")


def test_cli_labels_experimental_game_and_replay(tmp_path, capsys):
    path = tmp_path / "copenhagen.json"
    main(["play", "--ruleset", COPENHAGEN.value, "--max-plies", "1", "--output", str(path)])
    output = capsys.readouterr()
    assert "Experimental Copenhagen" in output.err
    assert "10" in output.out.splitlines()[0]
    main(["replay", str(path)])
    replayed = capsys.readouterr()
    assert replayed == output
