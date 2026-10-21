"""Published fort positions, counterexamples, and explicit draw event recording."""

import json
from dataclasses import replace
from pathlib import Path

import pytest

from hnef.cli import main
from hnef.engine import (
    GameState,
    Move,
    Outcome,
    Player,
    Ruleset,
    adjudicate_draw,
    apply_move,
    initial_state,
    legal_moves,
    prove_exit_fort,
)
from hnef.replay import Replay, random_game

A, D = Player.ATTACKERS, Player.DEFENDERS
V1, V2 = Ruleset.COPENHAGEN, Ruleset.COPENHAGEN_V2
FIXTURES = json.loads((Path(__file__).parent / "fixtures" / "exit-forts.json").read_text())


def rotate(square, turns):
    for _ in range(turns):
        square = square[1], 10 - square[0]
    return square


def pieces_of(board):
    return {
        (r, c): piece for r, row in enumerate(board) for c, piece in enumerate(row) if piece != "."
    }


def position(pieces, player=D, ruleset=V2, turns=0):
    board = [["."] * 11 for _ in range(11)]
    for square, piece in pieces.items():
        r, c = rotate(square, turns)
        board[r][c] = piece
    return GameState(tuple("".join(row) for row in board), player, ruleset=ruleset)


@pytest.mark.parametrize("fixture", FIXTURES, ids=lambda fixture: fixture["name"])
@pytest.mark.parametrize("turns", range(4))
def test_published_forts_are_proved_on_every_edge(fixture, turns):
    pieces = pieces_of(fixture["board"])
    # The published diagrams omit opponents; add one outside the fort.
    pieces[(0, 5)] = "A"
    state = position(pieces, turns=turns)
    proof = prove_exit_fort(state.board)
    assert proof is not None
    assert set(proof.room) == {rotate(tuple(square), turns) for square in fixture["room"]}
    assert all(state.at(square) == "D" for square in proof.wall)
    assert any(state.at(square) == "K" for square in proof.room)


@pytest.mark.parametrize("fixture", FIXTURES, ids=lambda fixture: fixture["name"])
@pytest.mark.parametrize("turns", range(4))
def test_closing_a_fort_wins_and_prevents_further_moves(fixture, turns):
    pieces = pieces_of(fixture["board"])
    source, destination = (tuple(square) for square in fixture["closing_move"])
    del pieces[destination]
    pieces[source] = "D"
    pieces[(0, 5)] = "A"
    state = position(pieces, turns=turns)
    assert prove_exit_fort(state.board) is None
    move = Move(rotate(source, turns), rotate(destination, turns))
    successor = apply_move(state, move)
    assert successor.outcome == Outcome(D, "exit_fort")
    assert successor.player is A
    assert successor.outcome.reward(D) == 1
    assert successor.outcome.reward(A) == -1
    assert state.outcome is None
    assert legal_moves(successor) == ()
    with pytest.raises(ValueError, match="Illegal move"):
        apply_move(successor, move)


@pytest.mark.parametrize("fixture", FIXTURES, ids=lambda fixture: fixture["name"])
def test_each_published_wall_piece_is_needed(fixture):
    pieces = pieces_of(fixture["board"])
    pieces[(0, 5)] = "A"
    for square, piece in pieces.items():
        if piece == "D":
            reduced = {key: value for key, value in pieces.items() if key != square}
            assert prove_exit_fort(position(reduced).board) is None, square


def test_immobile_king_is_not_an_exit_fort():
    pieces = pieces_of(FIXTURES[0]["board"])
    pieces[(10, 5)] = "D"
    pieces[(0, 5)] = "A"
    assert prove_exit_fort(position(pieces).board) is None


def test_interior_fort_is_not_an_exit_fort():
    pieces = {(r - 2, c): piece for (r, c), piece in pieces_of(FIXTURES[0]["board"]).items()}
    pieces[(9, 4)] = "D"
    pieces[(9, 5)] = "D"
    pieces[(0, 5)] = "A"
    assert prove_exit_fort(position(pieces).board) is None


def test_attacker_inside_fort_prevents_proof():
    pieces = pieces_of(FIXTURES[1]["board"])
    pieces[(9, 5)] = "A"
    assert prove_exit_fort(position(pieces).board) is None


def test_open_edge_and_corner_escape_are_not_reported_as_forts():
    for king in ((0, 5), (0, 0)):
        assert prove_exit_fort(position({king: "K", (5, 4): "D"}).board) is None


def test_no_king_is_not_a_fort():
    assert prove_exit_fort(("...........",) * 11) is None


def test_fort_proof_does_not_depend_on_sacrificial_outside_defenders():
    pieces = pieces_of(FIXTURES[0]["board"])
    pieces[(0, 5)] = "A"
    pieces[(4, 4)] = "D"  # The hypothetical opponent can capture this immediately.
    proof = prove_exit_fort(position(pieces).board)
    assert proof is not None
    assert (4, 4) not in proof.wall


@pytest.mark.parametrize("fixture", FIXTURES, ids=lambda fixture: fixture["name"])
def test_certified_wall_survives_every_legal_attacker_move(fixture):
    pieces = pieces_of(fixture["board"])
    pieces.update({square: "A" for square in ((7, 3), (7, 6), (8, 3), (8, 7), (10, 2), (10, 8))})
    state = position(pieces, A)
    proof = prove_exit_fort(state.board)
    assert proof is not None
    assert legal_moves(state)
    for move in legal_moves(state):
        successor = apply_move(state, move)
        assert all(successor.at(square) == "D" for square in proof.wall), move
        assert successor.outcome == Outcome(D, "exit_fort"), move
        assert prove_exit_fort(successor.board) is not None, move


@pytest.mark.parametrize("fixture", FIXTURES, ids=lambda fixture: fixture["name"])
def test_king_can_move_inside_certified_room_without_weakening_wall(fixture):
    pieces = pieces_of(fixture["board"])
    pieces[(0, 5)] = "A"
    state = position(pieces)
    proof = prove_exit_fort(state.board)
    king_moves = [move for move in legal_moves(state) if state.at(move.source) == "K"]
    assert proof is not None and king_moves
    for move in king_moves:
        assert move.destination in proof.room
        successor = apply_move(state, move)
        assert all(successor.at(square) == "D" for square in proof.wall)
        if any(coordinate in (0, 10) for coordinate in move.destination):
            assert successor.outcome == Outcome(D, "exit_fort")
        else:
            # Room safety persists, but the rule requires the king itself on an edge.
            assert successor.outcome is None
            successor = apply_move(successor, Move((0, 5), (0, 6)))
            successor = apply_move(successor, Move(move.destination, move.source))
            assert successor.outcome == Outcome(D, "exit_fort")


def test_safe_for_one_move_is_not_an_unbreakable_wall():
    pieces = pieces_of(FIXTURES[1]["board"])
    del pieces[(8, 4)]  # Exposes the top defender to a future two-sided capture.
    pieces.update({(8, 2): "A", (6, 6): "A"})
    state = position(pieces, A)
    assert prove_exit_fort(state.board) is None
    state = apply_move(state, Move((8, 2), (8, 4)))
    assert state.outcome is None
    assert state.at((8, 5)) == "D"  # No immediate capture, but the fort is unsound.
    state = apply_move(state, Move((10, 5), (9, 5)))
    state = apply_move(state, Move((6, 6), (8, 6)))
    assert state.at((8, 5)) == "."
    assert prove_exit_fort(state.board) is None


def test_shieldwall_formation_is_not_an_exit_fort():
    pieces = {(0, 2): "D", (0, 3): "K", (0, 4): "D", (0, 1): "A", (0, 5): "A"}
    pieces.update({(1, c): "A" for c in (2, 3, 4)})
    assert prove_exit_fort(position(pieces).board) is None


def test_v1_keeps_its_original_nonadjudication_behavior():
    pieces = pieces_of(FIXTURES[0]["board"])
    pieces[(0, 5)] = "A"
    v1 = position(pieces, ruleset=V1)
    v2 = position(pieces)
    move = Move((10, 4), (10, 5))
    assert apply_move(v1, move).outcome is None
    assert apply_move(v2, move).outcome == Outcome(D, "exit_fort")


def test_fort_win_precedes_repetition():
    pieces = pieces_of(FIXTURES[0]["board"])
    pieces[(0, 5)] = "A"
    state = position(pieces)
    move = Move((10, 4), (10, 5))
    terminal_key = apply_move(state, move).history[-1]
    state = replace(state, history=(terminal_key, terminal_key, *state.history))
    assert apply_move(state, move).outcome == Outcome(D, "exit_fort")


def test_low_material_is_not_automatically_drawn():
    state = position({(0, 5): "K", (7, 7): "A"})
    successor = apply_move(state, Move((0, 5), (0, 0)))
    assert successor.outcome == Outcome(D, "king_escaped")


def test_adjudicated_draw_preserves_position_and_has_zero_reward():
    state = initial_state(V2)
    successor = adjudicate_draw(state)
    assert successor.outcome == Outcome(None, "adjudicated_draw")
    assert successor.outcome.reward(A) == successor.outcome.reward(D) == 0
    assert successor.board == state.board
    assert successor.player == state.player
    assert successor.ply == state.ply
    assert successor.history == state.history
    assert successor.key != state.key
    assert state.outcome is None
    assert legal_moves(successor) == ()


@pytest.mark.parametrize("ruleset", [Ruleset.MINI, V1])
def test_draw_adjudication_does_not_change_old_profiles(ruleset):
    with pytest.raises(ValueError, match="requires"):
        adjudicate_draw(initial_state(ruleset))


def test_adjudication_cannot_override_a_win_or_existing_draw():
    state = position({(0, 5): "K", (7, 7): "A"})
    win = apply_move(state, Move((0, 5), (0, 0)))
    for finished in (win, adjudicate_draw(initial_state(V2))):
        with pytest.raises(ValueError, match="already finished"):
            adjudicate_draw(finished)


def test_draw_replay_contains_an_explicit_verified_event():
    record = random_game(42, 4, ruleset=V2)
    drawn = record.adjudicate_draw("Players agreed the blocked endgame is a draw.")
    data = json.loads(drawn.to_json())
    assert data["version"] == 2
    assert data["adjudication"] == {"kind": "draw", "note": drawn.adjudication_note}
    assert drawn.actions == record.actions
    assert drawn.state_keys[:-1] == record.state_keys
    assert drawn.verify().ply == record.verify().ply == 4
    assert Replay.from_json(drawn.to_json()) == drawn
    with pytest.raises(ValueError, match="already finished"):
        drawn.adjudicate_draw("Again")


@pytest.mark.parametrize("note", ["", "  ", 3, [], None])
def test_draw_record_requires_a_written_explanation(note):
    with pytest.raises(ValueError):
        random_game(1, 1, ruleset=V2).adjudicate_draw(note)


@pytest.mark.parametrize(
    "corruption", ["kind", "note", "hash", "missing_event", "version", "extra"]
)
def test_invalid_adjudication_records_are_rejected(corruption):
    record = random_game(1, 1, ruleset=V2).adjudicate_draw("Referee decision")
    data = json.loads(record.to_json())
    if corruption == "kind":
        data["adjudication"]["kind"] = "win"
    elif corruption == "note":
        data["adjudication"]["note"] = ""
    elif corruption == "hash":
        data["state_keys"][-1] = "wrong"
    elif corruption == "missing_event":
        del data["adjudication"]
    elif corruption == "version":
        data["version"] = 1
    else:
        data["adjudication"]["unrecognized"] = True
    with pytest.raises(ValueError):
        Replay.from_json(json.dumps(data))


def test_old_copenhagen_replay_verifies_byte_for_byte():
    text = (Path(__file__).parent / "fixtures" / "copenhagen-v1-seed42.json").read_text()
    record = Replay.from_json(text)
    assert record == random_game(42, 30, ruleset=V1)
    assert record.to_json() == text


def test_v2_replay_without_adjudication_uses_original_schema():
    record = random_game(42, 4, ruleset=V2)
    assert json.loads(record.to_json())["version"] == 1
    assert Replay.from_json(record.to_json()) == record


def test_cli_can_record_and_replay_a_draw(tmp_path, capsys):
    original, result = tmp_path / "game.json", tmp_path / "draw.json"
    text = random_game(1, 1, ruleset=V2).to_json()
    original.write_text(text)
    main(
        [
            "adjudicate-draw",
            str(original),
            "--note",
            "Agreed by both players",
            "--output",
            str(result),
        ]
    )
    output = capsys.readouterr()
    assert "adjudicated_draw" in output.out
    main(["replay", str(result)])
    assert capsys.readouterr() == output
    assert original.read_text() == text
