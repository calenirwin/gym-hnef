"""Validate the exact positions and captures shipped in the Learn illustrations."""

from html.parser import HTMLParser
from pathlib import Path

import pytest

from hnef.engine import GameState, Move, Player, Ruleset, apply_move, legal_moves


class Lessons(HTMLParser):
    def __init__(self):
        super().__init__()
        self.positions = []

    def handle_starttag(self, tag, attrs):
        attrs = dict(attrs)
        if attrs.get("class") == "tactic-demo":
            self.positions.append(attrs)


parser = Lessons()
parser.feed((Path(__file__).parents[1] / "src/hnef/web/learn.html").read_text())


def square(coordinate):
    return 11 - int(coordinate[1:]), ord(coordinate[0]) - ord("a")


@pytest.mark.parametrize("lesson", parser.positions)
@pytest.mark.parametrize("ruleset", [Ruleset.COPENHAGEN, Ruleset.COPENHAGEN_V2])
def test_shipped_tactic(lesson, ruleset):
    board = [["."] * 11 for _ in range(11)]
    for item in lesson["data-pieces"].split():
        coordinate, piece = item.split(":")
        row, col = square(coordinate)
        board[row][col] = piece
    state = GameState(tuple("".join(row) for row in board), Player.ATTACKERS, ruleset=ruleset)
    move = Move(*(square(c) for c in lesson["data-move"].split()))
    assert move in legal_moves(state)
    result = apply_move(state, move)
    expected = [list(row) for row in state.board]
    expected[move.destination[0]][move.destination[1]] = state.at(move.source)
    expected[move.source[0]][move.source[1]] = "."
    for coordinate in lesson["data-captured"].split():
        row, col = square(coordinate)
        assert state.at((row, col)) in ("D", "K")
        expected[row][col] = "."
    assert result.board == tuple("".join(row) for row in expected)
    if "f5" in lesson["data-captured"].split():
        assert result.outcome.winner == Player.ATTACKERS
        assert result.outcome.reason == "king_captured"
    else:
        assert result.outcome is None


def test_three_shipped_examples():
    assert len(parser.positions) == 3
