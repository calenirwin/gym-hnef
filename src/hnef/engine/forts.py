"""Conservative proof of a permanent exit fort, not a heuristic fort score.

Keep defenders stationary and let the king move within its room. Grant attackers
unlimited pieces on every outside square, ignoring how they would get there.
Remove any defender this stronger opponent could capture, then repeat. If a
wall still isolates a mobile edge king, no real attack can break it. Failure to
prove a fort means unknown, not that the position is breakable.
"""

from dataclasses import dataclass

from .copenhagen import shieldwall_captures
from .state import Player, Square


@dataclass(frozen=True)
class ExitFort:
    """A safe room and its surviving defender boundary, in row-major order."""

    room: tuple[Square, ...]
    wall: tuple[Square, ...]


def _neighbors(square: Square, size: int) -> tuple[Square, ...]:
    r, c = square
    return tuple(
        (nr, nc)
        for nr, nc in ((r - 1, c), (r + 1, c), (r, c - 1), (r, c + 1))
        if 0 <= nr < size and 0 <= nc < size
    )


def _room(king: Square, defenders: set[Square], size: int) -> set[Square]:
    reached = {king}
    frontier = [king]
    while frontier:
        for square in _neighbors(frontier.pop(), size):
            if square not in defenders and square not in reached:
                reached.add(square)
                frontier.append(square)
    return reached


def prove_exit_fort(board: tuple[str, ...]) -> ExitFort | None:
    """Prove an edge fort using fixed defenders and overestimated attacker power.

    The caller supplies a validated Copenhagen board. No piece-count shortcut,
    current-move-only safety test, or time limit is used to infer a win.
    """
    size = len(board)
    kings = [(r, c) for r in range(size) for c in range(size) if board[r][c] == "K"]
    if len(kings) != 1:
        return None
    king = kings[0]
    last = size - 1
    corners = {(0, 0), (0, last), (last, 0), (last, last)}
    if king in corners or not any(coordinate in (0, last) for coordinate in king):
        return None
    if not any(board[r][c] == "." for r, c in _neighbors(king, size)):
        return None

    squares = {(r, c) for r in range(size) for c in range(size)}
    defenders = {(r, c) for r, c in squares if board[r][c] == "D"}
    attackers = {(r, c) for r, c in squares if board[r][c] == "A"}
    restricted = corners | {(size // 2, size // 2)}
    edges = {square for square in squares if any(i in (0, last) for i in square)}
    room = _room(king, defenders, size)
    if room & (attackers | corners):
        return None
    wall = {neighbor for square in room for neighbor in _neighbors(square, size)} & defenders
    if not wall:
        return None
    while defenders:
        # Even inaccessible outside pockets are granted to the hypothetical attacker.
        available = squares - defenders - room - restricted
        hostile = available | restricted
        vulnerable = set()
        for r, c in defenders:
            for dr, dc in ((1, 0), (0, 1)):
                first, second = (r - dr, c - dc), (r + dr, c + dc)
                if (
                    first in hostile
                    and second in hostile
                    and (first in available or second in available)
                ):
                    vulnerable.add((r, c))

        hypothetical = [["."] * size for _ in range(size)]
        for r, c in available:
            hypothetical[r][c] = "A"
        for r, c in defenders:
            hypothetical[r][c] = "D"
        hypothetical[king[0]][king[1]] = "K"
        for destination in available & edges:
            vulnerable.update(shieldwall_captures(hypothetical, Player.ATTACKERS, destination))

        # Keep the original room fixed: a newly opened square may already contain
        # an attacking piece. Re-flooding an enlarged room could hide that threat.
        if vulnerable & wall:
            return None
        if not vulnerable:
            return ExitFort(tuple(sorted(room)), tuple(sorted(wall)))
        defenders -= vulnerable
    return None
