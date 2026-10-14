"""Copenhagen shield walls and enclosure; see docs/copenhagen-rules.md."""

from .state import Player, Square


def shieldwall_captures(board: list[list[str]], player: Player, destination: Square) -> set[Square]:
    """Only a move to an end bracket can close a shield wall, not a backing move."""
    size = len(board)
    last = size - 1
    # Edges are paired with their inward-facing direction.
    edges = (
        ([(0, c) for c in range(size)], (1, 0)),
        ([(last, c) for c in range(size)], (-1, 0)),
        ([(r, 0) for r in range(size)], (0, 1)),
        ([(r, last) for r in range(size)], (0, -1)),
    )
    captured: set[Square] = set()
    for edge, inward in edges:
        if destination not in edge:
            continue
        start = edge.index(destination)
        for direction in (-1, 1):
            index = start + direction
            victims = []
            while 0 < index < last:
                r, c = edge[index]
                if not player.opponent.owns(board[r][c]):
                    break
                victims.append((r, c))
                index += direction
            if len(victims) < 2 or not 0 <= index <= last:
                continue
            r, c = edge[index]
            bracketed = index in (0, last) or player.owns(board[r][c])
            backed = all(player.owns(board[r + inward[0]][c + inward[1]]) for r, c in victims)
            if bracketed and backed:
                captured.update((r, c) for r, c in victims if board[r][c] != "K")
    return captured


def defenders_encircled(board: tuple[str, ...]) -> bool:
    """All defenders must be separated from every board edge by attackers.

    Treat defenders as traversable so a defender's own piece cannot falsely
    complete the attackers' ring. Disconnected groups are checked together.
    """
    size = len(board)
    frontier = [
        (r, c)
        for r in range(size)
        for c in range(size)
        if (r in (0, size - 1) or c in (0, size - 1)) and board[r][c] != "A"
    ]
    reached = set(frontier)
    while frontier:
        r, c = frontier.pop()
        if board[r][c] in "DK":
            return False
        for nr, nc in ((r - 1, c), (r + 1, c), (r, c - 1), (r, c + 1)):
            if (
                0 <= nr < size
                and 0 <= nc < size
                and board[nr][nc] != "A"
                and (nr, nc) not in reached
            ):
                reached.add((nr, nc))
                frontier.append((nr, nc))
    return True
