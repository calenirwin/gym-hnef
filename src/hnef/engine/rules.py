"""Deterministic transitions for mini-v1 and experimental Copenhagen."""

from .copenhagen import defenders_encircled, shieldwall_captures
from .forts import prove_exit_fort
from .state import GameState, Move, Outcome, Player, Square, position_key
from .variants import Ruleset

DIRECTIONS = ((-1, 0), (0, -1), (0, 1), (1, 0))


def _inside(square: Square, size: int) -> bool:
    return all(0 <= coordinate < size for coordinate in square)


def _add(square: Square, direction: Square) -> Square:
    return square[0] + direction[0], square[1] + direction[1]


def legal_moves(state: GameState) -> tuple[Move, ...]:
    if state.outcome is not None:
        return ()
    size = state.ruleset.size
    restricted = state.ruleset.restricted
    moves = []
    for row in range(size):
        for column in range(size):
            source = (row, column)
            piece = state.at(source)
            if not state.player.owns(piece):
                continue
            for direction in DIRECTIONS:
                destination = _add(source, direction)
                while _inside(destination, size) and state.at(destination) == ".":
                    if destination not in restricted or piece == "K":
                        moves.append(Move(source, destination))
                    destination = _add(destination, direction)
    return tuple(moves)


def apply_move(state: GameState, move: Move) -> GameState:
    if move not in legal_moves(state):
        raise ValueError(f"Illegal move: {move}")
    ruleset = state.ruleset
    size, throne = ruleset.size, ruleset.throne
    copenhagen = ruleset.is_copenhagen
    board = [list(row) for row in state.board]
    source, destination = move.source, move.destination
    piece = board[source[0]][source[1]]
    board[source[0]][source[1]] = "."
    board[destination[0]][destination[1]] = piece

    def at(square: Square) -> str:
        return board[square[0]][square[1]]

    captured: set[Square] = set()
    # Evaluate every capture on the same post-move board before removing pieces.
    for direction in DIRECTIONS:
        victim = _add(destination, direction)
        support = _add(victim, direction)
        if not _inside(victim, size) or not _inside(support, size):
            continue
        if at(victim) == "K" or not state.player.opponent.owns(at(victim)):
            continue
        hostile = (
            state.player.owns(at(support))
            or (support == throne and (state.player is Player.DEFENDERS or at(throne) == "."))
            or (copenhagen and support in ruleset.corners)
        )
        if hostile:
            captured.add(victim)

    if copenhagen:
        captured.update(shieldwall_captures(board, state.player, destination))

    king = next((r, c) for r in range(size) for c in range(size) if board[r][c] == "K")
    neighbors = tuple(_add(king, direction) for direction in DIRECTIONS)
    if state.player is Player.ATTACKERS and destination in neighbors:
        if copenhagen or king == throne or throne in neighbors:
            king_captured = all(
                _inside(square, size) and (at(square) == "A" or square == throne)
                for square in neighbors
            )
        else:
            opposite = (2 * king[0] - destination[0], 2 * king[1] - destination[1])
            king_captured = _inside(opposite, size) and at(opposite) == "A"
        if king_captured:
            captured.add(king)
    for row, column in captured:
        board[row][column] = "."

    result = None
    if king in captured:
        result = Outcome(Player.ATTACKERS, "king_captured")
    elif piece == "K" and (
        destination in ruleset.corners
        if copenhagen
        else any(coordinate in (0, size - 1) for coordinate in destination)
    ):
        result = Outcome(Player.DEFENDERS, "king_escaped")
    next_board = tuple("".join(row) for row in board)
    next_player = state.player.opponent
    key = position_key(next_board, next_player, ruleset)
    history = (*state.history, key)
    successor = GameState(next_board, next_player, state.ply + 1, history, result, ruleset)
    if result is None:
        if ruleset.adjudicates_forts and prove_exit_fort(next_board) is not None:
            result = Outcome(Player.DEFENDERS, "exit_fort")
        elif copenhagen and defenders_encircled(next_board):
            result = Outcome(Player.ATTACKERS, "encircled")
        elif not legal_moves(successor):
            result = Outcome(state.player, "no_legal_moves")
        elif history.count(key) >= 3:
            result = (
                Outcome(Player.ATTACKERS, "defender_repetition")
                if copenhagen
                else Outcome(None, "repetition")
            )
        if result is not None:
            successor = GameState(next_board, next_player, state.ply + 1, history, result, ruleset)
    return successor


def adjudicate_draw(state: GameState) -> GameState:
    """Record an external draw judgment, not an automatic dead-position proof.

    A referee or application with both players' agreement supplies the judgment.
    Use Replay.adjudicate_draw(note) to preserve that judgment in a saved game.
    """
    if state.ruleset is not Ruleset.COPENHAGEN_V2:
        raise ValueError("Recorded draw adjudication requires Copenhagen experimental v2")
    if state.outcome is not None:
        raise ValueError("Cannot adjudicate an already finished game")
    return GameState(
        state.board,
        state.player,
        state.ply,
        state.history,
        Outcome(None, "adjudicated_draw"),
        state.ruleset,
    )
