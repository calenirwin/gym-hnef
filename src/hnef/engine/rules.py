"""Project mini-v1 rules, specified in docs/mini-rules.md (not Copenhagen)."""

from .state import SIZE, THRONE, GameState, Move, Outcome, Player, Square, position_key

DIRECTIONS = ((-1, 0), (0, -1), (0, 1), (1, 0))


def _inside(square: Square) -> bool:
    return all(0 <= coordinate < SIZE for coordinate in square)


def _add(square: Square, direction: Square) -> Square:
    return square[0] + direction[0], square[1] + direction[1]


def legal_moves(state: GameState) -> tuple[Move, ...]:
    if state.outcome is not None:
        return ()
    moves = []
    for row in range(SIZE):
        for column in range(SIZE):
            source = (row, column)
            piece = state.at(source)
            if not state.player.owns(piece):
                continue
            for direction in DIRECTIONS:
                destination = _add(source, direction)
                while _inside(destination) and state.at(destination) == ".":
                    if destination != THRONE or piece == "K":
                        moves.append(Move(source, destination))
                    destination = _add(destination, direction)
    return tuple(moves)


def apply_move(state: GameState, move: Move) -> GameState:
    if move not in legal_moves(state):
        raise ValueError(f"Illegal move: {move}")
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
        if not _inside(victim) or not _inside(support):
            continue
        if at(victim) == "K" or not state.player.opponent.owns(at(victim)):
            continue
        hostile = state.player.owns(at(support)) or (
            support == THRONE and (state.player is Player.DEFENDERS or at(THRONE) == ".")
        )
        if hostile:
            captured.add(victim)

    king = next((r, c) for r in range(SIZE) for c in range(SIZE) if board[r][c] == "K")
    neighbors = tuple(_add(king, direction) for direction in DIRECTIONS)
    if state.player is Player.ATTACKERS and destination in neighbors:
        if king == THRONE or THRONE in neighbors:
            king_captured = all(
                _inside(square) and (at(square) == "A" or square == THRONE) for square in neighbors
            )
        else:
            opposite = (2 * king[0] - destination[0], 2 * king[1] - destination[1])
            king_captured = _inside(opposite) and at(opposite) == "A"
        if king_captured:
            captured.add(king)
    for row, column in captured:
        board[row][column] = "."

    result = None
    if king in captured:
        result = Outcome(Player.ATTACKERS, "king_captured")
    elif piece == "K" and any(coordinate in (0, SIZE - 1) for coordinate in destination):
        result = Outcome(Player.DEFENDERS, "king_escaped")
    next_board = tuple("".join(row) for row in board)
    next_player = state.player.opponent
    key = position_key(next_board, next_player)
    history = (*state.history, key)
    successor = GameState(next_board, next_player, state.ply + 1, history, result)
    if result is None:
        if not legal_moves(successor):
            result = Outcome(state.player, "no_legal_moves")
        elif history.count(key) >= 3:
            result = Outcome(None, "repetition")
        if result is not None:
            successor = GameState(next_board, next_player, state.ply + 1, history, result)
    return successor
