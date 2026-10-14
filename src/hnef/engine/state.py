"""Immutable, dependency-free types for the project's mini ruleset."""

import json
from dataclasses import dataclass
from enum import StrEnum
from hashlib import sha256

Square = tuple[int, int]
RULESET = "mini-v1"
SIZE = 5
THRONE = (2, 2)


class Player(StrEnum):
    ATTACKERS = "attackers"
    DEFENDERS = "defenders"

    @property
    def opponent(self) -> "Player":
        return Player.DEFENDERS if self is Player.ATTACKERS else Player.ATTACKERS

    def owns(self, piece: str) -> bool:
        return piece == "A" if self is Player.ATTACKERS else piece in ("D", "K")


@dataclass(frozen=True)
class Move:
    source: Square
    destination: Square

    def __post_init__(self) -> None:
        for name in ("source", "destination"):
            square = tuple(getattr(self, name))
            if len(square) != 2 or any(type(value) is not int for value in square):
                raise ValueError("Squares must contain two integer coordinates")
            object.__setattr__(self, name, square)


@dataclass(frozen=True)
class Outcome:
    winner: Player | None
    reason: str

    def __post_init__(self) -> None:
        if self.winner is not None and not isinstance(self.winner, Player):
            raise ValueError("Winner must be a Player or None")
        reasons = {
            "king_escaped": Player.DEFENDERS,
            "king_captured": Player.ATTACKERS,
            "repetition": None,
        }
        if self.reason == "no_legal_moves":
            if not isinstance(self.winner, Player):
                raise ValueError("A blocked player loses to the other player")
        elif self.reason not in reasons or self.winner != reasons[self.reason]:
            raise ValueError("Invalid outcome")

    def reward(self, player: Player) -> int:
        """Zero-sum terminal value from an explicit player's perspective."""
        if self.winner is None:
            return 0
        return 1 if self.winner == player else -1


def position_key(board: tuple[str, ...], player: Player) -> str:
    """Stable repetition identity, including side to move and rules version."""
    payload = f"{RULESET}|{player.value}|{'/'.join(board)}"
    return sha256(payload.encode()).hexdigest()


@dataclass(frozen=True)
class GameState:
    board: tuple[str, ...]
    player: Player = Player.ATTACKERS
    ply: int = 0
    history: tuple[str, ...] = ()
    outcome: Outcome | None = None

    def __post_init__(self) -> None:
        board = tuple(self.board)
        if len(board) != SIZE or any(
            not isinstance(row, str) or len(row) != SIZE or set(row) - set(".ADK") for row in board
        ):
            raise ValueError("mini-v1 requires five rows of five . A D K characters")
        if not isinstance(self.player, Player) or type(self.ply) is not int or self.ply < 0:
            raise ValueError("Invalid player or ply")
        if self.outcome is not None and not isinstance(self.outcome, Outcome):
            raise ValueError("Invalid outcome")
        kings = sum(row.count("K") for row in board)
        captured = self.outcome is not None and self.outcome.reason == "king_captured"
        if kings != (0 if captured else 1):
            raise ValueError("A live board must contain exactly one king")
        if board[2][2] not in ".K":
            raise ValueError("Only the king may occupy the throne")
        object.__setattr__(self, "board", board)
        key = position_key(board, self.player)
        history = tuple(self.history) or (key,)
        if (
            any(
                not isinstance(item, str) or len(item) != 64 or set(item) - set("0123456789abcdef")
                for item in history
            )
            or history[-1] != key
        ):
            raise ValueError("History must contain position keys and end at the current position")
        object.__setattr__(self, "history", history)

    @property
    def key(self) -> str:
        """Full state identity for replay/search, including repetition context."""
        result = None if self.outcome is None else (self.outcome.winner, self.outcome.reason)
        payload = (RULESET, self.board, self.player, self.ply, self.history, result)
        return sha256(json.dumps(payload, separators=(",", ":")).encode()).hexdigest()

    def at(self, square: Square) -> str:
        return self.board[square[0]][square[1]]


def initial_state() -> GameState:
    return GameState((".A.A.", "A.D.A", ".DKD.", "A.D.A", ".A.A."))
