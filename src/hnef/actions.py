"""Generated action vocabulary: size² × 2(size − 1), no diagonal/padding actions."""

from collections.abc import Mapping
from dataclasses import dataclass, field
from types import MappingProxyType

from .engine import GameState, Move, legal_moves


@dataclass(frozen=True)
class ActionCodec:
    size: int
    moves: tuple[Move, ...] = field(init=False)
    _ids: Mapping[Move, int] = field(init=False, repr=False, compare=False)

    def __post_init__(self) -> None:
        if type(self.size) is not int or self.size < 2:
            raise ValueError("Board size must be an integer of at least two")
        squares = tuple((r, c) for r in range(self.size) for c in range(self.size))
        moves = tuple(
            Move(source, destination)
            for source in squares
            for destination in squares
            if source != destination
            and (source[0] == destination[0] or source[1] == destination[1])
        )
        object.__setattr__(self, "moves", moves)
        object.__setattr__(
            self, "_ids", MappingProxyType({move: i for i, move in enumerate(moves)})
        )

    def encode(self, move: Move) -> int:
        try:
            return self._ids[move]
        except KeyError as error:
            raise ValueError(f"Move is outside the action vocabulary: {move}") from error

    def decode(self, action: int) -> Move:
        if type(action) is not int or not 0 <= action < len(self.moves):
            raise ValueError("Action ID is out of range")
        return self.moves[action]

    def mask(self, state: GameState) -> tuple[bool, ...]:
        if len(state.board) != self.size:
            raise ValueError("Codec and board sizes differ")
        valid = {self.encode(move) for move in legal_moves(state)}
        return tuple(action in valid for action in range(len(self.moves)))
