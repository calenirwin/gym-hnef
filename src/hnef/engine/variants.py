"""Versioned rule identities: old profiles retain their replay semantics."""

from enum import StrEnum


class Ruleset(StrEnum):
    MINI = "mini-v1"
    COPENHAGEN = "copenhagen-experimental-v1"
    COPENHAGEN_V2 = "copenhagen-experimental-v2"

    @property
    def is_copenhagen(self) -> bool:
        return self is not Ruleset.MINI

    @property
    def adjudicates_forts(self) -> bool:
        return self is Ruleset.COPENHAGEN_V2

    @property
    def size(self) -> int:
        return 5 if self is Ruleset.MINI else 11

    @property
    def throne(self) -> tuple[int, int]:
        return self.size // 2, self.size // 2

    @property
    def corners(self) -> frozenset[tuple[int, int]]:
        last = self.size - 1
        return frozenset(((0, 0), (0, last), (last, 0), (last, last)))

    @property
    def restricted(self) -> frozenset[tuple[int, int]]:
        squares = frozenset((self.throne,))
        return squares | self.corners if self.is_copenhagen else squares
