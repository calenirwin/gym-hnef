"""Versioned, verified records starting from each ruleset's initial position."""

import json
from dataclasses import dataclass
from functools import lru_cache
from random import Random

from .actions import ActionCodec
from .engine import GameState, Ruleset, apply_move, initial_state, legal_moves


@lru_cache(maxsize=2)
def _codec(ruleset: Ruleset) -> ActionCodec:
    return ActionCodec(ruleset.size)


@dataclass(frozen=True)
class Replay:
    actions: tuple[int, ...]
    state_keys: tuple[str, ...]
    ruleset: Ruleset = Ruleset.MINI

    def __post_init__(self) -> None:
        object.__setattr__(self, "ruleset", Ruleset(self.ruleset))
        object.__setattr__(self, "actions", tuple(self.actions))
        object.__setattr__(self, "state_keys", tuple(self.state_keys))
        if len(self.state_keys) != len(self.actions) + 1:
            raise ValueError("Replay needs one state key per position, including the opening")
        for action in self.actions:
            _codec(self.ruleset).decode(action)
        if any(not isinstance(key, str) for key in self.state_keys):
            raise ValueError("State keys must be strings")

    def verify(self) -> GameState:
        state = initial_state(self.ruleset)
        codec = _codec(self.ruleset)
        if self.state_keys[0] != state.key:
            raise ValueError(f"Replay opening does not match {self.ruleset.value}")
        for ply, action in enumerate(self.actions, start=1):
            state = apply_move(state, codec.decode(action))
            if self.state_keys[ply] != state.key:
                raise ValueError(f"Replay state mismatch at ply {ply}")
        return state

    def to_json(self) -> str:
        return (
            json.dumps(
                {
                    "version": 1,
                    "ruleset": self.ruleset.value,
                    "actions": self.actions,
                    "state_keys": self.state_keys,
                },
                indent=2,
            )
            + "\n"
        )

    @classmethod
    def from_json(cls, text: str) -> "Replay":
        data = json.loads(text)
        if (
            not isinstance(data, dict)
            or set(data) != {"version", "ruleset", "actions", "state_keys"}
            or type(data["version"]) is not int
            or data["version"] != 1
            or data["ruleset"] not in tuple(Ruleset)
            or not isinstance(data["actions"], list)
            or not isinstance(data["state_keys"], list)
        ):
            raise ValueError("Unsupported replay format or ruleset")
        replay = cls(tuple(data["actions"]), tuple(data["state_keys"]), Ruleset(data["ruleset"]))
        replay.verify()
        return replay


def random_game(seed: int, max_plies: int = 500, *, ruleset: Ruleset = Ruleset.MINI) -> Replay:
    """A ply limit stops collection; it does not invent a terminal game result."""
    if type(max_plies) is not int or max_plies < 1:
        raise ValueError("max_plies must be a positive integer")
    rng = Random(seed)
    state = initial_state(ruleset)
    codec = _codec(state.ruleset)
    actions = []
    keys = [state.key]
    while state.outcome is None and len(actions) < max_plies:
        move = rng.choice(legal_moves(state))
        state = apply_move(state, move)
        actions.append(codec.encode(move))
        keys.append(state.key)
    return Replay(tuple(actions), tuple(keys), state.ruleset)
