"""Versioned, verified records starting from each ruleset's initial position."""

import json
from dataclasses import dataclass
from functools import lru_cache
from random import Random

from .actions import ActionCodec
from .engine import GameState, Ruleset, adjudicate_draw, apply_move, initial_state, legal_moves


@lru_cache(maxsize=3)
def _codec(ruleset: Ruleset) -> ActionCodec:
    return ActionCodec(ruleset.size)


@dataclass(frozen=True)
class Replay:
    actions: tuple[int, ...]
    state_keys: tuple[str, ...]
    ruleset: Ruleset = Ruleset.MINI
    adjudication_note: str | None = None

    def __post_init__(self) -> None:
        object.__setattr__(self, "ruleset", Ruleset(self.ruleset))
        object.__setattr__(self, "actions", tuple(self.actions))
        object.__setattr__(self, "state_keys", tuple(self.state_keys))
        if self.adjudication_note is not None and (
            not isinstance(self.adjudication_note, str)
            or not self.adjudication_note.strip()
            or self.ruleset is not Ruleset.COPENHAGEN_V2
        ):
            raise ValueError("A draw adjudication needs a nonempty note and Copenhagen v2")
        if len(self.state_keys) != len(self.actions) + 1 + (self.adjudication_note is not None):
            raise ValueError("Replay needs a key for the opening, each move, and any adjudication")
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
        if self.adjudication_note is not None:
            state = adjudicate_draw(state)
            if self.state_keys[-1] != state.key:
                raise ValueError("Replay state mismatch after draw adjudication")
        return state

    def adjudicate_draw(self, note: str) -> "Replay":
        """Append a documented external judgment without adding a fictitious move."""
        state = adjudicate_draw(self.verify())
        return Replay(self.actions, (*self.state_keys, state.key), self.ruleset, note)

    def to_json(self) -> str:
        data = {
            "version": 1,
            "ruleset": self.ruleset.value,
            "actions": self.actions,
            "state_keys": self.state_keys,
        }
        if self.adjudication_note is not None:
            data["version"] = 2
            data["adjudication"] = {"kind": "draw", "note": self.adjudication_note}
        return (
            json.dumps(
                data,
                indent=2,
            )
            + "\n"
        )

    @classmethod
    def from_json(cls, text: str) -> "Replay":
        data = json.loads(text)
        if not isinstance(data, dict):
            raise ValueError("Unsupported replay format or ruleset")
        fields = {"version", "ruleset", "actions", "state_keys"}
        note = None
        if data.get("version") == 2:
            fields.add("adjudication")
            event = data.get("adjudication")
            if (
                not isinstance(event, dict)
                or set(event) != {"kind", "note"}
                or event["kind"] != "draw"
                or not isinstance(event["note"], str)
                or not event["note"].strip()
            ):
                raise ValueError("Invalid draw adjudication event")
            note = event["note"]
        if (
            set(data) != fields
            or type(data["version"]) is not int
            or data["version"] not in (1, 2)
            or data["ruleset"] not in tuple(Ruleset)
            or not isinstance(data["actions"], list)
            or not isinstance(data["state_keys"], list)
        ):
            raise ValueError("Unsupported replay format or ruleset")
        replay = cls(
            tuple(data["actions"]), tuple(data["state_keys"]), Ruleset(data["ruleset"]), note
        )
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
