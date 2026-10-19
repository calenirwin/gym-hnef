"""Dependency-free bridge used by the static demo's Python WebAssembly worker."""

import json

from .actions import ActionCodec
from .engine import apply_move, initial_state
from .policies import HeuristicPolicy, RandomPolicy
from .replay import Replay

codec = ActionCodec(5)
state = initial_state()
actions = []
keys = [state.key]
states = [state]


def snapshot():
    outcome = state.outcome
    mask = codec.mask(state)
    return {
        "board": state.board,
        "player": state.player.value,
        "ply": state.ply,
        "key": state.key,
        "ruleset": state.ruleset.value,
        "last_move": None
        if not actions
        else {
            "source": codec.decode(actions[-1]).source,
            "destination": codec.decode(actions[-1]).destination,
        },
        "outcome": None
        if outcome is None
        else {"winner": outcome.winner, "reason": outcome.reason},
        "legal": [
            {"id": index, "source": move.source, "destination": move.destination}
            for index, move in enumerate(codec.moves)
            if mask[index]
        ],
    }


def features():
    # Keep byte-for-byte semantics aligned with learning.network.encode; parity-tested.
    board = [float(cell == piece) for piece in "ADK" for row in state.board for cell in row]
    board += [
        float(state.player.value == "defenders"),
        state.ply / (state.ply + 1),
        state.history.count(state.history[-1]) / 3,
    ]
    return {
        "board": board,
        "history": [[byte / 127.5 - 1 for byte in bytes.fromhex(k)] for k in state.history],
        "mask": codec.mask(state),
    }


def command(message):
    global state, actions, keys, states
    data = json.loads(message)
    if data["kind"] == "reset":
        state, actions = initial_state(), []
        keys = [state.key]
        states = [state]
    elif data["kind"] == "step":
        state = apply_move(state, codec.decode(data["action"]))
        actions.append(data["action"])
        keys.append(state.key)
        states.append(state)
    elif data["kind"] == "undo":
        ply = data["ply"]
        if type(ply) is not int or not 0 <= ply < state.ply:
            raise ValueError("Undo must target an earlier ply")
        state = states[ply]
        actions, keys, states = actions[:ply], keys[: ply + 1], states[: ply + 1]
    elif data["kind"] == "reference":
        policy = (
            HeuristicPolicy(data["seed"])
            if data["policy"] == "heuristic"
            else RandomPolicy(data["seed"])
        )
        action = codec.encode(policy.choose_move(state))
        return command(json.dumps({"kind": "step", "action": action}))
    elif data["kind"] == "features":
        return json.dumps(features())
    elif data["kind"] == "replay":
        replay = Replay(tuple(actions), tuple(keys))
        replay.verify()
        return replay.to_json()
    return json.dumps(snapshot())
