"""Dependency-free Copenhagen training-data scaffold; no learned model yet.

Run: python -m hnef.learning.copenhagen --output /tmp/copenhagen-collection
"""

import argparse
import hashlib
import json
import platform
from pathlib import Path
from random import Random

from hnef.actions import ActionCodec
from hnef.engine import Player, Ruleset, apply_move, initial_state
from hnef.opponents import opponent
from hnef.replay import Replay

from .artifacts import provenance, write_json

RULESET = Ruleset.COPENHAGEN_V2
FEATURES = "copenhagen-absolute-board-full-history-v1"
CODEC = ActionCodec(11)
CODEC_SHA256 = hashlib.sha256(
    json.dumps([[m.source, m.destination] for m in CODEC.moves], separators=(",", ":")).encode()
).hexdigest()


def encode(state):
    """Lossless history is retained alongside absolute piece planes and scalar inputs."""
    if state.ruleset is not RULESET:
        raise ValueError("This feature contract requires Copenhagen experimental v2")
    return {
        "planes": [[int(cell == piece) for row in state.board for cell in row] for piece in "ADK"],
        "mover": state.player.value,
        "ply": state.ply,
        "repetitions": state.history.count(state.history[-1]),
        "history": list(state.history),
        "legal_mask": list(CODEC.mask(state)),
    }


def stream_seed(seed, role):
    payload = json.dumps(["copenhagen-collection-v1", seed, role], separators=(",", ":"))
    return int.from_bytes(hashlib.sha256(payload.encode()).digest()[:8], "big")


def terminal_target(outcome, learner):
    """None is censored data, never a draw target or an invented value estimate."""
    return None if outcome is None else outcome.reward(learner)


def collect_episode(seed, *, learner, max_plies, reference="random-v1"):
    if type(seed) is not int or not 0 <= seed < 100_000:
        raise ValueError("Training seeds must be integers in [0, 100000); evaluation is reserved")
    if type(max_plies) is not int or not 1 <= max_plies <= 128:
        raise ValueError("Scaffold collection is bounded to 1..128 plies")
    learner = Player(learner)
    if reference not in ("random-v1", "heuristic-v1"):
        raise ValueError("Use a fixed random or heuristic reference")
    seeds = {role: stream_seed(seed, role) for role in ("learner", "reference")}
    rng = Random(seeds["learner"])
    reference_policy = opponent(reference).create(RULESET, seeds["reference"])
    state = initial_state(RULESET)
    samples, actions, keys = [], [], [state.key]
    while state.outcome is None and state.ply < max_plies:
        if state.player is learner:
            observation = encode(state)
            legal = [i for i, valid in enumerate(observation["legal_mask"]) if valid]
            action = rng.choice(legal)
            move = CODEC.decode(action)
            samples.append({"state_key": state.key, "observation": observation, "action": action})
        else:
            move = reference_policy.choose_move(state)
            action = CODEC.encode(move)
        state = apply_move(state, move)
        actions.append(action)
        keys.append(state.key)
    replay = Replay(tuple(actions), tuple(keys), RULESET)
    assert replay.verify() == state
    target = terminal_target(state.outcome, learner)
    for sample in samples:
        sample["terminal_target"] = target
    return {
        "seed": seed,
        "streams": seeds,
        "learner": learner.value,
        "reference": reference,
        "truncated": state.outcome is None,
        "terminal_reason": state.outcome.reason if state.outcome else None,
        "samples": samples,
    }, replay


def collect(output, *, episodes=4, max_plies=16, seed=0):
    if type(episodes) is not int or not 1 <= episodes <= 16:
        raise ValueError("Scaffold collection is bounded to 1..16 episodes")
    if type(seed) is not int or seed < 0 or seed + episodes > 100_000:
        raise ValueError("Training seed range overlaps reserved evaluation seeds")
    if type(max_plies) is not int or not 1 <= max_plies <= 128:
        raise ValueError("Scaffold collection is bounded to 1..128 plies")
    output = Path(output)
    output.mkdir(parents=True, exist_ok=False)
    manifest = {
        "schema": 1,
        "purpose": "collection sanity only; no training or checkpoint",
        "ruleset": RULESET.value,
        "features": FEATURES,
        "action_count": len(CODEC.moves),
        "codec_sha256": CODEC_SHA256,
        "behavior": "uniform legal learner versus random-v1; alternate learner side",
        "python": platform.python_version(),
        "source": provenance(),
        "config": {"episodes": episodes, "max_plies": max_plies, "seed": seed},
        "episodes": [],
    }
    for index in range(episodes):
        data, replay = collect_episode(
            seed + index,
            learner=Player.ATTACKERS if index % 2 == 0 else Player.DEFENDERS,
            max_plies=max_plies,
        )
        data_name, replay_name = f"episode-{index}.json", f"replay-{index}.json"
        write_json(output / data_name, data)
        (output / replay_name).write_text(replay.to_json())
        manifest["episodes"].append(
            {
                "data": data_name,
                "replay": replay_name,
                "data_sha256": hashlib.sha256((output / data_name).read_bytes()).hexdigest(),
                "replay_sha256": hashlib.sha256((output / replay_name).read_bytes()).hexdigest(),
                "learner": data["learner"],
                "truncated": data["truncated"],
                "terminal_reason": data["terminal_reason"],
                "plies": len(replay.actions),
                "samples": len(data["samples"]),
                "labeled_samples": sum(s["terminal_target"] is not None for s in data["samples"]),
            }
        )
    write_json(output / "manifest.json", manifest)
    return manifest


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = collect(args.output)
    print(json.dumps(result, indent=2))
