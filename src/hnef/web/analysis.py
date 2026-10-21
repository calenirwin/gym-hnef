"""Bounded, read-only comparison of reference policies on a verified position."""

import json

from hnef.engine import apply_move, legal_moves
from hnef.policies import MCTSPolicy, evaluate, make_policy
from hnef.replay import Replay

from . import codec_for, game_response


def analyze_response(payload):
    if not isinstance(payload, dict):
        raise ValueError("Expected an experiment object")
    seed = payload.get("seed", 42)
    iterations = payload.get("iterations", 32)
    rollout_plies = payload.get("rollout_plies", 24)
    if type(seed) is not int or not 0 <= seed < 2**32:
        raise ValueError("Seed must be an integer from 0 to 4294967295")
    if type(iterations) is not int or iterations not in (8, 32, 64):
        raise ValueError("Choose 8, 32, or 64 simulations")
    if type(rollout_plies) is not int or rollout_plies not in (0, 12, 24, 48):
        raise ValueError("Choose 0, 12, 24, or 48 rollout plies")
    # Never allow an experiment payload to apply a move to the supplied history.
    if "replay" in payload:
        position = game_response({"operation": "import", "replay": payload["replay"]})
    else:
        position = game_response(
            {"ruleset": payload.get("ruleset", "mini-v1"), "actions": payload.get("actions", [])}
        )
    if position["outcome"] or position["truncated"]:
        raise ValueError("Choose an unfinished position below the 500-ply limit")
    state = Replay.from_json(json.dumps(position["replay"])).verify()
    codec = codec_for(state.ruleset)
    decision_seed = seed * 1009 + state.ply

    def describe(move):
        return {
            "action": codec.encode(move),
            "source": move.source,
            "destination": move.destination,
        }

    decisions = []
    for name in ("random", "heuristic"):
        move = make_policy(name, decision_seed).choose_move(state)
        decisions.append(
            {
                "policy": name,
                **describe(move),
                "heuristic_score": evaluate(apply_move(state, move), state.player),
            }
        )
    search = MCTSPolicy(decision_seed, iterations=iterations, rollout_plies=rollout_plies).search(
        state
    )
    decisions.append(
        {
            "policy": "mcts",
            **describe(search.move),
            "heuristic_score": evaluate(apply_move(state, search.move), state.player),
        }
    )
    return {
        "schema_version": 1,
        "policy_version": "reference-v1",
        "position": position,
        "seed": seed,
        "decision_seed": decision_seed,
        "iterations": iterations,
        "rollout_plies": rollout_plies,
        "legal_count": len(legal_moves(state)),
        "decisions": decisions,
        "search": {
            "rollout_cutoffs": search.rollout_cutoffs,
            "children": [
                {**describe(child.move), "visits": child.visits, "mean_value": child.mean_value}
                for child in search.children
            ],
        },
    }
