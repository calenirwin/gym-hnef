"""Held-out seeded comparisons, separated by side, with verified saved replays."""

import time
from collections import Counter
from pathlib import Path
from random import Random

import torch

from hnef.engine import Player
from hnef.matches import play_game
from hnef.policies import make_policy

from .artifacts import now, provenance, sha256, write_json
from .network import NeuralPolicy, checkpoint_model


class TimedPolicy:
    def __init__(self, policy):
        self.policy, self.seconds, self.decisions = policy, 0.0, 0

    def choose_move(self, state):
        start = time.perf_counter()
        move = self.policy.choose_move(state)
        self.seconds += time.perf_counter() - start
        self.decisions += 1
        return move


def evaluate_checkpoint(
    checkpoint,
    output,
    *,
    games=50,
    seed=100_000,
    opponents=("random", "heuristic"),
    max_plies=128,
    iterations=64,
    rollout_plies=48,
    greedy=False,
):
    if type(games) is not int or games < 1 or type(max_plies) is not int or max_plies < 1:
        raise ValueError("games and max_plies must be positive integers")
    if not opponents or any(name not in ("random", "heuristic", "mcts") for name in opponents):
        raise ValueError("Choose random, heuristic, or mcts opponents")
    torch.set_num_threads(1)
    model, data = checkpoint_model(Path(checkpoint))
    output = Path(output)
    output.mkdir(parents=True, exist_ok=False)
    (output / "replays").mkdir()
    report = {
        "schema": 1,
        "created_at": now(),
        "checkpoint_sha256": sha256(checkpoint),
        "checkpoint_update": data["update"],
        "ruleset": data["ruleset"],
        "training_source": data["source"],
        "evaluation_source": provenance(),
        "config": {
            "games_per_matchup": games,
            "seed": seed,
            "max_plies": max_plies,
            "iterations": iterations,
            "rollout_plies": rollout_plies,
            "selection": "greedy" if greedy else "sampled",
        },
        "matchups": [],
    }
    for opponent in opponents:
        for side in Player:
            counts, reasons = Counter(), Counter()
            plies, seconds, decisions = 0, 0.0, 0
            for game_seed in range(seed, seed + games):
                rng = Random(game_seed)
                candidate = TimedPolicy(NeuralPolicy(model, rng.getrandbits(63), greedy=greedy))
                reference = make_policy(
                    opponent,
                    rng.getrandbits(63),
                    iterations=iterations,
                    rollout_plies=rollout_plies,
                )
                players = (
                    (candidate, reference) if side is Player.ATTACKERS else (reference, candidate)
                )
                replay = play_game(*players, max_plies=max_plies)
                state = replay.verify()
                name = f"{opponent}-{side.value}-{game_seed}.json"
                (output / "replays" / name).write_text(replay.to_json())
                outcome = state.outcome
                result = (
                    "truncated"
                    if outcome is None
                    else (
                        "draw"
                        if outcome.winner is None
                        else ("win" if outcome.winner == side else "loss")
                    )
                )
                counts[result] += 1
                reasons[outcome.reason if outcome else "collection_limit"] += 1
                plies += state.ply
                seconds += candidate.seconds
                decisions += candidate.decisions
            matchup = {
                "opponent": opponent,
                "candidate_side": side.value,
                "games": games,
                "results": {key: counts[key] for key in ("win", "loss", "draw", "truncated")},
                "outcomes": dict(reasons),
                "plies": plies,
                "verified_replays": games,
                "candidate_decisions": decisions,
                "mean_decision_ms": 1000 * seconds / decisions if decisions else None,
            }
            report["matchups"].append(matchup)
            print(f"{opponent}/{side.value}: {dict(counts)}", flush=True)
    write_json(output / "report.json", report)
    return report
