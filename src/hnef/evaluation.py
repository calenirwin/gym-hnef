"""Small dependency-free Copenhagen baseline experiment, not a strength ranking.

Run: python -m hnef.evaluation --output /tmp/copenhagen-smoke
"""

import argparse
import hashlib
import json
import platform
import time
from collections import Counter
from dataclasses import asdict
from math import ceil
from pathlib import Path

from .engine import Ruleset
from .matches import play_game
from .opponents import opponent


class MeasuredPolicy:
    def __init__(self, policy):
        self.policy = policy
        self.samples = []

    def choose_move(self, state):
        start = time.perf_counter()
        move = self.policy.choose_move(state)
        self.samples.append(1000 * (time.perf_counter() - start))
        return move


def policy_seed(seed: int, identifier: str) -> int:
    """Identity-paired stream: stable across seats, opponents and registry ordering."""
    payload = json.dumps(["identity-paired-v1", seed, identifier], separators=(",", ":"))
    return int.from_bytes(hashlib.sha256(payload.encode()).digest()[:8], "big")


def summarize(games):
    """Per-opponent, per-seat denominators; nearest-rank latency percentiles."""
    rows = []
    for identifier in sorted({g[side] for g in games for side in ("attackers", "defenders")}):
        for index, side in enumerate(("attackers", "defenders")):
            selected = [g for g in games if g[side] == identifier]
            if not selected:
                continue
            results, reasons = Counter(), Counter()
            samples = sorted(t for g in selected for t in g["decision_ms"][index])
            for game in selected:
                outcome = game["outcome"]
                if outcome is None:
                    results["unfinished"] += 1
                else:
                    reasons[outcome["reason"]] += 1
                    results[
                        "draw"
                        if outcome["winner"] is None
                        else "win"
                        if outcome["winner"] == side
                        else "loss"
                    ] += 1
            rows.append(
                {
                    "opponent": identifier,
                    "side": side,
                    "games": len(selected),
                    "results": {k: results[k] for k in ("win", "loss", "draw", "unfinished")},
                    "terminal_reasons": dict(sorted(reasons.items())),
                    "cutoff_rate": results["unfinished"] / len(selected),
                    "latency_ms": {
                        "decisions": len(samples),
                        "mean": sum(samples) / len(samples) if samples else None,
                        "p50": samples[ceil(0.5 * len(samples)) - 1] if samples else None,
                        "p95": samples[ceil(0.95 * len(samples)) - 1] if samples else None,
                        "max": samples[-1] if samples else None,
                    },
                }
            )
    return rows


def smoke(output, *, seeds=(100_000, 100_001), max_plies=32):
    """Paired seeds and sides; retain every replay and distinguish unfinished games."""
    seeds = tuple(seeds)
    if not seeds or any(type(seed) is not int or seed < 0 for seed in seeds):
        raise ValueError("Provide nonnegative integer seeds")
    if len(set(seeds)) != len(seeds):
        raise ValueError("Seeds must be unique")
    if type(max_plies) is not int or max_plies < 1:
        raise ValueError("max_plies must be a positive integer")
    output = Path(output)
    output.mkdir(parents=True, exist_ok=False)
    ruleset = Ruleset.COPENHAGEN_V2
    ids = ("random-v1", "heuristic-v1", "mcts-v1-i4-r4")
    root = Path(__file__).parent
    source_hash = hashlib.sha256()
    for path in sorted(root.rglob("*.py")):
        source_hash.update(path.relative_to(root).as_posix().encode() + b"\0")
        source_hash.update(path.read_bytes())
    report = {
        "schema": 2,
        "seed_scheme": (
            "identity-paired-v1: SHA256(JSON([scheme, seed, opponent ID])) first 8 bytes big-endian"
        ),
        "latency_percentiles": "nearest rank over decisions, not per-game averages",
        "purpose": "pipeline smoke; no strength or complete Copenhagen claim",
        "ruleset": ruleset.value,
        "python": platform.python_version(),
        "platform": platform.platform(),
        "source_sha256": source_hash.hexdigest(),
        "seeds": seeds,
        "max_plies": max_plies,
        "opponents": [asdict(opponent(identifier)) for identifier in ids],
        "games": [],
    }
    for first in range(len(ids)):
        for second in range(first + 1, len(ids)):
            for attacker, defender in ((ids[first], ids[second]), (ids[second], ids[first])):
                for seed in seeds:
                    policy_seeds = [policy_seed(seed, name) for name in (attacker, defender)]
                    players = [
                        MeasuredPolicy(opponent(name).create(ruleset, stream))
                        for name, stream in zip((attacker, defender), policy_seeds, strict=True)
                    ]
                    replay = play_game(*players, ruleset=ruleset, max_plies=max_plies)
                    state = replay.verify()
                    name = f"{attacker}--{defender}--{seed}.json"
                    text = replay.to_json()
                    (output / name).write_text(text)
                    report["games"].append(
                        {
                            "attackers": attacker,
                            "defenders": defender,
                            "seed": seed,
                            "policy_seeds": dict(
                                zip(("attackers", "defenders"), policy_seeds, strict=True)
                            ),
                            "plies": state.ply,
                            "outcome": asdict(state.outcome) if state.outcome else None,
                            "truncated": state.outcome is None,
                            "replay": name,
                            "replay_sha256": hashlib.sha256(text.encode()).hexdigest(),
                            "decision_ms": [player.samples for player in players],
                        }
                    )
    report["summary"] = summarize(report["games"])
    (output / "report.json").write_text(json.dumps(report, indent=2) + "\n")
    return report


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--max-plies", type=int, default=32)
    args = parser.parse_args()
    result = smoke(args.output, max_plies=args.max_plies)
    print(f"Verified {len(result['games'])} Copenhagen replays; report: {args.output}/report.json")
