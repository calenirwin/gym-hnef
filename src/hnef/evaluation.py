"""Small dependency-free Copenhagen baseline experiment, not a strength ranking.

Run: python -m hnef.evaluation --output /tmp/copenhagen-smoke
"""

import argparse
import hashlib
import json
import platform
import time
from dataclasses import asdict
from pathlib import Path
from random import Random

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
        "schema": 1,
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
                    rng = Random(seed)
                    players = [
                        MeasuredPolicy(opponent(name).create(ruleset, rng.getrandbits(64)))
                        for name in (attacker, defender)
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
                            "plies": state.ply,
                            "outcome": asdict(state.outcome) if state.outcome else None,
                            "truncated": state.outcome is None,
                            "replay": name,
                            "replay_sha256": hashlib.sha256(text.encode()).hexdigest(),
                            "decision_ms": [player.samples for player in players],
                        }
                    )
    (output / "report.json").write_text(json.dumps(report, indent=2) + "\n")
    return report


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--max-plies", type=int, default=32)
    args = parser.parse_args()
    result = smoke(args.output, max_plies=args.max_plies)
    print(f"Verified {len(result['games'])} Copenhagen replays; report: {args.output}/report.json")
