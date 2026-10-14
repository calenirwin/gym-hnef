"""Run seeded games, verify saved games, and benchmark the engine."""

import argparse
import json
from collections import Counter
from pathlib import Path

from .engine import GameState
from .replay import Replay, random_game


def _positive(value: str) -> int:
    number = int(value)
    if number < 1:
        raise argparse.ArgumentTypeError("must be positive")
    return number


def _show(state: GameState) -> None:
    print("  0 1 2 3 4")
    for index, row in enumerate(state.board):
        print(f"{index} {' '.join(row)}")
    if state.outcome is None:
        print(f"Ply {state.ply}: {state.player.value} to move (collection limit reached)")
    else:
        winner = state.outcome.winner.value if state.outcome.winner else "draw"
        print(f"Ply {state.ply}: {winner} / {state.outcome.reason}")


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Hnef mini-v1 engine")
    commands = parser.add_subparsers(dest="command", required=True)
    play = commands.add_parser("play", help="run a random-vs-random mini game")
    play.add_argument("--seed", type=int, default=0)
    play.add_argument("--max-plies", type=_positive, default=500)
    play.add_argument("--output", type=Path)
    replay = commands.add_parser("replay", help="verify and display a saved game")
    replay.add_argument("path", type=Path)
    benchmark = commands.add_parser(
        "benchmark", help="verify seeded random games and their replays"
    )
    benchmark.add_argument("--games", type=_positive, default=1000)
    benchmark.add_argument("--seed", type=int, default=0)
    benchmark.add_argument("--max-plies", type=_positive, default=500)
    args = parser.parse_args(argv)
    try:
        if args.command == "replay":
            _show(Replay.from_json(args.path.read_text()).verify())
        elif args.command == "play":
            record = random_game(args.seed, args.max_plies)
            state = record.verify()
            if args.output:
                args.output.write_text(record.to_json())
            _show(state)
        else:
            counts: Counter[str] = Counter()
            plies = 0
            for seed in range(args.seed, args.seed + args.games):
                state = random_game(seed, args.max_plies).verify()
                plies += state.ply
                if state.outcome is None:
                    counts["truncated"] += 1
                elif state.outcome.winner is None:
                    counts["draws"] += 1
                else:
                    counts[state.outcome.winner.value] += 1
            print(json.dumps({"games": args.games, "plies": plies, "results": counts}, indent=2))
    except (ValueError, OSError) as error:
        parser.error(str(error))
