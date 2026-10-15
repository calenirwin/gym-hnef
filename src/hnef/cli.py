"""Run seeded games, verify saved games, and benchmark the engine."""

import argparse
import json
import sys
from collections import Counter
from pathlib import Path

from .engine import GameState, Ruleset
from .replay import Replay, random_game


def _positive(value: str) -> int:
    number = int(value)
    if number < 1:
        raise argparse.ArgumentTypeError("must be positive")
    return number


def _show(state: GameState) -> None:
    _warn_experimental(state.ruleset)
    width = len(str(state.ruleset.size - 1))
    print(" " * (width + 1) + " ".join(f"{i:>{width}}" for i in range(state.ruleset.size)))
    for index, row in enumerate(state.board):
        print(f"{index:>{width}} " + " ".join(f"{piece:>{width}}" for piece in row))
    if state.outcome is None:
        print(f"Ply {state.ply}: {state.player.value} to move (collection limit reached)")
    else:
        winner = state.outcome.winner.value if state.outcome.winner else "draw"
        print(f"Ply {state.ply}: {winner} / {state.outcome.reason}")


def _warn_experimental(ruleset: Ruleset) -> None:
    if ruleset.is_copenhagen:
        limits = (
            "exit forts and dead-position draws are not adjudicated."
            if ruleset is Ruleset.COPENHAGEN
            else "fort detection is conservative; draws require external adjudication."
        )
        print(
            f"Experimental Copenhagen: {limits}",
            file=sys.stderr,
        )


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Hnef mini and experimental Copenhagen engine")
    commands = parser.add_subparsers(dest="command", required=True)
    play = commands.add_parser("play", help="run a random-vs-random game")
    play.add_argument("--seed", type=int, default=0)
    play.add_argument("--max-plies", type=_positive, default=500)
    play.add_argument("--output", type=Path)
    replay = commands.add_parser("replay", help="verify and display a saved game")
    replay.add_argument("path", type=Path)
    adjudicate = commands.add_parser(
        "adjudicate-draw", help="record an external draw judgment in a Copenhagen v2 replay"
    )
    adjudicate.add_argument("path", type=Path)
    adjudicate.add_argument("--note", required=True, help="explain the referee/player agreement")
    adjudicate.add_argument("--output", type=Path, required=True)
    benchmark = commands.add_parser(
        "benchmark", help="verify seeded random games and their replays"
    )
    benchmark.add_argument("--games", type=_positive, default=1000)
    benchmark.add_argument("--seed", type=int, default=0)
    benchmark.add_argument("--max-plies", type=_positive, default=500)
    for command in (play, benchmark):
        command.add_argument("--ruleset", choices=list(Ruleset), default=Ruleset.MINI, type=Ruleset)
    args = parser.parse_args(argv)
    try:
        if args.command == "replay":
            _show(Replay.from_json(args.path.read_text()).verify())
        elif args.command == "adjudicate-draw":
            record = Replay.from_json(args.path.read_text()).adjudicate_draw(args.note)
            args.output.write_text(record.to_json())
            _show(record.verify())
        elif args.command == "play":
            record = random_game(args.seed, args.max_plies, ruleset=args.ruleset)
            state = record.verify()
            if args.output:
                args.output.write_text(record.to_json())
            _show(state)
        else:
            _warn_experimental(args.ruleset)
            counts: Counter[str] = Counter()
            plies = 0
            for seed in range(args.seed, args.seed + args.games):
                state = random_game(seed, args.max_plies, ruleset=args.ruleset).verify()
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
