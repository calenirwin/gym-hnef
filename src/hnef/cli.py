"""Run seeded games, verify saved games, and benchmark the engine."""

import argparse
import json
import sys
from collections import Counter
from pathlib import Path

from .engine import GameState, Ruleset
from .matches import policy_game
from .policies import POLICY_NAMES
from .replay import Replay


def _positive(value: str) -> int:
    number = int(value)
    if number < 1:
        raise argparse.ArgumentTypeError("must be positive")
    return number


def _nonnegative(value: str) -> int:
    number = int(value)
    if number < 0:
        raise argparse.ArgumentTypeError("must be nonnegative")
    return number


def _game(args, seed):
    return policy_game(
        seed,
        args.max_plies,
        ruleset=args.ruleset,
        attackers=args.attackers,
        defenders=args.defenders,
        iterations=args.iterations,
        rollout_plies=args.rollout_plies,
    )


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
    web = commands.add_parser("web", help="open the local browser game")
    web.add_argument("--port", type=_positive, default=8000)
    play = commands.add_parser("play", help="run a seeded game between reference policies")
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
        "benchmark", help="compare seeded policies and verify every replay"
    )
    benchmark.add_argument("--games", type=_positive, default=1000)
    benchmark.add_argument("--seed", type=int, default=0)
    benchmark.add_argument("--max-plies", type=_positive, default=500)
    benchmark.add_argument("--output", type=Path, help="save the JSON benchmark report")
    for command in (play, benchmark):
        command.add_argument("--ruleset", choices=list(Ruleset), default=Ruleset.MINI, type=Ruleset)
        command.add_argument("--attackers", choices=POLICY_NAMES, default="random")
        command.add_argument("--defenders", choices=POLICY_NAMES, default="random")
        command.add_argument(
            "--iterations", type=_positive, default=64, help="MCTS iterations per move"
        )
        command.add_argument(
            "--rollout-plies", type=_nonnegative, default=48, help="MCTS rollout cap"
        )
    args = parser.parse_args(argv)
    try:
        if args.command == "web":
            from .web import serve

            serve(args.port)
        elif args.command == "replay":
            _show(Replay.from_json(args.path.read_text()).verify())
        elif args.command == "adjudicate-draw":
            record = Replay.from_json(args.path.read_text()).adjudicate_draw(args.note)
            args.output.write_text(record.to_json())
            _show(record.verify())
        elif args.command == "play":
            record = _game(args, args.seed)
            state = record.verify()
            if args.output:
                args.output.write_text(record.to_json())
            _show(state)
        else:
            _warn_experimental(args.ruleset)
            counts: Counter[str] = Counter()
            plies = 0
            for seed in range(args.seed, args.seed + args.games):
                state = _game(args, seed).verify()
                plies += state.ply
                if state.outcome is None:
                    counts["truncated"] += 1
                elif state.outcome.winner is None:
                    counts["draws"] += 1
                else:
                    counts[state.outcome.winner.value] += 1
            report = {"games": args.games, "plies": plies, "results": counts}
            if args.attackers != "random" or args.defenders != "random":
                report["config"] = {
                    "policy_version": 1,
                    "ruleset": args.ruleset.value,
                    "attackers": args.attackers,
                    "defenders": args.defenders,
                    "seed": args.seed,
                    "max_plies": args.max_plies,
                    "iterations": args.iterations,
                    "rollout_plies": args.rollout_plies,
                }
            text = json.dumps(report, indent=2) + "\n"
            if args.output:
                args.output.write_text(text)
            print(text, end="")
    except (ValueError, OSError) as error:
        parser.error(str(error))
