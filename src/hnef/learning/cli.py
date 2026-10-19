"""Training, evaluation, model catalog, and static model export commands."""

import argparse
import json
from pathlib import Path


def positive(value):
    number = int(value)
    if number < 1:
        raise argparse.ArgumentTypeError("must be positive")
    return number


def main(argv=None):
    parser = argparse.ArgumentParser(description="CPU mini-v1 RL baseline and model artifacts")
    commands = parser.add_subparsers(dest="command", required=True)
    training = commands.add_parser("train")
    training.add_argument("--run", type=Path, required=True)
    training.add_argument("--updates", type=positive, default=100)
    training.add_argument("--episodes-per-update", type=positive, default=16)
    training.add_argument("--max-plies", type=positive, default=128)
    training.add_argument("--seed", type=int, default=42)
    training.add_argument(
        "--opponents", nargs="+", choices=("random", "heuristic", "self"), default=["random"]
    )
    training.add_argument("--resume", action="store_true")
    evaluation = commands.add_parser("evaluate")
    evaluation.add_argument("checkpoint", type=Path)
    evaluation.add_argument("--output", type=Path, required=True)
    evaluation.add_argument("--games", type=positive, default=50)
    evaluation.add_argument("--seed", type=int, default=100_000)
    evaluation.add_argument(
        "--opponents",
        nargs="+",
        choices=("random", "heuristic", "mcts"),
        default=["random", "heuristic"],
    )
    evaluation.add_argument("--max-plies", type=positive, default=128)
    evaluation.add_argument("--iterations", type=positive, default=64)
    evaluation.add_argument("--rollout-plies", type=positive, default=48)
    evaluation.add_argument("--greedy", action="store_true")
    promotion = commands.add_parser("promote")
    promotion.add_argument("checkpoint", type=Path)
    promotion.add_argument("--report", type=Path, required=True)
    promotion.add_argument("--models", type=Path, default=Path("models"))
    promotion.add_argument("--alias", default="candidate")
    catalog = commands.add_parser("list")
    catalog.add_argument("--runs", type=Path, default=Path("runs"))
    catalog.add_argument("--models", type=Path, default=Path("models"))
    export = commands.add_parser("export-web")
    export.add_argument("checkpoint", type=Path)
    export.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    try:
        if args.command == "train":
            from .train import TrainConfig, train

            config = TrainConfig(
                seed=args.seed,
                episodes_per_update=args.episodes_per_update,
                max_plies=args.max_plies,
                opponents=tuple(args.opponents),
            )
            print(train(args.run, updates=args.updates, config=config, resume=args.resume))
        elif args.command == "evaluate":
            from .evaluate import evaluate_checkpoint

            evaluate_checkpoint(
                args.checkpoint,
                args.output,
                games=args.games,
                seed=args.seed,
                opponents=args.opponents,
                max_plies=args.max_plies,
                iterations=args.iterations,
                rollout_plies=args.rollout_plies,
                greedy=args.greedy,
            )
        elif args.command == "promote":
            from .artifacts import promote

            print(promote(args.checkpoint, args.report, args.models, args.alias))
        elif args.command == "export-web":
            from .artifacts import sha256, write_json
            from .network import checkpoint_model

            model, data = checkpoint_model(args.checkpoint)
            write_json(
                args.output,
                {
                    "schema": 1,
                    "features": data["features"],
                    "ruleset": data["ruleset"],
                    "network": data["network"],
                    "checkpoint_sha256": sha256(args.checkpoint),
                    "update": data["update"],
                    "weights": {key: tensor.tolist() for key, tensor in model.state_dict().items()},
                },
            )
        else:
            runs = [
                {
                    "run": str(path.parent),
                    **json.loads(path.read_text()),
                    "latest": json.loads((path.parent / "latest.json").read_text()),
                }
                for path in sorted(args.runs.glob("*/run.json"))
            ]
            releases = [
                json.loads(path.read_text()) for path in sorted(args.models.glob("*/manifest.json"))
            ]
            print(json.dumps({"runs": runs, "releases": releases}, indent=2))
    except (ValueError, OSError, KeyError, RuntimeError) as error:
        parser.error(str(error))
