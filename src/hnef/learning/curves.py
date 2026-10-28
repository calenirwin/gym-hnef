"""Read-only, exact one-ply policy diagnostics on training-only Copenhagen starts."""

import argparse
import hashlib
import json
import time
from pathlib import Path

import numpy as np

from hnef.engine import Player, apply_move, legal_moves

from .artifacts import write_json
from .copenhagen import CODEC, RULESET
from .onpolicy import ALGORITHM, cpu_caps, features, starts
from .plumbing import contract, load


def training_positions(path):
    cases = json.loads(Path(path).read_text())["cases"]
    if not cases or any(
        c.get("split") != "training" or c.get("ruleset") != RULESET.value or not c.get("board")
        for c in cases
    ):
        raise ValueError("Curve diagnostics require explicit training-only Copenhagen positions")
    ids = [c["id"] for c in cases]
    if len(set(ids)) != len(ids):
        raise ValueError("Training position IDs must be unique")
    states, digest = starts(path)
    # Never consume witness actions or intended reward annotations.
    winning = []
    for state in states:
        actions = set()
        for move in legal_moves(state):
            successor = apply_move(state, move)
            if successor.outcome is not None and successor.outcome.winner is state.player:
                actions.add(CODEC.encode(move))
        winning.append(actions)
    return ids, states, winning, digest


def measure(model, ids, states, winning):
    rows = []
    for identifier, state, actions in zip(ids, states, winning, strict=True):
        x, mask = features(state)
        probs, _ = model.forward(x[None], mask[None])
        if not np.isfinite(probs).all() or np.any(probs[0, ~mask] != 0):
            raise ValueError("Invalid masked probabilities")
        rows.append(
            {
                "id": identifier,
                "side": state.player.value,
                "legal_actions": int(mask.sum()),
                "immediate_winning_actions": len(actions),
                # None means no immediate-win opportunity, not a failed choice.
                "immediate_win_probability": float(probs[0, list(actions)].sum())
                if actions
                else None,
                "uniform_win_probability": len(actions) / int(mask.sum()) if actions else None,
            }
        )
    summary = []
    for side in Player:
        values = [
            r["immediate_win_probability"]
            for r in rows
            if r["side"] == side.value and r["immediate_win_probability"] is not None
        ]
        summary.append(
            {
                "side": side.value,
                "opportunity_positions": len(values),
                "mean_immediate_win_probability": float(np.mean(values)) if values else None,
            }
        )
    return rows, summary


def curves(data, checkpoints, output):
    cpu_caps()
    paths = [Path(p).resolve() for p in checkpoints]
    if not 2 <= len(paths) <= 9 or len(set(paths)) != len(paths):
        raise ValueError("Use 2..9 unique checkpoint paths in increasing update order")
    start = time.monotonic()
    ids, states, winning, digest = training_positions(data)
    expected = {**contract(digest), "schema": 2, "algorithm": ALGORITHM}
    records, config, last_step = [], None, -1
    for path in paths:
        if time.monotonic() - start > 60:
            raise TimeoutError("60-second curve budget exhausted")
        model, meta, _ = load(path, expected)
        current = {k: v for k, v in meta.items() if k not in ("step", "rng")}
        if config is not None and config != current:
            raise ValueError("Curve checkpoints differ in seed or training configuration")
        config = current
        if type(meta["step"]) is not int or meta["step"] <= last_step:
            raise ValueError("Checkpoint update steps must strictly increase")
        last_step = meta["step"]
        rows, summary = measure(model, ids, states, winning)
        records.append(
            {
                "step": last_step,
                "checkpoint": str(path),
                "checkpoint_sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
                "positions": rows,
                "summary": summary,
            }
        )
    report = {
        "schema": 1,
        "purpose": "training-only one-ply learning curves; not strength evaluation",
        "contract": expected,
        "training_config": config,
        "records": records,
        "elapsed_seconds": time.monotonic() - start,
        "limitations": (
            "Synthetic training positions; exact one-ply opportunities under experimental v2, "
            "not minimax values or full-game win rates. "
            "No sampling, updates or checkpoint selection."
        ),
    }
    output = Path(output)
    output.mkdir(parents=True, exist_ok=False)
    write_json(output / "report.json", report)
    return report


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--checkpoints", type=Path, nargs="+", required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = curves(args.data, args.checkpoints, args.output)
    print(
        json.dumps(
            {"records": [{"step": r["step"], "summary": r["summary"]} for r in result["records"]]},
            indent=2,
        )
    )
