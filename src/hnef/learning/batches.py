"""Bounded, matched-decision comparison of episode and independent-rollout updates."""

import argparse
import json
import time
from pathlib import Path

import numpy as np

from hnef.engine import Player

from .artifacts import write_json
from .copenhagen import stream_seed
from .curves import measure, training_positions
from .onpolicy import collect, cpu_caps, gradients, update
from .plumbing import LinearModel, contract, load, save

ALGORITHM = "same-player-td0-matched-cells-v1"
MODES = ("episode-v1", "cell-batch-v1")
BEHAVIOR = "immediate-win-prior4-v1"
DECISIONS_PER_CELL = 8
TOTAL_CELLS = 16


def cell(model, state, seed, index, mode):
    """Eight decisions from independent fresh rollouts at one start/reference.

    A terminal rollout is followed by a fresh one until the cell quota is met.
    The final rollout horizon is capped by remaining decisions, with normal TD
    bootstrapping. Every batch sample was collected under the same frozen model.
    """
    if mode not in MODES or not 0 <= index < TOTAL_CELLS:
        raise ValueError("Invalid cell mode/index")
    reference = "random-v1" if index < 8 else "heuristic-v1"
    remaining, rollout, records, all_samples = DECISIONS_PER_CELL, 0, [], []
    before_actor, before_critic = model.actor.copy(), model.critic.copy()
    while remaining:
        learner_seed = stream_seed(seed, f"matched-cell-{index}-learner-{rollout}")
        reference_seed = stream_seed(seed, f"matched-cell-{index}-reference-{rollout}")
        samples, trace = collect(
            model,
            state,
            np.random.default_rng(learner_seed),
            reference_seed,
            min(4, remaining),
            0.99,
            reference,
            BEHAVIOR,
        )
        _, _, contribution = gradients(model, samples)
        if mode == "episode-v1":
            update(model, samples, 0.05)
        all_samples.extend(samples)
        records.append(
            dict(
                learner_seed=learner_seed,
                reference_seed=reference_seed,
                decisions=len(samples),
                trace=trace,
                actor_contribution_norm=contribution["actor_gradient_norm"],
                critic_contribution_norm=contribution["critic_gradient_norm"],
            )
        )
        remaining -= len(samples)
        rollout += 1
    if mode == "cell-batch-v1":
        # One mean over decisions, not an unweighted mean of unequal rollouts.
        assert np.array_equal(model.actor, before_actor)
        assert np.array_equal(model.critic, before_critic)
        update(model, all_samples, 0.05)
    return dict(
        cell=index,
        reference=reference,
        learner=state.player.value,
        decisions=DECISIONS_PER_CELL,
        rollouts=records,
        updates=rollout if mode == "episode-v1" else 1,
        actor_delta_norm=float(np.linalg.norm(model.actor - before_actor)),
        critic_delta_norm=float(np.linalg.norm(model.critic - before_critic)),
    )


def setup(data, mode, seed):
    if mode not in MODES or type(seed) is not int or not 0 <= seed < 100000:
        raise ValueError("Use a supported mode and training seed below 100000")
    ids, states, winning, digest = training_positions(data, "broad")
    if len(states) != 8 or any(sum(s.player is side for s in states) != 4 for side in Player):
        raise ValueError("Require eight broad training starts, four per side")
    expected = {
        **contract(digest),
        "schema": 3,
        "algorithm": ALGORITHM,
        "mode": mode,
        "seed": seed,
        "behavior": BEHAVIOR,
        "decisions_per_cell": DECISIONS_PER_CELL,
        "total_cells": TOTAL_CELLS,
        "max_rollout_horizon": 4,
        "gamma": 0.99,
        "learning_rate": 0.05,
        "optimizer": "sgd-no-momentum-v1",
        "rng_policy": "independent-cell-rollout-streams-v1",
        "start_ids": ids,
    }
    return ids, states, winning, expected


def train(output, data, *, mode, seed, cells=8, resume=None):
    cpu_caps()
    if type(cells) is not int or not 1 <= cells <= 8:
        raise ValueError("Use 1..8 cells per invocation")
    started = time.monotonic()
    ids, states, winning, expected = setup(data, mode, seed)
    if resume:
        model, meta, rng = load(resume, expected)
    else:
        model, rng = LinearModel(), np.random.default_rng(seed)
        meta = {**expected, "step": 0, "optimizer_steps": 0}
    if type(meta["step"]) is not int or not 0 <= meta["step"] <= TOTAL_CELLS - cells:
        raise ValueError("Total cell cap is 16")
    output = Path(output)
    output.mkdir(parents=True, exist_ok=False)
    save(output / f"checkpoint-{meta['step']:04}.npz", model, meta, rng)
    initial_positions, initial_summary = measure(model, ids, states, winning)
    records = []
    for _ in range(cells):
        if time.monotonic() - started > 60:
            raise TimeoutError("60-second training budget exhausted")
        index = meta["step"]
        record = cell(model, states[index % 8], seed, index, mode)
        record["start_id"] = ids[index % 8]
        meta["step"] += 1
        meta["optimizer_steps"] += record["updates"]
        positions, summary = measure(model, ids, states, winning)
        record["raw_positions"], record["raw_summary"] = positions, summary
        records.append(record)
        save(output / f"checkpoint-{meta['step']:04}.npz", model, meta, rng)
    report = dict(
        schema=1,
        contract=expected,
        records=records,
        initial_positions=initial_positions,
        initial_summary=initial_summary,
        elapsed_seconds=time.monotonic() - started,
        limitation="Synthetic training only. Equal decision quotas, not equal rollout "
        "or optimizer counts. Rule-assisted behavior; probes measure unassisted raw "
        "policy. A batch contribution is not an isolated causal parameter update.",
    )
    write_json(output / "report.json", report)
    return report


def verify(run, data):
    """Reproduce existing cells in memory; no new training artifacts or evaluation."""
    cpu_caps()
    started = time.monotonic()
    run = Path(run)
    report = json.loads((run / "report.json").read_text())
    cfg = report["contract"]
    ids, states, winning, expected = setup(data, cfg["mode"], cfg["seed"])
    if cfg != expected or not 1 <= len(report["records"]) <= 8:
        raise ValueError("Replay contract/count mismatch")
    first = report["records"][0]["cell"]
    if type(first) is not int or not 0 <= first < TOTAL_CELLS:
        raise ValueError("Invalid initial cell index")
    initial, initial_meta, _ = load(run / f"checkpoint-{first:04}.npz", expected)
    if initial_meta["step"] != first:
        raise ValueError("Initial checkpoint step mismatch")
    initial_positions, initial_summary = measure(initial, ids, states, winning)
    if (
        report.get("initial_positions") != initial_positions
        or report.get("initial_summary") != initial_summary
    ):
        raise ValueError("Initial baseline probe replay mismatch")
    last = None
    for record in report["records"]:
        if time.monotonic() - started > 60:
            raise TimeoutError("60-second verification budget exhausted")
        index = record["cell"]
        if last is not None and index != last + 1:
            raise ValueError("Replay cells must be consecutive")
        last = index
        model, meta, rng = load(run / f"checkpoint-{index:04}.npz", expected)
        after, after_meta, after_rng = load(run / f"checkpoint-{index + 1:04}.npz", expected)
        replay = cell(model, states[index % 8], cfg["seed"], index, cfg["mode"])
        replay["start_id"] = ids[index % 8]
        replay["raw_positions"], replay["raw_summary"] = measure(model, ids, states, winning)
        if json.loads(json.dumps(replay)) != record:
            raise ValueError("Cell trace/gradient/probe replay mismatch")
        if not all(
            np.array_equal(getattr(model, k), getattr(after, k)) for k in ("actor", "critic")
        ):
            raise ValueError("Cell parameter replay mismatch")
        if meta["step"] != index or after_meta["step"] != index + 1:
            raise ValueError("Cell step mismatch")
        if after_meta["optimizer_steps"] != meta["optimizer_steps"] + record["updates"]:
            raise ValueError("Optimizer step mismatch")
        if rng.bit_generator.state != after_rng.bit_generator.state:
            raise ValueError("Unexpected checkpoint RNG mutation")
    return dict(
        cells=len(report["records"]),
        decisions=8 * len(report["records"]),
        elapsed_seconds=time.monotonic() - started,
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("train", "verify"))
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--mode", choices=MODES, default="episode-v1")
    parser.add_argument("--seed", type=int, default=53)
    parser.add_argument("--cells", type=int, default=8)
    parser.add_argument("--resume", type=Path)
    args = parser.parse_args()
    result = (
        train(
            args.output,
            args.data,
            mode=args.mode,
            seed=args.seed,
            cells=args.cells,
            resume=args.resume,
        )
        if args.command == "train"
        else verify(args.output, args.data)
    )
    print(json.dumps({k: v for k, v in result.items() if k != "records"}))
