"""Read-only gradient accumulation and TD residuals on verified saved curve phases.

Run with the exact source revision matching checkpoints on PYTHONPATH. No new
training, heldout probes, parameter updates or checkpoint selection are performed.
"""

import argparse
import json
import time
from pathlib import Path

import numpy as np

from hnef.learning import batches
from hnef.learning.artifacts import sha256, write_json
from hnef.learning.onpolicy import collect, cpu_caps, features, gradients
from hnef.learning.plumbing import load


def projection(model, state, winning, delta):
    """First-order raw winning-mass change under an actual parameter delta."""
    x, mask = features(state)
    p, _ = model.forward(x[None], mask[None])
    p = p[0]
    indicator = np.zeros_like(p)
    indicator[list(winning)] = 1
    mass = float(p @ indicator)
    return float((x @ delta) @ (p * (indicator - mass)))


def residuals(samples, trace):
    """Classify only the final sample as terminal/cutoff, not its whole rollout."""
    rows = []
    for i, sample in enumerate(samples):
        last = i == len(samples) - 1
        kind = "bootstrap-intermediate"
        if last:
            kind = f"terminal-{trace['reward']}" if trace["terminated"] else "bootstrap-cutoff"
        value, target = sample[3:5]
        rows.append(dict(kind=kind, value=value, target=target, residual=target - value))
    return rows


def diagnose(run, data, output):
    cpu_caps()
    started = time.monotonic()
    run = Path(run)
    # Replays all traces, gradients, parameters, raw probes and initial baselines.
    verification = batches.verify(run, data)
    saved = json.loads((run / "report.json").read_text())
    cfg = saved["contract"]
    if cfg.get("budget_profile") != batches.CURVE_PROFILE or cfg["mode"] != "cell-normalized-v1":
        raise ValueError("Require fixed normalized curve profile")
    ids, states, winning, expected = batches.setup(
        data, cfg["mode"], cfg["seed"], cfg["budget_profile"]
    )
    records = []
    for record in saved["records"]:
        if time.monotonic() - started > 60:
            raise TimeoutError("60-second diagnostic cap exhausted")
        index = record["cell"]
        model, _, _ = load(run / f"checkpoint-{index:04}.npz", expected)
        after, _, _ = load(run / f"checkpoint-{index + 1:04}.npz", expected)
        actor_delta, critic_delta = after.actor - model.actor, after.critic - model.critic
        da, dc = np.zeros_like(actor_delta), np.zeros_like(critic_delta)
        parts, remaining = [], 8
        for rollout in record["rollouts"]:
            samples, trace = collect(
                model,
                states[index % 8],
                np.random.default_rng(rollout["learner_seed"]),
                rollout["reference_seed"],
                min(4, remaining),
                0.99,
                record["reference"],
                batches.BEHAVIOR,
            )
            if json.loads(json.dumps(trace)) != rollout["trace"]:
                raise ValueError("Diagnostic trace mismatch")
            a, c, _ = gradients(model, samples)
            dose = 0.05 * len(samples) / 8 * rollout["clip_scale"]
            da -= dose * a
            dc -= dose * c
            parts.append(
                dict(residuals=residuals(samples, trace), clip_scale=rollout["clip_scale"])
            )
            remaining -= len(samples)
        if (
            remaining
            or not np.allclose(da, actor_delta, rtol=0, atol=1e-15)
            or not np.allclose(dc, critic_delta, rtol=0, atol=1e-15)
        ):
            raise ValueError("Accumulated gradient mismatch")
        before_rows = (
            saved["initial_positions"]
            if index == saved["records"][0]["cell"]
            else saved["records"][index - saved["records"][0]["cell"] - 1]["raw_positions"]
        )
        effects = []
        for i, (identifier, state, wins) in enumerate(zip(ids, states, winning, strict=True)):
            if not wins:
                continue
            effects.append(
                dict(
                    target_id=identifier,
                    target_side=state.player.value,
                    own_start=identifier == record["start_id"],
                    predicted=projection(model, state, wins, da),
                    observed=record["raw_positions"][i]["immediate_win_probability"]
                    - before_rows[i]["immediate_win_probability"],
                )
            )
        records.append(
            dict(
                cell=index,
                cycle=index // 16 + 1,
                start_id=record["start_id"],
                side=record["learner"],
                reference=record["reference"],
                effects=effects,
                rollouts=parts,
            )
        )
    report = dict(
        schema=1,
        contract=cfg,
        input_sha256=sha256(run / "report.json"),
        verification=verification,
        records=records,
        elapsed_seconds=time.monotonic() - started,
        limitation="Training-only local first-order projections are descriptive, not isolated causal effects or strength evidence. Residuals are sampled on-policy TD errors, not full-game value accuracy.",
    )
    output = Path(output)
    output.mkdir(parents=True, exist_ok=False)
    write_json(output / "diagnosis.json", report)
    return report


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", type=Path, required=True)
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = diagnose(args.run, args.data, args.output)
    print(json.dumps({"cells": len(result["records"]), "seconds": result["elapsed_seconds"]}))
