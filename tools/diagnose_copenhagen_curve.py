"""Read-only gradient accumulation and TD residuals on verified saved curve phases.

Run with the exact source revision matching checkpoints on PYTHONPATH. No new
training, heldout probes, parameter updates or checkpoint selection are performed.
"""

import argparse
import hashlib
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


def critic_alignment(samples, kinds, scales, before, after):
    """Fixed-target half-MSE projections; no re-evaluated bootstrap targets."""
    if not samples or not len(samples) == len(kinds) == len(scales):
        raise ValueError("Invalid critic diagnostic sample counts")
    x = np.asarray([s[0] for s in samples])
    y = np.asarray([s[4] for s in samples])
    v = np.tanh(x @ before)
    nv = np.tanh(x @ after)
    if not np.allclose(v, [s[3] for s in samples], rtol=0, atol=1e-12):
        raise ValueError("Critic sample values do not match checkpoint")
    g = ((v - y) * (1 - v * v))[:, None] * x
    updates = -0.05 / 8 * np.asarray(scales)[:, None] * g
    if not np.allclose(updates.sum(axis=0), after - before, rtol=0, atol=1e-15):
        raise ValueError("Per-sample critic accumulation mismatch")
    groups = {}
    for kind in sorted(set(kinds)):
        take = np.asarray([k == kind for k in kinds])
        groups[kind] = (take, g[take].mean(axis=0), updates[take].sum(axis=0))
    rows = []
    for target, (take, gradient, _) in groups.items():
        influences = []
        for source, (_, _, delta) in groups.items():
            norm = float(np.linalg.norm(gradient) * np.linalg.norm(delta))
            influences.append(
                dict(
                    source=source,
                    gradient_cosine=float(-gradient @ delta / norm) if norm else None,
                    predicted_half_mse_change=float(gradient @ delta),
                )
            )
        rows.append(
            dict(
                target=target,
                count=int(take.sum()),
                before_half_mse=float(np.square(v[take] - y[take]).mean() / 2),
                after_half_mse=float(np.square(nv[take] - y[take]).mean() / 2),
                influences=influences,
            )
        )
    # Same encoded state can have different sampled outcomes under its policy.
    # Within this frozen cell its prediction is identical; MSE = variance + bias².
    terminal = {}
    for i, kind in enumerate(kinds):
        if not kind.startswith("terminal-"):
            continue
        key = hashlib.sha256(np.asarray(x[i], dtype="<f8").tobytes()).hexdigest()
        terminal.setdefault(key, []).append(i)
    aliases = []
    for key, indices in terminal.items():
        targets, values = y[indices], v[indices]
        if not np.all(values == values[0]):
            raise ValueError("Identical critic inputs have inconsistent predictions")
        variance = float(np.var(targets))
        bias = float((values[0] - targets.mean()) ** 2)
        mse = float(np.square(values - targets).mean())
        if not np.isclose(mse, variance + bias, rtol=0, atol=1e-12):
            raise ValueError("Terminal variance decomposition mismatch")
        aliases.append(
            dict(
                feature_sha256=key,
                count=len(indices),
                wins=int((targets == 1).sum()),
                losses=int((targets == -1).sum()),
                target_mean=float(targets.mean()),
                prediction=float(values[0]),
                target_variance=variance,
                squared_mean_error=bias,
                mse=mse,
            )
        )
    return dict(groups=rows, terminal_inputs=aliases)


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
        critic_samples, critic_kinds, critic_scales = [], [], []
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
            critic_samples.extend(samples)
            critic_kinds.extend(r["kind"] for r in residuals(samples, trace))
            critic_scales.extend([rollout["clip_scale"]] * len(samples))
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
                critic_alignment=critic_alignment(
                    critic_samples, critic_kinds, critic_scales, model.critic, after.critic
                ),
            )
        )
    report = dict(
        schema=2,
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
