"""Fixed first-cycle terminal panel; replay saved runs, never train or use heldout."""

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np

from hnef.learning import batches
from hnef.learning.artifacts import write_json
from hnef.learning.onpolicy import collect, cpu_caps
from hnef.learning.plumbing import load


def effects(x, targets, before, after):
    """Actual half-MSE changes on identical features and terminal rewards."""
    old = 0.5 * (np.tanh(x @ before) - targets) ** 2
    new = 0.5 * (np.tanh(x @ after) - targets) ** 2
    return old, new, new - old


def allocate(x, targets, before, components):
    """Exact straight-line allocation for scalar tanh critic half-MSE.

    Every component follows the same simultaneous parameter path. This avoids
    arbitrary component ordering, but is not a causal intervention or unique
    explanation. Stable tanh difference avoids cancellation near zero updates.
    """
    names = sorted(components)
    if not names:
        raise ValueError("No critic components")
    dz_parts = np.array([x @ components[k] for k in names])
    dz = dz_parts.sum(axis=0)
    v = np.tanh(x @ before)
    t = np.tanh(dz)
    dv = np.tanh(x @ before + dz) - v
    small = np.abs(dz) < 0.1
    dv[small] = t[small] * (1 - v[small] * v[small]) / (1 + v[small] * t[small])
    ratio = np.divide(dv, dz, out=1 - v * v, where=dz != 0)
    slope = ratio * (v - targets + 0.5 * dv)
    result = {k: dz_parts[i] * slope for i, k in enumerate(names)}
    if not all(np.isfinite(a).all() for a in result.values()):
        raise ValueError("Nonfinite allocation")
    return result


def components(record, model, state):
    """Reconstruct actual clipped sample contributions, preserving rollout weights."""
    parts, remaining = {}, 8
    for rollout in record["rollouts"]:
        samples, trace = collect(
            model,
            state,
            np.random.default_rng(rollout["learner_seed"]),
            rollout["reference_seed"],
            min(4, remaining),
            0.99,
            record["reference"],
            batches.BEHAVIOR,
        )
        if json.loads(json.dumps(trace)) != rollout["trace"]:
            raise ValueError("Attribution trace mismatch")
        for i, sample in enumerate(samples):
            x, _, _, value, target = sample[:5]
            kind = "bootstrap-intermediate"
            if i == len(samples) - 1:
                kind = f"terminal-{trace['reward']}" if trace["terminated"] else "bootstrap-cutoff"
            delta = -0.05 / 8 * rollout["clip_scale"] * (value - target) * (1 - value * value) * x
            parts[kind] = parts.get(kind, np.zeros_like(model.critic)) + delta
        remaining -= len(samples)
    if remaining:
        raise ValueError("Attribution decision quota mismatch")
    return parts


def diagnose(runs, data, output, *, attribute_outcomes=False):
    cpu_caps()
    runs = [Path(r).resolve() for r in runs]
    if not 3 <= len(runs) <= 8 or len(set(runs)) != len(runs):
        raise ValueError("Require 3..8 unique consecutive eight-cell phases")
    reports, previous = [], None
    for phase, run in enumerate(runs):
        batches.verify(run, data)
        report = json.loads((run / "report.json").read_text())
        cfg = report["contract"]
        if (
            cfg.get("budget_profile") != batches.CURVE_PROFILE
            or cfg["mode"] != "cell-normalized-v1"
        ):
            raise ValueError("Require fixed normalized curve profile")
        if [r["cell"] for r in report["records"]] != list(range(phase * 8, (phase + 1) * 8)):
            raise ValueError("Phases must start at zero and be consecutive")
        if reports and cfg != reports[0]["contract"]:
            raise ValueError("Phase contracts differ")
        initial = load(run / f"checkpoint-{phase * 8:04}.npz", cfg)
        if previous:
            a, am, ar = previous
            b, bm, br = initial
            if (
                am != bm
                or ar.bit_generator.state != br.bit_generator.state
                or not np.array_equal(a.actor, b.actor)
                or not np.array_equal(a.critic, b.critic)
            ):
                raise ValueError("Phase boundary mismatch")
        previous = load(run / f"checkpoint-{(phase + 1) * 8:04}.npz", cfg)
        reports.append(report)
    cfg = reports[0]["contract"]
    _, states, _, _ = batches.setup(data, cfg["mode"], cfg["seed"], cfg["budget_profile"])
    panel = []
    for phase in range(2):
        for record in reports[phase]["records"]:
            index = record["cell"]
            model, _, _ = load(runs[phase] / f"checkpoint-{index:04}.npz", cfg)
            remaining = 8
            for number, rollout in enumerate(record["rollouts"]):
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
                    raise ValueError("Panel trace mismatch")
                if trace["terminated"]:
                    sample = samples[-1]
                    if sample[4] != trace["reward"]:
                        raise ValueError("Terminal target mismatch")
                    panel.append(
                        dict(
                            cell=index,
                            rollout=number,
                            start_id=record["start_id"],
                            side=record["learner"],
                            target=sample[4],
                            features=sample[0].tolist(),
                        )
                    )
                remaining -= len(samples)
    if not panel:
        raise ValueError("First cycle has no terminal samples")
    digest = hashlib.sha256(json.dumps(panel, sort_keys=True).encode()).hexdigest()
    x, y = np.array([r["features"] for r in panel]), np.array([r["target"] for r in panel])
    groups = {}
    for i, row in enumerate(panel):
        groups.setdefault((row["start_id"], row["side"], row["target"]), []).append(i)
    rows = []
    for phase in range(2, len(runs)):
        for record in reports[phase]["records"]:
            index = record["cell"]
            before, _, _ = load(runs[phase] / f"checkpoint-{index:04}.npz", cfg)
            after, _, _ = load(runs[phase] / f"checkpoint-{index + 1:04}.npz", cfg)
            old, new, delta = effects(x, y, before.critic, after.critic)
            allocation = None
            if attribute_outcomes:
                parts = components(record, before, states[index % 8])
                if not np.allclose(
                    sum(parts.values()), after.critic - before.critic, rtol=0, atol=1e-15
                ):
                    raise ValueError("Outcome components do not reconstruct critic update")
                allocation = allocate(x, y, before.critic, parts)
                if not np.allclose(sum(allocation.values()), delta, rtol=0, atol=1e-12):
                    raise ValueError("Outcome allocations do not sum to observed error change")
            rows.append(
                dict(
                    cell=index,
                    start_id=record["start_id"],
                    groups=[
                        dict(
                            start_id=k[0],
                            side=k[1],
                            target=k[2],
                            count=len(indices),
                            own_start=k[0] == record["start_id"],
                            before_half_mse=float(old[indices].mean()),
                            after_half_mse=float(new[indices].mean()),
                            change=float(delta[indices].mean()),
                            **(
                                {
                                    "outcomes": {
                                        k: float(v[indices].mean()) for k, v in allocation.items()
                                    }
                                }
                                if allocation is not None
                                else {}
                            ),
                        )
                        for k, indices in groups.items()
                    ],
                )
            )
    # Every panel group must telescope from checkpoint16 to the last checkpoint.
    first, _, _ = load(runs[1] / "checkpoint-0016.npz", cfg)
    old, new, total = effects(x, y, first.critic, previous[0].critic)
    for k, indices in groups.items():
        changes = [
            g["change"]
            for row in rows
            for g in row["groups"]
            if (g["start_id"], g["side"], g["target"]) == k
        ]
        if not np.isclose(sum(changes), total[indices].mean(), rtol=0, atol=1e-12):
            raise ValueError("Panel changes do not telescope")
    result = dict(
        schema=2 if attribute_outcomes else 1,
        contract=cfg,
        panel_definition="all-first16-terminal-samples-v1",
        panel_sha256=digest,
        panel=panel,
        records=rows,
        limitation="Frozen observed terminal samples, including duplicates; not an unbiased policy-value estimate. No bootstrap targets or heldout. Own/other update attribution is descriptive, not a counterfactual training intervention.",
    )
    output = Path(output)
    output.mkdir(parents=True, exist_ok=False)
    write_json(output / "panel.json", result)
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--attribute-outcomes", action="store_true")
    parser.add_argument("--runs", nargs="+", type=Path, required=True)
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    r = diagnose(args.runs, args.data, args.output, attribute_outcomes=args.attribute_outcomes)
    print(
        json.dumps(
            dict(
                panel_samples=len(r["panel"]),
                later_updates=len(r["records"]),
                panel_sha256=r["panel_sha256"],
            )
        )
    )
