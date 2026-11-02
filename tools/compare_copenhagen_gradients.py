"""Read-only local optimizer factorial on saved, frozen-policy rollout batches.

Use the trainer source revision matching the input checkpoints on PYTHONPATH.
No model checkpoint is written; local probe changes are not a learning curve.
"""

import argparse
import ast
import json
import time
from pathlib import Path

import numpy as np

from hnef.learning import batches
from hnef.learning.artifacts import sha256, write_json
from hnef.learning.curves import measure
from hnef.learning.onpolicy import collect, cpu_caps, gradients
from hnef.learning.plumbing import LinearModel, load

VERIFIER = Path(__file__).resolve().parents[1] / "src/hnef/learning/batches.py"
ARMS = (
    ("decision", "aggregate"),
    ("episode", "aggregate"),
    ("decision", "per-rollout"),
    ("episode", "per-rollout"),
)


def strict_verify(run, data, verifier=VERIFIER):
    """Apply current verifier to identical original training code, without hash overrides."""
    old = ast.parse(Path(batches.__file__).read_text())
    new = ast.parse(Path(verifier).read_text())

    def training(tree):
        return ast.dump(
            ast.Module(
                body=[
                    n for n in tree.body if not isinstance(n, ast.FunctionDef) or n.name != "verify"
                ],
                type_ignores=[],
            ),
            include_attributes=False,
        )

    if training(old) != training(new):
        raise ValueError("Verifier compatibility requires identical training code")
    node = next(n for n in new.body if isinstance(n, ast.FunctionDef) and n.name == "verify")
    namespace = batches.__dict__.copy()
    exec(compile(ast.Module(body=[node], type_ignores=[]), str(verifier), "exec"), namespace)
    return namespace["verify"](run, data)


def clipped(actor, critic):
    norm = np.sqrt(np.square(actor).sum() + np.square(critic).sum())
    if not np.isfinite(norm):
        raise ValueError("Nonfinite gradient")
    scale = min(1.0, 1 / max(float(norm), 1e-12))
    return actor * scale, critic * scale


def direction(parts, counts, weighting, clipping):
    """Equal total learning-rate exposure in all four arms."""
    if weighting not in ("decision", "episode") or clipping not in ("aggregate", "per-rollout"):
        raise ValueError("Unknown factorial arm")
    if not parts or len(parts) != len(counts) or any(type(n) is not int or n <= 0 for n in counts):
        raise ValueError("Invalid rollout counts")
    weights = np.array(counts, dtype=float) if weighting == "decision" else np.ones(len(parts))
    weights /= weights.sum()
    used = [clipped(a, c) for a, c in parts] if clipping == "per-rollout" else parts
    actor = sum(w * a for w, (a, _) in zip(weights, used, strict=True))
    critic = sum(w * c for w, (_, c) in zip(weights, used, strict=True))
    return clipped(actor, critic) if clipping == "aggregate" else (actor, critic)


def candidate(model, actor, critic, dose=1.0):
    result = LinearModel()
    result.actor = model.actor - 0.05 * dose * actor
    result.critic = model.critic - 0.05 * dose * critic
    if not np.isfinite(result.actor).all() or not np.isfinite(result.critic).all():
        raise ValueError("Nonfinite local counterfactual")
    return result


def compare(run, data, output):
    cpu_caps()
    started = time.monotonic()
    run = Path(run)
    # Includes the strengthened report-level initial baseline check.
    verification = strict_verify(run, data)
    report = json.loads((run / "report.json").read_text())
    cfg = report["contract"]
    if cfg["mode"] != "cell-batch-v1":
        raise ValueError("Only genuinely frozen-policy batch inputs are supported")
    ids, states, winning, expected = batches.setup(data, cfg["mode"], cfg["seed"])
    records = []
    for record in report["records"]:
        if time.monotonic() - started > 60:
            raise TimeoutError("60-second local comparison cap exhausted")
        index = record["cell"]
        before = run / f"checkpoint-{index:04}.npz"
        after = run / f"checkpoint-{index + 1:04}.npz"
        model, _, _ = load(before, expected)
        saved, _, _ = load(after, expected)
        parts, counts, remaining = [], [], 8
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
                raise ValueError("Frozen rollout replay mismatch")
            a, c, _ = gradients(model, samples)
            parts.append((a, c))
            counts.append(len(samples))
            remaining -= len(samples)
        if remaining != 0:
            raise ValueError("Decision quota mismatch")
        baseline_positions, baseline_summary = measure(model, ids, states, winning)
        arms = []
        for weighting, clipping in ARMS:
            a, c = direction(parts, counts, weighting, clipping)
            counterfactual = candidate(model, a, c)
            if (weighting, clipping) == ("decision", "aggregate"):
                if not all(
                    np.allclose(getattr(counterfactual, k), getattr(saved, k), rtol=0, atol=1e-12)
                    for k in ("actor", "critic")
                ):
                    raise ValueError("Actual batch update reconstruction failed")
            positions, summary = measure(counterfactual, ids, states, winning)
            arms.append(
                dict(
                    weighting=weighting,
                    clipping=clipping,
                    actor_delta_norm=float(np.linalg.norm(0.05 * a)),
                    critic_delta_norm=float(np.linalg.norm(0.05 * c)),
                    raw_positions=positions,
                    raw_summary=summary,
                )
            )
        a, c = direction(parts, counts, "episode", "per-rollout")
        one = candidate(model, a, c)
        sequential = LinearModel()
        sequential.actor, sequential.critic = model.actor.copy(), model.critic.copy()
        for ga, gc in parts:
            ga, gc = clipped(ga, gc)
            sequential.actor -= 0.05 * ga / len(parts)
            sequential.critic -= 0.05 * gc / len(parts)
        error = max(
            float(np.max(np.abs(getattr(one, k) - getattr(sequential, k))))
            for k in ("actor", "critic")
        )
        if error > 1e-12:
            raise ValueError("Fixed-gradient split-step equivalence failed")
        # Explicitly unmatched dose diagnostic, not a fifth normalized factorial arm.
        _, dose_summary = measure(candidate(model, a, c, dose=len(parts)), ids, states, winning)
        records.append(
            dict(
                cell=index,
                start_id=ids[index % 8],
                learner=record["learner"],
                rollout_counts=counts,
                baseline_positions=baseline_positions,
                baseline_summary=baseline_summary,
                arms=arms,
                split_step_max_error=error,
                unmatched_episode_dose_multiplier=len(parts),
                unmatched_dose_summary=dose_summary,
                before_sha256=sha256(before),
                after_sha256=sha256(after),
            )
        )
    result = dict(
        schema=1,
        purpose="local frozen-data optimizer counterfactuals, not trained models",
        contract=expected,
        records=records,
        input_report_sha256=sha256(run / "report.json"),
        verifier_sha256=sha256(VERIFIER),
        comparison_sha256=sha256(__file__),
        strict_verification=verification,
        elapsed_seconds=time.monotonic() - started,
        limitations="Same states/actions/targets and initial parameters for each local arm. "
        "No trajectory refresh, cumulative training curve, heldout evaluation or strength claim. "
        "Split-step equivalence holds for fixed gradients and this stateless SGD, not live online training. "
        "The unmatched dose diagnostic changes total step size and is not a normalized arm.",
    )
    output = Path(output)
    output.mkdir(parents=True, exist_ok=False)
    write_json(output / "report.json", result)
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    for key in ("data", "run", "output"):
        parser.add_argument("--" + key, type=Path, required=True)
    args = parser.parse_args()
    r = compare(args.run, args.data, args.output)
    print(json.dumps(dict(cells=len(r["records"]), seconds=r["elapsed_seconds"])))
