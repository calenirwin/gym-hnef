"""Read-only replay diagnostics for one bounded training invocation.

Run with PYTHONPATH=src and OPENBLAS_NUM_THREADS=OMP_NUM_THREADS=MKL_NUM_THREADS=1.
Lives outside the training source fingerprint: old checkpoints remain strictly checked.
"""

import argparse
import json
import time
from pathlib import Path

import numpy as np

from hnef.engine import apply_move, legal_moves
from hnef.learning.artifacts import sha256, write_json
from hnef.learning.copenhagen import CODEC
from hnef.learning.curves import measure, training_positions
from hnef.learning.onpolicy import ALGORITHM, collect, cpu_caps, gradients, trace_state, update
from hnef.learning.plumbing import contract, load


def clipping(actor_norm, critic_norm):
    joint = min(1.0, 1 / max(float(np.hypot(actor_norm, critic_norm)), 1e-12))
    actor_only = min(1.0, 1 / max(actor_norm, 1e-12))
    return joint / actor_only


def visits(initial, trace, samples):
    """Exact immediate wins at sampled learner states, not minimax opportunities."""
    state, cursor, rows = initial, 0, []
    for sample in samples:
        action = sample[2]
        wins = {
            CODEC.encode(m)
            for m in legal_moves(state)
            if (s := apply_move(state, m)).outcome is not None and s.outcome.winner is state.player
        }
        if trace["actions"][cursor] != action:
            raise ValueError("Sample/trace action mismatch")
        rows.append(
            dict(
                opportunity=bool(wins), selected_win=action in wins, advantage=sample[4] - sample[3]
            )
        )
        state = apply_move(state, CODEC.decode(trace["actions"][cursor]))
        cursor += 1
        if state.outcome is None:
            state = apply_move(state, CODEC.decode(trace["actions"][cursor]))
            cursor += 1
    if cursor != len(trace["actions"]):
        raise ValueError("Unconsumed trace actions")
    return rows


def diagnose(data, run, output):
    cpu_caps()
    started = time.monotonic()
    run = Path(run)
    report = json.loads((run / "report.json").read_text())
    ids, states, winning, digest = training_positions(data)
    broad = [
        i
        for i, c in enumerate(json.loads(Path(data).read_text())["cases"])
        if c.get("stage") == "broad"
    ]
    if not broad or not 1 <= len(report["records"]) <= 32:
        raise ValueError("Require broad training probes and 1..32 updates")
    expected = {**contract(digest), "schema": 2, "algorithm": ALGORITHM}
    if any(report["contract"].get(k) != v for k, v in expected.items()):
        raise ValueError("Training report contract mismatch")
    records, last = [], None
    for record in report["records"]:
        if time.monotonic() - started > 60:
            raise TimeoutError("60-second diagnostic cap exhausted")
        step = record["episode"]
        if last is not None and step != last + 1:
            raise ValueError("Updates must be consecutive")
        last = step
        before = run / f"checkpoint-{step - 1:04}.npz"
        after = run / f"checkpoint-{step:04}.npz"
        model, meta, rng = load(before, report["contract"])
        saved, next_meta, next_rng = load(after, report["contract"])
        if meta["step"] != step - 1 or next_meta["step"] != step:
            raise ValueError("Checkpoint step mismatch")
        trace_path = run / record["trace"]
        trace = json.loads(trace_path.read_text())
        trace_state(trace)
        initial = states[record["start_index"]]
        samples, replay = collect(
            model,
            initial,
            rng,
            trace["opponent_seed"],
            meta["horizon"],
            meta["gamma"],
            record["opponent"],
            meta.get("behavior", "policy-v1"),
        )
        if json.loads(json.dumps(replay)) != trace:
            raise ValueError("Exact collection replay failed")
        sample_rows = visits(initial, trace, samples)
        _, _, metrics = gradients(model, samples)
        probe_args = (
            [ids[i] for i in broad],
            [states[i] for i in broad],
            [winning[i] for i in broad],
        )
        _, pre = measure(model, *probe_args)
        update(model, samples, meta["learning_rate"])
        if not all(
            np.array_equal(getattr(model, k), getattr(saved, k)) for k in ("actor", "critic")
        ):
            raise ValueError("Exact optimizer replay failed")
        if rng.bit_generator.state != next_rng.bit_generator.state:
            raise ValueError("RNG replay mismatch")
        _, post = measure(model, *probe_args)
        records.append(
            dict(
                step=step,
                stage=record["stage"],
                learner=record["learner"],
                reward=record["reward"],
                truncated=record["truncated"],
                samples=sample_rows,
                actor_clip_ratio=clipping(
                    metrics["actor_gradient_norm"], metrics["critic_gradient_norm"]
                ),
                broad_probability_delta={
                    a["side"]: b["mean_immediate_win_probability"]
                    - a["mean_immediate_win_probability"]
                    for a, b in zip(pre, post, strict=True)
                },
                before_sha256=sha256(before),
                after_sha256=sha256(after),
                trace_sha256=sha256(trace_path),
            )
        )
    result = dict(
        schema=1,
        purpose="training-only replay diagnosis; not strength evaluation",
        contract=expected,
        input_report_sha256=sha256(run / "report.json"),
        records=records,
        elapsed_seconds=time.monotonic() - started,
        limitations="Separate actor/critic parameters; clipping ratio isolates shared "
        "norm scaling only. Probability deltas measure update interference on synthetic "
        "training probes, not causal critic ablation or full-game strength.",
    )
    output = Path(output)
    output.mkdir(parents=True, exist_ok=False)
    write_json(output / "report.json", result)
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("data", "run", "output"):
        parser.add_argument("--" + name, type=Path, required=True)
    args = parser.parse_args()
    result = diagnose(args.data, args.run, args.output)
    print(json.dumps(dict(updates=len(result["records"]), seconds=result["elapsed_seconds"])))
