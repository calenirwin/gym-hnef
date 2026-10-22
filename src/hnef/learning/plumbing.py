"""Bounded NumPy policy/value optimizer smoke, not reinforcement learning or a bot.

Run with OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1.
"""

import argparse
import hashlib
import json
import os
import time
from pathlib import Path

import numpy as np

from hnef.engine import GameState, Move, Player, apply_move

from .artifacts import provenance, write_json
from .copenhagen import CODEC, CODEC_SHA256, RULESET, encode

FEATURES = "copenhagen-linear-board-repetition-count-v1"
WIDTH = 367  # Three piece planes, mover, normalized ply/repetition, bias.


def dataset(path):
    raw = Path(path).read_bytes()
    cases = json.loads(raw)["cases"]
    if not 1 <= len(cases) <= 8:
        raise ValueError("Dataset must contain 1..8 synthetic transitions")
    rows, masks, actions, values = [], [], [], []
    for case in cases:
        state = GameState(tuple(case["board"]), Player(case["player"]), ruleset=RULESET)
        action = CODEC.encode(Move(*case["move"]))
        obs = encode(state)
        if not obs["legal_mask"][action]:
            raise ValueError("Training target must be legal")
        terminal = apply_move(state, CODEC.decode(action))
        if terminal.outcome is None or terminal.outcome.reason != case["expected_reason"]:
            raise ValueError("Synthetic transition must reach its stated terminal outcome")
        rows.append(
            [v for plane in obs["planes"] for v in plane]
            + [
                float(state.player is Player.DEFENDERS),
                state.ply / (state.ply + 1),
                obs["repetitions"] / 3,
                1.0,
            ]
        )
        masks.append(obs["legal_mask"])
        actions.append(action)
        values.append(terminal.outcome.reward(state.player))
    return (
        np.array(rows, dtype=np.float64),
        np.array(masks, dtype=bool),
        np.array(actions),
        np.array(values, dtype=np.float64),
    ), hashlib.sha256(raw).hexdigest()


class LinearModel:
    def __init__(self):
        self.actor = np.zeros((WIDTH, len(CODEC.moves)), dtype=np.float64)
        self.critic = np.zeros(WIDTH, dtype=np.float64)

    def forward(self, x, mask):
        if not mask.any(axis=1).all():
            raise ValueError("Cannot infer with an empty legal mask")
        if not np.isfinite(self.actor).all() or not np.isfinite(self.critic).all():
            raise ValueError("Nonfinite parameters")
        logits = x @ self.actor
        logits = np.where(mask, logits, -np.inf)
        probs = np.exp(logits - logits.max(axis=1, keepdims=True))
        probs /= probs.sum(axis=1, keepdims=True)
        return probs, np.tanh(x @ self.critic)

    def update(self, x, mask, actions, targets, lr):
        if not mask[np.arange(len(x)), actions].all():
            raise ValueError("Illegal target action")
        probs, values = self.forward(x, mask)
        selected = probs[np.arange(len(x)), actions]
        loss = -np.log(selected).mean() + 0.5 * np.square(values - targets).mean()
        delta = probs.copy()
        delta[np.arange(len(x)), actions] -= 1
        actor_grad = x.T @ delta / len(x)
        critic_grad = x.T @ ((values - targets) * (1 - values**2)) / len(x)
        norm = float(np.sqrt(np.square(actor_grad).sum() + np.square(critic_grad).sum()))
        if not np.isfinite(loss) or not np.isfinite(norm):
            raise ValueError("Nonfinite loss or gradient")
        scale = min(1.0, 1 / max(norm, 1e-12))
        self.actor -= lr * scale * actor_grad
        self.critic -= lr * scale * critic_grad
        if not np.isfinite(self.actor).all() or not np.isfinite(self.critic).all():
            raise ValueError("Nonfinite update")
        return {"loss": float(loss), "gradient_norm": norm}


def contract(data_hash):
    return {
        "schema": 1,
        "ruleset": RULESET.value,
        "features": FEATURES,
        "codec_sha256": CODEC_SHA256,
        "dataset_sha256": data_hash,
        "source_sha256": provenance()["source_sha256"],
        "numpy": np.__version__,
    }


def save(path, model, metadata, rng):
    metadata = {**metadata, "rng": rng.bit_generator.state}
    # Exclusive creation preserves every initialized/intermediate checkpoint.
    with Path(path).open("xb") as file:
        np.savez_compressed(
            file,
            actor=model.actor,
            critic=model.critic,
            metadata=np.array(json.dumps(metadata, allow_nan=False)),
        )


def load(path, expected):
    if Path(path).stat().st_size > 20_000_000:
        raise ValueError("Checkpoint exceeds smoke size cap")
    with np.load(path, allow_pickle=False) as data:
        meta = json.loads(str(data["metadata"]))
        if any(meta.get(k) != v for k, v in expected.items()):
            raise ValueError("Checkpoint contract mismatch")
        model = LinearModel()
        for name in ("actor", "critic"):
            value = data[name]
            if (
                value.shape != getattr(model, name).shape
                or value.dtype != np.float64
                or not np.isfinite(value).all()
            ):
                raise ValueError("Invalid checkpoint parameters")
            setattr(model, name, value.copy())
    if meta["optimizer"] != "sgd-no-momentum-v1" or not 0 < meta["learning_rate"] <= 0.1:
        raise ValueError("Invalid optimizer state")
    rng = np.random.default_rng()
    rng.bit_generator.state = meta["rng"]
    return model, meta, rng


def run(output, data_path, *, steps=4, seed=7, resume=None):
    if any(
        os.environ.get(k) != "1"
        for k in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS")
    ):
        raise ValueError("Set all three documented CPU thread caps to 1 before Python starts")
    if type(steps) is not int or not 1 <= steps <= 8:
        raise ValueError("Smoke accepts 1..8 optimizer steps")
    if type(seed) is not int or not 0 <= seed < 100_000:
        raise ValueError("Use a reserved training seed below 100000")
    start = time.monotonic()
    batch, data_hash = dataset(data_path)
    expected = contract(data_hash)
    if resume:
        model, meta, rng = load(resume, expected)
    else:
        model, rng = LinearModel(), np.random.default_rng(seed)
        meta = {
            **expected,
            "seed": seed,
            "step": 0,
            "optimizer": "sgd-no-momentum-v1",
            "learning_rate": 0.05,
            "batch_size": 2,
        }
    if meta["step"] + steps > 8:
        raise ValueError("Total optimizer step cap is 8")
    output = Path(output)
    output.mkdir(parents=True, exist_ok=False)
    save(output / f"checkpoint-{meta['step']:04}.npz", model, meta, rng)
    initial_probs, initial_values = model.forward(batch[0], batch[1])
    initial_loss = float(
        -np.log(initial_probs[np.arange(len(batch[0])), batch[2]]).mean()
        + 0.5 * np.square(initial_values - batch[3]).mean()
    )
    records = []
    for _ in range(steps):
        if time.monotonic() - start > 30:
            raise TimeoutError("30-second smoke budget exhausted; prior checkpoints retained")
        indices = rng.choice(
            len(batch[0]), size=min(meta["batch_size"], len(batch[0])), replace=False
        )
        metrics = model.update(*(array[indices] for array in batch), meta["learning_rate"])
        meta["step"] += 1
        path = output / f"checkpoint-{meta['step']:04}.npz"
        save(path, model, meta, rng)
        # Check loading at every step without consuming the training RNG.
        recovered, _, _ = load(path, expected)
        if not (
            np.array_equal(model.actor, recovered.actor)
            and np.array_equal(model.critic, recovered.critic)
        ):
            raise ValueError("Checkpoint round-trip differs")
        records.append({"step": meta["step"], "indices": indices.tolist(), **metrics})
    probs, values = model.forward(batch[0], batch[1])
    report = {
        "purpose": "supervised synthetic policy/value plumbing smoke; not RL strength",
        "contract": expected,
        "seed": meta["seed"],
        "optimizer": meta["optimizer"],
        "learning_rate": meta["learning_rate"],
        "steps_this_run": steps,
        "total_steps": meta["step"],
        "transitions": len(batch[0]),
        "terminals": len(batch[0]),
        "cutoffs": 0,
        "records": records,
        "elapsed_seconds": time.monotonic() - start,
        "checkpoint": path.name,
        "checkpoint_bytes": path.stat().st_size,
        "checkpoint_sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
        "initial_dataset_loss": initial_loss,
        "final_dataset_loss": float(
            -np.log(probs[np.arange(len(probs)), batch[2]]).mean()
            + 0.5 * np.square(values - batch[3]).mean()
        ),
        "illegal_probability_mass": float(probs[~batch[1]].sum()),
        "target_probabilities": probs[np.arange(len(probs)), batch[2]].tolist(),
        "values": values.tolist(),
        "validation_failures": [],
    }
    write_json(output / "report.json", report)
    return report


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--data", required=True, type=Path)
    parser.add_argument("--resume", type=Path)
    parser.add_argument("--steps", type=int, default=4)
    args = parser.parse_args()
    print(json.dumps(run(args.output, args.data, steps=args.steps, resume=args.resume), indent=2))
