"""Bounded on-policy TD actor-critic on synthetic Copenhagen starts, versus random."""

import argparse
import hashlib
import json
import os
import time
from collections import Counter
from pathlib import Path

import numpy as np

from hnef.engine import GameState, Player, apply_move
from hnef.policies import RandomPolicy

from .artifacts import write_json
from .copenhagen import CODEC, RULESET, encode, stream_seed
from .plumbing import LinearModel, contract, load, save

ALGORITHM = "same-player-td0-ac-v1"


def features(state):
    obs = encode(state)
    x = np.array(
        [v for plane in obs["planes"] for v in plane]
        + [
            float(state.player is Player.DEFENDERS),
            state.ply / (state.ply + 1),
            obs["repetitions"] / 3,
            1.0,
        ],
        dtype=np.float64,
    )
    return x, np.array(obs["legal_mask"], dtype=bool)


def target(outcome, learner, next_value, gamma):
    # One discount per learner decision, which includes the fixed opponent reply.
    return float(outcome.reward(learner)) if outcome is not None else gamma * next_value


def starts(path):
    raw = Path(path).read_bytes()
    records = [c for c in json.loads(raw)["cases"] if c.get("board") is not None]
    if not 1 <= len(records) <= 16:
        raise ValueError("Expected 1..16 concrete curriculum positions")
    # Deliberately do not read move/expected-outcome/teacher labels.
    states = [GameState(tuple(c["board"]), Player(c["player"]), ruleset=RULESET) for c in records]
    return states, hashlib.sha256(raw).hexdigest()


def trace_state(trace):
    if trace.get("schema") != 1 or trace.get("ruleset") != RULESET.value:
        raise ValueError("Unsupported curriculum trace contract")
    state = GameState(tuple(trace["board"]), Player(trace["player"]), ruleset=RULESET)
    if state.key != trace["keys"][0]:
        raise ValueError("Invalid initial trace identity")
    if len(trace["keys"]) != len(trace["actions"]) + 1:
        raise ValueError("Invalid trace length")
    for action, key in zip(trace["actions"], trace["keys"][1:], strict=True):
        state = apply_move(state, CODEC.decode(action))
        if state.key != key:
            raise ValueError("Trace transition differs")
    if (
        trace["terminated"] != (state.outcome is not None)
        or trace["truncated"] != (state.outcome is None)
        or trace["reason"] != (state.outcome.reason if state.outcome else None)
        or trace["reward"]
        != (state.outcome.reward(Player(trace["player"])) if state.outcome else None)
    ):
        raise ValueError("Trace result differs from engine adjudication")
    return state


def collect(model, initial, rng, opponent_seed, horizon, gamma):
    learner = initial.player
    reference = RandomPolicy(opponent_seed)
    state, samples = initial, []
    trace = {
        "schema": 1,
        "ruleset": RULESET.value,
        "board": initial.board,
        "player": learner.value,
        "actions": [],
        "keys": [initial.key],
        "opponent_seed": opponent_seed,
    }
    for _ in range(horizon):
        x, mask = features(state)
        probs, value = model.forward(x[None], mask[None])
        action = int(rng.choice(len(probs[0]), p=probs[0]))
        state = apply_move(state, CODEC.decode(action))
        trace["actions"].append(action)
        trace["keys"].append(state.key)
        if state.outcome is None:
            reply = reference.choose_move(state)
            state = apply_move(state, reply)
            trace["actions"].append(CODEC.encode(reply))
            trace["keys"].append(state.key)
        next_value = 0.0
        if state.outcome is None:
            assert state.player is learner
            nx, nm = features(state)
            _, nv = model.forward(nx[None], nm[None])
            next_value = float(nv[0])
        y = target(state.outcome, learner, next_value, gamma)
        samples.append((x, mask, action, float(value[0]), y))
        if state.outcome is not None:
            break
    trace.update(
        terminated=state.outcome is not None,
        truncated=state.outcome is None,
        reason=state.outcome.reason if state.outcome else None,
        reward=state.outcome.reward(learner) if state.outcome else None,
        bootstrap_value=next_value,
        final_target=y,
        learner_decisions=len(samples),
    )
    assert trace_state(trace) == state
    return samples, trace


def gradients(model, samples):
    x, mask, actions, old_values, targets = (np.array(a) for a in zip(*samples, strict=True))
    probs, values = model.forward(x, mask)
    if not np.allclose(values, old_values, rtol=1e-12, atol=1e-12):
        raise ValueError("Model changed since on-policy collection")
    advantages = targets - old_values  # Detached target and baseline.
    delta = probs.copy()
    delta[np.arange(len(x)), actions] -= 1
    actor_grad = x.T @ (delta * advantages[:, None]) / len(x)
    critic_grad = x.T @ ((values - targets) * (1 - values**2)) / len(x)
    loss = -(np.log(probs[np.arange(len(x)), actions]) * advantages).mean()
    loss += 0.5 * np.square(values - targets).mean()
    norm = float(np.sqrt(np.square(actor_grad).sum() + np.square(critic_grad).sum()))
    if not np.isfinite(loss) or not np.isfinite(norm):
        raise ValueError("Nonfinite loss/gradient")
    return (
        actor_grad,
        critic_grad,
        {
            "loss": float(loss),
            "gradient_norm": norm,
            "mean_abs_advantage": float(np.abs(advantages).mean()),
        },
    )


def update(model, samples, lr):
    actor, critic, metrics = gradients(model, samples)
    scale = min(1.0, 1 / max(metrics["gradient_norm"], 1e-12))
    model.actor -= lr * scale * actor
    model.critic -= lr * scale * critic
    if not np.isfinite(model.actor).all() or not np.isfinite(model.critic).all():
        raise ValueError("Nonfinite update")
    return metrics


def cpu_caps():
    if any(
        os.environ.get(k) != "1"
        for k in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS")
    ):
        raise ValueError("Set BLAS/OMP/MKL CPU thread caps to 1 before launch")


def train(output, data, *, episodes=32, horizon=4, seed=17, resume=None):
    cpu_caps()
    if type(episodes) is not int or not 1 <= episodes <= 32:
        raise ValueError("Use 1..32 episodes")
    if type(horizon) is not int or not 1 <= horizon <= 8:
        raise ValueError("Use 1..8 learner decisions per episode")
    if type(seed) is not int or not 0 <= seed < 100000:
        raise ValueError("Training seed must be below 100000")
    start = time.monotonic()
    positions, digest = starts(data)
    expected = {
        **contract(digest),
        "schema": 2,
        "algorithm": ALGORITHM,
        "horizon": horizon,
        "gamma": 0.99,
    }
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
            "batch_size": 1,
        }
    if meta["step"] + episodes > 64:
        raise ValueError("Total episode/update cap is 64")
    output = Path(output)
    output.mkdir(parents=True, exist_ok=False)
    save(output / f"checkpoint-{meta['step']:04}.npz", model, meta, rng)
    records = []
    for _ in range(episodes):
        if time.monotonic() - start > 60:
            raise TimeoutError("60-second training budget exhausted")
        index = meta["step"] % len(positions)
        reference_seed = stream_seed(meta["seed"], f"td-reference-{meta['step']}")
        samples, trace = collect(
            model, positions[index], rng, reference_seed, horizon, meta["gamma"]
        )
        metrics = update(model, samples, meta["learning_rate"])
        meta["step"] += 1
        name = f"trace-{meta['step']:04}.json"
        write_json(output / name, trace)
        records.append(
            {
                "episode": meta["step"],
                "start_index": index,
                "trace": name,
                "learner": positions[index].player.value,
                "plies": len(trace["actions"]),
                "decisions": len(samples),
                "terminated": trace["terminated"],
                "truncated": trace["truncated"],
                "reason": trace["reason"],
                "reward": trace["reward"],
                **metrics,
            }
        )
        path = output / f"checkpoint-{meta['step']:04}.npz"
        save(path, model, meta, rng)
    report = {
        "purpose": "on-policy synthetic-start TD actor-critic pilot; no strength claim",
        "contract": expected,
        "seed": meta["seed"],
        "records": records,
        "elapsed_seconds": time.monotonic() - start,
        "checkpoint": path.name,
        "checkpoint_bytes": path.stat().st_size,
        "checkpoint_sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
        "episodes": episodes,
        "total_updates": meta["step"],
        "terminals": sum(r["terminated"] for r in records),
        "cutoffs": sum(r["truncated"] for r in records),
        "transitions": sum(r["plies"] for r in records),
        "nonzero_updates": sum(r["gradient_norm"] > 0 for r in records),
    }
    write_json(output / "report.json", report)
    return report


def symmetric_keys(board):
    array = np.array([list(row) for row in board])
    return {
        tuple("".join(row) for row in transform)
        for k in range(4)
        for transform in (np.rot90(array, k), np.fliplr(np.rot90(array, k)))
    }


def evaluate(checkpoint, training_data, heldout, output, *, seeds=(100000, 100001)):
    cpu_caps()
    if not seeds or len(seeds) > 4 or any(type(s) is not int or s < 100000 for s in seeds):
        raise ValueError("Use 1..4 reserved evaluation seeds >=100000")
    if len(set(seeds)) != len(seeds):
        raise ValueError("Evaluation seeds must be unique")
    start = time.monotonic()
    training, digest = starts(training_data)
    testing, heldout_hash = starts(heldout)
    forbidden = {key for state in training for key in symmetric_keys(state.board)}
    if any(state.board in forbidden for state in testing):
        raise ValueError("Training/evaluation symmetry overlap")
    expected = {**contract(digest), "schema": 2, "algorithm": ALGORITHM}
    candidate, meta, _ = load(checkpoint, expected)
    output = Path(output)
    output.mkdir(parents=True, exist_ok=False)
    records = []
    for label, model in (("initialized", LinearModel()), ("candidate", candidate)):
        for index, state in enumerate(testing):
            for seed in seeds:
                if time.monotonic() - start > 60:
                    raise TimeoutError("60-second evaluation budget exhausted")
                # Same learner and reference streams for initialized/candidate control.
                rng = np.random.default_rng(stream_seed(seed, f"eval-learner-{index}"))
                ref_seed = stream_seed(seed, f"eval-reference-{index}")
                _, trace = collect(model, state, rng, ref_seed, meta["horizon"], meta["gamma"])
                name = f"{label}-{index}-{seed}.json"
                write_json(output / name, trace)
                records.append(
                    {
                        "model": label,
                        "start_index": index,
                        "seed": seed,
                        "side": state.player.value,
                        "reward": trace["reward"],
                        "reason": trace["reason"],
                        "truncated": trace["truncated"],
                        "plies": len(trace["actions"]),
                        "trace": name,
                    }
                )
    summary = []
    for label in ("initialized", "candidate"):
        for side in Player:
            games = [r for r in records if r["model"] == label and r["side"] == side.value]
            counts = Counter(
                "unfinished"
                if g["reward"] is None
                else "win"
                if g["reward"] == 1
                else "loss"
                if g["reward"] == -1
                else "draw"
                for g in games
            )
            summary.append(
                {
                    "model": label,
                    "side": side.value,
                    "games": len(games),
                    "results": dict(counts),
                    "terminal_reasons": dict(Counter(g["reason"] for g in games if g["reason"])),
                }
            )
    report = {
        "checkpoint_sha256": hashlib.sha256(Path(checkpoint).read_bytes()).hexdigest(),
        "contract": expected,
        "heldout_sha256": heldout_hash,
        "seeds": seeds,
        "horizon": meta["horizon"],
        "gamma": meta["gamma"],
        "summary": summary,
        "records": records,
        "elapsed_seconds": time.monotonic() - start,
        "limitation": "tiny synthetic-start evaluation, not full-game strength",
    }
    write_json(output / "report.json", report)
    return report


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("train", "evaluate"))
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--episodes", type=int, default=32)
    parser.add_argument("--seed", type=int, default=17)
    parser.add_argument("--horizon", type=int, default=4)
    parser.add_argument("--resume", type=Path)
    parser.add_argument("--checkpoint", type=Path)
    parser.add_argument("--heldout", type=Path)
    args = parser.parse_args()
    if args.command == "train":
        result = train(
            args.output,
            args.data,
            episodes=args.episodes,
            horizon=args.horizon,
            seed=args.seed,
            resume=args.resume,
        )
    else:
        if args.checkpoint is None or args.heldout is None:
            parser.error("evaluate requires --checkpoint and --heldout")
        result = evaluate(args.checkpoint, args.data, args.heldout, args.output)
    print(json.dumps({k: v for k, v in result.items() if k != "records"}, indent=2))
