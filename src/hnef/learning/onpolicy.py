"""Bounded on-policy TD actor-critic on synthetic Copenhagen starts."""

import argparse
import hashlib
import json
import os
import time
from collections import Counter
from pathlib import Path

import numpy as np

from hnef.engine import GameState, Player, apply_move
from hnef.opponents import opponent
from hnef.policies import RandomPolicy

from .artifacts import write_json
from .copenhagen import CODEC, RULESET, encode, stream_seed
from .plumbing import LinearModel, contract, load, save

ALGORITHM = "same-player-td0-ac-v1"
REFERENCES = ("random-v1", "heuristic-v1", "mcts-v1-i4-r4")
TRAINING_REFERENCES = (*REFERENCES[:2], "mixed-random-heuristic-v1")
CURRICULA = (
    "fixture-v1",
    "constrained32-broad32-v1",
    "broad64-v1",
    "constrained16-intermediate16-broad32-v1",
)


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
    if not 1 <= len(records) <= 24:
        raise ValueError("Expected 1..24 concrete curriculum positions")
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


def collect(model, initial, rng, opponent_seed, horizon, gamma, reference_id="random-v1"):
    learner = initial.player
    reference = (
        RandomPolicy(opponent_seed)
        if reference_id == "random-v1"
        else opponent(reference_id).create(RULESET, opponent_seed)
    )
    state, samples = initial, []
    trace = {
        "schema": 1,
        "ruleset": RULESET.value,
        "board": initial.board,
        "player": learner.value,
        "actions": [],
        "keys": [initial.key],
        "opponent_seed": opponent_seed,
        "opponent": reference_id,
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
            "actor_gradient_norm": float(np.linalg.norm(actor_grad)),
            "critic_gradient_norm": float(np.linalg.norm(critic_grad)),
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


def training_reference(identifier, step, positions):
    """Cross reference with every start/side; never alternate it with the mover."""
    if identifier not in TRAINING_REFERENCES:
        raise ValueError("Unknown bounded training reference")
    if type(step) is not int or step < 0 or type(positions) is not int or positions < 1:
        raise ValueError("Invalid curriculum schedule index")
    if identifier == "mixed-random-heuristic-v1":
        return REFERENCES[(step // positions) % 2]
    return identifier


def update_coverage(records):
    """Observed terminal update coverage, never intended fixture/witness labels."""
    return [
        {
            "side": side.value,
            "result": result,
            "episodes": len(games),
            "nonzero_actor_updates": sum(g["actor_gradient_norm"] > 0 for g in games),
            "nonzero_critic_updates": sum(g["critic_gradient_norm"] > 0 for g in games),
        }
        for side in Player
        for result, reward in (("win", 1), ("loss", -1))
        for games in [[r for r in records if r["learner"] == side.value and r["reward"] == reward]]
    ]


def curriculum_groups(data, positions, curriculum, both_sides):
    if curriculum not in CURRICULA:
        raise ValueError("Unknown curriculum contract")
    if curriculum == "fixture-v1":
        return {"fixture": list(range(len(positions)))}
    if both_sides:
        raise ValueError("Staged starts already contain both movers")
    cases = json.loads(Path(data).read_text())["cases"]
    stage_names = (
        ("constrained", "intermediate", "broad") if len(cases) == 24 else ("constrained", "broad")
    )
    if (
        len(cases) not in (16, 24)
        or len(positions) != len(cases)
        or (curriculum == "constrained16-intermediate16-broad32-v1" and len(cases) != 24)
        or any(
            c.get("split") != "training"
            or c.get("ruleset") != RULESET.value
            or c.get("stage") not in stage_names
            for c in cases
        )
    ):
        raise ValueError(
            "Staging requires 16 or 24 explicit training-only, versioned stage records"
        )
    groups = {
        stage: [i for i, c in enumerate(cases) if c["stage"] == stage] for stage in stage_names
    }
    for indices in groups.values():
        if len(indices) != 8 or Counter(positions[i].player for i in indices) != {
            Player.ATTACKERS: 4,
            Player.DEFENDERS: 4,
        }:
            raise ValueError("Each stage requires eight starts, four per learner side")
    return groups


def curriculum_choice(curriculum, groups, step):
    if curriculum not in CURRICULA or type(step) is not int or not 0 <= step < 64:
        raise ValueError("Invalid curriculum step or contract")
    if curriculum == "fixture-v1":
        stage = "fixture"
    elif curriculum == "broad64-v1":
        stage = "broad"
    elif curriculum == "constrained16-intermediate16-broad32-v1":
        stage = "constrained" if step < 16 else "intermediate" if step < 32 else "broad"
    else:
        stage = "constrained" if step < 32 else "broad"
    indices = groups[stage]
    return stage, indices[step % len(indices)], len(indices)


def train(
    output,
    data,
    *,
    episodes=32,
    horizon=4,
    seed=17,
    resume=None,
    reference_id="random-v1",
    both_sides=False,
    curriculum="fixture-v1",
):
    cpu_caps()
    if type(episodes) is not int or not 1 <= episodes <= 32:
        raise ValueError("Use 1..32 episodes")
    if type(horizon) is not int or not 1 <= horizon <= 8:
        raise ValueError("Use 1..8 learner decisions per episode")
    if type(seed) is not int or not 0 <= seed < 100000:
        raise ValueError("Training seed must be below 100000")
    if reference_id not in TRAINING_REFERENCES or type(both_sides) is not bool:
        raise ValueError("Training requires a bounded reference and boolean both_sides")
    start = time.monotonic()
    positions, digest = starts(data)
    if both_sides:
        # A fresh synthetic start, not a continuation with an edited mover/history.
        positions = [
            GameState(state.board, side, ruleset=RULESET) for state in positions for side in Player
        ]
    groups = curriculum_groups(data, positions, curriculum, both_sides)
    expected = {
        **contract(digest),
        "schema": 2,
        "algorithm": ALGORITHM,
        "horizon": horizon,
        "gamma": 0.99,
        "reference_id": reference_id,
        "start_schedule": "both-sides-v1" if both_sides else "fixture-mover-v1",
        "curriculum": curriculum,
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
        stage, index, stage_size = curriculum_choice(curriculum, groups, meta["step"])
        reference_seed = stream_seed(meta["seed"], f"td-reference-{meta['step']}")
        actual_reference = training_reference(reference_id, meta["step"], stage_size)
        samples, trace = collect(
            model, positions[index], rng, reference_seed, horizon, meta["gamma"], actual_reference
        )
        metrics = update(model, samples, meta["learning_rate"])
        meta["step"] += 1
        name = f"trace-{meta['step']:04}.json"
        write_json(output / name, trace)
        records.append(
            {
                "episode": meta["step"],
                "stage": stage,
                "opponent": actual_reference,
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
        "terminal_update_coverage": update_coverage(records),
        "reference_side_coverage": [
            {
                "opponent": ref,
                "side": side.value,
                "episodes": len(games),
                "start_indices": sorted({g["start_index"] for g in games}),
            }
            for ref in sorted({r["opponent"] for r in records})
            for side in Player
            for games in [
                [r for r in records if r["opponent"] == ref and r["learner"] == side.value]
            ]
        ],
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


def result_name(reward):
    return {None: "unfinished", 1: "win", -1: "loss", 0: "draw"}[reward]


def evaluation_summary(records):
    """No completed-game-only denominator or implicit draw for censored games."""
    summary, paired = [], []
    references = sorted({r["opponent"] for r in records})
    for reference_id in references:
        for side in Player:
            groups = {}
            for label in ("initialized", "candidate"):
                games = [
                    r
                    for r in records
                    if r["model"] == label
                    and r["side"] == side.value
                    and r["opponent"] == reference_id
                ]
                if not games:
                    continue
                counts = Counter(result_name(g["reward"]) for g in games)
                counts = {name: counts[name] for name in ("win", "loss", "draw", "unfinished")}
                groups[label] = {(r["start_index"], r["seed"]): r for r in games}
                if len(groups[label]) != len(games):
                    raise ValueError("Duplicate evaluation pair")
                n = len(games)
                summary.append(
                    {
                        "model": label,
                        "opponent": reference_id,
                        "side": side.value,
                        "games": n,
                        "results": counts,
                        "cutoff_rate": counts["unfinished"] / n,
                        "win_rate_bounds": [
                            counts["win"] / n,
                            (counts["win"] + counts["unfinished"]) / n,
                        ],
                        "terminal_reasons": dict(
                            Counter(g["reason"] for g in games if g["reason"])
                        ),
                    }
                )
            if not groups:
                continue
            if set(groups) != {"initialized", "candidate"} or (
                groups["initialized"].keys() != groups["candidate"].keys()
            ):
                raise ValueError("Unpaired evaluation records")
            low = high = 0
            transitions = Counter()
            for key, candidate in groups["candidate"].items():
                initial = groups["initialized"][key]
                c, i = candidate["reward"], initial["reward"]
                # Finite-sample identification bounds, not confidence intervals.
                low += int(c == 1) - int(i is None or i == 1)
                high += int(c is None or c == 1) - int(i == 1)
                transitions[f"{result_name(i)}->{result_name(c)}"] += 1
            n = len(groups["candidate"])
            paired.append(
                {
                    "opponent": reference_id,
                    "side": side.value,
                    "pairs": n,
                    "win_rate_delta_bounds": [low / n, high / n],
                    "outcome_transitions": dict(transitions),
                }
            )
    return summary, paired


def evaluate(
    checkpoint,
    training_data,
    heldout,
    output,
    *,
    seeds=(100000, 100001),
    references=("random-v1",),
    horizon=None,
):
    cpu_caps()
    if not seeds or len(seeds) > 4 or any(type(s) is not int or s < 100000 for s in seeds):
        raise ValueError("Use 1..4 reserved evaluation seeds >=100000")
    if len(set(seeds)) != len(seeds):
        raise ValueError("Evaluation seeds must be unique")
    if not references or any(r not in REFERENCES for r in references):
        raise ValueError("Use bounded registered evaluation opponents")
    if len(set(references)) != len(references):
        raise ValueError("Evaluation opponents must be unique")
    if horizon is not None and (type(horizon) is not int or not 1 <= horizon <= 32):
        raise ValueError("Evaluation horizon must be 1..32 learner decisions")
    start = time.monotonic()
    training, digest = starts(training_data)
    testing, heldout_hash = starts(heldout)
    forbidden = {key for state in training for key in symmetric_keys(state.board)}
    if any(state.board in forbidden for state in testing):
        raise ValueError("Training/evaluation symmetry overlap")
    # New training metadata binds the opponent and start schedule; earlier schema-2
    # artifacts remain source-bound. Evaluation does not change the training contract.
    expected = {**contract(digest), "schema": 2, "algorithm": ALGORITHM}
    candidate, meta, _ = load(checkpoint, expected)
    horizon = meta["horizon"] if horizon is None else horizon
    output = Path(output)
    output.mkdir(parents=True, exist_ok=False)
    records = []
    for reference_id in references:
        for label, model in (("initialized", LinearModel()), ("candidate", candidate)):
            for index, state in enumerate(testing):
                for seed in seeds:
                    if time.monotonic() - start > 60:
                        raise TimeoutError("60-second evaluation budget exhausted")
                    # Pair candidate/control within a reference; separate reference streams.
                    rng = np.random.default_rng(stream_seed(seed, f"eval-learner-{index}"))
                    ref_seed = stream_seed(seed, f"eval-{reference_id}-{index}")
                    episode_start = time.monotonic()
                    _, trace = collect(
                        model, state, rng, ref_seed, horizon, meta["gamma"], reference_id
                    )
                    elapsed = time.monotonic() - episode_start
                    name = f"{reference_id}-{label}-{index}-{seed}.json"
                    write_json(output / name, trace)
                    records.append(
                        {
                            "model": label,
                            "opponent": reference_id,
                            "start_index": index,
                            "seed": seed,
                            "learner_seed": stream_seed(seed, f"eval-learner-{index}"),
                            "opponent_seed": ref_seed,
                            "side": state.player.value,
                            "reward": trace["reward"],
                            "reason": trace["reason"],
                            "truncated": trace["truncated"],
                            "plies": len(trace["actions"]),
                            "trace": name,
                            "elapsed_seconds": elapsed,
                        }
                    )
    summary, paired = evaluation_summary(records)
    report = {
        "evaluation_schema": 2,
        "checkpoint_sha256": hashlib.sha256(Path(checkpoint).read_bytes()).hexdigest(),
        "contract": expected,
        "training_config": meta,
        "heldout_sha256": heldout_hash,
        "seeds": seeds,
        "opponents": references,
        "horizon": horizon,
        "gamma": meta["gamma"],
        "summary": summary,
        "paired": paired,
        "records": records,
        "elapsed_seconds": time.monotonic() - start,
        "limitation": "synthetic diagnostic; bounds express censoring, not confidence or strength",
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
    parser.add_argument(
        "--horizon", type=int, help="train default 4; evaluation default checkpoint horizon"
    )
    parser.add_argument("--reference", choices=TRAINING_REFERENCES, default="random-v1")
    parser.add_argument("--both-sides", action="store_true")
    parser.add_argument("--curriculum", choices=CURRICULA, default="fixture-v1")
    parser.add_argument("--opponents", choices=REFERENCES, nargs="+", default=["random-v1"])
    parser.add_argument("--resume", type=Path)
    parser.add_argument("--evaluation-seeds", type=int, nargs="+", default=[100000, 100001])
    parser.add_argument("--checkpoint", type=Path)
    parser.add_argument("--heldout", type=Path)
    args = parser.parse_args()
    if args.command == "train":
        result = train(
            args.output,
            args.data,
            episodes=args.episodes,
            horizon=4 if args.horizon is None else args.horizon,
            reference_id=args.reference,
            both_sides=args.both_sides,
            curriculum=args.curriculum,
            seed=args.seed,
            resume=args.resume,
        )
    else:
        if args.checkpoint is None or args.heldout is None:
            parser.error("evaluate requires --checkpoint and --heldout")
        result = evaluate(
            args.checkpoint,
            args.data,
            args.heldout,
            args.output,
            references=args.opponents,
            horizon=args.horizon,
            seeds=tuple(args.evaluation_seeds),
        )
    print(json.dumps({k: v for k, v in result.items() if k != "records"}, indent=2))
