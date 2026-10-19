"""On-policy episodic actor/critic baseline, alternating the learning player's side."""

import json
import math
from collections import Counter
from dataclasses import asdict, dataclass
from pathlib import Path
from random import Random

import torch
from torch.distributions import Categorical

from hnef.engine import Player, apply_move, initial_state
from hnef.policies import make_policy
from hnef.replay import Replay

from .artifacts import now, provenance, write_json
from .network import CODEC, ActorCritic, NeuralPolicy, checkpoint_model, encode, network_metadata


@dataclass(frozen=True)
class TrainConfig:
    seed: int = 42
    episodes_per_update: int = 16
    max_plies: int = 128
    learning_rate: float = 0.001
    entropy_coefficient: float = 0.02
    value_coefficient: float = 0.5
    opponents: tuple[str, ...] = ("random",)

    def __post_init__(self):
        object.__setattr__(self, "opponents", tuple(self.opponents))
        if type(self.seed) is not int or self.seed < 0:
            raise ValueError("seed must be a nonnegative integer")
        for value in (self.episodes_per_update, self.max_plies):
            if type(value) is not int or value < 1:
                raise ValueError("Episode count and ply limit must be positive integers")
        if not math.isfinite(self.learning_rate) or self.learning_rate <= 0:
            raise ValueError("Learning rate must be finite and positive")
        for value in (self.entropy_coefficient, self.value_coefficient):
            if not math.isfinite(value) or value < 0:
                raise ValueError("Loss coefficients must be finite and nonnegative")
        if not self.opponents or any(
            x not in ("random", "heuristic", "self") for x in self.opponents
        ):
            raise ValueError("Opponents must be random, heuristic, or self")


def terminal_targets(states, outcome):
    """Undiscounted Monte Carlo returns, from each recorded mover's perspective."""
    if outcome is None:
        return []  # Discard unknown cutoff returns, never invent draw labels.
    return [float(outcome.reward(state.player)) for state in states]


def collect(model, generator, config, episode):
    learner = Player.ATTACKERS if episode % 2 == 0 else Player.DEFENDERS
    rng = Random(config.seed * 1_000_003 + episode)
    opponent_name = config.opponents[rng.randrange(len(config.opponents))]
    policy = NeuralPolicy(model)
    policy.generator = generator
    opponent = (
        policy if opponent_name == "self" else make_policy(opponent_name, rng.getrandbits(63))
    )
    state = initial_state()
    states, actions, all_actions, keys = [], [], [], [state.key]
    while state.outcome is None and state.ply < config.max_plies:
        own_turn = state.player == learner
        move = (policy if own_turn else opponent).choose_move(state)
        action = CODEC.encode(move)
        if own_turn:
            states.append(state)
            actions.append(action)
        state = apply_move(state, move)
        all_actions.append(action)
        keys.append(state.key)
    replay = Replay(tuple(all_actions), tuple(keys))
    replay.verify()
    returns = terminal_targets(states, state.outcome)
    result = (
        "truncated"
        if state.outcome is None
        else (
            "draw"
            if state.outcome.winner is None
            else ("win" if state.outcome.winner == learner else "loss")
        )
    )
    return states if returns else [], actions if returns else [], returns, result, state.ply, replay


def update_model(model, optimizer, states, actions, targets, config):
    logits, values = model(*encode(states))
    distribution = Categorical(logits=logits)
    returns = torch.tensor(targets, dtype=torch.float32)
    advantages = returns - values.detach()
    policy_loss = -(distribution.log_prob(torch.tensor(actions)) * advantages).mean()
    value_loss = (values - returns).square().mean()
    entropy = distribution.entropy().mean()
    loss = (
        policy_loss + config.value_coefficient * value_loss - config.entropy_coefficient * entropy
    )
    if not torch.isfinite(loss):
        raise ValueError("Nonfinite training loss")
    optimizer.zero_grad()
    loss.backward()
    norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0, error_if_nonfinite=True)
    optimizer.step()
    return {
        "loss": loss.item(),
        "policy_loss": policy_loss.item(),
        "value_loss": value_loss.item(),
        "entropy": entropy.item(),
        "gradient_norm": norm.item(),
    }


def train(run: Path, *, updates=100, config=None, resume=False):
    if type(updates) is not int or updates < 1:
        raise ValueError("updates must be a positive integer")
    torch.set_num_threads(1)
    torch.use_deterministic_algorithms(True)
    run = Path(run)
    if resume:
        pointer = json.loads((run / "latest.json").read_text())
        model, saved = checkpoint_model(run / pointer["checkpoint"])
        config = TrainConfig(**saved["config"])
        optimizer = torch.optim.Adam(model.parameters(), lr=config.learning_rate)
        optimizer.load_state_dict(saved["optimizer"])
        generator = torch.Generator()
        generator.set_state(saved["sampling_rng"])
        torch.set_rng_state(saved["torch_rng"])
        start, source = saved["update"], saved["source"]
        if source["source_sha256"] != provenance()["source_sha256"]:
            raise ValueError(
                "Source changed; start a new run rather than silently changing a resume"
            )
    else:
        config = config or TrainConfig()
        run.mkdir(parents=True, exist_ok=False)
        torch.manual_seed(config.seed)
        model = ActorCritic()
        optimizer = torch.optim.Adam(model.parameters(), lr=config.learning_rate)
        generator = torch.Generator().manual_seed(config.seed + 1)
        start, source = 0, provenance()
        write_json(
            run / "run.json",
            {
                "schema": 1,
                "created_at": now(),
                "algorithm": "episodic-ac-v1",
                "config": asdict(config),
                **network_metadata(model),
                "source": source,
                "torch_version": str(torch.__version__),
            },
        )

    def save(update):
        name = f"checkpoint-{update:06d}.pt"
        path = run / name
        if path.exists():
            raise ValueError(f"Refusing to overwrite checkpoint {path}")
        temporary = run / f".{name}.tmp"
        torch.save(
            {
                "schema": 1,
                "model": model.state_dict(),
                "optimizer": optimizer.state_dict(),
                "sampling_rng": generator.get_state(),
                "torch_rng": torch.get_rng_state(),
                "config": asdict(config),
                "update": update,
                "source": source,
                **network_metadata(model),
            },
            temporary,
        )
        temporary.replace(path)
        write_json(run / "latest.json", {"checkpoint": name, "update": update})

    if not resume:
        save(0)
    for update in range(start, start + updates):
        states, actions, targets = [], [], []
        counts = Counter()
        plies = 0
        for index in range(config.episodes_per_update):
            episode = update * config.episodes_per_update + index
            batch_states, batch_actions, batch_targets, result, length, replay = collect(
                model, generator, config, episode
            )
            states.extend(batch_states)
            actions.extend(batch_actions)
            targets.extend(batch_targets)
            counts[result] += 1
            plies += length
            if index == 0:
                (run / "sample-replay.json").write_text(replay.to_json())
        metrics = update_model(model, optimizer, states, actions, targets, config) if states else {}
        record = {
            "update": update + 1,
            "episodes": (update + 1) * config.episodes_per_update,
            "results": dict(counts),
            "plies": plies,
            "training_positions": len(states),
            "skipped_update": not bool(states),
            **metrics,
        }
        # One immutable metrics file per update; a crash cannot duplicate JSONL entries.
        write_json(run / "metrics" / f"{update + 1:06d}.json", record)
        save(update + 1)
        if (update + 1) % 10 == 0 or update == start + updates - 1:
            print(json.dumps(record), flush=True)
    return run / f"checkpoint-{start + updates:06d}.pt"
