"""Small masked actor/critic with an ordered, complete position-history input."""

from dataclasses import asdict, dataclass
from pathlib import Path

import torch
from torch import nn
from torch.nn.utils.rnn import pack_padded_sequence, pad_sequence

from hnef.actions import ActionCodec
from hnef.engine import GameState, Ruleset

CODEC = ActionCodec(5)
FEATURE_VERSION = "mini-board-hash-history-v1"


@dataclass(frozen=True)
class NetworkConfig:
    width: int = 64
    history_width: int = 16


def encode(states):
    """Absolute board, mover, elapsed plies, current repetitions, and all history.

    A GRU consumes all ordered SHA-256 position identities as 32 normalized bytes.
    These identities include the side to move. No history window is discarded.
    This is a learned summary, not a guarantee of learned repetition reasoning.
    """
    boards, histories, masks = [], [], []
    for state in states:
        if state.ruleset is not Ruleset.MINI:
            raise ValueError("This network supports mini-v1 only")
        board = [float(cell == piece) for piece in "ADK" for row in state.board for cell in row]
        board += [
            float(state.player.value == "defenders"),
            state.ply / (state.ply + 1),
            state.history.count(state.history[-1]) / 3,
        ]
        boards.append(torch.tensor(board))
        histories.append(
            torch.tensor([[byte / 127.5 - 1 for byte in bytes.fromhex(k)] for k in state.history])
        )
        masks.append(CODEC.mask(state))
    lengths = torch.tensor([len(history) for history in histories])
    return (
        torch.stack(boards),
        pad_sequence(histories, batch_first=True),
        lengths,
        torch.tensor(masks, dtype=torch.bool),
    )


class ActorCritic(nn.Module):
    def __init__(self, config=None):
        super().__init__()
        config = config or NetworkConfig()
        self.config = config
        self.board = nn.Sequential(nn.Linear(78, config.width), nn.Tanh())
        self.history = nn.GRU(32, config.history_width, batch_first=True)
        self.shared = nn.Sequential(
            nn.Linear(config.width + config.history_width, config.width), nn.Tanh()
        )
        self.actor = nn.Linear(config.width, len(CODEC.moves))
        self.critic = nn.Linear(config.width, 1)

    def forward(self, boards, histories, lengths, masks):
        if not masks.any(dim=1).all():
            raise ValueError("Cannot infer an action in a terminal or immobile position")
        packed = pack_padded_sequence(histories, lengths, batch_first=True, enforce_sorted=False)
        _, hidden = self.history(packed)
        features = self.shared(torch.cat((self.board(boards), hidden[-1]), dim=1))
        logits = self.actor(features).masked_fill(~masks, float("-inf"))
        return logits, self.critic(features).squeeze(-1).tanh()


class NeuralPolicy:
    def __init__(self, model, seed=0, *, greedy=False):
        self.model = model
        self.generator = torch.Generator().manual_seed(seed)
        self.greedy = greedy

    @torch.inference_mode()
    def choose_move(self, state: GameState):
        logits, _ = self.model(*encode([state]))
        action = (
            logits[0].argmax().item()
            if self.greedy
            else torch.multinomial(logits[0].softmax(-1), 1, generator=self.generator).item()
        )
        return CODEC.decode(action)


def checkpoint_model(path: Path):
    data = torch.load(path, map_location="cpu", weights_only=True)
    if (
        data.get("schema") != 1
        or data.get("features") != FEATURE_VERSION
        or data.get("ruleset") != Ruleset.MINI.value
    ):
        raise ValueError("Unsupported checkpoint format, features, or ruleset")
    model = ActorCritic(NetworkConfig(**data["network"]))
    model.load_state_dict(data["model"])
    if not all(torch.isfinite(tensor).all() for tensor in model.state_dict().values()):
        raise ValueError("Checkpoint contains nonfinite model parameters")
    model.eval()
    return model, data


def network_metadata(model):
    return {
        "network": asdict(model.config),
        "features": FEATURE_VERSION,
        "ruleset": Ruleset.MINI.value,
    }
