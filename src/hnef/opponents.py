"""Versioned baseline registry; only runnable, rules-compatible opponents belong here."""

from dataclasses import dataclass

from .engine import Ruleset
from .policies import Policy, make_policy


@dataclass(frozen=True)
class OpponentSpec:
    identifier: str
    policy: str
    rulesets: tuple[Ruleset, ...]
    iterations: int = 64
    rollout_plies: int = 48

    def create(self, ruleset: Ruleset, seed: int) -> Policy:
        if Ruleset(ruleset) not in self.rulesets:
            raise ValueError(f"{self.identifier} does not support {ruleset}")
        return make_policy(
            self.policy, seed, iterations=self.iterations, rollout_plies=self.rollout_plies
        )


# IDs include budgets so different search costs cannot silently share a result label.
OPPONENTS = (
    OpponentSpec("random-v1", "random", tuple(Ruleset)),
    OpponentSpec("heuristic-v1", "heuristic", tuple(Ruleset)),
    OpponentSpec("mcts-v1-i4-r4", "mcts", tuple(Ruleset), 4, 4),
    OpponentSpec("mcts-v1-i64-r48", "mcts", tuple(Ruleset), 64, 48),
)


def opponent(identifier: str) -> OpponentSpec:
    for spec in OPPONENTS:
        if spec.identifier == identifier:
            return spec
    raise ValueError(f"Unknown opponent: {identifier}")
