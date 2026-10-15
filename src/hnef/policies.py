"""Dependency-free reference opponents with explicit reward perspectives."""

import math
from dataclasses import dataclass, field
from random import Random
from typing import Protocol

from .engine import GameState, Move, Player, apply_move, legal_moves

POLICY_NAMES = ("random", "heuristic", "mcts")


class Policy(Protocol):
    def choose_move(self, state: GameState) -> Move: ...


def _moves(state: GameState) -> tuple[Move, ...]:
    moves = legal_moves(state)
    if not moves:
        raise ValueError("Cannot choose a move in a terminal or immobile position")
    return moves


def _best(items, score, rng):
    scored = [(item, score(item)) for item in items]
    maximum = max(value for _, value in scored)
    return rng.choice([item for item, value in scored if value == maximum])


class RandomPolicy:
    def __init__(self, seed: int = 0):
        self.rng = Random(seed)

    def choose_move(self, state: GameState) -> Move:
        return self.rng.choice(_moves(state))


def evaluate(state: GameState, player: Player) -> float:
    """Handwritten v1 score, not a calibrated win probability; terminal values ±1."""
    player = Player(player)
    if state.outcome is not None:
        return float(state.outcome.reward(player))
    size = state.ruleset.size
    board = "".join(state.board)
    king_index = board.index("K")
    row, column = divmod(king_index, size)
    if state.ruleset.is_copenhagen:
        distance = min(abs(row - r) + abs(column - c) for r, c in state.ruleset.corners)
        progress = 1 - distance / (size - 1)
        initial_attackers, initial_defenders = 24, 12
    else:
        distance = min(row, column, size - 1 - row, size - 1 - column)
        progress = 1 - distance / (size // 2)
        initial_attackers, initial_defenders = 8, 4
    material = board.count("D") / initial_defenders - board.count("A") / initial_attackers
    escape_lanes = 0
    adjacent_attackers = 0
    for dr, dc in ((-1, 0), (0, -1), (0, 1), (1, 0)):
        r, c = row + dr, column + dc
        if 0 <= r < size and 0 <= c < size and state.board[r][c] == "A":
            adjacent_attackers += 1
        while 0 <= r < size and 0 <= c < size and state.board[r][c] == ".":
            escape = (
                (r, c) in state.ruleset.corners
                if state.ruleset.is_copenhagen
                else r in (0, size - 1) or c in (0, size - 1)
            )
            if escape:
                escape_lanes += 1
                break
            r, c = r + dr, c + dc
    score = 0.35 * material + 0.25 * progress + 0.3 * escape_lanes / 4
    score -= 0.1 * adjacent_attackers / 4
    score = max(-0.9, min(0.9, score))
    return score if player is Player.DEFENDERS else -score


class HeuristicPolicy:
    def __init__(self, seed: int = 0):
        self.rng = Random(seed)

    def choose_move(self, state: GameState) -> Move:
        return _best(
            _moves(state), lambda move: evaluate(apply_move(state, move), state.player), self.rng
        )


@dataclass(frozen=True)
class MoveStats:
    move: Move
    visits: int
    mean_value: float  # Always from the root player's perspective.


@dataclass(frozen=True)
class SearchResult:
    move: Move
    iterations: int
    rollout_cutoffs: int
    children: tuple[MoveStats, ...]


@dataclass
class _Node:
    state: GameState
    visits: int = 0
    value_sum: float = 0.0
    children: dict[Move, "_Node"] = field(default_factory=dict)
    untried: list[Move] = field(init=False)

    def __post_init__(self):
        self.untried = list(legal_moves(self.state))


class MCTSPolicy:
    """UCT with random rollouts and a fresh, history-preserving tree per decision."""

    def __init__(
        self,
        seed: int = 0,
        *,
        iterations: int = 64,
        rollout_plies: int = 48,
        exploration: float = math.sqrt(2),
    ):
        if type(iterations) is not int or iterations < 1:
            raise ValueError("iterations must be a positive integer")
        if type(rollout_plies) is not int or rollout_plies < 0:
            raise ValueError("rollout_plies must be a nonnegative integer")
        if (
            isinstance(exploration, bool)
            or not isinstance(exploration, (int, float))
            or not math.isfinite(exploration)
            or exploration < 0
        ):
            raise ValueError("exploration must be a finite nonnegative number")
        self.iterations = iterations
        self.rollout_plies = rollout_plies
        self.exploration = exploration
        self.rng = Random(seed)

    def choose_move(self, state: GameState) -> Move:
        return self.search(state).move

    def _select(self, node: _Node, root_player: Player) -> _Node:
        # Opponent nodes minimize root utility; exploration stays positive.
        sign = 1 if node.state.player is root_player else -1
        return _best(
            node.children.values(),
            lambda child: (
                sign * child.value_sum / child.visits
                + self.exploration * math.sqrt(math.log(node.visits) / child.visits)
            ),
            self.rng,
        )

    def _rollout(self, state: GameState, root_player: Player) -> tuple[float, bool]:
        for _ in range(self.rollout_plies):
            if state.outcome is not None:
                break
            state = apply_move(state, self.rng.choice(legal_moves(state)))
        if state.outcome is not None:
            return float(state.outcome.reward(root_player)), False
        # A neutral search estimate does not change the engine outcome to a draw.
        return 0.0, True

    def search(self, state: GameState) -> SearchResult:
        _moves(state)
        root = _Node(state)
        cutoffs = 0
        for _ in range(self.iterations):
            node = root
            path = [node]
            while node.state.outcome is None and not node.untried:
                node = self._select(node, state.player)
                path.append(node)
            if node.untried:
                move = node.untried.pop(self.rng.randrange(len(node.untried)))
                child = _Node(apply_move(node.state, move))
                node.children[move] = child
                node = child
                path.append(node)
            value, cutoff = self._rollout(node.state, state.player)
            cutoffs += cutoff
            for ancestor in path:
                ancestor.visits += 1
                ancestor.value_sum += value
        winning = [
            move
            for move, child in root.children.items()
            if child.state.outcome is not None and child.state.outcome.winner is state.player
        ]
        # Prefer a discovered immediate proof; otherwise use the robust child
        # (most visits), breaking ties by mean root value and then seeded choice.
        move = _best(
            winning or list(root.children),
            lambda candidate: (
                root.children[candidate].visits,
                root.children[candidate].value_sum / root.children[candidate].visits,
            ),
            self.rng,
        )
        return SearchResult(
            move,
            root.visits,
            cutoffs,
            tuple(
                MoveStats(move, child.visits, child.value_sum / child.visits)
                for move, child in root.children.items()
            ),
        )


def make_policy(name: str, seed: int = 0, *, iterations=64, rollout_plies=48) -> Policy:
    if name == "random":
        return RandomPolicy(seed)
    if name == "heuristic":
        return HeuristicPolicy(seed)
    if name == "mcts":
        return MCTSPolicy(seed, iterations=iterations, rollout_plies=rollout_plies)
    raise ValueError(f"Unknown policy: {name}")
