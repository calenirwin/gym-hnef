"""Turn-based AEC adapter; see docs/rl-environment.md for observation semantics."""

import numpy as np
from gymnasium import spaces
from gymnasium.utils import seeding
from pettingzoo import AECEnv
from pettingzoo.utils import wrappers

from hnef.actions import ActionCodec
from hnef.engine import GameState, Player, Ruleset, apply_move, initial_state


def env(**kwargs):
    """Create an AEC environment with action-bound and call-order checks."""
    return wrappers.OrderEnforcingWrapper(wrappers.AssertOutOfBoundsWrapper(raw_env(**kwargs)))


class raw_env(AECEnv):
    metadata = {
        "name": "hnef_v0",
        "render_modes": ["ansi", "human"],
        "is_parallelizable": False,
    }

    def __init__(self, *, ruleset=Ruleset.MINI, max_plies=500, render_mode=None):
        super().__init__()
        self.ruleset = Ruleset(ruleset)
        if type(max_plies) is not int or max_plies < 1:
            raise ValueError("max_plies must be a positive integer")
        if render_mode not in (None, "ansi", "human"):
            raise ValueError("render_mode must be None, 'ansi', or 'human'")
        self.max_plies = max_plies
        self.render_mode = render_mode
        self.possible_agents = [player.value for player in Player]
        self.codec = ActionCodec(self.ruleset.size)
        self._game_state = None
        self._action_spaces = {
            agent: spaces.Discrete(len(self.codec.moves)) for agent in self.possible_agents
        }
        self._observation_spaces = {
            agent: spaces.Dict(
                {
                    "observation": spaces.Box(
                        0, 1, (7, self.ruleset.size, self.ruleset.size), np.int8
                    ),
                    "action_mask": spaces.MultiBinary(len(self.codec.moves)),
                    "action_history": spaces.Box(
                        -1, len(self.codec.moves) - 1, (max_plies,), np.int32
                    ),
                }
            )
            for agent in self.possible_agents
        }

    @property
    def game_state(self) -> GameState:
        """Immutable engine state, including repetition history, for search/debugging."""
        if self._game_state is None:
            raise RuntimeError("Call reset() before accessing the game")
        return self._game_state

    def action_space(self, agent):
        return self._action_spaces[agent]

    def observation_space(self, agent):
        return self._observation_spaces[agent]

    def reset(self, seed=None, options=None):
        # Openings and transitions are deterministic. Seed sampling independently
        # for each agent, without touching NumPy's process-global RNG.
        if seed is not None or not hasattr(self, "np_random"):
            self.np_random, self.np_random_seed = seeding.np_random(seed)
            for agent in self.possible_agents:
                self.action_space(agent).seed(int(self.np_random.integers(2**32)))
                self.observation_space(agent).seed(int(self.np_random.integers(2**32)))
        self._game_state = initial_state(self.ruleset)
        self._action_history = np.full(self.max_plies, -1, dtype=np.int32)
        self.agents = self.possible_agents[:]
        self.agent_selection = self.game_state.player.value
        self._skip_agent_selection = None
        self.rewards = dict.fromkeys(self.agents, 0.0)
        self._cumulative_rewards = dict.fromkeys(self.agents, 0.0)
        self.terminations = dict.fromkeys(self.agents, False)
        self.truncations = dict.fromkeys(self.agents, False)
        self._update_infos()
        if self.render_mode == "human":
            self.render()

    def _update_infos(self):
        state = self.game_state
        outcome = state.outcome
        self.infos = {
            agent: {
                "ruleset": self.ruleset.value,
                "experimental": self.ruleset.is_copenhagen,
                "ply": state.ply,
                "state_key": state.key,
                "winner": outcome.winner.value if outcome and outcome.winner else None,
                "reason": outcome.reason if outcome else None,
                "time_limit_reached": self.truncations[agent],
            }
            for agent in self.agents
        }

    def observe(self, agent):
        if agent not in self.possible_agents:
            raise KeyError(agent)
        state = self.game_state
        board = np.array([list(row) for row in state.board])
        planes = np.zeros((7, self.ruleset.size, self.ruleset.size), dtype=np.int8)
        for index, piece in enumerate("ADK"):
            planes[index] = board == piece
        planes[3, *self.ruleset.throne] = 1
        if self.ruleset.is_copenhagen:
            for corner in self.ruleset.corners:
                planes[4, *corner] = 1
        planes[5] = state.player == Player.DEFENDERS
        planes[6] = agent == Player.DEFENDERS.value
        mask = np.zeros(len(self.codec.moves), dtype=np.int8)
        if (
            agent in self.agents
            and agent == self.agent_selection
            and not self.terminations[agent]
            and not self.truncations[agent]
        ):
            mask[:] = self.codec.mask(state)
        return {
            "observation": planes,
            "action_mask": mask,
            "action_history": self._action_history.copy(),
        }

    def step(self, action):
        state = self.game_state
        if not self.agents:
            raise RuntimeError("Episode is over; call reset()")
        agent = self.agent_selection
        if self.terminations[agent] or self.truncations[agent]:
            self._was_dead_step(action)
            return
        if isinstance(action, (bool, np.bool_)) or not isinstance(action, (int, np.integer)):
            raise ValueError("Action must be an integer ID")
        action = int(action)
        # Validate before changing bookkeeping, so an illegal action is atomic.
        successor = apply_move(state, self.codec.decode(action))
        self._cumulative_rewards[agent] = 0.0
        self._clear_rewards()
        self._action_history[state.ply] = action
        self._game_state = successor
        if successor.outcome is not None:
            self.terminations = dict.fromkeys(self.agents, True)
            self.rewards = {
                player: float(successor.outcome.reward(Player(player))) for player in self.agents
            }
        elif successor.ply >= self.max_plies:
            self.truncations = dict.fromkeys(self.agents, True)
        self.agent_selection = successor.player.value
        self._update_infos()
        self._accumulate_rewards()
        self._deads_step_first()
        if self.render_mode == "human":
            self.render()

    def render(self):
        if self.render_mode is None:
            return None
        state = self.game_state
        status = (
            f"{state.outcome.winner or 'draw'}: {state.outcome.reason}"
            if state.outcome
            else ("truncated" if state.ply >= self.max_plies else f"{state.player} to move")
        )
        output = f"{self.ruleset.value} | ply {state.ply} | {status}\n" + "\n".join(
            " ".join(row) for row in state.board
        )
        if self.render_mode == "human":
            print(output)
            return None
        return output

    def close(self):
        pass
