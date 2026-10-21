# PettingZoo environment v0

Install the optional training dependencies with `uv sync --locked --extra rl`
or `pip install '.[rl]'`. Importing the engine or running the CLI does not require
NumPy, Gymnasium, or PettingZoo.

```python
from hnef.envs.pettingzoo import env

game = env(max_plies=500)
game.reset(seed=42)
returns = {agent: 0.0 for agent in game.possible_agents}
for agent in game.agent_iter():
    observation, reward, terminated, truncated, info = game.last()
    returns[agent] += reward
    action = (
        None if terminated or truncated
        else game.action_space(agent).sample(observation["action_mask"])
    )
    game.step(action)
game.close()
print(returns)
```

The environment follows the [PettingZoo AEC API](https://pettingzoo.farama.org/api/aec/).
Agents are named `attackers` and `defenders`; attackers move first. `env()` adds
PettingZoo's call-order and action-bound wrappers. `raw_env()` exposes the same
game without those wrappers. This is a sequential game and cannot be converted
to a simultaneous Parallel environment.

## Observations and actions

The observation is a Gymnasium `Dict` with three entries:

| Entry | Shape / dtype | Meaning |
| --- | --- | --- |
| `observation` | `(7, size, size)`, int8 | Binary board planes listed below |
| `action_mask` | `(200,)` or `(2420,)`, int8 | One for each currently legal action |
| `action_history` | `(max_plies,)`, int32 | Action IDs from the opening, padded with −1 |

Planes 0–6 encode attackers, ordinary defenders, king, throne, restricted corners,
defenders-to-move, and observer-is-defender, respectively. The last two planes
are constant across the board. Mini's corner plane is zero because its corners
are ordinary escape squares. Coordinates remain fixed for both players: no
rotation or perspective-dependent action numbering.

Actions use the engine's stable `ActionCodec`: all distinct source/destination
pairs in the same row or column, in row-major order. The environment accepts
Python and NumPy integer IDs. Illegal moves raise an error without changing the
state; they do not become passes or fabricated losses. Sample with the mask.
Masks are zero for inactive players, terminated players, and truncated players.
Returned arrays are independent copies.

Repetition depends on history. **The seven board planes alone are not a complete
Markov observation.** The action history reconstructs the exact engine state
from the known opening and ruleset, including repetition and elapsed plies.
A training integration must retain this entry or explicitly use a history-aware
policy; wrappers that discard it lose that information. There is no arbitrary
starting-position reset option. `options` is reserved and currently ignored.

`game.unwrapped.game_state` exposes the immutable engine state for search and
debugging. `game.unwrapped.codec` converts between IDs and `Move` objects. Info
contains ruleset, experimental status, ply, full state digest, winner, outcome
reason, and whether the collection limit was reached. Episode history can be
replayed with the core engine; there is no hidden stochastic transition.

## Rewards, endings, and reproducibility

Rewards are zero during play, +1 for the winner, −1 for the loser, and zero for
a draw. Both agents receive their final reward via `last()`, then must take a
`None` step for removal. The game ends when `agents` is empty. Do not stop the
loop immediately after the winning move: that would miss final rewards.

`max_plies` is a positive collection limit, not a rule declaring a draw. A
nonterminal game at that limit has `truncated=True`, `terminated=False`, no
winner, and no outcome reason. A genuine terminal move at the limit takes
precedence and is not truncated. Reset restores the opening and all bookkeeping.

`reset(seed=...)` also seeds each agent's action/observation spaces independently,
so the example's masked sampling is reproducible without global RNG changes.
`reset()` without a seed continues existing sampling streams. Game transitions
themselves are deterministic. Rendering is optional: `ansi` returns text and
`human` prints it; `close()` needs no external resource cleanup.

Mini-v1 is the stable default. Pass `ruleset="copenhagen-experimental-v2"` (or
the preserved v1 profile) to opt into 11×11 play. The adapter does not expand
[Copenhagen adjudication coverage](copenhagen-rules.md): fort detection remains
conservative, and external draw judgments are not automatic agent actions.

## Validation

CI installs the RL extra and runs PettingZoo's official `api_test` and `seed_test`
for all three rule profiles, alongside engine parity, frozen replay, reward,
repetition, truncation, invalid-action, rendering, and observation-isolation tests.
The official tests emit advisory warnings for dictionary observations and the
descriptive agent names; these are intentional API-supported choices. Without
the extra, pytest skips this adapter module and runs the core suite.
