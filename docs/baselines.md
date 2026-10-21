# Reference opponents, version 1

The engine now has three dependency-free policies: seeded uniform `random`,
one-ply `heuristic`, and random-rollout `mcts`. Mini-v1 remains the default.
These are reference opponents for future training and evaluation, not trained
models or a claim of solved play.

## Run games and comparisons

```sh
uv run --no-sync hnef play --attackers mcts --defenders heuristic \
  --iterations 64 --rollout-plies 48 --seed 42 --output /tmp/mcts-game.json
uv run --no-sync hnef replay /tmp/mcts-game.json

uv run --no-sync hnef benchmark --attackers mcts --defenders random \
  --games 50 --seed 0 --max-plies 200 --output /tmp/mcts-attackers.json
uv run --no-sync hnef benchmark --attackers random --defenders mcts \
  --games 50 --seed 0 --max-plies 200 --output /tmp/mcts-defenders.json
```

Benchmarking verifies every full replay. Reports separate attacker wins,
defender wins, genuine draws, and collection truncations. Policy reports include
the policy version, ruleset, player assignments, starting seed, collection limit,
and search budgets. Run each opponent on both sides: this asymmetric game does
not support interpreting a single side's win rate as overall policy strength.

`play` still writes the existing replay schema. A replay records the actions and
state digests needed to reproduce a game; policy configuration is in benchmark
reports, not embedded in that schema. With both policies set to random, the old
seeded sequences and benchmark output structure are preserved. Otherwise each
player has an independent RNG stream derived from the game seed. A benchmark
uses consecutive game seeds and fresh policies for each game.

## Python and PettingZoo

```python
from hnef.engine import initial_state, apply_move
from hnef.policies import MCTSPolicy

state = initial_state()
policy = MCTSPolicy(seed=42, iterations=64, rollout_plies=48)
search = policy.search(state)
successor = apply_move(state, search.move)
print(search.iterations, search.rollout_cutoffs, search.children)
```

All policies implement `choose_move(state) -> Move`. `play_game(attackers,
defenders, ...)` in `hnef.matches` accepts custom policies and returns a `Replay`.
`policy_game(seed, attackers="mcts", defenders="random", ...)` constructs and
seeds built-in policies. Invalid moves are rejected by the engine, and terminal
or immobile input positions cannot be searched. Inputs are immutable.

For an active PettingZoo turn, use:

```python
action = game.unwrapped.codec.encode(policy.choose_move(game.unwrapped.game_state))
game.step(action)
```

Use one policy instance per player and handle terminal `None` steps as in the
[AEC guide](rl-environment.md). Search reads the complete immutable state,
including repetition history, rather than reconstructing it from board planes.

## What the policies do

The heuristic scores each legal successor from the mover's perspective. Terminal
outcomes take precedence: +1 for a win, −1 for a loss, zero for a draw. Its
nonterminal defender score combines:

- 0.35 × normalized defender-minus-attacker material, using each profile's opening counts.
- 0.25 × king progress toward the nearest escape square (edge for mini, corner for Copenhagen).
- 0.30 × the fraction of four rook directions with a clear escape route.
- −0.10 × the fraction of adjacent squares occupied by attackers.

Scores are clamped to [−0.9, 0.9] and negated for attackers. Ties use the policy's
seeded RNG. This score is an untuned preference, not a win probability; the
one-ply policy does not explicitly search the opponent's reply.

MCTS uses [UCT](https://sites.ualberta.ca/~szepesva/papers/ecml06.pdf) selection,
one randomly chosen untried expansion per iteration, uniform random rollouts,
and visit/value backpropagation. Values are always stored from the root player's
perspective; opponent nodes minimize that value while retaining a positive
exploration bonus. The exploration coefficient defaults to √2 on the [−1, 1]
utility scale. A discovered immediate winning child takes precedence; otherwise
the final move has the most visits, then highest mean utility, then a seeded tie.

`iterations` (default 64) counts simulations per decision. `rollout_plies`
(default 48) bounds moves *after expansion*, not total tree depth. A zero rollout
cap is supported for debugging. A rollout ending without a terminal result has
neutral utility zero and increments `rollout_cutoffs`. This is an unknown-value
estimate, not a declared draw. Search never changes game outcomes or repetition
rules, and never calls the heuristic evaluator.

Each decision builds a fresh tree, preserving complete history in every node.
There are no board-only transposition merges, tree reuse, trained priors, GPU
dependencies, or wall-clock deadline. A small budget can leave legal root moves
unexplored and miss immediate wins; cutoff-heavy random rollouts can also give
poor estimates. Copenhagen is opt-in and retains its experimental adjudication
limits; these budgets are primarily a mini-game starting point.

## Validation and measurements

Tests cover both players' immediate wins, opponent-node utility direction,
visit accounting, final-ply rollout results, repetition history, legal actions,
determinism, immutable inputs, configuration validation, replay verification,
and CLI report round trips. The frozen random replays retain their old semantics.

Measured on Python 3.12.3 on October 14, 2026, using mini-v1, seeds 0–49, and
a 200-ply collection limit. MCTS used 64 iterations and 48 rollout plies. Every
game was replay-verified; none ended in a draw or truncation.

| Attackers | Defenders | Attacker wins | Defender wins | Total plies |
| --- | --- | ---: | ---: | ---: |
| random | random | 7 | 43 | 1,205 |
| heuristic | random | 46 | 4 | 594 |
| random | heuristic | 0 | 50 | 218 |
| mcts | random | 27 | 23 | 955 |
| random | mcts | 0 | 50 | 308 |

[Raw reports](baseline-results-v1.json) retain the parameters. These small samples
describe these specific opponents and budgets, not Copenhagen balance or general
playing strength. The heuristic had more attacker wins than MCTS against random
in this sample. No parameters were tuned to these results; deeper search, stronger
rollouts, and head-to-head evaluation remain separate experiments.
