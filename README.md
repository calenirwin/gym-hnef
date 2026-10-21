# Hnef

[Play Hnef · local two-player or AI](https://hnef-mini-lab.nelac.chatgpt.site).

Modernizing the Hnefatafl reinforcement-learning project built by Tindur
Sigurdason and Calen Irwin for CISC-856 at Queen's University, Winter 2021.

The new `hnef` package is a dependency-free, deterministic **5×5 mini-game
engine**, with an opt-in **experimental Copenhagen 11×11** ruleset. The v2 profile
recognizes provably permanent exit forts and records external draw judgments;
automatic adjudication remains conservative. An optional PettingZoo adapter
supports training experiments. Seeded random, heuristic, and MCTS opponents are
available. A CPU actor–critic baseline and a static browser game are now available;
see [training and model artifacts](docs/rl-baseline.md) and [demo hosting](docs/hosting.md).
See [the modernization checkpoint](docs/modernization.md),
[the exact mini rules](docs/mini-rules.md), and [Copenhagen coverage](docs/copenhagen-rules.md).
For the latest local verification and prioritized next work, see the
[October 18 assessment](docs/status-2026-10-18.md).

## Play in your browser

Use [the live game](https://hnef-mini-lab.nelac.chatgpt.site), or rebuild the same
static app without installing training dependencies:

```sh
python3 tools/fetch_browser_runtime.py
python3 tools/build_demo.py --output web-dist
python3 -m http.server 8768 --directory web-dist
```

Open http://localhost:8768. Choose **Two players · Pass & play** for a shared
board, or play either side against the computer. Undo and verified replay export
are available. This is the **5×5 mini variant**; refresh starts a new game.
See [the deployment stack and GPU extension path](docs/deployment-stack.md).

## Quick start

With Python 3.11+ and [uv](https://docs.astral.sh/uv/getting-started/installation/):

```sh
uv sync --locked
uv run hnef play --seed 42 --output /tmp/hnef-game.json
uv run hnef replay /tmp/hnef-game.json
uv run hnef benchmark --games 1000 --seed 0
```

To exercise the 11×11 implementation (not yet full Copenhagen):

```sh
uv run hnef play --ruleset copenhagen-experimental-v2 --seed 42
```

The CLI defaults to random-vs-random games. Choose opponents with
`--attackers mcts --defenders heuristic`; see [baseline opponents](docs/baselines.md)
for search budgets and reproducible comparisons. A benchmark verifies every replay and
reports attacker wins, defender wins, repetition draws, and collection-limit
truncations separately. Saved games contain action IDs and a digest of every state.

The engine can also run directly without installing any dependencies:

```sh
PYTHONPATH=src python3 -m hnef play --seed 42
```

```python
from hnef.engine import initial_state, legal_moves, apply_move

state = initial_state()
successor = apply_move(state, legal_moves(state)[0])
assert state.ply == 0  # The parent is unchanged.
assert successor.ply == 1
```

## Development

For the first learning experiment, install `--extra train` and use `hnef-rl`:

```sh
uv sync --locked --extra rl --extra train --python 3.12
uv run --no-sync hnef-rl train --run runs/my-first-baseline --updates 100
uv run --no-sync hnef-rl evaluate runs/my-first-baseline/checkpoint-000100.pt \
  --output runs/my-first-baseline/evaluations/heldout --games 50
uv run --no-sync hnef-rl list
```

The first measured model is still weak on the attacking side; the heuristic remains
the default demo opponent. Read [the results and limitations](docs/rl-baseline.md)
before interpreting defender win rates as learned strength.

```sh
uv sync --locked --extra rl
uv run --no-sync pytest
uv run --no-sync ruff check .
uv run --no-sync ruff format --check .
```

For masked multi-agent play, use `from hnef.envs.pettingzoo import env`.
See [the RL environment guide](docs/rl-environment.md) for a runnable example,
observation planes and history, rewards, and termination/truncation handling.

Tests and lint target the modern package. The wheel includes only `hnef`, with
no mandatory Gym, NumPy, TensorFlow, or rendering dependency. The generated action codec
supports 200 actions for mini and 2,420 for Copenhagen. Mini remains the default;
each replay identifies its ruleset, and old mini replays remain compatible.

## Original school project

The original `gym_hnef/`, training modules, `run.ipynb`, `tests.ipynb`, and
`best_weights_1.h5` remain intact for reference. Their original dependencies and
APIs are not installed by the modern package. The old instructions were to upload
`run.ipynb` to a cloud notebook such as Colab and follow its cells.
