# Hnef

Modernizing the Hnefatafl reinforcement-learning project built by Tindur
Sigurdason and Calen Irwin for CISC-856 at Queen's University, Winter 2021.

The new `hnef` package is a dependency-free, deterministic **5×5 mini-game
engine**, with an opt-in **experimental Copenhagen 11×11** ruleset. The v2 profile
recognizes provably permanent exit forts and records external draw judgments;
automatic adjudication remains conservative. An optional PettingZoo adapter
supports training experiments. Seeded random, heuristic, and MCTS opponents are
available. A local browser game connects those opponents to the real engine;
optional CPU actor–critic training is available separately.
See [the modernization checkpoint](docs/modernization.md),
[the exact mini rules](docs/mini-rules.md), and [Copenhagen coverage](docs/copenhagen-rules.md).

## Quick start

### Play in your browser

```sh
uv sync --locked
uv run hnef web
```

Open http://127.0.0.1:8000. Play either side against random, heuristic, or MCTS
bots; share the board with a second player; or watch a bot match. The UI includes
legal-move highlights, a move journal, position review, takebacks, board rotation,
browser-local autosave, and engine-compatible replay downloads. Mini is the
default; experimental Copenhagen v2 is available in the battlefield selector.
Use **+** to open setup, then press **Play** to start. The minimal playing table
uses [Albion colours, typography, and original engraved pieces](docs/albion-ui.md).

Open **Learn** to study search, DQN/Rainbow, PPO, AlphaZero, MuZero, Dreamer,
and population training, with dated research references and interactive examples.
The **Experiments** topic compares the three implemented bots on the same position
with adjustable search budgets and downloadable diagnostics. Other methods are
study material, with proposed Hnef experiments.

No web dependencies or frontend build are required. Use `--port 8001` to choose
another port, or `PYTHONPATH=src python3 -m hnef web` without installation.
See [the browser game guide](docs/web-game.md) for controls and current limits.

### Run simulations

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

## Integration and separate static demo

The selected product baseline is the raven/hart/king playing table in
`src/hnef/web/`, including its Copenhagen selector and Learn pages. Its artwork,
UI, autosave, replay import and engine contracts are preserved.

The separate `web/` directory retains the previously deployed 5×5 Pyodide Mini
Lab for compatibility. `tools/build_demo.py`, the Pages workflow and the Caddy
container build that static mini artifact only; they do not publish the playing
table. The local playing table requires its Python HTTP API and is not directly
compatible with GitHub Pages without an adapter. A separate
[static original-table candidate](docs/STATIC-TABLE.md) now runs the same Python
functions in a dedicated Pyodide worker. Its build and CI do not publish. See
[integration scope and release gates](docs/integration-baseline.md).

CPU training and evaluated model exports from `main` are retained without adding
an experimental learned opponent to the selected playing table:

```sh
uv sync --locked --extra rl --extra train
uv run --no-sync hnef-rl --help
```

See [the learning guide](docs/rl-baseline.md). Existing static-demo deployment
instructions remain in [Pages](docs/PAGES.md) and [Railway](docs/DEPLOYMENT.md).
No public release of the selected playing table is part of this integration.

## Original school project

The original `gym_hnef/`, training modules, `run.ipynb`, `tests.ipynb`, and
`best_weights_1.h5` remain intact for reference. Their original dependencies and
APIs are not installed by the modern package. The old instructions were to upload
`run.ipynb` to a cloud notebook such as Colab and follow its cells.
