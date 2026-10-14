# Hnef

Modernizing the Hnefatafl reinforcement-learning project built by Tindur
Sigurdason and Calen Irwin for CISC-856 at Queen's University, Winter 2021.

The new `hnef` package is a dependency-free, deterministic **5×5 mini-game
engine**. Copenhagen 11×11, the modern RL adapter, training, and the web game
are future milestones. See [the modernization checkpoint](docs/modernization.md)
and [the exact mini rules](docs/mini-rules.md).

## Quick start

With Python 3.11+ and [uv](https://docs.astral.sh/uv/getting-started/installation/):

```sh
uv sync --locked
uv run hnef play --seed 42 --output /tmp/hnef-game.json
uv run hnef replay /tmp/hnef-game.json
uv run hnef benchmark --games 1000 --seed 0
```

The CLI runs random-vs-random games. A benchmark verifies every replay and
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
uv run pytest
uv run ruff check .
uv run ruff format --check .
```

Tests and lint target the modern package. The wheel includes only `hnef`, with
no Gym, NumPy, TensorFlow, or rendering dependency. The generated action codec
supports 200 actions for mini and 2,420 for an 11×11 board; only mini gameplay is
implemented so far.

## Original school project

The original `gym_hnef/`, training modules, `run.ipynb`, `tests.ipynb`, and
`best_weights_1.h5` remain intact for reference. Their original dependencies and
APIs are not installed by the modern package. The old instructions were to upload
`run.ipynb` to a cloud notebook such as Colab and follow its cells.
