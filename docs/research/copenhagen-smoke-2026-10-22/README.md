# Local Copenhagen baseline smoke

Command (22 October 2026):

```sh
PYTHONPATH=src python -m hnef.evaluation --output docs/research/copenhagen-smoke-2026-10-22
```

12 games, 315 plies; 7 collection-limit truncations, 5 terminal results. Every
saved replay verified. No trained model, strength ranking or automatic dead-position
claim. These short trajectories primarily validate the evaluation pipeline.

Measured local mean decision time: random 0.35 ms (107 decisions), heuristic
106.70 ms (94), MCTS at 4 iterations × 4 rollout plies 26.95 ms (114). These unequal
budgets and CPU measurements are not a fair strength comparison or mobile browser
performance guarantee. Raw decision timings, outcomes, source fingerprint and
replay hashes are in `report.json`.

Local validation: 402 tests passed, 2 optional modules skipped (Torch training and
PettingZoo dependencies absent); Ruff check and format passed. No dependencies
were added to the project, no training was run, no site was published.
