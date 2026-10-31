# Identity-paired evaluator smoke and profile

Schema 2; old schema-1 evidence remains in the sibling folder unchanged.
Command:

```sh
PYTHONPATH=src python -m cProfile -o /tmp/copenhagen-profile.pstats -m hnef.evaluation --output docs/research/copenhagen-smoke-v2-2026-10-22
```

Same bounds: 12 games, two seeds, 32 plies, MCTS 4 iterations × 4 rollout plies.
340 plies completed, 9 collection cutoffs; all replays verified. These are pipeline
checks, not a strength ranking. **All latency values in this report include
cProfile overhead** and must not be compared directly to uninstrumented results.

Schema 1 assigned RNG streams by seat: swapping sides changed an opponent's RNG
seed. Schema 2 derives each stream from the evaluation seed and versioned opponent
ID using SHA-256 (first eight bytes, big-endian). The same opponent gets the same
seed in both seats and across matchups, independent of registry ordering. Equal
seeds do not imply equal actions or consumption counts on different positions.
Derived attacker/defender seeds are saved per game. This is a versioned evaluation
change, not an engine/replay change; old smoke outcomes are not directly pooled.

`report.json` includes per-opponent/per-seat wins, losses, draws, unfinished games,
terminal reasons, cutoff fraction and decision-weighted mean/p50/p95/max latency.
Percentiles use nearest rank. Empty latency sets return null. No win-rate or Elo
ranking is inferred from censored short games.

`profile.json` records the bounded run's hot functions. cProfile recorded about
99 million calls and 38.9 seconds of internal profiled time. The smoke's cumulative
time was 41.2 seconds, legal move generation 35.6 seconds across 32,198 calls, and
heuristic decisions 30.1 seconds across 106 decisions. Cumulative times overlap;
do not add them. `apply_move` validates by regenerating moves, and the heuristic
applies every candidate. Move generation is the first optimization candidate,
not larger search budgets. Any optimization must preserve legality, deterministic
ordering and replay digests. No optimization or increased budget is included here.
