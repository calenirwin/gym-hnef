# Copenhagen opponents: inventory and staged work

Audit: 22 October 2026, base `52563a78b37fa2e367b5fe69ad2c294836b38722`.
See [the rules completion gate](copenhagen-completion.md). No new trained
Copenhagen model is claimed by this audit.

## What exists

| Implementation | Actual artifact / training | Copenhagen and packaging |
| --- | --- | --- |
| `policies.RandomPolicy` | Seeded uniform legal moves; no learned weights | All modern profiles; Python/Pyodide |
| `HeuristicPolicy` | One-ply handwritten material, escape and king-pressure score | All profiles; not a calibrated probability or RL |
| `MCTSPolicy` | UCT, fresh history-preserving tree, random rollouts; default 64 iterations/48 rollout plies | All profiles; neutral search cutoff estimates never change game outcomes; browser worker available |
| `learning/` episodic actor–critic | Mini-only 64-unit board/shared layers, 16-unit history GRU, 200-action head; terminal Monte Carlo targets, Adam and entropy bonus | `encode` rejects Copenhagen; fixed 5×5 codec/78 inputs. No 2,420-action Copenhagen checkpoint |
| Legacy `agent.py`, `mcts.py`, `model.py` | RandomAgent and neural policy/value MCTS with residual CNN/self-play replay training code | Separate old environment; not modern Copenhagen-certified; not in published package |
| DQN/Rainbow, PPO, AlphaZero, MuZero, Dreamer, population methods | Learn-page explanations/proposed experiments | No modern runnable implementation/checkpoint for these methods |

The modern mini run is evidenced by `docs/rl-baseline-results-v1.json` and
`web/baseline/mini-ac-random-s42.json`: 200 updates, 3,200 games, one training seed,
128-ply cap. Export is 769,407 bytes; SHA-256
`c5feb2ab4907382810c99ee3db2635d41a632646dc57d0e2f333deb5e260b311`, linked to
checkpoint `e6fb5c378486d01c81512fd20c11e88bf2321967bc7f3f7ef6de5ed514b4a172`.
The export is present and versioned. **`runs/` and `models/` are absent in this
checkout**: old documentation saying resumable checkpoints live “on this machine”
is not current evidence. Recovery requires the original checkpoint archive;
portable inference weights do not contain optimizer/RNG state.

The saved evaluation has 420 replay-verified games reported historically, split
by side: initialized/final versus random and heuristic, plus 20 final MCTS games.
It does not establish improvement; one seed, severe side asymmetry, small samples,
and no initialized MCTS comparison. Full evaluation replays are not present here.
The modern trainer records config, source/dependency provenance, RNG and optimizer;
resume determinism is tested in one CPU runtime, not promised across platforms.
No reliable wall-clock training bill is recorded. Do not infer cost from episode count.

Legacy `best_weights_1.h5` is present (51,772,256 bytes), SHA-256
`0f7e0ed2b0cc2e0f83f9f3df5132b13abb4c2aa271c9a0ee5c81b9d4c98649cd`.
Read-only HDF5 inspection found TensorFlow/Keras 2.4.0 attributes, first convolution
(3,3,5,256), policy kernel (162,6561), value kernel (20,1). The 6,561 output is
consistent with the legacy 9×9 all-square-pairs action space, not modern 11×11's
2,420 orthogonal actions. This is evidence of weights, not validated training
quality or reproducibility. No matching run/seed/evaluation manifest was found.
The current legacy config specifies 20 MCTS simulations, 5 episodes, 10 training
loops and a long 256-filter stack; it is not proof of the checkpoint's actual run.
No TensorFlow install or legacy execution was attempted.

## Runnable foundation added here

`hnef.opponents` supplies immutable versioned baseline specs with supported
rulesets, factories and explicit search budgets. Existing UI/CLI names remain
unchanged. Only implemented opponents enter this registry; it does not advertise
planned models. A future learned spec must identify checkpoint checksum, ruleset,
feature/action versions, inference backend and measured cost before admission.

```sh
PYTHONPATH=src python -m hnef.evaluation --output /tmp/copenhagen-smoke
```

This dependency-free smoke runs all three baseline pairs in both seat assignments,
two held-out seeds, 32-ply cap and MCTS 4×4. It saves every verified replay, hashes,
source fingerprint, Python/platform, exact specs and per-decision timing.
The checked-in result is `research/copenhagen-smoke-2026-10-22/report.json`.
It tests infrastructure, not strength: short caps censor outcomes heavily. Tests
verify reproducibility, distinct seat pairs, replay identity, cutoff separation,
compatibility rejection and refusal to overwrite results. The existing engine CI
collects these tests; no new compute service is required.

## Stages and acceptance

1. **Correctness and baseline evidence, now.** Complete the rules contract and
   corpus alongside registry/evaluation work. Add tactical held-out positions
   separate from training; measure both colors, seed ranges and cutoff rates.
   Increase match length/budget only after profiling. Use paired seeds, multiple
   training seeds (at least three for comparisons), confidence intervals and
   initialized controls; do not rank a model from this smoke sample.
2. **First Copenhagen learner.** Generalize board/codec/features and training,
   evaluation, artifact loading and export together under new schema IDs; retain
   mini readers. Add legal-mask, reward-sign, finite-gradient, true weight-update,
   exact-resume and replay tests. Start with small episodic actor–critic CPU runs
   versus fixed random/heuristic pools. Report discarded truncated trajectories;
   consider bootstrapping only with an explicit change to estimator semantics.
   A model trained on experimental rules must say so, even after stable rules ship.
3. **Controlled variant suite.** Compare shared versus side-specific heads,
   board MLP versus small CNN, repetition-aware features, fixed versus snapshot
   opponent pools, and tactical curriculum versus none. These are useful
   variants without pretending each is a distinct RL algorithm. Keep one-factor
   ablations and fixed training/evaluation budgets. Then implement masked PPO and
   DQN with terminal/truncation and perspective tests; assess self-play search
   policy/value learning after stronger baselines. MuZero/Dreamer and large
   population runs remain research proposals, not near-term model promises.
4. **Browser admission.** Keep training out of the browser. Export inference-only
   weights with checksummed manifest and strict rules/feature/codec checks. Test
   Python/JS or WASM numerical/action-mask parity, seeded sampling, loading size,
   worker cancellation, memory, mobile p50/p95 latency and responsiveness. The
   existing JS exporter is mini-specific and cannot be relabeled Copenhagen.
   Reject incompatible bundles; keep heuristic fallback. No learned opponent is
   added to the published table by this PR.

Rules research/corpus and learner architecture work can advance independently;
training claims and stable release remain gated on their evidence. No paid
training, accounts, new credentials or hosting changes. Use local CPU and existing
free CI smoke jobs; cap each proposed local pilot before running it. Hosting budget
remains CAD20–30/month maximum, not a training allowance. Longer or paid training
needs a concrete time/cost proposal and specific approval.

## Decisions and unblockers

Need expert answers to the four rule questions in the completion audit, and an
archive location if exact mini checkpoint resume is desired. Neither blocks the
baseline smoke. Torch is not installed in this environment; no large Torch download
was made. No training smoke or newly trained model is claimed. Existing free CI's
learning job can validate the untouched mini trainer.

“King’s Table”, a dimensional square table and a crown/king mark remain naming
explorations only. Preserve the current published name, domain, pieces and light
visual treatment; handle logo studies in a separate design task. No other tafl
variant is implemented here.

## Evaluation follow-up

The registry now names supported profiles explicitly, so adding an enum member
cannot silently certify an opponent. Schema-2 evaluation uses opponent-identity
seed pairing, records derived seeds and reports side-specific results, terminal
reasons, cutoff rates and latency distributions. See the
[bounded profile and exact seed contract](research/copenhagen-smoke-v2-2026-10-22/README.md).
It preserves the original smoke evidence and does not increase search or match
budgets. See also [published-ruling research](copenhagen-rulings-research.md) and
the [initial held-out corpus](../tests/fixtures/copenhagen-heldout-v1/README.md).

The [first learner contract and bounded pilot proposal](copenhagen-first-learner.md)
now has a dependency-free, replay-verified collection scaffold. It produces data,
not learned weights; its initial sanity episodes yielded no terminal labels.

A [bounded synthetic policy/value optimizer smoke](research/copenhagen-optimizer-smoke/README.md)
now retains an initialized and fitted resumable linear checkpoint. This is
supervised training plumbing on four authored terminal transitions, **not** a
modern trained Copenhagen RL opponent or a browser/model-registry candidate.

A subsequent [schema-2 on-policy TD pilot](research/copenhagen-onpolicy-pilot/README.md)
now exists, including initialized/final resumable checkpoints and held-out controls.
It collected 246 transitions from synthetic starts, three terminals and 29 cutoffs;
held-out results matched initialization. It is an experimental partial-history
linear learner, not evidence that Copenhagen strength or full adjudication is solved.
It remains outside the public opponent registry and UI.
