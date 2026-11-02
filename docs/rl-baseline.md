# RL baseline and model catalog

The first modern baseline is **episodic actor–critic on mini-v1**, trained on CPU.
It is an experiment with reproducible artifacts, not a strong opponent yet.

## Run it

```sh
uv sync --locked --extra rl --extra train --python 3.12
uv run --no-sync hnef-rl train --run runs/mini-ac-random-s42 \
  --updates 200 --episodes-per-update 16 --seed 42
uv run --no-sync hnef-rl evaluate runs/mini-ac-random-s42/checkpoint-000200.pt \
  --output runs/mini-ac-random-s42/evaluations/heldout --games 50
uv run --no-sync hnef-rl list
```

The named first run is historical; its resumable checkpoint directory is not
present in the October 22 audit checkout. The versioned browser inference export
and result reports remain available. Choose a new run name for a fresh experiment. A run directory cannot silently be overwritten. Resume adds
updates to the saved configuration, optimizer, sampling RNG, and Torch RNG:

```sh
uv run --no-sync hnef-rl train --run runs/mini-ac-random-s42 --resume --updates 100
```

Exact source/dependency fingerprints are checked on resume. Source changes need
a new run. Bitwise resume is tested in the same CPU runtime, not promised across
PyTorch/platform versions. Torch uses one CPU thread and deterministic algorithms.

Use `--opponents random heuristic self` to uniformly sample an opponent pool;
`self` samples the current policy, held fixed during each collection batch.
The first recorded run used **random only**, without imitation, search guidance,
reward shaping, or GPU training. The two sides share network parameters, and the
learning side alternates each episode. Only that side's actions receive an update.

The network has a 64-unit board encoder, 16-unit GRU over the complete ordered
position-digest history, a 64-unit shared layer, a 200-action masked policy head,
and a tanh value head. Inputs retain side to move, elapsed plies, current repetition
count, and all history digests as normalized bytes. There is no sliding history
window. The GRU is a learned compression of opaque identities, not a proof that
the policy understands repetition. Tactical board reasoning comes from the board
input; this deliberately small baseline leaves substantial modeling work open.

Terminal targets are undiscounted +1/−1/0 from the recorded mover's perspective.
The actor uses a detached learned-value baseline; the critic fits terminal returns.
An entropy bonus encourages exploration. Each batch gets one Adam update and
gradient clipping at 1.0. Nonterminal collection cutoffs are counted and excluded
from the update, not treated as draws. This complete-episode estimator can favor
shorter completed episodes if truncations become frequent; measure them before
using it for longer games. Every collected episode is replay-verified.

## Artifact layout

```text
runs/<run-name>/
  run.json                    # config, runtime, feature/rules versions, source fingerprint
  checkpoint-000000.pt         # initialized control
  checkpoint-000001.pt         # immutable complete training state per update
  latest.json                 # atomic pointer to latest checkpoint
  metrics/000001.json          # losses, outcomes, cutoffs, position counts
  sample-replay.json
  evaluations/<evaluation>/
    report.json               # checkpoint hash, seeds, both sides, budgets, latency
    replays/*.json
models/
  mini-ac-<full-sha256>/
    checkpoint.pt
    manifest.json
    evaluation-<report-sha256>.json
  candidate.json              # explicit alias, never automatically called champion
  mini-ac-random-s42-web.json  # portable inference weights
```

Runs and model binaries are ignored by Git. Keep small result reports and model
cards in `docs/`; archive valuable checkpoints separately before changing machines.
The first run occupies about 67 MB; its release and browser export together occupy
about 1.1 MB. Those local checkpoint directories are absent from the October 22 audit checkout.
The retained Mini Lab sources contain the inference export and public evaluation
card; the published original playing table does not load that learned opponent.

```sh
uv run --no-sync hnef-rl promote runs/mini-ac-random-s42/checkpoint-000200.pt \
  --report runs/mini-ac-random-s42/evaluations/final/report.json --alias candidate
uv run --no-sync hnef-rl export-web runs/mini-ac-random-s42/checkpoint-000200.pt \
  --output models/mini-ac-random-s42-web.json
```

Promotion checks that the evaluation belongs to the checkpoint hash. It exports
an experimental candidate; it does not assert a quality threshold or redeploy a
site. Aliases resolve with checksum verification. Checkpoints load with Torch's
restricted `weights_only=True` loader and reject mismatched schemas/rules/features
and nonfinite parameters.

## First measurement — October 18, 2026

Run `mini-ac-random-s42`: seed 42, 200 updates × 16 episodes = **3,200 training
games**, 128-ply cap, learning rate 0.001, entropy coefficient 0.02, value coefficient
0.5. The initialized checkpoint is evaluated with the same held-out seed set.
The final checkpoint SHA-256 is
`e6fb5c378486d01c81512fd20c11e88bf2321967bc7f3f7ef6de5ed514b4a172`.

Results below are candidate wins, not attacker wins. Random/heuristic comparisons
use seeds 100000–100049, 50 games per side. MCTS uses seeds 100000–100009,
64 iterations and 48 rollout plies, ten games per side. All policies use seeded
sampling; the learned policy is not greedily selected.

| Opponent | Candidate side | Initialized wins | Trained wins |
| --- | --- | ---: | ---: |
| Random | Attackers | 0/50 | 1/50 |
| Random | Defenders | 47/50 | 48/50 |
| Heuristic | Attackers | 0/50 | 0/50 |
| Heuristic | Defenders | 7/50 | 5/50 |
| MCTS | Attackers | Not measured | 0/10 |
| MCTS | Defenders | Not measured | 8/10 |

No evaluation draws or truncations occurred. All 420 evaluation games (200 initial,
200 final, 20 MCTS) replay-verified. Final inference averaged approximately
1.0–1.7 ms per decision on this machine; this is CPU inference, not browser latency.

These results **do not establish improvement over initialization**. This is one
training seed and a small evaluation sample. High defender wins against random
mostly reflect the asymmetric mini-game. The important missing capability is
attacker tactics, and the heuristic remains the demo's default opponent.

Next experiments: bootstrap tactical competence from the heuristic or a small
curated position curriculum, compare attacker/defender-specific value learning,
then train against a fixed reference/checkpoint pool. Reserve new evaluation seeds
for final comparisons and use several training seeds. Keep the initialized control,
report both sides, and distinguish a useful training signal from a better opponent.

## Verification

287 tests pass with the RL and training extras installed. New coverage checks legal
masking, mover-specific return signs, history/feature parity, actual parameter
updates, exact uninterrupted-versus-resumed training, cutoff exclusion, evaluation
and release identity, corrupted releases, and JavaScript/PyTorch inference parity.
The parity test uses Node and the repository's `web/` sources. Core and PettingZoo
CI stay separate from the CPU learning job. Ruff and wheel/source builds pass.
