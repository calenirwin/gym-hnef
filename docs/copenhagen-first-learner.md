# First Copenhagen learner: data contract and bounded pilot

This PR adds a **collection scaffold**, not a trained opponent. Run it without
Torch, NumPy, accounts or downloads:

```sh
PYTHONPATH=src python -m hnef.learning.copenhagen --output /tmp/copenhagen-collection
```

The sanity run is four episodes capped at 16 plies. Collection guards allow at
most 16 episodes and 128 plies per episode through the Python API. Output paths
cannot be overwritten. The checked-in manifest/replays are in
`research/copenhagen-collection-sanity/`; full feature records from this run are
local at `/workspace/shared/copenhagen-collection-sanity`. Regenerate those records
with the command; the manifest contains their checksums, not their entire payloads.

## Implemented contract

* Rules are exactly `copenhagen-experimental-v2`; no fort/draw semantics change.
* Features `copenhagen-absolute-board-full-history-v1`: three absolute row-major
  A/D/K planes (121 cells each), mover, raw ply, repetition count and the complete
  ordered position-key history. This retains the engine's repetition information;
  it does not claim a neural network can reason perfectly over opaque hashes.
* Actions use the existing 2,420-entry orthogonal codec. Its ordered source/target
  vocabulary is hashed into the manifest. Every sample includes a legal mask.
  The uniform learner samples only legal entries; no invalid-action repair.
* Learner side alternates across episodes. Learner/reference RNG streams are
  SHA-256-derived with separate role names. Training seeds are below 100000;
  evaluation seeds from the baseline smoke are reserved. Python/runtime and source
  fingerprint are recorded. Fixed random is the sanity reference; the collection
  API also admits the existing fixed heuristic, with an explicit ID.
* Each learner action stores its full pre-move state identity and observation.
  Each whole episode produces an engine-verified replay. Finished episodes assign
  +1/-1/0 from the learner's perspective. Unfinished episodes retain **null**
  targets; they are not zero-reward examples. No update is run on these records.

The four-episode sanity produced 64 plies and 32 learner samples, all censored
at the collection cap, with **zero labeled training samples**. This is useful
contract evidence, not evidence that the proposed estimator can learn Copenhagen.
It also warns against silently treating capped episodes as draws.

## Proposed first actual learner experiment (not run)

Start with a small masked episodic actor–critic: 64-unit board encoder, 16-unit
history GRU, 64-unit shared layer, 2,420-action policy head and scalar value head.
Convert the mover to a scalar and normalize ply/repetition explicitly in a new
network feature version; keep the raw collection contract intact. Mask illegal
logits before sampling and entropy/log-prob computation. Reject terminal inputs
with an empty legal set. Use undiscounted terminal returns and exclude truncated
trajectories from this initial estimator, reporting how much data is excluded.

Before any quality run, implement checkpoint/load/resume and train/evaluate as
one versioned contract, separate from mini schema 1. Require legal masking,
reward perspective, finite loss/gradients, actual parameter changes, checkpoint
feature/rules/codec rejection, exact same-runtime resume and initialized controls.
Do not reuse mini weights as Copenhagen weights or silently resize its head.

First proposed CPU pilot: one thread, seeds 0/1/2, at most one update of four
128-ply episodes per seed against fixed random, with a hard aggregate five-minute
wall limit. Save full manifests and initialized controls. If no completed learner
trajectories exist, skip the update and report the pipeline as incomplete rather
than invent returns. This budget is a proposal, not an incurred training cost.
Torch is absent locally; do not download it here merely to run this pilot. The
existing free learning CI can eventually run tiny tensor/optimizer tests once
implemented, without introducing a new compute service.

Only after the pipeline gate should we choose longer episodes, bootstrapped
estimators or a separate training curriculum. Each changes either cost or the
estimator and needs measured justification. The public held-out corpus must never
become curriculum data. Keep final evaluation seeds/positions separate from model
selection; report both colors, terminal reasons, cutoffs, time and initialized
versus trained controls against fixed baselines. A tiny pilot cannot justify a
strength claim, stable rules label or browser release. No model registry entry is
added until a real checksummed checkpoint and compatible inference path exist.

## Completed bounded fallback

The proposed on-policy actor–critic pilot above is still not implemented. A
subsequent full-game probe yielded no terminal labels, so a separate
[NumPy supervised optimizer plumbing smoke](research/copenhagen-optimizer-smoke/README.md)
now demonstrates real parameter updates, legal masks, finite gradients and exact
checkpoint resume on independent synthetic terminal transitions. Its linear,
partial-history feature version and teacher-action loss are deliberately named
separately. It is not evidence of a trained competitive Copenhagen RL opponent.
