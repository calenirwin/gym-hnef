# Bounded Copenhagen learning and evaluation

The synthetic-start learner is a research tool, not a released opponent. It uses
`copenhagen-experimental-v2`; unresolved fort, dead-position and repetition
questions remain in the [completion audit](copenhagen-completion.md). Public game
rules, saved games and browser opponents are unaffected.

## Training with terminal signal

A random opponent often leaves short trajectories unfinished. The trainer also
supports the registered `heuristic-v1` opponent, which can find immediate wins.
This supplies real terminal rewards, including learner losses, without replacing
unknown outcomes with draw labels. It does not guarantee balanced wins/losses or
that the learner becomes stronger.

`--both-sides` visits each training board with attackers and defenders to move in
that order. Each is a **fresh synthetic start** with its own initial history, not a
claim that changing the mover yields a reachable continuation. Fixture teacher
moves and expected outcomes are ignored. No held-out boards are used for training.
The opponent ID and start schedule are checkpoint contract fields: resume rejects
changes to either, as well as changes to the horizon, source or other fingerprints.

```sh
export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 PYTHONPATH=src
timeout 65s python -m hnef.learning.onpolicy train \
  --data tests/fixtures/copenhagen-training-plumbing-v1/cases.json \
  --output /tmp/copenhagen-train --episodes 32 --horizon 8 --seed 17 \
  --reference heuristic-v1 --both-sides
```

The training cap remains 32 episodes per invocation, 64 cumulative updates and
8 learner decisions per episode. A decision includes one opponent reply if the
game remains live. TD targets bootstrap nonterminal successors, including cutoffs,
from the actual next state of the same player; true terminals override bootstrap.
The model still has partial-history linear features. Full engine history is
preserved for adjudication, but the feature representation is not Markov-complete.

## Held-out comparison

Evaluation can use a longer horizon than training, without modifying the saved
training configuration. Candidate and initialized models receive the same learner
and reference seeds within each opponent/start/seed pair. Opponent identities have
separate reference streams. All policies remain fixed during evaluation.

```sh
timeout 65s python -m hnef.learning.onpolicy evaluate \
  --data tests/fixtures/copenhagen-training-plumbing-v1/cases.json \
  --heldout tests/fixtures/copenhagen-heldout-v1/cases.json \
  --checkpoint /tmp/copenhagen-train/checkpoint-0032.npz \
  --output /tmp/copenhagen-evaluation --horizon 16 \
  --opponents random-v1 heuristic-v1
```

The API accepts up to four unique reserved seeds (at least 100000); the CLI uses
100000 and 100001. Evaluation allows up to 32 learner decisions and the bounded
registry IDs `random-v1`, `heuristic-v1`, `mcts-v1-i4-r4`. The latter is only a tiny
search-cost diagnostic, not evidence of strong MCTS play. Duplicate IDs/seeds,
unsupported opponents and invalid horizons fail before output creation. Existing
output directories are never overwritten. Train/evaluation boards must be distinct
under every rotation/reflection, regardless of mover.

Reports include:

- Outcome counts and terminal reasons by model, opponent and learner side, with
  unfinished games retained in every denominator.
- Cutoff rates and finite-sample win-rate bounds: `wins / games` through
  `(wins + unfinished) / games`. These bound unknown outcomes of the sampled games;
  they are **not confidence intervals** or population-strength estimates.
- Paired initialized-to-candidate outcome transitions and bounds on the win-rate
  difference. Unfinished pairs cannot silently become draws, ties or improvements.
- Checkpoint/data/source fingerprints, separate training/evaluation horizons,
  recorded RNG seeds, episode wall times and uniquely named verifiable traces.

Evaluation report schema 2 changes the reference stream naming and trace names to
include opponent identity. Do not compare its trajectories as if they were seeded
identically to schema 1. Checkpoint schema 2 remains source-bound; older source
fingerprints are not silently migrated. Reproduce old artifacts with their
recorded source or train a new matching checkpoint before evaluating current code.

Both commands check a 60-second budget at episode boundaries. Use the outer
65-second timeout shown above for a hard process cap and set thread limits before
Python starts. Results/checkpoints belong in experiment output directories; record
measured findings and review decisions in the PR rather than adding dated result
narratives to this guide.

## Interpreting a small experiment

Predeclare training seeds, opponents, horizons and which checkpoints will be
compared before opening evaluation results. A matched random-versus-heuristic
training comparison can measure terminal-signal coverage, not isolate every cause
of strength. Report attacker and defender wins and losses separately: more
terminals can simply mean more predictable losses.

The existing public held-out corpus is a small authored diagnostic, not independent
referee certification or a hidden final benchmark. Reusing the same initialized
control across training seeds does not create independent control observations.
Do not tune on these results. Before a strength claim, require broader training-only
coverage, attacker-winning signal, full-history features, multiple seeds and a
separate frozen final evaluation including opening-start trajectories.
