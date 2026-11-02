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
100000 and 100001 by default; override with `--evaluation-seeds`. Evaluation allows up to 32 learner decisions and the bounded
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

## Balanced tactical curriculum

The [training-only tactics](../tests/fixtures/copenhagen-training-tactics-v1/README.md)
contain explicit winning/loss witnesses for both learner sides. The witnesses are
mechanically verified rule continuations, not supervised targets or forced-outcome
labels. Actual terminal rewards drive updates. Attacker-winning starts deliberately
have low branching; this is a teaching aid, not a full-game benchmark.

`mixed-random-heuristic-v1` alternates **complete passes through all starts** between
random and heuristic references. Every start and both learner sides therefore meet
both opponents in a full two-pass cycle. Alternating opponent on every episode
would confound it with the alternating mover and is not used. Partial runs may be
unbalanced; reports list actual opponent/side exposure and visited start indices.
Resume retains absolute step and schedule identity across pass boundaries.

```sh
timeout 65s python -m hnef.learning.onpolicy train \
  --data tests/fixtures/copenhagen-training-tactics-v1/cases.json \
  --output /tmp/copenhagen-balanced --episodes 32 --horizon 4 --seed 17 \
  --reference mixed-random-heuristic-v1
```

Both movers are already explicit in these eight starts; omit `--both-sides`.
For a matched comparator, use `--reference heuristic-v1` with the same other
settings and a new output directory. Reports count observed terminal wins/losses
and nonzero actor/critic gradients separately for each learner side. A high terminal
rate consisting only of losses is not balanced update coverage.

The [fresh validation set](../tests/fixtures/copenhagen-validation-tactics-v1/README.md)
is disjoint under rotations/reflections from training and earlier public diagnostics.
Use it with the evaluation command after freezing a manifest containing all seeds,
source/data fingerprints, budgets and the rule to evaluate every final checkpoint.
Its public synthetic tactical families do not justify a general strength claim.
Never select a checkpoint or curriculum from these validation outcomes.

```sh
timeout 65s python -m hnef.learning.onpolicy evaluate \
  --data tests/fixtures/copenhagen-training-tactics-v1/cases.json \
  --heldout tests/fixtures/copenhagen-validation-tactics-v1/cases.json \
  --checkpoint /tmp/copenhagen-balanced/checkpoint-0032.npz \
  --output /tmp/copenhagen-validation --horizon 16 \
  --opponents random-v1 heuristic-v1 --evaluation-seeds 100010 100011
```

## Training-only learning curves

The [broader training starts](../tests/fixtures/copenhagen-training-broad-v1/README.md)
add controlled distractor moves to the existing teaching positions. The read-only
`hnef.learning.curves` command enumerates legal one-ply successors and measures
policy probability assigned to immediate wins. It does not use witness labels,
sample actions, update weights, consume a training RNG or select checkpoints.
Positions without an immediate winning move are marked `null` and excluded from
the opportunity mean; they are not counted as failed winning choices.

```sh
timeout 65s python -m hnef.learning.onpolicy train \
  --data tests/fixtures/copenhagen-training-broad-v1/cases.json \
  --output /tmp/copenhagen-broad-1 --episodes 32 --horizon 4 --seed 17 \
  --reference mixed-random-heuristic-v1
timeout 65s python -m hnef.learning.onpolicy train \
  --data tests/fixtures/copenhagen-training-broad-v1/cases.json \
  --output /tmp/copenhagen-broad-2 --episodes 32 --horizon 4 \
  --reference mixed-random-heuristic-v1 \
  --resume /tmp/copenhagen-broad-1/checkpoint-0032.npz
timeout 65s python -m hnef.learning.curves \
  --data tests/fixtures/copenhagen-training-broad-v1/cases.json \
  --checkpoints /tmp/copenhagen-broad-1/checkpoint-0000.npz \
    /tmp/copenhagen-broad-1/checkpoint-0016.npz \
    /tmp/copenhagen-broad-1/checkpoint-0032.npz \
    /tmp/copenhagen-broad-2/checkpoint-0048.npz \
    /tmp/copenhagen-broad-2/checkpoint-0064.npz \
  --output /tmp/copenhagen-broad-curves
```

Use the single-thread environment settings above. The existing cumulative training
cap remains 64 updates. Curves accept 2–9 unique checkpoints with strictly increasing
steps, the same seed/configuration and matching source/data/rules/encoding contracts.
They require explicitly training-only data and refuse to overwrite output. Reports
include per-position branching, immediate-winning-action counts, uniform-policy
reference probabilities, checkpoint hashes and separate means for each learner side.

Freeze seeds, checkpoints, budget and an experiment-specific improvement gate
before running. Include terminal win/loss update coverage as well as probability
curves: a small change in a training metric alone is not readiness for a strength
claim. If broader-branching learning fails that gate, diagnose training instead of
spending a fresh evaluation set. If it passes, freeze a new, disjoint evaluation
separately before use. These curves are training-fit evidence under the current
experimental rules, not minimax values, generalization or full-game win rates.

## Fixed staged training

The [staged manifest](../tests/fixtures/copenhagen-training-staged-v1/README.md)
binds both constrained and broader training starts to one checkpoint contract.
`--curriculum constrained32-broad32-v1` makes a fixed transition after 32 updates;
`--curriculum broad64-v1` is the same-budget broader-only control. Both retain the
64-update cap. No dataset swap, optimizer reset or automatic mastery decision
occurs at the boundary. Resume rejects a different curriculum ID.

```sh
timeout 65s python -m hnef.learning.onpolicy train \
  --data tests/fixtures/copenhagen-training-staged-v1/cases.json \
  --output /tmp/copenhagen-stage-1 --episodes 32 --horizon 4 --seed 17 \
  --reference mixed-random-heuristic-v1 --curriculum constrained32-broad32-v1
timeout 65s python -m hnef.learning.onpolicy train \
  --data tests/fixtures/copenhagen-training-staged-v1/cases.json \
  --output /tmp/copenhagen-stage-2 --episodes 32 --horizon 4 \
  --reference mixed-random-heuristic-v1 --curriculum constrained32-broad32-v1 \
  --resume /tmp/copenhagen-stage-1/checkpoint-0032.npz
timeout 65s python -m hnef.learning.curves \
  --data tests/fixtures/copenhagen-training-staged-v1/cases.json --stage broad \
  --checkpoints /tmp/copenhagen-stage-1/checkpoint-0000.npz \
    /tmp/copenhagen-stage-1/checkpoint-0032.npz \
    /tmp/copenhagen-stage-2/checkpoint-0064.npz \
  --output /tmp/copenhagen-stage-curves
```

Use the documented single-thread environment. For the control, use new output
folders and `broad64-v1` for both training invocations. Freeze seeds, curve points,
criteria and budgets before either arm runs. Report probability curves on the same
broader subset, actual stage-specific terminal updates and reference/side exposure.
In particular, broader-phase wins/losses must be observed directly; successes in
the constrained phase cannot satisfy a broader-phase coverage gate. Preserve all
fixed seeds rather than selecting a favorable model. No validation or final
benchmark is needed to diagnose a failed training-only gate.

## Intermediate defender stage

The [three-stage manifest](../tests/fixtures/copenhagen-training-defender-staged-v1/README.md)
adds an intermediate branch count for only the two defender escape starts. Use
`--curriculum constrained16-intermediate16-broad32-v1` for 16 constrained, 16
intermediate and 32 broader updates. For the control, use the same manifest with
`--curriculum constrained32-broad32-v1`. Both use the existing two 32-update
invocations and the same documented resume process; do not change curriculum IDs
between invocations. Keep `--stage broad` for comparison curves.

The fixed manifest has 24 starts (the explicit loader cap), but trajectory, thread
and time budgets are unchanged. Each 16-update phase contains a complete crossed
start/opponent cycle; the final 32-update phase contains two. Attacker starts and
defender loss positions in the intermediate phase are unchanged. Check counts and
terminal witnesses, freeze criteria/seeds before running, and preserve all outcomes.
This compares two fixed teaching schedules, not a learned promotion decision.

### Replay diagnosis of training updates

`PYTHONPATH=src python tools/diagnose_copenhagen.py --data <training-cases.json>
--run <training-invocation-directory> --output <new-directory>` diagnoses up to
32 saved updates without training a new model. Set `OPENBLAS_NUM_THREADS=1`,
`OMP_NUM_THREADS=1` and `MKL_NUM_THREADS=1`. The command has a 60-second cap.
It requires matching source/data/encoding contracts, reproduces collection,
optimizer parameters and RNG state, and records input hashes. Use the source
revision matching the saved training checkpoint; contracts are not bypassed.

The report counts exact one-ply winning opportunities and choices at visited
learner states, records their signed advantages, and measures each update's
change in winning-action probability on broad **training** probes. Negative
probe deltas indicate interference with that probe, not proof of weaker play.
Actor and critic parameters are separate. The reported clipping ratio isolates
how the critic gradient reduces actor step size through the shared norm cap;
it is not an actor/critic gradient cosine or a causal critic ablation.
Synthetic training diagnostics do not replace heldout full-game evaluation.

### Fixed tactical-prior behavior experiment

Training accepts `--behavior immediate-win-prior4-v1` (default `policy-v1`).
The experimental behavior policy is `q(a|s) = p(a|s) w(s,a) / sum_b p(b|s) w(s,b)`,
where `w=4` for exact engine-adjudicated immediate winning moves and `w=1`
otherwise. Illegal actions remain masked. This is rule-assisted sampling, not
an unassisted learner discovering tactics. It does not decide unresolved rules.

The prior is fixed with respect to the parameters. The actor uses `q-one_hot`
in its loss gradient and samples from that same `q`; no off-policy forced moves
or omitted importance correction are hidden in the update. The critic targets
the same behavior's trajectories with explicit truncation bootstrapping. Traces
record the behavior identifier and each selected action's actual probability;
checkpoints bind the behavior and reject switching it on resume. Source
fingerprints still prevent silently resuming older implementations.

Training curves and the existing evaluation command measure the **raw** policy
`p`, with the tactical prior removed. Gate any heldout run on prespecified raw
training-policy improvement, balanced actual win/loss updates, and exact replay.
Assisted collection success alone is not evidence of learned playing strength.

The replay diagnostic's schema 2 adds per-visited-state legal/winning action
counts, raw and behavior winning-action probability, selected-action probability,
value/target and exact state identity. Summaries group by original training start
and initial versus continuation decision. `conditional_expected_wins` sums the
actual behavior probabilities at visited histories; `observed_minus_expected`
compares this sum with selected wins. Histories and policies adapt, so these are
not independent trials, binomial significance tests or guarantees of wins.
Raw probability mass on assisted histories is not the counterfactual outcome of
an unassisted rollout. Per-position probe deltas separate training-fit changes
on each broad start; they do not measure heldout generalization. The command
still replays existing updates only and never alters checkpoints or rules.

### Matched-decision rollout batches

`python -m hnef.learning.batches train --data <training-cases.json> --output
<new-directory> --mode cell-batch-v1 --seed 53 --cells 8` compares with
`--mode episode-v1`. Use the same documented one-thread environment variables
and `PYTHONPATH=src`. Resume into a new directory with the last checkpoint and
identical mode/seed. `verify --data <training-cases.json> --output <run-directory>`
replays the saved invocation in memory, checking traces, gradients, probes,
parameters and checkpoint state without writing new model artifacts.

The comparison uses eight broad training starts and two reference opponents.
Each position/opponent pair receives exactly eight learner decisions. A rollout
ends at a terminal result or at four learner decisions; the last horizon is
shortened to meet the quota exactly. After a terminal rollout a fresh independent
rollout fills the remaining quota. Cutoffs retain TD bootstrapping, not draw
labels. Seeds are independently derived for each pair/rollout and each actor.
Reference assignment crosses every start and therefore both learner sides.

In `cell-batch-v1`, parameters stay frozen during collection for one pair and
one decision-weighted gradient update follows. In `episode-v1`, each rollout
gets its own update. Both use the fixed tactical-prior behavior and its correct
on-policy gradient; reported probes remove that prior. Equal sampling budgets
do not imply equal optimizer steps, rollout counts or aggregate gradient
weights. Learning-rate and update-frequency effects are part of this comparison;
it cannot isolate variance reduction alone. Per-rollout gradient contributions
are distinguished from actual aggregate cell parameter changes.

The new schema-3 algorithm binds mode, seed, position order, RNG derivation and
budgets. It does not migrate earlier checkpoints; source fingerprints remain
strict. Limits are eight pairs (64 learner decisions) per invocation, sixteen
pairs (128 decisions) total, and 60 seconds per invocation. No public opponent
or game rule changes follow from a synthetic training comparison.

### Local optimizer factorial on saved batches

`tools/compare_copenhagen_gradients.py --data <training-cases.json> --run
<saved-cell-batch-invocation> --output <new-directory>` separates local optimizer
operations using identical starting parameters, states, actions and TD targets.
Its four arms cross decision versus rollout weighting with clipping after
averaging versus clipping each rollout before averaging. All four have the same
total learning-rate exposure. A separately labeled unmatched-dose diagnostic
shows the effect of multiplying that exposure by the rollout count.

The command verifies that decision-weighted aggregate clipping reproduces the
saved batch update, and that splitting a fixed gradient across stateless SGD
steps reproduces the equivalent single step. This identity does **not** cover
online policy refresh, changing targets or momentum. Local probe effects are not
cumulative learning curves, trained models or heldout strength evidence. No
checkpoint is written and these diagnostics cannot satisfy a learning gate.

Use the original trainer source revision matching the artifacts on PYTHONPATH,
with the standard one-thread environment caps. The tool applies the strengthened
current verifier only after AST-checking that all other batch-module code is
identical; recorded source contracts are enforced without overrides. It checks
report-level initial baselines as well as traces, parameters and per-cell probes.
Only saved frozen-policy batch runs are supported, with the existing eight-cell
and 60-second invocation limits. No new rollout is sampled for training.

### Online comparison with matched update exposure

The `episode-normalized-v1` and `cell-normalized-v1` modes use the versioned
`same-player-td0-matched-cells-v2` contract. Both clip each rollout's joint actor
and critic gradient to norm one, weight it by `rollout_decisions / 8`, and spend
exactly 0.05 total learning-rate exposure per position/opponent block.

`episode-normalized-v1` applies each weighted contribution immediately; the
following independent rollout uses refreshed policy and value parameters.
`cell-normalized-v1` collects under frozen parameters, accumulates the same kind
of clipped, decision-weighted contributions, then applies them once. Neither
mode clips the accumulated result again. Policy/value refresh timing is the
intentional difference; trajectories and TD targets can consequently diverge.
Equal nominal exposure does not imply identical parameter distances or outcomes.

Use the existing `hnef.learning.batches train` and `verify` commands with either
new mode. Per-rollout weights, clipping scales and effective learning rates,
and the per-block exposure total, are recorded and checked during exact replay.
Resume binds the normalization, clipping, refresh timing and source fingerprints.
The old modes retain their own semantics for reproduction; no old checkpoint is
relabeled. The existing sample/time caps, training-only probes and strengthened
initial-baseline checks remain in force. A local optimizer identity does not
substitute for an online learning gate or heldout opponent evaluation.

### Fixed normalized training curve

The optional `--profile normalized-curve512-v1` explicitly extends only
`--mode cell-normalized-v1` to 64 blocks (512 learner decisions). The default
profiles keep their 16-block cap. The extended contract binds the profile and
reference schedule: eight starts against random, then the same eight against
heuristic, repeated four times. Global block indices keep rollout streams
independent across cycles. Resume cannot switch profiles or extend the cap.

Each invocation still accepts at most eight blocks and 60 seconds. For example:

```sh
python -m hnef.learning.batches train --data <training-cases.json> \
  --output <phase-1> --mode cell-normalized-v1 --seed 137 \
  --profile normalized-curve512-v1 --cells 8
python -m hnef.learning.batches train --data <training-cases.json> \
  --output <phase-2> --mode cell-normalized-v1 --seed 137 \
  --profile normalized-curve512-v1 --cells 8 \
  --resume <phase-1>/checkpoint-0008.npz
python -m hnef.learning.batches verify --data <training-cases.json> --output <phase-2>
```

Set the documented one-thread environment before running. Declare seeds,
checkpoints, aggregate CPU budget and final gate before collecting trajectories.
The reports expose raw-policy training probes at every block; intermediate
checkpoints are diagnostics, not opportunities to select a favorable model.
Verify every invocation and compare adjacent boundary checkpoints, including
metadata and RNG, before reporting a curve. Synthetic training improvement
alone cannot establish heldout generalization or competitive strength.

### Saved-curve gradient and critic diagnostics

`tools/diagnose_copenhagen_curve.py --run <saved-phase> --data
<training-cases.json> --output <new-directory>` accepts the fixed normalized
curve profile. Run with the matching trainer source on PYTHONPATH and the
one-thread environment. Each invocation has a 60-second limit and first verifies
all saved traces, parameters, probes and report-level baselines.

It reconstructs the weighted, clipped actor/critic update and checks it against
the saved parameter change. For each immediate-win training probe, it reports
the first-order winning-probability change and the observed change, identifying
whether the update came from that same start. These local projections can be
summed to inspect reinforcement or interference across starts; they are not
isolated causal effects or evidence of opponent strength.

Residual rows contain target, value and signed TD error. Only the final sample
of a terminal rollout is marked terminal; earlier samples remain bootstrapped.
Cutoffs have their own bootstrap category and are never labeled draws. These
sampled, changing-policy errors do not establish full-game value calibration.
The diagnostic does not write model checkpoints, train a new policy or read
heldout positions. Freeze the input phases and aggregate time cap before running
multiple invocations; report all seeds rather than choosing favorable results.
