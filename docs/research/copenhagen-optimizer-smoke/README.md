# Bounded Copenhagen optimizer plumbing smoke — 22 October 2026

This is **supervised policy/value fitting on synthetic terminal transitions**,
not a reinforcement-learning run, playable model release or strength result.
The full-game probe (four seeds, 128-ply cap) produced 512 transitions, four
cutoffs and no terminals in 1.262 seconds. No cutoff received a draw target.
The fallback uses four independently authored terminal transitions from
`tests/fixtures/copenhagen-training-plumbing-v1/cases.json`, outside the held-out
corpus. Their legal one-ply outcomes are engine-checked; opening reachability is
not claimed. Tests exclude rotation/reflection duplicates of held-out boards.

## Actual run

* NumPy 2.3.5, CPU only; OpenBLAS/OMP/MKL thread caps set to one **before** Python.
* Four synthetic transitions, four terminals, zero cutoffs; four SGD updates,
  batch size two, seed 7, learning rate 0.05, global gradient norm clipped to one.
* End-to-end wall time including Python startup: **0.673 seconds**, under an
  external 35-second process timeout. Runner also checks a 30-second budget at
  step boundaries. APIs cap dataset size at eight transitions and total updates
  at eight, including resumed steps.
* Full-dataset policy/value loss: **4.718330 → 4.391950**. Illegal probability
  mass: **0**. All measured loss/gradient values and parameters finite.
* Final resumable checkpoint: **10,084 bytes**. Initialized, step-two and step-four
  checkpoints are retained here with report/config/source/codec/data fingerprints.
* Independent four-step versus two-plus-two execution produced bitwise-identical
  actor, critic, metadata and RNG state in the same runtime. Every training step
  also reloads its checkpoint and checks parameter equality. A timed repeated run
  is the artifact retained here. These repeat/resume checks are validation runs,
  not additional claimed training improvement.
* Final-run validation failures: **none**. Development caught and corrected one
  multiline-condition syntax error before the successful run. Tests deliberately
  exercise invalid masks, nonfinite gradients/parameters, contract mismatch,
  overwrite refusal and budget/thread guards; their expected exceptions are not
  experiment failures.

`report.json` contains losses, sampled indices, output probabilities, checkpoint
hash and per-run time. `execution.json` records the outer timing/caps and probe.
Checkpoint files use NPZ without pickle, plus JSON metadata for SGD configuration,
step, seed and NumPy RNG state. Load checks rules, feature version, ordered codec,
dataset, source and NumPy fingerprints; cross-runtime bitwise equivalence is not
promised. Source changes intentionally require a fresh run.

## Model and loss

The small implementation has a linear masked 2,420-action softmax head and a
linear tanh value head over 367 inputs: A/D/K planes, mover, normalized ply,
current repetition count and bias. The feature ID is
`copenhagen-linear-board-repetition-count-v1`. Unlike the collection record, this
model does **not** consume full ordered history and is not a Markov-complete
Copenhagen agent. No mini weights are reused. Initial weights are zero; SGD fits
teacher-action cross entropy plus half squared terminal-value error. This is
supervised optimization, not an actor–critic policy-gradient estimator. All four
positive targets make it especially unsuitable for general value calibration.

```sh
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 PYTHONPATH=src \
  timeout 35s python -m hnef.learning.plumbing \
  --data tests/fixtures/copenhagen-training-plumbing-v1/cases.json \
  --output /tmp/copenhagen-plumbing --steps 4
```

To verify resume, run two steps to a new folder, then another two to a different
folder with `--resume <first-folder>/checkpoint-0002.npz`. Use the matching source
commit and NumPy version when loading the retained checkpoint. No network model
or framework downloads, paid compute, new account, deployment or UI change.

## Next substantive gate — stop here for this task

Choose a genuine on-policy learning estimator and a training-only curriculum with
varied terminal returns, or specify and test truncation bootstrapping rigorously.
The existing Monte Carlo proposal suffers heavy cutoff censoring on Copenhagen.
Before scaling, demonstrate terminal learning signal for both colors, legal
masking, full-history feature semantics, deterministic resume and initialized
controls on data excluded from final evaluation. Then compare multiple training
seeds against fixed baselines with side-specific terminal/cutoff and latency
reports. This smoke proves optimizer/checkpoint plumbing only and admits no
learned opponent to the registry or browser. Experimental rules remain unchanged.
