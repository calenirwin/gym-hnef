# Copenhagen on-policy review milestone — 22 October 2026

This is a genuine **on-policy TD actor–critic pilot on synthetic starts**, against
fixed random opponents. It is not a full-game strength result, complete-rule
certification or browser opponent release. The earlier supervised smoke remains
separate and unchanged.

## Measured result

| Measurement | Result |
| --- | --- |
| Training seed / starts | 17 / four training-only synthetic positions |
| Episodes / updates | 32 / 32 (28 nonzero-gradient updates) |
| Learner horizon | 4 decisions, each including the opponent reply if still live |
| Engine transitions | 246 |
| Terminal signal | Two defender wins, one attacker loss, all corner escapes |
| Cutoffs | 29 unfinished games; no draw labels |
| Training wall time, including startup | 2.312 seconds |
| Held-out evaluation wall time | 0.673 seconds |
| Final resumable checkpoint | 60,428 bytes; SHA-256 in `train/report.json` |
| Resume check | 32 uninterrupted versus 16 + 16: identical nonzero parameters, metadata and RNG |

Evaluation uses six held-out concrete positions and seeds 100000/100001, paired
between initialized and candidate policies, with no updates during evaluation.
All rotations/reflections of training boards are rejected from evaluation.

| Policy | Side | Wins | Losses | Draws | Unfinished |
| --- | --- | ---: | ---: | ---: | ---: |
| Initialized | Attackers | 0 | 2 | 0 | 4 |
| Candidate | Attackers | 0 | 2 | 0 | 4 |
| Initialized | Defenders | 0 | 0 | 0 | 6 |
| Candidate | Defenders | 0 | 0 | 0 | 6 |

**No improvement is demonstrated.** This is a small public diagnostic corpus,
not a hidden final benchmark. No tuning or checkpoint selection was performed
from these results. Training and evaluation boards are synthetic with no claimed
opening reachability. Existing teacher moves/outcome labels in fixture files are
ignored: the learner samples actions from its own masked policy, and all outcomes
come from the unchanged Copenhagen-v2 engine. Both policies play fixed random
opponents. The curriculum lacks broad attacker-winning signal.

## Exact estimator and cutoff semantics

One learner decision comprises its move plus one fixed-opponent reply unless a
true terminal result occurs first. The successor is therefore the **same player's
next decision**, not an opponent-perspective value. The policy/critic stay fixed
through each episode, followed by one gradient update over the collected decisions.

For each decision, with gamma = 0.99 per such decision:

* True terminal: target is the engine reward from the learner's perspective;
  bootstrap is zero, even if this occurs on the collection limit.
* Nonterminal, including a collection cutoff: target is gamma times the frozen
  critic's value of the actual final successor. There is no reset observation,
  invented terminal reward, sign reversal or automatic draw.
* Advantage is target minus the frozen current value. The actor uses the sampled
  action's log probability weighted by this detached advantage; the critic uses
  half squared target error. SGD learning rate 0.05, global gradient clip one.

This follows the termination/truncation distinction described in
[Gymnasium's time-limit guidance](https://gymnasium.farama.org/tutorials/gymnasium_basics/handling_time_limits/).
It is approximate TD learning, not an unbiased Monte Carlo return or a proof of
convergence. In particular, the inherited linear features include only the current
repetition count, **not full ordered history**; they are not Markov-complete for
Copenhagen. Full engine history is nevertheless preserved in each transition and
used for exact game adjudication. Unknown fort/dead-position cases remain unknown.

## Commands and artifact contract

NumPy is optional and was already installed; no framework/model download occurred.
Set CPU caps before Python starts. Each invocation has an internal 60-second check
at episode boundaries and was wrapped with a hard 65-second process timeout.
Training accepts at most 32 episodes per invocation, 64 cumulative updates and
8 learner decisions per episode. Evaluation accepts at most four reserved seeds.

```sh
export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 PYTHONPATH=src
timeout 65s python -m hnef.learning.onpolicy train \
  --data tests/fixtures/copenhagen-training-plumbing-v1/cases.json \
  --output /tmp/copenhagen-td --episodes 32 --horizon 4 --seed 17
timeout 65s python -m hnef.learning.onpolicy evaluate \
  --data tests/fixtures/copenhagen-training-plumbing-v1/cases.json \
  --heldout tests/fixtures/copenhagen-heldout-v1/cases.json \
  --checkpoint /tmp/copenhagen-td/checkpoint-0032.npz \
  --output /tmp/copenhagen-td-evaluation
```

Resume into a new output folder with `--resume <checkpoint>` and the same horizon.
Resume restores the saved seed and RNG; a new `--seed` does not reseed it. The
checkpoint uses **schema 2** with algorithm `same-player-td0-ac-v1`, distinct from
the schema-1 supervised smoke. Loading into the old optimizer is rejected. Rules,
features, ordered action codec, dataset, source and NumPy version are checked;
training also checks gamma/horizon. Same-runtime resume is tested, not promised
across platforms. Initialized and final checkpoints are retained here; intermediate
checkpoints remain in `/workspace/shared/copenhagen-onpolicy-schema2/train`.

`train/` and `evaluation/` contain every custom-start trace and reports. Traces
record board, mover, actions, full-state digests, opponent seed and explicit
terminal/cutoff status; the verifier reconstructs every engine transition and
checks final rewards/status. This is a separate curriculum trace format, not a
change to saved-game replay schemas. `execution.json` records actual commands,
outer timings and process caps. `resume-verification.json` records exact equality.

## Tests and remaining gates

Local final suite: 428 passed, two optional suites skipped; lint/format passed.
New tests cover cutoff bootstrap from a distinguishable final observation, reward
perspective, terminal wins/losses overriding horizon bootstrap, no opponent move
after a win, illegal-action probability, a finite-difference policy-gradient
check, negative advantages, exact resume with **nonzero** learned weights,
algorithm-contract rejection, trace integrity and train/evaluation separation.
No validation failures occurred in the final run.

Next substantive gate: improve training-only state coverage so both colors see
wins and losses, add full-history feature semantics, and compare multiple seeds
against initialized controls and stronger fixed opponents. Then assess longer
opening-start trajectories with measured cutoff rates and a separately frozen
final evaluation set. Do not tune on this public held-out diagnostic corpus.
There is no basis yet for a strength claim, stable rules label, UI integration or
more compute. This milestone is ready for review; no merge/deploy was performed.
