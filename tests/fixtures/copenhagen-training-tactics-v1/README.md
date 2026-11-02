# Training-only tactical starts

Four synthetic boards, each with both movers, provide eight starts. Two king-capture
positions give the attacking learner two legal moves, one immediately winning;
two corner approaches expose defender escapes. Opposite movers supply legal
loss witnesses. All coordinates, legal-move counts, one-ply winning-move counts,
terminal reasons and reward signs are checked by `test_copenhagen_curriculum.py`.

`witness` proves only that the listed continuation legally reaches its stated
outcome. It is **not** a minimax label, a forced-loss claim, a training target or
evidence of reachability from the opening. Moves, witnesses and expected rewards
are ignored by the learner; actual sampled engine outcomes determine TD updates.
Source links cite ordinary Copenhagen capture/escape rules, not referee approval
of these authored boards. No fort/draw/repetition interpretation is added.

The constrained attacker branching is deliberate curriculum scaffolding. It does
not represent the branching or difficulty of full Copenhagen games. Material and
blocker arrangements are synthetic. Nonzero actor and critic updates from wins
and losses must be observed separately for each side; a fixture's intended outcome
alone does not satisfy that acceptance check.

The separate validation corpus and earlier public held-out corpus are excluded
under all board rotations/reflections. Do not use this training corpus as evidence
of generalization. Its paired movers are already explicit: the documented command
does not also pass `--both-sides`, which would duplicate each board/mover pair.
