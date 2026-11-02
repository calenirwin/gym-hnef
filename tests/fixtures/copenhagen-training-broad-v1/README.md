# Broader-branching training starts

These eight starts are controlled perturbations of
[`copenhagen-training-tactics-v1`](../copenhagen-training-tactics-v1/README.md).
Each adds an attacker at (8, 8) and a defender at (8, 2), using zero-based row/column
coordinates. Both movers remain explicit. The added pieces create distractor
moves without changing the verified legal capture/escape witness. Tests check the
exact perturbation, increased branching, witnesses and symmetry separation from
both previously viewed public diagnostic/validation sets.

There are 17–89 legal actions per start. Four starts offer an immediate winning
move for the learner; the opposite-mover starts have legal loss witnesses but no
initial immediate win. Witnesses do not classify forced outcomes and are not read
by training or curve diagnostics. These are synthetic teaching positions, not
claims of opening reachability or independent generalization examples.

Use this set for training-only learning curves. The additional branching tests
whether sparse winning feedback remains useful beyond the earlier constrained
attacker starts. Do not interpret probability of an immediate winning action as a
full-game win rate. Preserve separate observations of terminal update coverage,
cutoffs and both learner sides. Freeze any decision to proceed to new evaluation
before viewing the curves; never tune on an already viewed public validation set.
