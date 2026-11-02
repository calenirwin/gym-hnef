# Synthetic training-only plumbing dataset

Four independently authored one-move terminal transitions: two corner escapes
and two four-sided king captures. Rules are Copenhagen experimental v2. These are
custom positions with no claimed trajectory from the opening. They are **not**
held-out evaluation data and must not be used to measure competitive strength.

The moves are supplied teacher actions. The loader verifies legality and actual
engine terminal outcomes, and computes value targets from the mover's perspective.
All four targets are positive: this tiny dataset intentionally demonstrates
policy/value optimization, not balanced value calibration or on-policy RL.

No positions were copied from the held-out corpus. A test rejects equality under
all rotations/reflections with every concrete held-out board. Sharing basic rules
such as corner escape is unavoidable; it does not make these independent expert
adjudications. Do not use the held-out corpus as a curriculum or optimization input.
