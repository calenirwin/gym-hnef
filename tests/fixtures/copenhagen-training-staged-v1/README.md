# Fixed constrained-to-broader training contract

This manifest contains the eight verified constrained starts followed by the eight
verified broader starts, copied unchanged except for `stage` and `stage_source`.
Tests compare both subsets with their original corpora, which retain their exact
terminal witnesses and separation from public diagnostic/validation boards.
Both movers are already explicit; do not add `--both-sides`.

`constrained32-broad32-v1` uses constrained starts for updates 1–32 and broader starts
for updates 33–64. `broad64-v1` uses only broader starts for all 64 updates. These
are fixed schedules, not adaptive mastery promotion. Both bind the entire manifest
and schedule ID into the checkpoint contract. Resume across update 32 keeps the
parameters and RNG; switching schedules or replacing the dataset is rejected.

Each active stage has eight starts, four per side. Mixed opponents alternate whole
passes through the active stage, preserving every start/opponent pair in each
32-update phase. The broader-only control gets twice as many broader-state updates
as the staged arm under the same total budget: this compares allocation strategies,
not an isolated causal effect with equal broader-state exposure.

Use `hnef.learning.curves --stage broad` to measure the same broader training subset
at every checkpoint. Its contract still hashes the whole manifest. Those declared
training positions do not become held-out evidence merely because a model has not
yet reached their training phase. Check winning/losing updates in the broader phase
separately; easy-phase coverage must not hide sparse later rewards. All positions
remain synthetic; no opening reachability, minimax or competitive-strength claim.
