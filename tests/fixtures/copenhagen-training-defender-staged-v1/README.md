# Intermediate defender curriculum

This training-only manifest has three eight-start stages. Constrained and broader
records are copied unchanged from `copenhagen-training-staged-v1`. The intermediate
stage retains the four attacker starts and the two defender loss starts unchanged.
Only the defender corner-escape starts gain a defender at (1,1) or (1,9), using
zero-based row/column coordinates. Their branching is 13 and 32 legal moves, between
constrained (2/21) and broader (17/38). Exact winning/loss witnesses and all initial
legal/winning-action counts are checked; added blockers are synthetic teaching
material, not an opening-reachability claim or forced-outcome labels.

`constrained16-intermediate16-broad32-v1` allocates 16/16/32 updates. The existing
`constrained32-broad32-v1` control can use this same complete manifest and ignores
the intermediate records. Both therefore share data, rule and encoding fingerprints,
but have different explicit curriculum identities. Neither can resume as the other.
The first 16 updates are identical under matched seeds; regression tests compare
actual parameters and RNG. Each stage crosses every start with both references.

The loader's explicit maximum is 24 concrete positions to accommodate this fixed
manifest; the 64-update, eight-decision and time/thread caps are unchanged. Curves
use `--stage broad` for both arms and retain the entire manifest fingerprint. Use
separate broader-phase outcome coverage: earlier easy wins cannot satisfy that gate.

Tests exclude all board rotations/reflections of previously viewed diagnostic and
validation sets. This is training-only data in the same authored tactical families,
not independent evaluation. No strength claim follows from fitting it. Keep experiment-specific results in review notes rather than this fixture guide.
