# Copenhagen held-out corpus v1

`cases.json` contains six executable, authored rule examples and three unresolved
research records. This is a small initial corpus, not a Copenhagen certification
suite or a trained opponent strength benchmark.

Five transition examples have concrete immediate outcomes/cell assertions derived
from the cited rule: corner escape despite low material, king capture beside the
throne, shield-wall king immunity, occupied-throne protection, and voluntary entry
between attackers. The sixth has a known v2 continuation but **unknown** general
adjudication. `expected_outcome: null` means the supplied move does not terminate
under v2; it must never be exported as a draw label or a solved game value.

Each case supplies source, rule, origin and split. Coordinates are zero-based
row/column, from the top left. Initial history is the single supplied position;
these are constructed tactical states, not claimed legal opening-to-position
replays. Prescribed moves test adjudication; they are not exhaustive optimal-move
labels. No independent referee has approved the authored positions.

Research records intentionally have no board. They capture exact questions with
source/date locators, missing evidence and an explicit unknown adjudication.
Do not score them as passes for a full rules implementation or train on them.

Held-out means excluded from training/curriculum/model selection; these cases have
not been used in any training run. They remain public regression fixtures, so a
future credible strength evaluation needs a separate frozen final test set. The
older `exit-forts.json` published examples are existing calibration evidence and
are not counted as new held-out cases. Future additions must preserve provenance,
record independent adjudication where available, and keep unknowns separate.

Run `pytest tests/test_copenhagen_corpus.py`. No fixture modifies saved games,
engine transitions, public UI, ruleset IDs or replay formats.
