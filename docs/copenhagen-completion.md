# Copenhagen completion audit — 22 October 2026

Audit base: `52563a78b37fa2e367b5fe69ad2c294836b38722`. Scope is correctness,
opponent infrastructure and evidence, not a public release or a rename.

## Rules and evidence

Primary reference: [Aage Nielsen's Copenhagen rules](https://aagenielsen.dk/copenhagen_rules.php),
read 22 October 2026, page updated 21 August 2026. The
[printable rules](https://aagenielsen.dk/Copenhagen_Hnefatafl_11x11.pdf)
(January 2025, section 3.1) explicitly require a flanking shield-wall move.
The web prose alone is less explicit on that last condition.

The following describes code inspection and executable coverage, not independent
certification. `C` means `tests/test_copenhagen.py`, `A` means
`tests/test_adjudication.py`. All ordinary-rule C fixtures now run under **both**
v1 and v2; previously they exercised v1 only. Existing fort fixtures are in
`tests/fixtures/exit-forts.json`, with diagram URLs and explicit expected rooms.

| Published rule | Implementation | Evidence / remaining qualification |
| --- | --- | --- |
| 1, opening | `state.initial_state` | C `test_copenhagen_opening`: size, material, king, corners, codec/mask |
| 2, alternating turns; attackers first | `initial_state`, `rules.apply_move` | opening plus replay fixtures and transition tests in `test_engine.py` |
| 3, sliding movement | `rules.legal_moves` | C throne traversal; `test_engine.py` movement/blocking; no jumping |
| 4a, ordinary capture and armed king | `rules.apply_move` | C corner support and simultaneous capture; `test_engine.py` sandwiches, king participation, moving between enemies |
| 4b, shield walls | `copenhagen.shieldwall_captures` | C both ends/four edges, corner support, backing gaps, king participation/immunity, backing-only and unrelated moves |
| 5, restricted squares and hostility | `variants.restricted`, `legal_moves`, `apply_move` | C corner/throne occupancy and crossing; shared engine hostility tests; edge is not a capture support |
| 6a, corner escape | `apply_move` | C each rotated corner versus ordinary edge |
| 6b, exit fort | `forts.prove_exit_fort`, v2 only | A published diagrams on four edges, legal closing moves, broken walls, mobility, each possible attacker reply. **Partial:** permanent stationary walls only |
| 7a, king capture | `apply_move` | C four sides/three beside throne, two-sided noncapture, edge immunity |
| 7b, enclosure | `copenhagen.defenders_encircled` | C closed/open rings and disconnected groups; flood fill is documented implementation interpretation |
| 8, perpetual repetition | `apply_move` history | C third occurrence for either final mover. **Convention:** three equal boards with equal mover, opening included; prose supplies no threshold |
| 9, inability to move | `apply_move`, `legal_moves` | C trapped edge king and last attacker removed |
| 10, dead position | `adjudicate_draw`, `Replay.adjudicate_draw` | A external judgment, recorded note, integrity checks, low-material counterexample. **No automatic proof** |

No confirmed ordinary-rule transition defect was found in this pass. The tractable
fix is the missing current-profile regression coverage. No engine semantics,
ruleset IDs, replay schemas, UI or default opponent changed. Checked-in old replays
still verify byte-for-byte. A future adjudication change must introduce a new
profile, retain v1/v2 readers and record the selected profile in model manifests.

## Precise unresolved questions

* Exit forts: does “impossible ... to break” include optimal *moving* defensive
  responses, or only permanent structures? Rule 6b does not formalize that
  quantifier. The current proof holds defenders still and grants unlimited
  hypothetical outside attackers. It may reject a fort safe against the actual
  force or requiring defensive rearrangement. `None` is unknown, not breakable.
  Obtain independently adjudicated examples of both classes before widening it.
* Dead positions: does “not possible to end” mean no legal continuation can win,
  or neither side can force a win? How does this interact with mandatory
  repetition loss? Rule 10 gives no executable material/quiet-ply threshold.
  An agreed draw is a human judgment, not proof of either interpretation.
  A graph solver needs this definition, full repetition context, terminal
  precedence, and an explicit unknown result on resource exhaustion. Do not
  train on cutoff-as-draw labels or assume low material is sufficient.
* Repetition: confirm the threshold, whether consecutive cycles are required,
  and whether the side to move belongs to identity. The current third-occurrence
  convention is testable and versioned, but is not specified by rule 8.
* Simultaneous terminal conditions: confirm fort versus immobility/repetition
  precedence. V2 checks capture/escape, fort, enclosure, immobility, repetition.
  The source is not an executable ordering specification.

These are targeted questions for an experienced Copenhagen referee or rules
maintainer. No request has been sent and no outside agreement is assumed.

## Gate for removing experimental

1. Resolve the questions above in a dated, cited rule contract. Distinguish a
   complete automatic profile from a profile needing a referee; approval of a
   documented convention alone does not prove completeness.
2. Assemble independent positive/negative fort and dead-position fixtures,
   including mobile walls, attacker-count constraints and repetition conflicts.
   Require sound proofs or explicit referee decisions, never heuristic scores.
3. Implement/version the resulting adjudication; measure false positives and
   unresolved cases. Unknown cannot silently become a win/draw. Document a
   playable resolution path for every supported case.
4. Pass rule fixtures on every edge, replay compatibility, PettingZoo reward/mask
   checks, browser/Python parity and full CI. Have another qualified reviewer
   inspect the proof and ambiguous cases, not just the screenshots.
5. Only then choose a stable identifier and migrate opt-in new games. Keep old
   games/models bound to their original rules; do not relabel old evaluations.

Until then the public experimental label stays. This is a work gate, not a
promise that generic dead-position solving is inexpensive.

Follow-up: [published adjudication research](copenhagen-rulings-research.md)
records historical alternatives and current unanswered questions; it does not
silently promote them into rules. The [held-out corpus](../tests/fixtures/copenhagen-heldout-v1/README.md)
separates concrete immediate expectations from unresolved adjudication.
