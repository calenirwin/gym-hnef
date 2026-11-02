# Published adjudication research — 22 October 2026

Read-only research; no external people contacted. This is a source ledger, not
new rules. Current [primary rules](https://aagenielsen.dk/copenhagen_rules.php)
remain the baseline. Historical proposals cannot silently override that page.

| Question | Primary evidence located | What it settles / leaves open |
| --- | --- | --- |
| Moving-defence or non-solid exit forts | [Copenhagen forum, page 12](https://aagenielsen.dk/hnefataflforum/phpBB3/viewtopic.php?start=110&t=2), Hagbard, 30 September 2013 | A championship deadlock had no solid winning fort, yet attacking it would lose. The author identifies a gap and proposes defender-position repetition. This is evidence that tactical safety and a permanent wall differ; it is not a general current ruling that such positions win. |
| Repetition threshold and identity | Same page, Adam, 18 June 2013; Hagbard, 30 September; crust, 2 October | Discussion uses three whole-board occurrences, then proposes counting only white's configuration without captures. Support for the latter is historical, not an unambiguous current contract. Current rule 8 assigns repetition loss to white but omits identity/threshold details. |
| Dead-position meaning | [Copenhagen forum, page 16](https://aagenielsen.dk/hnefataflforum/phpBB3/viewtopic.php?start=150&t=2), Hagbard, 21 November 2022 | Reports agreed low-material draws after differing quiet intervals. These are examples of agreement, not a universal material bound, a forced-win theorem or an automatic quiet-move rule. |
| Terminal precedence | Same page, nath, 21 November 2018; Ytreza, 2 April 2026 | The posts raise draw/repetition conflicts; the 2026 example asks about a long-lived fort and describes personal alternatives. No authoritative complete precedence ordering was located. |

A [2019 repetition discussion](https://aagenielsen.dk/hnefataflforum/phpBB3/viewtopic.php?start=10&t=112)
also demonstrates competing accounts of who is threatening whom. It predates the
current defender-loss wording, so it should inform regression examples rather
than reintroduce an aggressor classifier.

[Abstract Play's own implementation description](https://abstractplay.com/wiki/doku.php?id=games%3Atafl)
says its exit-fort check waits until the end of the attacker's turn. That is useful
comparison evidence about another implementation, not a ruling requiring Hnef to
change. It does not establish completeness of either detector or resolve the
ordering of simultaneous outcomes. No code was copied from it.

## Versioned options requiring a decision

* Retain v2's board-plus-mover third-occurrence convention and explicitly document
  it in a future agreed stable contract, **or** specify defender-only no-capture
  repetition in a new profile. The historical latter proposal needs decisions on
  capture resets, sampling after whose turn, and treatment of the king. Neither
  option changes existing games here.
* For forts, retain sound stationary-wall proofs with unknown results, then seek
  independently adjudicated tactical/mobile cases. A more general defensive
  strategy solver needs a defined quantifier and resource bounds; timeout stays
  unknown. An opponent's losing attack is not itself an implementation defect.
* For draws, retain recorded external judgment unless a precise automatic rule is
  agreed. Separate no-winning-continuation proofs from neither-side-forces-win
  analysis. They are different predicates and may interact differently with
  repetition. No material shortcut or quiet-ply limit is introduced.
* Establish terminal precedence from concrete reachable examples before a new
  profile. V2's fort-before-repetition behavior remains a compatibility fact,
  not evidence that a referee sanctioned that order.

Research limitation: forum text was available through web retrieval, but the
2026 attachment click failed and direct forum retrieval returned HTTP 455.
Neither the 2013 diagram nor the 2026 position/history was transcribed. The corpus
records those as source-linked questions without invented board coordinates.
Failure to locate a definitive ruling is not proof that none exists.

See the [held-out corpus](../tests/fixtures/copenhagen-heldout-v1/README.md).
The existing published-fort diagrams remain calibration/regression fixtures, not
new independent held-out evidence. No experimental label or rule version changed.
