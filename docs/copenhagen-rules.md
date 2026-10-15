# Experimental Copenhagen engine

Select `copenhagen-experimental-v2` explicitly. It adds conservative exit-fort
detection and recorded external draw adjudication. **This is not a complete
automatic Copenhagen adjudicator.** Some valid forts may not be recognized, and
dead positions are not inferred automatically. Benchmark results remain
engineering checks, not valid Copenhagen strength or balance measurements.

`copenhagen-experimental-v1` remains available with its original behavior and
replay digests. The Python constant `Ruleset.COPENHAGEN` still means that old
profile; use `Ruleset.COPENHAGEN_V2` for the new behavior.

## Rule references and coverage

The baseline is [Aage Nielsen's published rules](https://aagenielsen.dk/copenhagen_rules.php),
read October 14, 2026 (page dated August 21, 2026), with the
[linked printable rules](https://aagenielsen.dk/Copenhagen_Hnefatafl_11x11.pdf),
section 3.1, clarifying that a shield-wall capture requires a flanking move.
We do not mix in other websites' repetition or draw variants.

| Rule | Implementation |
| --- | --- |
| Opening | 11×11, 24 attackers, 12 defenders and king; attackers first |
| Movement | Orthogonal sliding; king alone may occupy throne/corners; empty throne may be crossed |
| Ordinary capture | Sandwiches closed by the mover; hostile corners/throne; armed king; simultaneous removal |
| King | Corner escape; four attackers capture, or three beside the throne; immune on an edge |
| Shield wall | At least two contiguous edge pieces, bracketed and backed; final move at an end; king survives within the captured row |
| Enclosure | Attackers separate every defender, including king, from all edges |
| Repetition | Defender loss; exact software threshold specified below |
| Immobility | Player without a legal move loses |
| Exit fort | V2: permanent-wall proof, including both published diagram fixtures; conservative, not complete |
| Dead position | V2: explicit referee/player draw judgment with a saved explanation; no material-count heuristic |

## Software conventions

The prose rule forbids perpetual repetition without an executable threshold.
This version adjudicates the **third occurrence of a board and side to move**
as an attacker win (`defender_repetition`), regardless of who made the final move.
The opening counts. This convention is versioned so later adjudication changes
cannot silently alter recorded games.

Captures are collected from one post-move board before any piece is removed.
Enclosure uses a flood fill from all non-attacker edge squares, traversing empty
and defender squares. If any defender is reached, enclosure has not occurred.
This checks disconnected defender groups and does not mistake a defender for part
of the attackers' wall.

In v2, terminal checks run in this order: king capture/escape, proven exit fort,
enclosure, next-player immobility, repetition. V1 retains its original order
without the fort check. Collection limits leave a nonterminal state, reported as
`truncated`; they are not draws. A motionless edge king can still lose if its
entire side has no moves, even though the king is not captured.

Ruleset identity is part of both position and full-state digests. Unadjudicated
games retain JSON replay schema v1. A game with an explicit draw judgment uses
schema v2, adding one adjudication event and a final state digest. It does not add
a fictitious move, advance the ply, switch players, or change repetition history.
Checked-in mini and Copenhagen-v1 replays verify byte-for-byte compatibility.
Further changes to automatic adjudication must use a new ruleset identifier.

## Running it

```sh
uv run hnef play --ruleset copenhagen-experimental-v2 --seed 42 --output /tmp/copenhagen.json
uv run hnef replay /tmp/copenhagen.json
uv run hnef benchmark --ruleset copenhagen-experimental-v2 --games 100 --seed 0
```

The CLI prints the remaining rule limitations for experimental games, including
when loading a saved replay. Python callers can use:

```python
from hnef.engine import Ruleset, initial_state

state = initial_state(Ruleset.COPENHAGEN_V2)
assert state.ruleset.value == "copenhagen-experimental-v2"
```

## Draw adjudication

The rule source provides no executable material threshold for dead positions.
Its [rules forum records agreed low-material draws](https://aagenielsen.dk/hnefataflforum/phpBB3/viewtopic.php?start=150&t=2),
including different numbers of quiet rounds. A count of pieces or quiet moves is
therefore not treated as proof. Low-material positions can still contain a
winning escape or capture.

After a referee decision or both players' agreement, record it with:

```sh
uv run hnef adjudicate-draw /tmp/copenhagen.json \
  --note "Both players agreed the blocked endgame is a draw" \
  --output /tmp/copenhagen-adjudicated.json
```

This command **records external judgment**; it does not verify that the position
is dead or that the players agreed. The calling application is responsible for
that decision. A nonempty explanation is required, and a completed game cannot
be overwritten. Python callers can use `Replay.adjudicate_draw(note)` or
`adjudicate_draw(state)`; the replay method preserves the note. This makes draws
usable for a future referee/multiplayer flow without inventing an automatic rule.

## Remaining acceptance work

The [fort proof and limitations](fort-proof.md) describe the algorithm and its
test evidence. Expand coverage with additional independently adjudicated fort
positions, especially those requiring mobile defenders. A complete automatic
dead-position detector still needs a precise adjudication convention or solver;
the explicit judgment event is not a substitute for that proof. Keep full
Copenhagen training and tournament claims gated on those remaining limitations.
