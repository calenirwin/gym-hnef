# Experimental Copenhagen engine

Select `copenhagen-experimental-v1` explicitly. **This is not yet the full
Copenhagen ruleset.** Exit-fort wins and dead-position draws are not adjudicated.
Games can therefore continue beyond their proper end; benchmark results are
engineering checks, not valid Copenhagen strength or balance measurements.

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
| Exit fort | **Pending** |
| Dead position | **Pending** |

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

Terminal checks run in this order: king capture/escape, enclosure, next-player
immobility, repetition. Collection limits leave a nonterminal state, reported as
`truncated`; they are not draws. A motionless edge king can still lose if its
entire side has no moves, even though the king is not captured.

Ruleset identity is part of both position and full-state digests. JSON replay v1
already stores it, so the format is unchanged. Existing mini-v1 replay bytes,
action IDs, digests, and seeded results remain compatible. A checked-in replay
from before this change verifies that compatibility. Future changes to Copenhagen
adjudication must use a new ruleset identifier.

## Running it

```sh
uv run hnef play --ruleset copenhagen-experimental-v1 --seed 42 --output /tmp/copenhagen.json
uv run hnef replay /tmp/copenhagen.json
uv run hnef benchmark --ruleset copenhagen-experimental-v1 --games 100 --seed 0
```

The CLI prints the remaining rule limitations for experimental games, including
when loading a saved replay. Python callers can use:

```python
from hnef.engine import Ruleset, initial_state

state = initial_state(Ruleset.COPENHAGEN)
assert state.ruleset.value == "copenhagen-experimental-v1"
```

## Remaining acceptance work

Add authoritative exit-fort fixtures and negative cases for breakable structures,
immobile kings, shield-wall exposure, and interior forts. Define a sound detector
and document its coverage before enabling fort adjudication. Dead-position
adjudication likewise needs explicit criteria; low material alone does not prove
that no winning continuation exists. Only then promote a complete rules profile
for Copenhagen training or multiplayer use.
