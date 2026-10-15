# Exit-fort detection: a sufficient test

`prove_exit_fort(board)` returns an immutable `ExitFort` containing the king's
room and its defender boundary, or `None` when this algorithm cannot prove a fort.
It is a conservative structural test, not a general solver. `None` does not mean
the position is breakable. This implementation deliberately remains experimental.

## Construction

1. Require a king on a non-corner edge square and at least one currently empty
   adjacent square. A corner is handled by escape, not this detector.
2. Flood from the king through all non-defender squares. Reject the candidate if
   the resulting room contains an attacker or reaches a corner. Keep this room
   fixed throughout the analysis. Its adjacent defenders form the boundary.
3. Grant a hypothetical attacker unlimited pieces on **every** square outside
   the room except surviving defenders and restricted squares. This includes
   inaccessible pockets and ignores the opponent's actual piece count.
4. Find defenders that could be sandwiched using those hypothetical pieces and
   hostile restricted squares. Also find possible shield-wall captures, using
   the same capture geometry as the game engine.
5. If a boundary defender is vulnerable, return unknown. Otherwise remove the
   vulnerable outside defenders, granting their squares to the hypothetical
   opponent, and repeat until no additional defender can be removed.
6. The surviving boundary certifies the original room. Return both in sorted
   coordinate order for deterministic inspection.

There are at most twelve ordinary defenders in a reachable starting-game
position. Every nonfinal pass removes at least one, so the process terminates.
The implementation also handles larger custom test positions without a piece
count cutoff.

## Why the test is conservative

Hold the remaining defenders stationary. Every real attacker placement outside
the room is included in the hypothetical opponent's options. An attacker cannot
enter the room without first capturing a boundary defender; it cannot jump the
boundary. At the fixed point, neither ordinary nor shield-wall capture can remove
any remaining defender even under the stronger hypothetical opponent. Therefore
there is no first capture that could open the room.

The king has an actual move within the original room. King movement there cannot
expose the fixed defenders to an outside sandwich. Nor can it enable a shield
wall across a new king position: every adjacent square of a room square is either
inside the room or part of the boundary, so the opponent cannot supply the
required inward backing at the king's square.

The room must not be expanded after hypothetical captures. A newly opened square
could already contain an attacker. Re-flooding and then treating that square as
safe would erase the threat and could produce a false win.

This argument is specific to a permanent wall of stationary defenders. It does
not establish completeness for forts requiring active defensive rearrangement.
The stronger hypothetical opponent can also reject structures that are safe
against the actual number or location of attackers. Such positions continue
playing; the engine never converts unknown into a win or a draw.

## Evidence

Fixtures in `tests/fixtures/exit-forts.json` transcribe the two diagrams linked
from [rule 6b](https://aagenielsen.dk/copenhagen_rules.php). The fixtures retain
their individual source URLs. Tests rotate them to all four edges, close their
last gaps through legal moves, remove each essential wall piece, and distinguish
interior forts and immobile kings from exit forts. Further tests enumerate every
legal attacker move against both certified structures and exercise king mobility
within the protected room.

A breakable fixture demonstrates why checking only the current turn is not
enough: two attacker moves flank an initially safe defender and open the wall.
An outside defender that can be sacrificed does not invalidate a separately
permanent wall.
