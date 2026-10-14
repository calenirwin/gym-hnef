# Project mini-v1 rules

This is the project's 5×5 development variant, not Copenhagen or a claim about
historical tafl. It preserves the 2021 opening and edge-escape objective while
making previously inconsistent behavior explicit. Coordinates are zero-based
`(row, column)`, starting at the top left.

```text
. A . A .
A . D . A
. D K D .
A . D . A
. A . A .
```

- Attackers move first. Each turn moves one own piece any positive orthogonal
  distance through empty squares. Pieces cannot jump or move diagonally.
- Only the king may stop on the center throne. Other pieces may cross an empty
  throne. Corners are ordinary squares in this variant.
- Ordinary pieces are captured when an opponent moves next to them, sandwiching
  them against another opponent piece. The king participates as a defender.
  Multiple captures are simultaneous. Moving between enemies is safe.
- The throne supports captures against attackers whether occupied or empty.
  It supports captures against defenders only when empty. Edges and corners
  do not support captures.
- On the throne the king requires four adjacent attackers to be captured.
  Next to the throne it requires attackers on the other three sides. Elsewhere,
  two opposite attackers suffice. A capture must be completed by an attacker
  moving next to the king; the king cannot capture itself.
- The defenders win immediately when the king moves to any edge. The attackers
  win immediately upon capturing the king. Those wins take priority over other
  termination conditions.
- If the next player has no legal move, that player loses.
- The third occurrence of the same board **and side to move** is a draw. The
  opening counts as the first occurrence. This is a deliberate development rule,
  not Copenhagen's repetition adjudication.
- A benchmark's ply cap truncates collection. It is not an engine draw or win.

The immutable state includes repetition history. `position_key` identifies a
board, player, and rules version for repetition; `GameState.key` also includes
history, ply, and outcome for replay/search. Rewards explicitly take a player:
winner +1, loser −1, draw 0. There is no implicit reward tied to the next turn.

The generated action order is versioned with the replay schema: row-major source
squares followed by row-major destinations in the same row or column, excluding
the source. This gives 200 actions at 5×5 and 2,420 at 11×11. The 11×11 codec does
not imply complete Copenhagen support; see [experimental coverage](copenhagen-rules.md).

`GameState` can also represent custom test positions. Callers must supply a
nonterminal position with a movable side, or its explicit terminal outcome;
automatic adjudication happens when applying a move. Replay v1 starts from the
canonical opening and validates every move and full-state digest.
