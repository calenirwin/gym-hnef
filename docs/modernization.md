# Modernization checkpoint

## Where the earlier work stopped

The October 2026 conversation **Plan RL Web App** reported a new mini engine,
CLI/replay support, tooling, 42 passing tests, two expected legacy failures,
and 1,000 replayed random games. Publishing was blocked by GitHub 403 errors.
The saved code artifact was not available in this local checkout. On October 14,
the remote contained only `main` at `5959eeb` and none of the reported commits.
Those previous test results are historical reports, not validation of this code.

This increment reconstructs the missing foundation from the repository and the
agreed design. It is a new implementation, not a recovered copy of those commits.

## Available now

- Dependency-free immutable mini engine under `src/hnef/engine`.
- Explicit mini-v1 rules, including throne/king captures, escape, blocked players,
  and repetition. The school code and assets remain in their original locations.
- Stable position/state identities, explicit winner-based rewards, seeded random
  games, versioned JSON replay verification, and CLI benchmark commands.
- Generated action vocabulary and legal masks, with 200 mini actions and a
  tested 2,420-action vocabulary ready for an 11×11 engine.
- Isolated modern package, locked development dependencies, automated rules,
  regression, replay, and CLI tests, and CI.
- Opt-in experimental Copenhagen: opening, movement, restricted squares, corner
  escape, strong king capture, shield walls, enclosure, repetition, and no-move
  losses. Versioned state/replay support keeps mini-v1 compatible.
- Copenhagen experimental v2 adds a conservative permanent-wall exit-fort proof
  and explicit draw adjudication with a recorded explanation. V1 remains replay-compatible.

## Current milestone: Copenhagen rules

The current engine covers the core rules and now recognizes permanent exit forts
using a conservative proof. Draws can be recorded after external adjudication;
they are not guessed from material or quiet-move counts. The profiles remain
explicitly experimental: the fort detector is not complete, and automatic
dead-position recognition is unsolved here. Details, sources, and limitations are
in [Copenhagen coverage](copenhagen-rules.md) and the [fort proof](fort-proof.md).

1. Opening, corner escape, restricted/hostile corners and throne.
2. Four-sided king capture and edge behavior.
3. Shield-wall captures, including king immunity and simultaneous captures.
4. Encirclement, edge forts, repetition adjudication, blocked players, and dead positions.
5. Deterministic games and replay verification under a separate ruleset/version.

Do not call an incomplete adjudicator full Copenhagen. Keep old profiles reproducible.
Next, the PettingZoo AEC adapter and API compliance tests can use mini-v1 as the
stable baseline while preserving the experimental status of Copenhagen profiles.
Heuristic and pure MCTS baselines follow, then a small PyTorch policy/value network and self-play.
GPU training and the Albion-inspired human/AI web game follow those milestones.

## Validation of this increment

Locally verified on Python 3.12.3 on October 14, 2026:

- 74 pytest cases passed; Ruff lint and formatting checks passed.
- 1,000 random games, seeds 0–999, with a 500-ply collection limit: 22,019 total
  plies, 937 defender wins, 62 attacker wins, one repetition draw, zero truncations.
  Every recorded full-state digest matched when replayed from the opening.
- Locked environment sync, wheel/source builds, and CLI save/load passed.
  Both distributions exclude the legacy notebooks and model weights.
- The foundation was pushed as `efa1531` to `codex/modernize-engine`.
  [GitHub Actions passed on Python 3.11–3.13](https://github.com/calenirwin/gym-hnef/actions/runs/38623861231).

Random-play results describe this development variant and these policies only;
they do not establish Copenhagen balance or the strength of a trained agent.

### Experimental Copenhagen increment

- 145 tests pass, including rotated rule fixtures, both shield-wall closing
  directions, king immunity/participation, simultaneous captures, disconnected
  enclosures, repetition by either mover, and old mini replay compatibility.
- 100 Copenhagen random games (seeds 0–99, 500-ply cap), all replay-verified:
  39,848 plies; 51 defender wins, two attacker wins, 47 collection truncations.
  These results use incomplete adjudication and are not Copenhagen balance data.
- Ruff checks, locked sync, and source/wheel builds pass locally on Python 3.12.
- CI now also exercises a short Copenhagen replay benchmark on each Python version.

### Fort and draw-adjudication increment

- 197 tests pass. Both published fort diagrams are transcribed with source URLs,
  tested on all four edges, and challenged with every legal attacker move in
  threat positions. Negative cases cover missing walls, interior forts, immobile
  kings, attackers inside, and walls that can be broken in multiple moves.
- Explicit draw events preserve board, turn, ply, and repetition history, require
  an explanation in a replay, and cannot replace an existing terminal result.
- Frozen mini-v1 and Copenhagen-v1 replay fixtures remain byte-for-byte compatible.
- 100 v2 random games, seeds 0–99 and a 500-ply cap, replay-verified: 39,848 plies,
  51 defender wins, two attacker wins, and 47 truncations. These random games did
  not exercise fort wins; the targeted fixtures provide that evidence.
- CI exercises all three rule profiles on Python 3.11–3.13.

## Legacy issues to avoid

- `simulate_step()` checks terminal status on the old state while returning the
  successor; the modern engine computes the outcome on the resulting board.
- The old environment changes turns before deriving reward; modern outcomes
  accept the desired player's perspective explicitly.
- Legacy king capture can apply ordinary two-sided capture beside the throne;
  mini-v1 tests require the special three-attacker enclosure there.
- Python's process-dependent `hash()` and identities without side-to-move are
  unsuitable for persisted replay or repetition keys.

The old TensorFlow weights are historical artifacts, not a modern model baseline.
