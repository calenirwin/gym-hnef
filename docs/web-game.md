# The local playing table

Run `uv run hnef web` and open <http://127.0.0.1:8000>. The server binds only to
loopback. `--port` selects a different port. There are no additional runtime dependencies, external font services, analytics,
or frontend build steps. Artwork and fonts are bundled locally; see
[the Albion art direction and generation prompts](albion-ui.md).

## Learn

Open **Learn** or visit `/learn` for fourteen topics grouped by method family:
search and baselines, value learning, policy gradients, model-based learning,
self-play and populations, current research, and live experiments. The overview
explains how these choices combine; this is not a difficulty ladder ending at
AlphaZero. The three existing bots remain the only playable policies.

Topics cover random, heuristic, minimax/alpha–beta, MCTS, policy/value networks,
DQN/Rainbow, PPO, AlphaZero/Gumbel search, MuZero/EfficientZero, Dreamer, and
population training. The research page links dated primary sources, including
Dreamer 4, DiscoRL, and the 2026 regularized policy-optimization paper. Research
claims remain tied to their original benchmarks; applications to Hnef are
explicitly candidate experiments. The selected sources were checked on
October 16, 2026; this is not an exhaustive or automatically updated leaderboard.

Interactive teaching examples cover uniform sampling, the actual heuristic
formula, minimax replies, alternating-player value updates, both signs of PPO's
clipped objective, and a synthetic opponent mixture. They do not train networks.
Copy follows the Albion language guidance, with secondary detail in disclosures.

**Experiments** (`/learn#experiments`) runs the actual random, heuristic, and MCTS
policies on either opening or a snapshot of the browser's saved game. It reads
but never writes the save. The seed, simulation budget (8/32/64), and rollout
limit (0/12/24/48) are adjustable. Results include a position diagram, selected
moves, heuristic evaluations of their successors, explored root moves, MCTS
visits/mean utilities, and rollout cutoffs. No win-rate or strength ranking is
inferred from a single position. Settings changes invalidate displayed results.

**Download analysis** exports the verified replay, position identity, policy
version, seeds, budgets, decisions, and search statistics as JSON. It is an
analysis report, not a replay file; its `position.replay` contains the canonical
replay. **Analysis JSON** exposes the same report for inspection and copying.
Completed or collection-limited games cannot be analyzed. Imported draw
judgments and invalid histories are respected by engine replay validation.

Topics have shareable fragment links, previous/next navigation, a compact mobile
topic selector, and a readable no-JavaScript fallback. **Play against** opens game
setup with the selected opponent and preserves the game until **Play** is pressed.
As implementations arrive, update availability labels and add actual training
configuration and reproducible evaluations rather than implying that reading
material is an installed model.

## Play

- **Mini tafl** is the stable 5×5 development ruleset. **Copenhagen** selects the
  experimental v2 11×11 profile, with the same limitations documented in the engine.
- Play as attackers or defenders, play both sides locally, or watch two bots.
  Open **+** for setup. Changes apply when you select **Play**, replacing the current
  browser save. Download the current replay first if you want to keep it.
- Drag a piece to a marked destination, or click a piece and a highlighted destination. The selected piece can be clicked
  again to cancel. Arrow keys navigate cells, Enter selects, and Escape cancels.
- Random, heuristic, and MCTS use the existing Python policies. Interactive MCTS
  uses 32 iterations and 24 rollout plies. Seed and ply determine each bot decision,
  so retries and takebacks preserve the same decision at the same position.
- Pause/resume controls automatic bot moves; the right arrow steps one bot move
  while paused. An already running decision finishes before a pause takes effect.
- Open the history icon for the journal. Entries and left/right arrows inspect earlier
  positions. The board then shows a replay scrubber and previous/next controls.
  Left/Right step through history; Home/End jump to either end. **Return to game** (the skip-forward icon beside the turn cue) returns
  to the current game. Human play resumes automatically; watched matches remain
  paused until resumed. Reviewing never discards later moves.
- **Undo turn** (the backward curved arrow) removes your last move and the bot reply, or one ply for
  local and watched games. This branches the actual game and discards later moves.
- **Save replay** (the download icon) exports the complete current game's canonical JSON replay,
  even when inspecting an earlier position. Verify it with `hnef replay <path>`.
- **Open replay** inside the journal loads a JSON file; **Paste replay** accepts the same JSON text.
  The engine verifies every action and state key before replacing the current game.
  Imported games open in review with local two-player control available on return.
  Legacy Copenhagen v1 and recorded v2 draw judgments retain their original semantics.
- The crown beside a finished result opens an illustrated summary, rematch, and review.
  Piece motion respects the system reduced-motion preference. Hidden tabs suspend bot scheduling.

The browser remembers one game, its setup, and board orientation. A reload
reconstructs the game through the engine, including any imported draw judgment. Watched matches resume paused.
Connection failures offer Retry and never replace a valid save with a fresh game. If
browser storage is unavailable, play still works and the UI suggests downloading
the replay. The save belongs to this browser and server address; there is no account
or cross-device synchronization.

## Implementation and scope

`src/hnef/web/` contains a standard-library HTTP server and plain HTML/CSS/JS.
`POST /api/game` accepts a versioned ruleset and action history, reconstructs the
complete immutable state, optionally applies a human or bot action, and returns
legal moves, outcome, capture journal, and replay. The browser never duplicates
movement, capture, repetition, or winner logic. This preserves the same engine
behavior as CLI simulation and training.

`POST /api/analyze` accepts a ruleset and action history, or a canonical replay,
plus a seed and the bounded search settings above. It uses the same origin and
request-size checks as game requests. It reconstructs a verified immutable state
and compares policies without applying any chosen move. Defaults match the
playing table (32 simulations, 24 rollout plies), including decision seeding.

`GET /api/replay` reconstructs an action history and returns its canonical replay
as a JSON attachment. The UI exports the engine-provided replay through a local Blob
so recorded adjudication metadata is preserved and saving needs no extra server request.
The `import` operation validates canonical replay versions 1 and 2.

History is replayed per request rather than stored in server sessions. Reloads,
independent browser sessions, and undo therefore need no shared server state.
Requests are bounded to 500 plies and 128 KiB (imports are limited to 120 KB in the UI). Reaching the ply limit reports an
unfinished session, never an invented draw. The HTTP endpoint accepts local
same-origin requests; it is a development server, not a public hosting service.

This first UI does not include online multiplayer, trained opponents, accounts,
or in-browser external draw adjudication. Draw judgment remains
available through the existing CLI. Copenhagen retains its experimental label.

## Validation

The service tests cover canonical engine parity for every ruleset, repeatable bot
decisions, complete-history reconstruction, captures and terminal outcomes,
review/undo branches, collection-limit handling, malformed requests, HTTP origin
checks, static assets, and downloadable replay verification. Browser checks cover
human/bot play, both player seats, Copenhagen, local keyboard play, autosave reload,
history review, keyboard stepping, dragging, takebacks, bot match completion, result dialogs, rules,
replay import, and replay download. Imported draw judgments are checked across review and autosave.

Learning-page checks include engine-policy parity, deterministic analyses, complete
replay preservation, terminal/malformed input rejection, budget/origin checks,
and JavaScript calculations for adversarial replies, reward perspectives, PPO
clipping, and opponent mixtures. Run `node --test tests/web-learning.test.cjs`
alongside the Python suite; Node is only a development test dependency.
