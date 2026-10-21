# Albion playing surface review

Draft only; do not merge or publish until visual review. Based on main
`46e56323c532f76403f484ca77653b56de7fb183`, after PRs #8 and #9 merged.
Their branches and release were not modified.

The table shares the portfolio's Paper #F7F6F2 reading ground and Deep Blue
#052D46 ink. Sand #D8CDAA is concentrated on the board, with slightly clearer
squares and restrained existing grain. Original raven, hart, king, flourishes,
landscape and fonts are unchanged. Controls have a quiet white group and an
explicit deep-blue New game action. Secondary text uses #465962 rather than
faded opacity. Desktop board size leaves room for the control stack; mobile
board sizing and normal document flow are retained.

Learn uses larger text, deliberate spacing and a dark current-topic marker.
All existing history, resources, tactics and algorithm lessons remain intact.
No engine, transport, calculation, rules-dialog or reset behavior changed.

## Compare

Local static worker builds in Chromium, emulated viewports. `before` is the
combined released content at the base SHA; `after` is this implementation.
All surfaces below also have 320/390/768/1440px captures in their folders.

| Surface | Before | After |
| --- | --- | --- |
| Table, desktop | [Before](before/table-1440.png) | [After](after/table-1440.png) |
| Table, 390px | [Before](before/table-390.png) | [After](after/table-390.png) |
| Learn history | [Before](before/learn-history-1440.png) | [After](after/learn-history-1440.png) |
| Tactics, 390px | [Before](before/learn-tactics-390.png) | [After](after/learn-tactics-390.png) |
| Resources | [Before](before/learn-resources-1440.png) | [After](after/learn-resources-1440.png) |
| Rules dialog | — | [390px](after/rules-390.png), [1440px](after/rules-1440.png) |
| Experiments | [Before](before/learn-experiments-1440.png) | [After](after/learn-experiments-1440.png) |

## Validation

- Python: 325 tests pass, 2 optional-dependency suites skipped locally.
- JavaScript: 12 tests pass. Ruff and syntax checks pass.
- `tests/browser_table.py`: real-worker/API/tactical parity, play, AI, undo,
  reload, replay, rotation, cancellation/retry/recovery and Learn checks pass.
  [Gameplay report](gameplay-checks.json).
- `tests/browser_controls.py`: Mini and Copenhagen at 1440, 1280, 768, 390 and
  320px. Includes per-frame board bounds through calculation/cancel/retry,
  status/review/result changes, rules keyboard focus, reset confirmation,
  autosave/settings, pending calculation invalidation and reduced motion.
  11,226 sampled frames showed zero deviation.
  [Control report](controls.json), [bounds summary](bounds-summary.json).
- Board, toolbar and bottom New game bounds also remain identical through
  idle/busy/idle rendering at all four requested widths: [positions](control-positions.json).
- Learn history/playing/tactics/resources/experiments: no overflow or page errors
  at 320/390/768/1440; [visual report](visual-checks.json).
- [Token-pair contrast](contrast.json): primary 13.21:1, secondary 6.78:1 on paper.
  This calculation does not replace a full accessibility audit.
- Original-only static build succeeds; original assets byte-preserved.
- `git diff --check` passes.

Reproduce with `python tools/fetch_browser_runtime.py`, install Python Playwright
and Chromium, then run the existing browser suites. Set `CHROMIUM` only when using
a system browser. `python tools/build_table.py --output .demo-build/review` creates
a preview; serve that directory locally. No deploy command is part of this work.

The existing Original playing table workflow uploads the preview and browser
artifacts for 14 days. Screenshots here remain available after artifact expiry.
Physical devices, Safari/Firefox and full screen-reader testing remain outstanding.
