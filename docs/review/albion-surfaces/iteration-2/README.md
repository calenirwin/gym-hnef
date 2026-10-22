# Iteration 2: one playing area

Draft PR #10; no merge or deployment. Based on main
`46e56323c532f76403f484ca77653b56de7fb183`, retaining merged PRs #8 and #9.

The previous review placed the board, status, controls, New game and landscape
in separate bands on white. The board is now the dominant sand surface in an ink-blue
frame. Desktop controls share one adjacent sand area; at mobile widths that area
joins the bottom edge of the board. New game has a clear primary treatment, status
and secondary controls stay grouped, and the original landscape forms a lower
boundary behind the playing area. Learn has a bounded paper reading surface inside
the same frame. Frame, raised frame and reading-sand tokens match the portfolio.

Only presentation markup/CSS and a browser-test coordinate assertion changed.
Raven, hart, king, landscape, flourishes and fonts are byte-preserved. No engine,
transport, rules, game reset, calculation or Learn logic changes.

## Visual comparison

| Width | Previous | Current |
| --- | --- | --- |
| 1440 | [Previous table](../after/table-1440.png) | [Current table](table-1440.png) |
| 390 | [Previous table](../after/table-390.png) | [Current table](table-390.png) |
| 768 | [Previous table](../after/table-768.png) | [Current table](table-768.png) |
| 320 | [Previous table](../after/table-320.png) | [Current table](table-320.png) |

[Learn history](learn-history-390.png), [tactics](learn-tactics-1440.png),
[resources](learn-resources-1440.png), [experiments](learn-experiments-390.png).
All five Learn topics are captured at 320/390/768/1440. Previous before/after folders
are untouched. Rendered desktop and mobile compositions were visually inspected.

## Validation

- 325 Python tests pass; 2 optional-dependency suites skipped. 12 JavaScript tests pass.
- [Gameplay report](gameplay-checks.json): worker/API/tactical parity, human/AI,
  undo, Copenhagen, reload, replay import/export/review, rotation, cancel/deadline/
  recovery, startup retry, mobile keyboard, original art decoding, crash recovery,
  Learn saved position/cancel/retry/deep-link/pagination and local HTTP behavior.
- [Visual checks](visual-checks.json): no page errors or horizontal overflow for table
  and five Learn topics at 320/390/768/1440; board, toolbar and New game bounds match
  exactly through idle/busy/idle rendering at each width.
- [Controls suite](controls.json): Mini and Copenhagen at 1440/1280/768/390/320;
  rules keyboard focus, reset confirmation, saved settings, calculation invalidation,
  reduced motion and per-frame document bounds. See [bounds summary](bounds-summary.json).
  Retry may scroll into view at 320×568: the long-error assertion now compares
  document coordinates, consistent with the existing per-frame sampler. It still
  rejects board movement/resizing; user-driven scrolling is not a layout shift.
- [Contrast](contrast.json): sand/frame 7.30:1, mist/frame 7.52:1;
  primary/secondary text on sand 10.54:1 / 5.41:1.
- Original-only static preview build and `git diff --check` pass. Engine and artwork
  paths have no diff from the previous draft; required merged main is an ancestor.

Limitations: Chromium emulation, not physical devices. Safari/Firefox and a full
screen-reader audit remain outstanding. Contrast evidence covers text token pairs,
not every image pixel/state. No W2 approval is claimed.

Reproduce using `python tools/fetch_browser_runtime.py`,
`CHROMIUM=/usr/bin/chromium python tests/browser_table.py` and
`CHROMIUM=/usr/bin/chromium python tests/browser_controls.py` with Python Playwright
installed. Preview: `python tools/build_table.py --output .demo-build/review` and a
local HTTP server. No deployment command is part of this work.
