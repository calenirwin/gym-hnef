# Iteration 3: ink linen and warm paper

Same draft PR; no merge or deployment. Earlier galleries remain untouched.

The material is now a visible foundation: fine ink-blue weave, warm grained paper
for content and controls, and a slightly deeper paper board. All use the same
256px scale and upper-left lighting. The board frame and primary controls repeat
the same materials. This retains iteration 2's hierarchy and grouping. Hnef's
scenery and ornamental corners no longer compete with the board and pieces.
No marketing copy was added.

Both repositories carry byte-identical material CSS and tiles. These are original,
deterministic procedural assets, generated offline with [this source](generate-materials.py)
(Python + Pillow), not images generated at runtime. No runtime dependency added.

## Inspect at ordinary size

| View | Iteration 2 | Iteration 3 |
| --- | --- | --- |
| Desktop | [1440](../iteration-2/table-1440.png) | [1440](table-1440.png) |
| Mobile | [390](../iteration-2/table-390.png) | [390](table-390.png) |
| Narrow | [320](../iteration-2/table-320.png) | [320](table-320.png) |
| Tablet | [768](../iteration-2/table-768.png) | [768](table-768.png) |

[Material detail at 100%](material-detail-100-percent.png) is a DPR-1 browser crop,
not an enlarged image. Desktop, mobile and detail captures were visually inspected.
The first portfolio render exposed a cascade conflict that hid the material; this
was corrected and the captures repeated. Worst-pixel contrast testing also prompted
a darker secondary text token, #394E5B.

## Evidence

The real worker/API/tactical parity and gameplay suite passes, including AI play, undo, replay/reload/rotation, cancellation/retry/recovery, local HTTP and Learn behavior: [gameplay report](gameplay-checks.json). All 12 JavaScript tests pass. The controls suite covers Mini and Copenhagen at 1440/1280/768/390/320, keyboard rules/reset flows, reduced motion, calculation changes and per-frame board stability: [controls](controls.json), [bounds](bounds-summary.json). The separate [visual checks](visual-checks.json) cover table and five Learn topics at all four requested widths, no page errors/overflow, and identical board/toolbar/New game bounds across idle/busy/idle. Engine code and the original pieces are unchanged. The scenery and flourishes are removed from table markup; their source assets remain in Git. No Python-engine changes; its 325-test result belongs to iteration 2, not a fresh run here.

[Contrast](contrast.json) uses every pixel in each tile plus the worst lighting
endpoint: sand on blue 6.88:1, mist on blue 7.09:1, primary on paper 8.41:1,
secondary on paper 5.12:1, button text on blue 10.10:1. This is a conservative
text-surface check, not a complete accessibility audit.

[Performance](performance.json): texture payload including CSS is 65,117 bytes for
the blog and 100,635 for Hnef. Static lossless WebP tiles; no new JS, animation,
SVG turbulence, blend filters or backdrop filters. Observed initial-load CLS in one
local Chromium run was 0.00098 for blog and 0.00356 for Hnef; these small startup
shifts are distinct from the zero board deviation checked during game state changes.
Local timings are not a network or physical-device benchmark. Full asset hashes
and resource timings are recorded. `git diff --check` passes.

Remaining limits: physical devices, Safari/Firefox, high-DPI moiré assessment and a
full screen-reader audit remain outstanding. W2 approval is not assumed. Review the
material taste from the captures before any merge or publication.
