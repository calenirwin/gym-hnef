# Iteration 4: daylight and quiet space

Same unpublished draft PR. No merge or deployment. Iterations 1–3 are preserved.

Iteration 3 was rejected as too dark and overwhelming. This removes the repeated
linen/grain treatment and yellow panels. The shared palette is warm stone and mist
with dark readable type. Atmosphere comes from one original static SVG: daylight,
layered hills, cloud strokes and fine contours, with a quiet field around the
functional content. It is a focused illustration beside the desktop blog heading,
in normal flow below the introduction at smaller widths, and cropped at Hnef's
lower edge. No noise overlay or material texture sits behind text or controls.
The article list follows the reading grid with a subtle divider; Hnef's board and
controls share a continuous setting, without disconnected cards.

References: [Anthropic](https://www.anthropic.com/),
[Anthropic research](https://www.anthropic.com/research) and
[Railway](https://railway.com/). The direction uses calm editorial space and
contained atmosphere, not their artwork, branding or Railway's dark palette.
The optional Ubuntu wallpaper was not supplied or guessed.

## Review

| Width | Rejected iteration 3 | Iteration 4 |
| --- | --- | --- |
| Desktop | [1440](../iteration-3/table-1440.png) | [1440](table-1440.png) |
| Mobile | [390](../iteration-3/table-390.png) | [390](table-390.png) |
| Narrow | [320](../iteration-3/table-320.png) | [320](table-320.png) |
| Tablet | [768](../iteration-3/table-768.png) | [768](table-768.png) |

[Unscaled DPR-1 detail](atmosphere-detail-100-percent.png). Desktop/mobile/detail
renders were visually inspected. The first tablet capture exposed artwork behind
the blog heading; the art now moves below the introduction through 1100px. A Hnef
background-edge overflow was corrected by cropping the background within its box.
The repeated texture files were removed from active assets and remain in Git history.
No new UI slogans, labels or copy. Existing feature behavior is retained.

## Validation

[Gameplay suite](gameplay-checks.json): real-worker/API/tactical parity, AI play, undo, replay/reload/rotation, cancellation/retry/recovery, local HTTP and Learn behavior pass. 12 JavaScript tests pass. [Controls](controls.json) cover Mini/Copenhagen at 1440/1280/768/390/320, keyboard rules/reset flows, reduced motion and frame-by-frame board stability; [bounds summary](bounds-summary.json). [Visual checks](visual-checks.json) cover table and five Learn topics at all four widths with no overflow/page errors and unchanged board/toolbar/New game bounds across idle/busy/idle. Engine and original raven/hart/king assets have no diff. No fresh Python-engine run for this presentation-only change.

[Contrast](contrast.json): primary and secondary text on mist 9.81:1 / 5.59:1;
on stone 10.25:1 / 5.84:1; primary button text 10.80:1. Decorative art is outside
the reading areas. This is token-pair evidence, not a full accessibility audit.

[Performance](performance.json): one shared CSS file and one original SVG, about
5.5 KB combined, versus iteration 3's 65 KB blog / 101 KB Hnef texture payload.
No added runtime JS, dependency, animated artwork or filter. Local Chromium timing
and initial CLS are recorded in the report, not presented as a network benchmark.
Calculation-state zero-jitter checks are separate from initial font/startup layout.
`git diff --check` passes. Both products use byte-identical daylight CSS and SVG.

Remaining limits: physical devices, Safari/Firefox and full screen-reader testing.
The aesthetic still needs user review; no W2 approval is claimed. Remote CI for
new commits is separate from earlier-head results. Nothing is merged or deployed.
