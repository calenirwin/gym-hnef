# Iteration 5: ambient texture and soft edges

A focused refinement of the accepted iteration-4 direction. Same draft PR;
no merge or deployment. Layout, daylight illustration, light stone/mist palette,
typography, features and original pieces are retained. Prior captures are untouched.

The background gains faint non-directional grain and a broad gentle tonal variation.
A static original 256px alpha tile has a maximum opacity of 3/255 (1.18%); it adds
16,444 bytes. No directional weave or opaque paper surface. The initial 5/255 grain
was reduced after inspecting normal-size mobile captures. Offline generation is
Pillow RGBA pixels (35,60,67,alpha), seeded `random.Random(518)`, alpha from
`randrange(4)`, 256×256, saved as lossless WebP. Both repositories contain identical
CSS and tile bytes; no runtime JS, dependency, filter or animation was added.

The blog divider becomes a soft outline/shadow around the existing article-list
region, without moving its content. Hnef's existing controls get the same faint
group boundary; the board shadow is softened. Header active-nav underlines are
replaced by a muted fill and medium weight, retaining `aria-current`. Inline links
and keyboard focus remain distinguishable. These are two group treatments, not a
new card layout or navigation redesign.

## Compare

| View | Iteration 4 | Iteration 5 |
| --- | --- | --- |
| Desktop | [1440](../iteration-4/table-1440.png) | [1440](table-1440.png) |
| Mobile | [390](../iteration-4/table-390.png) | [390](table-390.png) |
| Narrow | [320](../iteration-4/table-320.png) | [320](table-320.png) |
| Tablet | [768](../iteration-4/table-768.png) | [768](table-768.png) |

[100% ambient detail](ambient-detail-100-percent.png) is a DPR-1 browser crop,
not an enlargement. Desktop/mobile/detail captures were visually inspected against
iteration 4. The texture is intentionally faint and should remain subordinate.

## Checks

[Gameplay report](gameplay-checks.json): worker/API/tactical parity, AI play, undo, replay/reload/rotation, cancellation/retry/recovery, local HTTP and Learn behavior pass. All 12 JavaScript tests pass. [Controls suite](controls.json) and [bounds summary](bounds-summary.json): Mini/Copenhagen at 1440/1280/768/390/320, keyboard rules/reset flows, reduced motion and per-frame board stability. [Visual checks](visual-checks.json): table and five Learn topics at all requested widths; no overflow/page errors and unchanged board/toolbar/New game bounds during idle/busy/idle. No engine or original-piece changes; no fresh Python-engine run for this CSS-only refinement.

[Contrast](contrast.json), conservatively including maximum grain over the darkest
mist endpoint: primary 9.63:1 and secondary 5.49:1. White primary-button text remains
10.80:1. [Performance resources](performance.json) record local decoded sizes and
request durations; not a network benchmark. No layout dimensions changed. Diff check
passes. Original artwork and gameplay code have no diff.

Limits: Chromium emulation; physical devices, Safari/Firefox and full screen-reader
review remain outstanding. Very faint texture varies with display calibration; review
at ordinary size. No W2 approval assumed. The previews remain unpublished.
