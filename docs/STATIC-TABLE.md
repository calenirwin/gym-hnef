# Original playing table: static candidate

This candidate runs the recovered Albion table on a static host while preserving
its original images, fonts, layout, canonical Python rules and search budgets.
It is stacked on integration PR #5, base
`09a94d4bcab93e2e0c30c870929b1c2f99210b63`. It has not replaced the live Mini.

## Build and run

Python 3.11+ builds the artifact. The pinned runtime downloader checks Pyodide
314.0.7 hashes. No model, GPU, database, credentials, backend service or additional
Python packages are loaded into the browser.

```sh
python tools/fetch_browser_runtime.py
python tools/build_table.py --output .demo-build/site/gym-hnef
python -m http.server 8000 --bind 127.0.0.1 --directory .demo-build/site
```

Open `http://127.0.0.1:8000/gym-hnef/` and `/gym-hnef/learn/`.
The output directory must be a dedicated child of `.demo-build`, excluding the
runtime directory. Output is rebuilt, so keep personal files elsewhere.

The original local server remains available with
`uv run --no-sync hnef web --port 8000`. It uses the HTTP API and retains its
local-only Host/Origin checks. Its generated page chooses HTTP transport; the
static build chooses a dedicated module worker. No public server configuration
or security-policy relaxation is needed for the static artifact.

## Transport and saved games

`transport.js` serializes requests and snapshots their payloads. The worker
imports the existing `game_response` / `analyze_response` functions directly.
Every command reconstructs canonical history; no alternate JavaScript rules or
Mini bridge are used. MCTS remains 32 simulations / 24 rollout plies, and Learn
retains its existing selectable budgets.

Startup has a 60-second deadline. Game calculations retain a 30-second deadline;
Learn retains 60 seconds. The UI reports loading/recovery, offers cancellation
while working, and keeps errors visible with Retry. Cancellation, deadline,
worker crash or initialization failure terminate the worker and reject pending
requests. Worker-generation checks discard late messages, and queued requests
from a cancelled generation do not execute after restart.

Only successful live-game responses update the transport's confirmed replay;
reviewing an earlier position cannot replace it. On restart that replay is
validated before the requested command runs. The existing UI writes autosave
only after success. Invalid requests, cancelled calculations and failed startup
do not overwrite the saved game. Retrying a move or opponent request reconstructs
its supplied history and preserves the deterministic seed. A failed calculation
is not silently applied later. The engine remains stateless between commands.

The `hnef-table-v1` storage key and replay formats are unchanged. Learn reads the
same save but does not modify it. Storage remains browser-origin-specific: moving
between localhost, a Pages domain or a different host requires replay export and
import. Existing Mini saves are not silently interpreted as original-table saves.

## Verification and limits

Run the engine suite with `uv run --no-sync pytest`; run transport/teaching tests:

```sh
node --test tests/web-transport.test.cjs tests/web-learning.test.cjs
```

The opt-in browser suite needs Playwright 1.62.0 and its Chromium browser. In a
separate test environment, install that pinned package and run
`python -m playwright install chromium`, then `python tests/browser_table.py`.
Set `CHROMIUM=/path/to/chromium` to use an existing browser. Test output is under
`.demo-build/table-qa-artifacts/`. The `Original playing table` CI workflow runs
these checks and uploads a separate artifact, with contents-read permission only.
It has no deployment job.

Coverage includes exact CPython/Pyodide API responses, both experimental Copenhagen
versions, seeded policies, replay adjudication/errors, 100/499/500-ply histories,
46 tactical positions (captures, shield walls, encirclement, corner escape,
repetition and rotated fort fixtures), real worker cancellation/deadline/restart,
autosave/replay UI, Learn and local HTTP compatibility. Test-only tactical code
is copied into the QA site; it is never included in the distributable artifact.
Artwork and font bytes are verified against the source, and subpath requests are
checked to remain beneath `/gym-hnef/`.

The CI browser job also runs `python tests/browser_resilience.py`. Its loopback
HTTP server uses fresh browser contexts and `Cache-Control: no-store`. It adds
150 ms response latency and streams each runtime response at 512 KiB/s in 32 KiB
chunks; this is per-response throttling, not a measured cellular link. Separate
attempts truncate the WASM body after 64 KiB with its original Content-Length.
Two failed attempts, a deliberately shortened 200 ms startup deadline, and a
successful retry must preserve the exact saved replay and add no stale AI move.
The normal product startup deadline stays 60 seconds.

The resilience test allocates and touches a bounded 64 MiB browser buffer, sends
Chromium's synthetic critical-memory-pressure notification when supported, and
compares the next MCTS result with CPython. Separately, `Page.crash` forces a
renderer failure during an AI request. A new page must restore the last confirmed
save and a subsequent AI request must add exactly one move. This is **not an
out-of-memory test**: it neither exhausts RAM nor establishes OS/browser behavior
on an actual low-memory phone. Supported/unsupported protocol status, browser
version, timings and runtime transfer counts are recorded in `resilience.json`
within the CI evidence artifact. No memory limit or application code is changed.

A 390px Chromium viewport is not physical-phone validation. Real mobile Safari
and Chrome, actual cellular-network conditions and genuine memory exhaustion
still need release review; synthetic tests cover only their stated conditions.
The feasibility probe found roughly 2-second Copenhagen MCTS and 8-second maximum
Learn analysis on a shared desktop-class runner; slower devices may hit deadlines.
No search budget is reduced to conceal latency. Copenhagen remains experimental,
with conservative fort recognition and no automatic dead-position draws.

## Merge and release sequence

1. Review and approve integration PR #5, which preserves the original baseline
   and retains Mini/training tooling separately.
2. Review this stacked transport PR, retarget to main after #5 merges, recheck its
   diff and CI, and obtain the scoped release confirmation before merging.
3. Separately review a publication change that selects the **original-table**
   artifact. The existing `web.yml`, Caddy/Docker and Railway configuration still
   build only the separate Mini and are unchanged by this candidate.
4. Only after release approval, publish a reviewed exact SHA and verify the actual
   public URL, original artwork, gameplay, saved games and Learn links.

No workflow dispatch, Pages settings, DNS, visibility, new permissions, credentials
or paid hosting are part of this PR. Keep the current live Mini until replacement
is explicitly approved. Preserve its source as a separate experiment.
