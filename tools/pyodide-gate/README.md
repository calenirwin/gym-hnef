# Original-table Pyodide feasibility gate

This opt-in experiment starts from integration commit
`09a94d4bcab93e2e0c30c870929b1c2f99210b63`. It does not adapt or publish the product.
All original UI, artwork, engine and API source files remain unchanged. No Mini
bridge, model, Torch, NumPy, database or external Python package is loaded into
the browser. The worker directly imports the original `game_response` and
`analyze_response` functions, including their existing module imports.

## Reproduce

Use Python 3.12, Playwright 1.62.0 and Chromium (the measured browser version was
154.0.8037.57). Playwright is an opt-in probe dependency, not a product dependency.
With those tools installed:

```sh
python tools/fetch_browser_runtime.py
CHROMIUM=/usr/bin/chromium python tools/pyodide-gate/run.py
```

The script checks the existing Pyodide runtime hashes, copies files into ignored
`.demo-build/pyodide-gate/`, starts a loopback-only ephemeral HTTP server, opens a
headless browser, and writes `results.json` there. It never dispatches deployment
or changes the production build. `results.json` beside this README records the
completed reference run; fixtures are generated from unchanged CPython functions.

## Gate findings

- All 33 complete response/error fixtures matched CPython exactly across Mini,
  Copenhagen v1 and Copenhagen v2: openings, moves, seeded random/heuristic/MCTS,
  12-ply replay import/restore, bounded Learn analysis, external draw adjudication,
  tampered replay and invalid inputs. Benchmark responses also match exactly.
- The direct imports work in Pyodide 314.0.7 / CPython 3.14.2. The source bundle is
  12 files / 47,698 bytes; the existing runtime is approximately 13 MiB on disk.
  Startup and all computation use same-origin requests only. No `loadPackage`,
  micropip, model or network API is needed.
- Measurements use a shared Linux x86_64 environment on AMD EPYC 9V74, headless
  Chromium, and a 390×844 viewport. This is not a physical phone or Safari test.
  Startup uses loopback assets with a fresh browser context, not a cold internet
  connection. Per-workload measurements contain three samples, not a load study.
- Main-page animation frames continue during worker MCTS/Learn work. The requested
  4× CDP page CPU throttle does not materially slow worker computation here;
  those samples must not be represented as a four-times-slower mobile CPU.
- A 10 ms consumer deadline rejects, but synchronous Python continues and a late
  reply arrives. Timeout alone is not cancellation. Terminating/recreating the
  worker cancels pending requests; importing a saved nonempty replay reproduces
  the response exactly. The probe uses its own storage key, not product autosave.
- Aborting the engine-bundle download rejects initialization. Invalid commands
  return matching error type/message; subsequent valid calls still work.

## Decision and next scope

Proceed to a narrowly scoped transport-adaptation PR after review of these gate
results. There is no dependency or measured parity blocker. The original functions
can run directly; extracting them from the HTTP module is optional cleanup, not
required for feasibility. Preserve the canonical rules and original design.

Next: add a transport adapter for the original game and Learn requests, a worker
startup state and generation-scoped request IDs, then test product autosave and
retry semantics. A deadline must terminate/recreate a stuck worker and restore
canonical saved replay before accepting more commands, or explicitly wait for
completion; never silently apply a timed-out late response. Coordinate recovery
ownership with W2 before implementing that behavior. This experiment changes no
existing recovery code and is not independent W2 sign-off.

Add an original-table static build with subpath-safe asset/navigation URLs and
pinned runtime/licences, leaving Mini separate. Preserve all original asset bytes,
rule identifiers, replay versions, default selections and save compatibility.
Do not dispatch the existing Mini deployment workflow for this artifact.

Release gates still include real mobile Safari/Chrome performance, realistic
network startup, long histories/500-ply boundary, tactical Copenhagen fixtures
(shield walls, forts, encirclement, repetition), full game/replay/autosave behavior,
Learn navigation and screenshot regression at `/gym-hnef/`. Current parity proves
only the tested cases; experimental Copenhagen is not a complete adjudicator.
The larger Learn workload and slower-device timeout risk deserve particular
attention. Do not silently reduce search budgets or change deterministic policy
behavior to hide latency.

No repository settings, credentials, permissions, hosting accounts or live sites
were changed. No merge or deployment is authorized by this experiment.

## Recorded timings

Unthrottled opening-position workloads, three warm samples each. Times below are
medians in milliseconds; raw samples and frame gaps are in `results.json`.

| Workload | CPython | Browser worker |
|---|---:|---:|
| opening | 0.6 | 1.6 |
| move | 1.2 | 3.3 |
| mcts | 748.8 | 1933.4 |
| learn-max | 3566.5 | 8309.7 |

Fresh-context loopback initialization: 3122.6 ms.
MCTS uses 32 simulations / 24 rollout plies; Learn maximum uses 64 / 48.
Search-workload page frame gaps had a 95th percentile of 16.7–16.8 ms and
a maximum of 33.4 ms without requested throttling. A viewport is not a phone.
