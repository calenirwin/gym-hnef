# Selected playing-table baseline

The product baseline is `feat/modernize-engine` at
`336c1e79e042eb34ee8313cef3456871eb98b08f`, which adds the browser game and learning
playground above engine commit `9348d34cb9bffc3cf33205e4e9212f1c92fab428`.
The integration branch starts there and merges existing `main`
`d5cca019a168bdefd663c689f3ec170f68d658f2`, preserving both histories.

## What is preserved

`src/hnef/web/` is the selected playing table, including the original raven
(`assets/attacker.png`), hart (`assets/defender.png`), king (`assets/king.png`),
board decorations, local fonts and licences. Its source and the rules engine are
unchanged from the selected branch. The default remains Mini; the existing
Copenhagen selector opens the 11×11 experimental v2 game. Existing local saves,
canonical replays, keyboard controls, rotation, bots and learning pages remain.

Copenhagen v2 is explicitly experimental: exit-fort recognition is conservative,
and automatic dead-position draws are not implemented. External adjudication is
recorded by the engine. Passing tests is not a claim of complete Copenhagen rules.

The merge retains optional CPU actor-critic training, browser inference tests,
locked dependencies, and CI from main. The separate top-level `web/` is the
previous Pyodide Mini Lab. Its model, runtime integrity checks, licences, build
scripts and deployment workflow are unchanged. It is not the selected UI.

## Run locally

Python 3.11–3.13 and uv are the development stack; CI uses uv 0.9.5. Node 22 runs
the teaching-math tests and separate static-demo checks. No database, credentials
or GPU is needed.

```sh
uv sync --locked --extra rl
uv run --no-sync hnef web --port 8000
# Open http://127.0.0.1:8000/; /learn is the learning playground.
uv run --no-sync pytest
node --test tests/web-learning.test.cjs
```

For optional CPU training, add `--extra train` to sync and run
`uv run --no-sync hnef-rl --help`. See [training](rl-baseline.md).

## Hosting and release gates

The selected UI requires its Python `/api/game`, `/api/replay` and `/api/analyze`
endpoints. Its current server binds loopback and validates local Host/Origin;
its asset/API paths are root-relative. It cannot be published unchanged to
GitHub Pages or served by the current static Caddy container.

[Pages](PAGES.md) and [Railway](DEPLOYMENT.md) instructions apply only to the
separate Mini Lab. The existing live Mini at https://calenirwin.com/gym-hnef/
remains unchanged. No workflow dispatch, main merge or publication is part of
this integration. No new permissions, secrets, paid services or DNS changes.

Review the integration PR against main, confirm the original design and game
behavior, then obtain release confirmation before merging. The integration
already contains both parent histories; do not cherry-pick the old Mini UI over
the playing table. A later hosting change must explicitly choose between a
browser-only API adapter for free static hosting and a hardened Python service
with a reviewed cost limit. Neither is implemented here. Preserve the overall
CAD 20–30 monthly ceiling and prefer free hosting.

The separate Mini opponent-error recovery issue remains outside this integration;
coordinate its existing W2 task before changing it. This merge does not claim
that task or change either application's recovery behavior.

## Verification on October 21, 2026

- Python 3.12: 362 tests passed with optional RL and training dependencies;
  nine non-failing PettingZoo observation/naming warnings.
- Five Node teaching-math tests passed; Ruff lint and formatting passed.
- Wheel and sdist built; every source playing-table asset is present byte-for-byte
  in the wheel.
- Static Mini runtime download, pinned hash validation, build and JS syntax checks
  passed. Its source and deployment configuration match existing main.
- Local Docker build reached runtime download but the build container had no
  network route. Remote container CI is the remaining container verification.

Browser smoke-test results and exact integration SHA are recorded in the PR.
