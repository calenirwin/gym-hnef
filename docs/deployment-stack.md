# Hnef MVP deployment

> Scope: this guide serves the separate `web/` 5×5 Mini Lab. It does not deploy
> the selected raven/hart/king playing table in `src/hnef/web/`. See
> [integration scope](integration-baseline.md) before any release.

The public game is https://hnef-mini-lab.nelac.chatgpt.site. The MVP is the
versioned **5×5 mini-v1 variant**, not the full 11×11 Copenhagen game.

## Stack and boundaries

- **UI:** static HTML, CSS and JavaScript; forest green, parchment and brass.
- **Rules:** the repository's dependency-free Python engine, bundled unchanged
  into `engine.json` and run by pinned Pyodide in a Web Worker. No duplicate JS rules.
- **Play:** two people on one device, or either side against heuristic, random,
  or the existing experimental learned baseline. Undo restores the full immutable
  state including repetition history; versus AI it takes back the human turn
  and its reply. Legal moves, outcomes, local replay export and piece counts are visible.
- **Hosting:** existing public Sites static hosting. The hosting manifest lives
  in the separate deployment checkout. No server, database, API key, account,
  GPU or paid cloud service is required to play.
- **Training:** optional Python/PyTorch plus PettingZoo, isolated from serving.
  No new training or stronger-model claim is part of this deployment.

Refresh begins a new game. There is no network multiplayer or cloud game storage.
Games stop at the demo's 256-ply collection limit as unfinished, not draws.
The first visit downloads about 13 MB of Python runtime assets; fonts use Google
Fonts with local fallbacks. All game decisions then run on the visitor's device.

## Rebuild from a fresh clone

Python 3.11+ and Node 22+ are sufficient; the static build needs no Python packages
or PyTorch installation:

```sh
python3 tools/fetch_browser_runtime.py
python3 tools/build_demo.py --output web-dist
python3 -m http.server 8768 --directory web-dist
```

The runtime downloader checks pinned SHA-256 hashes. The small public baseline
export and its evaluation report are tracked in `web/baseline/`; the build checks
their checkpoint identities match. Optional `--model` and `--report` arguments
select a replacement evaluated export. Training checkpoints are never shipped.
`web.yml` builds and uploads the same portable static artifact on pushes and PRs.
`web-dist/` can be uploaded to any static host with WebAssembly MIME support.
For Sites, preserve project `appgprj_6ad4decbf5048191b8b4937aa12f2fd2` and its
public audience; build into the selected Site checkout's `dist/`, then use the
Sites source helper to push, package, save a version, and deploy that exact version.

## Later cloud GPU models

Keep GPU training as a separate batch job using the existing `hnef-rl` training,
evaluation and export commands. Use a GPU-compatible PyTorch environment: the
repository's uv source currently pins a CPU Torch index for local reproducibility.
Use an explicit training environment/index override before expecting CUDA support.
Preserve mini-v1 action IDs, legal masks, complete history, feature schema and
checkpoint hashes when exporting a compatible model. Evaluate on both sides
against held-out random/heuristic/search opponents before promoting a release.

Small public models can continue to run in the browser. For larger models or
private weights, add a CPU Python inference API first (for example FastAPI in a
container), load immutable versioned bundles from object storage, and replace the
worker's policy request with an API call. Keep legal validation in the engine,
request timeouts, bounded search and a local heuristic fallback. A GPU serving
endpoint is a later measured choice. Online multiplayer will separately need an
authoritative game service; it is not provided by model inference.
