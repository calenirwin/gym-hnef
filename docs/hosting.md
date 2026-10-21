# Public demo hosting

**Live:** [Hnef Mini Lab](https://hnef-mini-lab.nelac.chatgpt.site).
Published publicly on October 18, 2026, as Sites version 1, source commit
`fa882ecb543870c6051957ea1513a16fabf6a0ba`. Deployment succeeded and a human
move plus AI reply were verified on the public URL.

The first demo is a **static site**. Its files are assembled from `web/`, the
actual dependency-free Python engine, and a versioned learned-policy export.
The selected Sites project is `appgprj_6ad4decbf5048191b8b4937aa12f2fd2`;
its separate deployment checkout is `/home/calen/Git/gym-hnef-demo`.

The board runs the Python rules in a WebAssembly worker with the pinned official
[Pyodide 314.0.7 runtime](https://pyodide.org/en/stable/usage/quickstart.html).
The browser loads that runtime from the same site. Learned inference is a small
JavaScript forward pass over exported weights, tested against PyTorch logits and
values; it keeps every legal mask and the complete history input. The browser
uses its own seeded sampling RNG, so it does not promise the same sampled games
as Torch for the same seed. The heuristic and random opponents execute the
existing Python policies without a second rule implementation.

This avoids an always-running Python service, GPU, database, API keys, and
cross-origin backend configuration for the first release. Runtime assets total
about 13 MB before HTTP compression; visitors pay an initial load cost, then play
locally in the tab. The site also loads optional Google Fonts; system fallbacks
remain available. Refresh starts a fresh game. There are no stored accounts,
leaderboards, multiplayer sessions, or game uploads. Replay download is local.

## October 19 MVP update

The source now includes local pass-and-play, undo, last-move highlighting and
piece counts. A fresh static build uses the checked-in evaluated browser export
in `web/baseline/` by default, so neither a training run nor PyTorch is needed.
See [the current deployment guide](deployment-stack.md) and the `web.yml` CI build.
The historical commands below describe the first publication.

## Rebuild the static artifact

```sh
uv run --no-sync python tools/fetch_browser_runtime.py
uv run --no-sync hnef-rl export-web runs/mini-ac-random-s42/checkpoint-000200.pt \
  --output models/mini-ac-random-s42-web.json
uv run --no-sync python tools/build_demo.py \
  --model models/mini-ac-random-s42-web.json \
  --report runs/mini-ac-random-s42/evaluations/final/report.json \
  --output /home/calen/Git/gym-hnef-demo/dist
```

The build checks checkpoint identity against the evaluation. It copies only
explicit modern engine modules, UI files, the model export, a public model card,
exact mini rules, and the browser runtime. It excludes training checkpoints,
optimizer/RNG states, legacy weights, notebooks, and local experiment directories.
Use the Sites skill/source helper for publishing from the separate checkout;
preserve `.openai/hosting.json` and the existing project ID.

These same static files can move to another static host if preferred. No provider
price is assumed here. A server is justified later for shared games, accounts,
telemetry with consent, private model weights, or search too expensive for mobile
devices. At that point use a CPU inference service and object storage for immutable
model bundles, with a pinned release hash per deployment. Train separately and
promote an evaluated candidate explicitly. Bigger Copenhagen models and GPU
training can follow measured demand instead of becoming launch dependencies.

## Checks performed locally

- The exact Python engine starts in the bundled module worker.
- Human moves receive heuristic and learned-policy responses.
- Playing defenders triggers the AI's opening attacker move.
- Downloaded browser replays verify with the Python CLI.
- The layout fits a phone viewport without horizontal overflow.
- JavaScript logits and value outputs match PyTorch within 2e-5 on tested positions;
  sampled actions always respect the legal mask.

The mini game stops collecting after 256 plies and labels it unfinished. This
does not change engine rules or declare a draw. Copenhagen is not exposed by
this first UI.
