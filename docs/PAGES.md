# Free GitHub Pages preparation

> Publication candidate: the recovered raven/hart/king playing table is served at
> `/gym-hnef/`, with the separate Mini Lab retained at `/gym-hnef/mini/`.
> This branch does not itself publish. The currently verified live revision is
> Mini at `d5cca019a168bdefd663c689f3ec170f68d658f2`.
> See [original-table tests and limitations](STATIC-TABLE.md).

The public `calenirwin/gym-hnef` repository is eligible for GitHub Pages on GitHub
Free. The app is entirely static: rules and opponents run in the browser's Python
worker. No server, database, GPU, secrets, paid plan or new domain is needed.
Prefer Pages within the overall CAD 20–30/month budget; retain
[Railway](DEPLOYMENT.md) only as an optional paid fallback.

## Controlled publication

PRs #1–#3 are merged into `main`. The owner approved Pages activation and the
required deployment permissions on October 21, 2026. The `Playable web app`
workflow packages a `github-pages` artifact on pushes and PRs, but publishes only
when manually dispatched on `main`. Build jobs retain `contents: read`; only the
`deploy-pages` job has `pages: write` and `id-token: write`. It waits for both the
static build and container checks. No personal token or stored secret is needed.

All UI, worker, engine, model and runtime URLs are relative. Serve the build at
`/gym-hnef/` including its trailing slash; no hardcoded root or base-path rewrite
is needed. Links and runtime imports continue to work at `/` for Railway/Sites.
The build retains Pyodide hash checks, licences and model identity checks. The combined artifact duplicates the pinned runtime to keep both apps independent
and remains below Pages' 1 GB published-site limit. Pages has a 100 GB/month soft
bandwidth limit; first-time runtime downloads are the main bandwidth consideration.
This hosting is suitable for the current public research game, not a future paid
SaaS or authoritative multiplayer service.

## Reproduce the project-path check

From the repository root, with Python 3.11+:

```sh
python tools/fetch_browser_runtime.py
python tools/build_release.py --output .demo-build/pages/gym-hnef
python -m http.server 8768 --bind 127.0.0.1 --directory .demo-build/pages
```

Open http://localhost:8768/gym-hnef/. Check original artwork, Copenhagen selection,
human move and AI reply, undo, replay import/export, autosave/reload, Learn analysis
and phone-width layout. Check separate Mini gameplay at `/gym-hnef/mini/`.
`python tests/browser_release.py` tests this combined artifact with Playwright. All first-party requests must stay under `/gym-hnef/`;
`/gym-hnef/runtime/pyodide.asm.wasm` must return `application/wasm`.
This local check does not substitute for checking the eventual public URL.

## Publish a reviewed main revision

Pages must use GitHub Actions as its publishing source (Settings > Pages).
The `github-pages` environment should restrict deployments to `main`. No DNS or
repository-visibility change is required. Once setup is complete:

1. Merge reviewed PR #5, then retarget/recheck PR #6 onto main and merge it,
   then retarget/recheck the stacked publication PR and merge it. Confirm explicit
   approval for those merges and publication. Check the intended `main` SHA and
   its engine, original-table (including resilience), and combined web CI results.
2. Manually run `Playable web app` on `main` (Actions > Run workflow), or use:
   `gh workflow run web.yml --ref main --repo calenirwin/gym-hnef`.
3. Watch the run through `deploy-pages`, record its SHA and deployment URL, and
   perform the public browser checks below. Dispatching another branch or pushing
   a commit does not publish. Failed build/container checks prevent deployment.

Use the deployment output URL as authoritative. The standard project URL is
`https://calenirwin.github.io/gym-hnef/`, but an existing account-level custom
domain can affect it. Verify public startup, play, undo, replay export, subpath
assets and WASM MIME before updating the README's live-game link. Keep the existing
Sites URL as a fallback. Verify `deployment.json` reports the dispatched main SHA;
check both apps' WASM MIME and the original art, Learn, replay and autosave routes.

For rollback, revert only the publication workflow selection in a reviewed main
commit to run `python tools/build_demo.py --output web-dist` and upload `web-dist/`
for both artifact steps, then manually dispatch that reviewed main revision.
This restores the preserved Mini implementation without deleting original-table
source or rewriting history. The previously successful Mini deployment is
[run 39636775926](https://github.com/calenirwin/gym-hnef/actions/runs/39636775926),
SHA `d5cca019a168bdefd663c689f3ec170f68d658f2`. Automatic deployment on push remains disabled.

Official references, checked October 21, 2026:
[Pages availability and project URLs](https://docs.github.com/en/pages/getting-started-with-github-pages/what-is-github-pages),
[custom workflow and deployment permissions](https://docs.github.com/en/pages/getting-started-with-github-pages/using-custom-workflows-with-github-pages),
[limits](https://docs.github.com/en/pages/getting-started-with-github-pages/github-pages-limits).
