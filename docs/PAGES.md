# Free GitHub Pages preparation

The public `calenirwin/gym-hnef` repository is eligible for GitHub Pages on GitHub
Free. The app is entirely static: rules and opponents run in the browser's Python
worker. No server, database, GPU, secrets, paid plan or new domain is needed.
Prefer Pages within the overall CAD 20–30/month budget; retain
[Railway](DEPLOYMENT.md) only as an optional paid fallback.

## Prepared, not published

The existing `Playable web app` workflow builds and uploads a `github-pages`
artifact using `actions/upload-pages-artifact@v4`. It has only `contents: read`
permissions and no deploy job. Pushes and PRs cannot publish this artifact.
The read-only Pages API check returned 404 on October 21, 2026; no site settings
were changed. That response does not establish whether account access is sufficient
to activate Pages later.

All UI, worker, engine, model and runtime URLs are relative. Serve the build at
`/gym-hnef/` including its trailing slash; no hardcoded root or base-path rewrite
is needed. Links and runtime imports continue to work at `/` for Railway/Sites.
The build retains Pyodide hash checks, licences and model identity checks. It is
about 14 MB, below Pages' 1 GB published-site limit. Pages has a 100 GB/month soft
bandwidth limit; first-time runtime downloads are the main bandwidth consideration.
This hosting is suitable for the current public research game, not a future paid
SaaS or authoritative multiplayer service.

## Reproduce the project-path check

From the repository root, with Python 3.11+:

```sh
python tools/fetch_browser_runtime.py
python tools/build_demo.py --output .demo-build/pages/gym-hnef
python -m http.server 8768 --bind 127.0.0.1 --directory .demo-build/pages
```

Open http://localhost:8768/gym-hnef/. Check engine startup, human move and AI reply,
full-turn undo, pass-and-play, learned opponent, replay export, rules/model links,
and phone-width layout. All first-party requests must stay under `/gym-hnef/`;
`/gym-hnef/runtime/pyodide.asm.wasm` must return `application/wasm`.
This local check does not substitute for checking the eventual public URL.

## Merge and publication order (after notifying Cale)

1. Merge PR #1, the modern app, into `main`.
2. Retarget PR #2 (Railway preparation) from `codex/playable-web-mvp` to `main`,
   resolve/rebase if the first merge was squashed, check CI, then merge it.
3. Retarget this Pages preparation PR from `codex/railway-readiness-20261021`
   to `main`, check its diff contains only Pages changes and CI passes, then merge.
   These merges alone do not deploy Pages.
4. Obtain approval for the publication step and the required deployment-job token
   permissions. In Settings > Pages, select GitHub Actions. Do not change DNS.
   Check the `github-pages` environment restricts deployment to `main`.
5. In a separately reviewed activation change, add the following job to
   `.github/workflows/web.yml`. The existing `workflow_dispatch` trigger permits
   deliberate publication; branch pushes and PRs remain build-only. Merge that
   activation change after review, then manually run `Playable web app` on `main`.

```yaml
  deploy-pages:
    if: github.event_name == 'workflow_dispatch' && github.ref == 'refs/heads/main'
    needs: build
    runs-on: ubuntu-latest
    permissions:
      pages: write
      id-token: write
    concurrency:
      group: github-pages
      cancel-in-progress: false
    environment:
      name: github-pages
      url: ${{ steps.deployment.outputs.page_url }}
    steps:
      - name: Deploy Pages artifact
        id: deployment
        uses: actions/deploy-pages@v4
```

Use the deployment output URL as authoritative. The standard project URL is
`https://calenirwin.github.io/gym-hnef/`, but an existing account-level custom
domain can affect it. Verify public startup, play, undo, replay export, subpath
assets and WASM MIME before updating the README's live-game link. Keep the current
Sites link until this succeeds. For rollback, manually dispatch a reviewed revert
on `main`; do not enable automatic deployment as part of this preparation.

Official references, checked October 21, 2026:
[Pages availability and project URLs](https://docs.github.com/en/pages/getting-started-with-github-pages/what-is-github-pages),
[custom workflow and deployment permissions](https://docs.github.com/en/pages/getting-started-with-github-pages/using-custom-workflows-with-github-pages),
[limits](https://docs.github.com/en/pages/getting-started-with-github-pages/github-pages-limits).
