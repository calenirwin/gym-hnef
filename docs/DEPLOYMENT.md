# Deployment preparation

GitHub Pages is the preferred free host within the total CAD 20–30/month budget.
See [Pages preparation and publication steps](PAGES.md). Railway below remains an
optional paid fallback; no Railway service is needed for Pages.

The modern 5×5 app and deployment preparation from PRs #1–#3 are merged
into `main`. Use `main` for new builds.
No Railway service has been provisioned or deployed by this change.

## Stack and local run

The UI is static HTML/CSS/JavaScript. The actual Python engine runs in a Pyodide
worker on the visitor's device. No application server, secrets, database, volume,
or GPU is needed. Python 3.12.14 builds the site; Caddy 2.10.2 serves it. Both
container images are pinned by digest. Update image versions/digests deliberately
and rerun the container smoke check when taking security updates.

```sh
docker build -t hnef-web .
docker run --rm --name hnef-web -p 127.0.0.1:8768:8080 hnef-web
```

Open http://localhost:8768. The image runs
`caddy run --config /etc/caddy/Caddyfile --adapter caddyfile`.
`PORT` is the only deployment environment variable; Railway supplies it and
Caddy listens on all interfaces. Local runs default to port 8080. No application
credentials or build secrets are required. Railway terminates public TLS.

The builder runs `python tools/fetch_browser_runtime.py` followed by
`python tools/build_demo.py --output /site`. It verifies the existing pinned
Pyodide hashes and model/evaluation identity, and preserves runtime licences and
`NOTICE.txt`. Only the generated static artifact enters the serving image.
Builds require outbound access to the container registry, jsDelivr and GitHub.
The allowlisted build context excludes legacy weights, notebooks and local runs.

## Later Railway setup

After choosing the hosting budget and authorizing deployment, connect the
repository, select a branch containing this change and PR #1, and use the
repository root. `railway.json` selects the Dockerfile and `/index.html`
healthcheck. Keep the image's start command; no install/build overrides or
pre-deploy commands are necessary. Begin with one replica and a Railway-generated
domain. Leave automatic PR environments off until their extra cost is wanted.
Use the reviewed `main` revision selected for that release.

The healthcheck confirms HTTP serving, not Python-worker startup. After a deploy,
verify a human move/AI reply, pass-and-play and undo in a browser. Check
`/runtime/pyodide.asm.wasm` returns `application/wasm`, and that runtime licences
remain available. Railway healthchecks gate a deployment; they are not continuous
browser monitoring. Keep prior successful releases available for rollback.

## Cost and scope

As checked October 21, 2026, Hobby is $5/month with $5 included resource usage;
usage above the allowance is billed additionally. Choose a monthly budget and
usage controls before enabling hosting. One static service needs no paid add-ons.
The roughly 14 MB artifact includes a roughly 13 MB browser runtime, so first-load
network egress and compression matter. Caddy enables gzip/zstd. Revisit caching
only with versioned asset URLs; the current filenames are reused between builds.

Games stay in the tab and refresh starts over. Training remains a separate,
optional CPU/PyTorch workflow; cloud training and online multiplayer are future
work. The existing public Sites deployment is unchanged.

References: [Railway configuration](https://docs.railway.com/config-as-code/reference),
[healthchecks](https://docs.railway.com/deployments/healthchecks),
[pricing](https://docs.railway.com/pricing/plans), and
[Caddy static serving](https://caddyserver.com/docs/caddyfile/directives/file_server).
