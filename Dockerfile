FROM python:3.12.14-slim-bookworm@sha256:392307d22300de8b5986851a12d9176dfc0fc073e65bf6523ebd7dcbeb23564e AS build
WORKDIR /build
COPY tools/fetch_browser_runtime.py tools/build_demo.py tools/
COPY web/ web/
COPY src/hnef/ src/hnef/
COPY docs/mini-rules.md docs/mini-rules.md
RUN python tools/fetch_browser_runtime.py && python tools/build_demo.py --output /site

FROM caddy:2.10.2-alpine@sha256:4c6e91c6ed0e2fa03efd5b44747b625fec79bc9cd06ac5235a779726618e530d
COPY Caddyfile /etc/caddy/Caddyfile
COPY --from=build /site/ /srv/
EXPOSE 8080
HEALTHCHECK --interval=30s --timeout=3s --start-period=5s --retries=3 \
  CMD wget -Y off -q -O /dev/null "http://127.0.0.1:${PORT:-8080}/index.html" || exit 1
CMD ["caddy", "run", "--config", "/etc/caddy/Caddyfile", "--adapter", "caddyfile"]
