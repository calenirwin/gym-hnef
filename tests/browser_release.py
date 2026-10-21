"""Smoke-test the original-only publication artifact under a project subpath."""

import functools
import os
import subprocess
import sys
import threading
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

from playwright.sync_api import sync_playwright

ROOT = Path(__file__).resolve().parents[1]
SITE = ROOT / ".demo-build/release-qa"
subprocess.run(
    [sys.executable, str(ROOT / "tools/build_release.py"), "--output", str(SITE / "gym-hnef")],
    check=True,
)


class QuietHandler(SimpleHTTPRequestHandler):
    def log_message(self, *args):
        pass


server = ThreadingHTTPServer(("127.0.0.1", 0), functools.partial(QuietHandler, directory=str(SITE)))
threading.Thread(target=server.serve_forever, daemon=True).start()
base = f"http://127.0.0.1:{server.server_port}/gym-hnef/"
try:
    with sync_playwright() as p:
        browser = p.chromium.launch(
            executable_path=os.environ.get("CHROMIUM") or None,
            headless=True,
            args=["--no-sandbox"],
        )
        page = browser.new_page()
        errors = []
        page.on("pageerror", lambda e: errors.append(str(e)))
        page.goto(base)
        page.wait_for_function("() => live && !busy", timeout=60000)
        assert page.title() == "Hnef"
        page.locator("#new-game").click()
        page.select_option("#ruleset", "copenhagen-experimental-v2")
        page.select_option("#mode", "local")
        page.locator('#setup-form button[type="submit"]').click()
        page.wait_for_function("() => live.board.length===11 && !busy")
        assert page.locator("#board .square").count() == 121
        assert page.locator('#board img[src$="attacker.png"]').count() == 24
        page.goto(base + "learn/#mcts")
        assert "Learn" in page.title()
        # The release must not include the separate Mini app or its model.
        assert not (SITE / "gym-hnef/mini").exists()
        assert not (SITE / "gym-hnef/model.json").exists()
        assert page.request.get(base + "mini/").status == 404
        response = page.request.get(base + "runtime/pyodide.asm.wasm")
        assert response.status == 200
        assert "application/wasm" in response.headers["content-type"]
        assert page.request.get(base + "runtime/LICENSE-PYODIDE.txt").status == 200
        metadata = page.request.get(base + "deployment.json").json()
        assert (
            metadata["source_sha"]
            == subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip()
        )
        assert "mini" not in metadata
        assert not errors, errors
        browser.close()
finally:
    server.shutdown()
print("PASS: original root/Copenhagen/Learn, no separate Mini route, WASM/licence, deployment SHA")
