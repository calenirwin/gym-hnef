"""Smoke-test the exact combined publication artifact under a project subpath."""

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
        # Mini stays a separate, functional app with all of its own relative assets.
        page.goto(base + "mini/")
        page.locator("#mode").wait_for()
        page.wait_for_function("() => !document.getElementById('mode').disabled", timeout=60000)
        page.select_option("#mode", "local")
        page.wait_for_function("() => !document.getElementById('new-game').disabled")
        page.locator('#board [aria-label$=", attacker"]').first.click()
        page.locator("#board .destination").first.click()
        page.wait_for_function("() => document.getElementById('ply').textContent==='PLY 01'")
        for prefix in ("", "mini/"):
            response = page.request.get(base + prefix + "runtime/pyodide.asm.wasm")
            assert response.status == 200
            assert "application/wasm" in response.headers["content-type"]
            assert page.request.get(base + prefix + "runtime/LICENSE-PYODIDE.txt").status == 200
        metadata = page.request.get(base + "deployment.json").json()
        assert (
            metadata["source_sha"]
            == subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip()
        )
        assert not errors, errors
        browser.close()
finally:
    server.shutdown()
print(
    "PASS: original root/Copenhagen/Learn, Mini gameplay, both WASM paths/licences, deployment SHA"
)
