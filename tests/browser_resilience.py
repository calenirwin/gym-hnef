"""Synthetic network, pressure-notification and forced-crash tests; not device/OOM QA."""

import asyncio
import functools
import json
import os
import subprocess
import sys
import threading
import time
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

from playwright.async_api import async_playwright
from playwright.sync_api import Error, sync_playwright

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from hnef.engine import Ruleset  # noqa: E402
from hnef.replay import random_game  # noqa: E402
from hnef.web import game_response  # noqa: E402

SITE = ROOT / ".demo-build/resilience-site"
OUT = ROOT / ".demo-build/table-qa-artifacts/resilience.json"
OUT.parent.mkdir(parents=True, exist_ok=True)
subprocess.run(
    [sys.executable, str(ROOT / "tools/build_table.py"), "--output", str(SITE / "gym-hnef")],
    check=True,
)
record = json.loads(random_game(42, max_plies=12, ruleset=Ruleset.COPENHAGEN_V2).to_json())
save = {
    "config": {"ruleset": record["ruleset"], "mode": "local", "policy": "mcts", "seed": 42},
    "actions": record["actions"],
    "replay": record,
    "flipped": False,
}
expected = game_response({"operation": "import", "replay": record})
network = {"mode": "fast", "requests": []}
report = {
    "conditions": {
        "cache": "fresh contexts; Cache-Control no-store on every response",
        "slow_runtime": (
            "150ms response latency; 512 KiB/s per runtime response, "
            "32 KiB chunks (not aggregate link bandwidth)"
        ),
        "interruption": "WASM HTTP body closed after 64 KiB with original Content-Length",
        "memory": (
            "64 MiB touched ArrayBuffer plus CDP critical pressure notification; no exhaustion"
        ),
        "crash": "CDP Page.crash during MCTS; forced renderer failure, not OOM",
    },
    "checks": [],
}


class NetworkHandler(SimpleHTTPRequestHandler):
    def log_message(self, *args):
        pass

    def end_headers(self):
        self.send_header("Cache-Control", "no-store")
        super().end_headers()

    def do_GET(self):
        mode = network["mode"]
        if "/runtime/" not in self.path:
            return super().do_GET()
        path = Path(self.translate_path(self.path))
        if not path.is_file():
            return super().do_GET()
        entry = {"path": self.path, "mode": mode, "bytes_sent": 0, "complete": False}
        network["requests"].append(entry)
        started = time.monotonic()
        try:
            if mode == "slow":
                time.sleep(0.15)
            self.send_response(200)
            self.send_header("Content-Type", self.guess_type(str(path)))
            self.send_header("Content-Length", str(path.stat().st_size))
            self.end_headers()
            with path.open("rb") as source:
                while chunk := source.read(32768):
                    self.wfile.write(chunk)
                    self.wfile.flush()
                    entry["bytes_sent"] += len(chunk)
                    if (
                        mode == "interrupt"
                        and path.suffix == ".wasm"
                        and entry["bytes_sent"] >= 65536
                    ):
                        self.close_connection = True
                        return
                    if mode == "slow":
                        time.sleep(len(chunk) / (512 * 1024))
            entry["complete"] = True
        except (BrokenPipeError, ConnectionResetError):
            entry["client_cancelled"] = True
        finally:
            entry["seconds"] = round(time.monotonic() - started, 3)


server = ThreadingHTTPServer(
    ("127.0.0.1", 0), functools.partial(NetworkHandler, directory=str(SITE))
)
threading.Thread(target=server.serve_forever, daemon=True).start()
base = f"http://127.0.0.1:{server.server_port}/gym-hnef/"


def wait(page, expression):
    page.wait_for_function("() => " + expression, timeout=90000)


def ready(page, key=expected["key"]):
    wait(page, "typeof live!=='undefined' && live && !busy")
    assert page.evaluate("live.key") == key
    assert page.locator("#error").is_hidden()


def mark(name, **data):
    report["checks"].append({"name": name, **data})
    print("PASS:", name, data, flush=True)


def seeded_context(browser):
    context = browser.new_context(viewport={"width": 390, "height": 844})
    # Seed once before app code, leaving every later page/retry to use its real saved state.
    context.add_init_script(
        "if (!localStorage.getItem('hnef-table-v1')) {"
        + "localStorage.setItem('hnef-table-v1', "
        + json.dumps(json.dumps(save))
        + ");}"
    )
    return context


async def crash_recovery(storage, before, expected_next):
    # Page.crash may never return a CDP response. Bound the test command explicitly.
    async with async_playwright() as p:
        browser = await p.chromium.launch(
            executable_path=os.environ.get("CHROMIUM") or None,
            headless=True,
            args=["--no-sandbox"],
        )
        context = await browser.new_context(storage_state=storage)
        page = await context.new_page()
        await page.goto(base)
        await page.wait_for_function("() => live && !busy", timeout=60000)
        assert await page.evaluate("live.key") == expected_next["key"]
        assert await page.evaluate("localStorage.getItem('hnef-table-v1')") == before
        await page.evaluate("() => {request('ai');}")
        await page.wait_for_function("() => busy && transport.status==='Calculating…'")
        cdp = await context.new_cdp_session(page)
        crashed = []
        page.on("crash", lambda _: crashed.append(True))
        try:
            await asyncio.wait_for(cdp.send("Page.crash"), timeout=5)
        except (Error, TimeoutError):
            pass  # A crashed renderer may never acknowledge its crash command.
        assert crashed, "No renderer crash event; cannot claim this test ran"
        await page.close()
        recovered = await context.new_page()
        await recovered.goto(base)
        await recovered.wait_for_function("() => live && !busy", timeout=60000)
        assert await recovered.evaluate("live.key") == expected_next["key"]
        assert await recovered.evaluate("live.ply") == 13
        actual_save = await recovered.evaluate("localStorage.getItem('hnef-table-v1')")
        assert json.loads(actual_save) == json.loads(before)
        next_result = game_response(
            {**save["config"], "actions": expected_next["actions"], "operation": "ai"}
        )
        await recovered.evaluate("() => {request('ai');}")
        await recovered.wait_for_function("() => live.ply===14 && !busy", timeout=60000)
        assert await recovered.evaluate("live.key") == next_result["key"]
        mark("forced renderer crash restores confirmed replay and no stale AI action")
        await context.close()
        await browser.close()


try:
    with sync_playwright() as p:
        browser = p.chromium.launch(
            executable_path=os.environ.get("CHROMIUM") or None, headless=True, args=["--no-sandbox"]
        )
        report["browser"] = browser.version
        network["mode"] = "slow"
        context = seeded_context(browser)
        page = context.new_page()
        started = time.monotonic()
        page.goto(base)
        ready(page)
        assert page.evaluate("live.ply") == 12
        for piece in ("attacker", "defender", "king"):
            assert page.evaluate(
                """async piece => {
              const image=new Image();image.src=transport.url(`assets/${piece}.png`);
              await image.decode();return image.naturalWidth>0;
            }""",
                piece,
            )
        mark(
            "cold-cache slow-runtime startup preserves replay/art",
            seconds=round(time.monotonic() - started, 3),
        )
        context.close()
        network["mode"] = "interrupt"
        context = seeded_context(browser)
        page = context.new_page()
        page.goto(base)
        original = json.dumps(save)
        for attempt in range(2):
            wait(
                page,
                "typeof busy!=='undefined' && !busy && !document.getElementById('error').hidden",
            )
            assert page.evaluate("localStorage.getItem('hnef-table-v1')") == original
            assert page.evaluate("live===null && transport.worker===null")
            if attempt == 0:
                page.locator("#retry").click()
        failed_wasm = [
            r
            for r in network["requests"]
            if r["mode"] == "interrupt" and r["path"].endswith(".wasm")
        ]
        assert len(failed_wasm) == 2 and all(
            r["bytes_sent"] == 65536 and not r["complete"] for r in failed_wasm
        )
        mark(
            "two interrupted cold WASM attempts preserve exact autosave", attempts=len(failed_wasm)
        )
        # Repeatable shortened deadline tests the real cancellation path, not a 60s wait.
        network["mode"] = "slow"
        page.evaluate("transport.startupTimeout=200")
        page.locator("#retry").click()
        wait(page, "!busy && !document.getElementById('error').hidden")
        assert "timed out" in page.locator("#error").inner_text()
        assert page.evaluate("transport.worker===null")
        assert page.evaluate("localStorage.getItem('hnef-table-v1')") == original
        network["mode"] = "fast"
        page.evaluate("transport.startupTimeout=60000")
        page.locator("#retry").click()
        ready(page)
        assert page.evaluate("live.ply") == 12
        mark("startup deadline and fourth-attempt recovery have no stale move")
        # Bounded allocation + a browser pressure signal; deliberately do not exhaust memory.
        cdp = context.new_cdp_session(page)
        allocated = page.evaluate("""() => {
          window.pressureBuffer=new Uint8Array(64*1024*1024);
          for(let i=0;i<pressureBuffer.length;i+=4096) pressureBuffer[i]=1;
          return pressureBuffer.byteLength;
        }""")
        try:
            cdp.send("Memory.simulatePressureNotification", {"level": "critical"})
            report["pressure_notification"] = "supported and sent"
        except Error as error:
            report["pressure_notification"] = "unsupported: " + str(error)
        assert allocated == 67108864
        expected_next = game_response(
            {**save["config"], "actions": record["actions"], "operation": "ai"}
        )
        page.evaluate("() => {request('ai');}")
        ready(page, expected_next["key"])
        assert page.evaluate("live.ply") == 13
        page.evaluate("window.pressureBuffer=null")
        mark(
            "bounded allocation and pressure notification preserve canonical MCTS result",
            allocated_bytes=allocated,
        )
        before = page.evaluate("localStorage.getItem('hnef-table-v1')")
        browser_storage = context.storage_state()
        context.close()
        browser.close()
    asyncio.run(crash_recovery(browser_storage, before, expected_next))
finally:
    server.shutdown()
report["runtime_requests"] = network["requests"]
OUT.write_text(json.dumps(report, indent=2) + "\n")
print("Evidence:", OUT)
