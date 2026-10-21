"""Real-browser original-table parity/recovery checks, also exercised in CI."""

import functools
import json
import os
import shutil
import subprocess
import sys
import threading
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

from playwright.sync_api import sync_playwright

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from table_parity import api_fixtures, tactical_results  # noqa: E402

from hnef.web import GameHandler  # noqa: E402

SITE = ROOT / ".demo-build/table-qa"
ARTIFACTS = ROOT / ".demo-build/table-qa-artifacts"
ARTIFACTS.mkdir(parents=True, exist_ok=True)
subprocess.run(
    [sys.executable, str(ROOT / "tools/build_table.py"), "--output", str(SITE / "gym-hnef")],
    check=True,
)
# Assert preserved images/fonts and package contract before adding test-only files.
for path in (ROOT / "src/hnef/web/assets").rglob("*"):
    if path.is_file():
        assert (
            path.read_bytes()
            == (
                SITE / "gym-hnef/assets" / path.relative_to(ROOT / "src/hnef/web/assets")
            ).read_bytes()
        )
assert not (SITE / "gym-hnef/_test").exists()
assert not (SITE / "gym-hnef/model.json").exists()
test_dir = SITE / "gym-hnef/_test"
test_dir.mkdir()
for source, dest in (
    ("tests/table-tactics-worker.mjs", "tactics.mjs"),
    ("tests/table_parity.py", "table_parity.py"),
    ("tests/fixtures/exit-forts.json", "exit-forts.json"),
):
    shutil.copyfile(ROOT / source, test_dir / dest)
fixtures = api_fixtures()
tactics = tactical_results(json.loads((ROOT / "tests/fixtures/exit-forts.json").read_text()))


class QuietHandler(SimpleHTTPRequestHandler):
    def log_message(self, *args):
        pass


static_server = ThreadingHTTPServer(
    ("127.0.0.1", 0), functools.partial(QuietHandler, directory=str(SITE))
)
local_server = ThreadingHTTPServer(("127.0.0.1", 0), GameHandler)
for server in (static_server, local_server):
    threading.Thread(target=server.serve_forever, daemon=True).start()
base = f"http://127.0.0.1:{static_server.server_port}/gym-hnef/"
report = {"api_parity": [], "tactical_count": len(tactics), "checks": []}


def mark(check):
    report["checks"].append(check)
    print("PASS:", check, flush=True)


def wait(page, condition):
    page.wait_for_function("() => " + condition, timeout=60000)


def move(page):
    wait(page, "view && !busy && !animating")
    action = page.evaluate("view.legal[0]")
    for r, c in (action["source"], action["destination"]):
        page.locator(f'#board [data-row="{r}"][data-col="{c}"]').click()


def new_copenhagen(page):
    page.locator("#new-game").click()
    page.select_option("#ruleset", "copenhagen-experimental-v2")
    page.select_option("#mode", "local")
    page.locator('#setup-form button[type="submit"]').click()
    wait(page, "live.board.length===11 && !busy")


try:
    with sync_playwright() as p:
        browser = p.chromium.launch(
            executable_path=os.environ.get("CHROMIUM") or None, headless=True, args=["--no-sandbox"]
        )
        report["browser"] = browser.version
        page = browser.new_page(viewport={"width": 1280, "height": 900}, accept_downloads=True)
        errors, requests = [], []
        page.on("pageerror", lambda error: errors.append(str(error)))
        page.on("request", lambda request: requests.append(request.url))
        page.goto(base)
        wait(page, "typeof live !== 'undefined' && live && !busy")
        for fixture in fixtures:
            actual = page.evaluate(
                """async f => {
              const r=await transport.request(f.kind,f.payload,{remember:false,timeout:60000});
              return {ok:r.ok,data:await r.json()};
            }""",
                fixture,
            )
            assert actual == fixture["expected"], fixture["name"]
            report["api_parity"].append(fixture["name"])
        actual = page.evaluate("""() => new Promise((resolve,reject)=>{
          const w=new Worker(new URL('_test/tactics.mjs',location.href),{type:'module'});
          const timer=setTimeout(()=>{
            w.terminate();reject(new Error('Tactical test timeout'));
          },60000);
          w.onmessage=({data})=>{
            clearTimeout(timer);w.terminate();
            data.error?reject(new Error(data.error)):resolve(data.result);
          };
          w.onerror=e=>{clearTimeout(timer);w.terminate();reject(new Error(e.message))};
          w.postMessage({});
        })""")
        assert actual == tactics
        mark("complete API and tactical parity")
        move(page)
        wait(page, "live.ply===2 && !busy && !animating")
        page.locator("#undo").click()
        wait(page, "live.ply===0 && !busy")
        new_copenhagen(page)
        assert page.locator("#board .square").count() == 121
        for ply in (1, 2):
            move(page)
            wait(page, f"live.ply==={ply} && !busy && !animating")
        saved = page.evaluate("localStorage.getItem('hnef-table-v1')")
        with page.expect_download() as download:
            page.locator("#download").click()
        replay_path = ARTIFACTS / "replay.json"
        download.value.save_as(str(replay_path))
        page.reload()
        wait(page, "live && live.ply===2 && !busy")
        assert json.loads(page.evaluate("localStorage.getItem('hnef-table-v1')")) == json.loads(
            saved
        )
        page.locator("#flip").click()
        page.locator("#history-button").click()
        page.locator("#replay-file").set_input_files(str(replay_path))
        wait(page, "reviewMode && view.ply===0 && !busy")
        page.locator("#review-forward").click()
        wait(page, "view.ply===1 && !busy")
        page.locator("#live").click()
        wait(page, "!reviewMode && view.ply===2")
        mark("human/AI, undo, Copenhagen, reload, replay import/export/review, rotation")
        # Explicit cancellation kills real work. Retry must resume from confirmed replay.
        page.evaluate("() => { config.policy='mcts'; request('ai'); }")
        wait(page, "busy && transport.status==='Calculating…'")
        page.locator("#cancel-request").click()
        wait(page, "!busy && !document.getElementById('error').hidden")
        assert page.evaluate("live.ply") == 2
        assert (
            json.loads(page.evaluate("localStorage.getItem('hnef-table-v1')"))["actions"]
            == json.loads(saved)["actions"]
        )
        assert page.evaluate("transport.worker===null")
        page.locator("#retry").click()
        wait(page, "live.ply===3 && !busy")
        assert page.locator("#error").is_hidden()
        # Deadline uses actual worker termination; test through UI retry at normal budget.
        page.evaluate("""() => {
          const original=transport.request.bind(transport);
          transport.request=(kind,payload,options)=>{
            transport.request=original;return original(kind,payload,{...options,timeout:1});
          };
          request('ai');
        }""")
        wait(page, "!busy && !document.getElementById('error').hidden")
        assert "timed out" in page.locator("#error").inner_text()
        assert page.evaluate("live.ply") == 3
        page.locator("#retry").click()
        wait(page, "live.ply===4 && !busy")
        mark("real worker cancel, deadline, replay recovery and UI retry")
        # A failed restart leaves browser autosave unchanged; Retry restarts and restores it.
        saved = page.evaluate("localStorage.getItem('hnef-table-v1')")
        page.route("**/engine.json", lambda route: route.abort())
        page.reload()
        wait(page, "typeof busy!=='undefined' && !busy && !document.getElementById('error').hidden")
        assert page.evaluate("localStorage.getItem('hnef-table-v1')") == saved
        page.unroute("**/engine.json")
        page.locator("#retry").click()
        wait(page, "live && live.ply===4 && !busy")
        mark("startup failure and retry preserve autosave")
        # Phone-width mouse and keyboard checks; explicitly not a physical-phone claim.
        page.set_viewport_size({"width": 390, "height": 844})
        move(page)
        wait(page, "live.ply===5 && !busy && !animating")
        assert page.evaluate("document.documentElement.scrollWidth <= innerWidth")
        cell = page.locator('#board [tabindex="0"]')
        cell.focus()
        page.keyboard.press("ArrowRight")
        assert page.evaluate("document.activeElement.getAttribute('role')") == "gridcell"
        for asset in ("attacker", "defender", "king"):
            assert page.evaluate(
                """async name=>{
                  const i=new Image();i.src=transport.url(`assets/${name}.png`);
                  await i.decode();return i.naturalWidth>0;
                }""",
                asset,
            )
        page.screenshot(path=str(ARTIFACTS / "mobile.png"), full_page=True)
        mark("390px play/keyboard/layout and decoded original artwork")
        # Real worker crash on the opponent turn; the next human turn must still schedule AI.
        page.locator("#new-game").click()
        page.select_option("#mode", "attackers")
        page.select_option("#policy", "mcts")
        page.once("dialog", lambda dialog: dialog.accept())
        page.locator('#setup-form button[type="submit"]').click()
        wait(page, "live.ply===0 && !busy")
        move(page)
        wait(page, "live.ply===1 && busy")
        page.evaluate("""() => transport.worker.dispatchEvent(
          new ErrorEvent('error',{cancelable:true,message:'Injected worker crash'}))""")
        wait(page, "!busy && !document.getElementById('error').hidden")
        assert page.evaluate("live.ply") == 1
        assert page.locator("#download").is_enabled()
        page.locator("#retry").click()
        wait(page, "live.ply===2 && !busy && !animating")
        move(page)
        wait(page, "live.ply===4 && !busy && !animating")
        before_import = page.evaluate("localStorage.getItem('hnef-table-v1')")
        page.evaluate("() => importReplay(JSON.stringify({version:1,actions:[]}))")
        assert page.evaluate("localStorage.getItem('hnef-table-v1')") == before_import
        mark("opponent worker crash, retry, subsequent AI scheduling and invalid-import safety")
        saved = page.evaluate("localStorage.getItem('hnef-table-v1')")
        page.goto(base + "learn/#experiments")
        wait(page, "typeof experimentRunning!=='undefined'")
        page.select_option("#experiment-position", "saved")
        page.select_option("#experiment-budget", "8")
        page.select_option("#experiment-rollout", "0")
        page.locator("#experiment-run").click()
        page.wait_for_selector("#experiment-results", timeout=60000)
        assert page.evaluate("localStorage.getItem('hnef-table-v1')") == saved
        page.select_option("#experiment-budget", "64")
        page.select_option("#experiment-rollout", "48")
        page.locator("#experiment-run").click()
        wait(page, "transport.status==='Analyzing…'")
        page.locator("#experiment-cancel").click()
        wait(page, "!experimentRunning")
        assert "cancelled" in page.locator("#experiment-error").inner_text()
        page.select_option("#experiment-budget", "8")
        page.select_option("#experiment-rollout", "0")
        page.locator("#experiment-run").click()
        page.wait_for_selector("#experiment-results", timeout=60000)
        page.select_option("#topic-select", "mcts")
        page.locator("#next-lesson").click()
        assert "/gym-hnef/learn/" in page.url
        assert page.url.endswith("#neural")
        page.reload()
        assert "Learn" in page.title()
        mark("Learn saved position, cancel/retry, deep-link and pagination")
        assert all(url.startswith(base) or url.startswith("blob:") for url in requests), requests
        assert not errors, errors
        # Local HTTP mode has the same controls and API; no worker is created.
        local = browser.new_page()
        local.goto(f"http://127.0.0.1:{local_server.server_port}/")
        wait(local, "live && !busy")
        assert local.evaluate("transport.mode") == "http"
        move(local)
        wait(local, "live.ply===2 && !busy")
        local.goto(f"http://127.0.0.1:{local_server.server_port}/learn/#experiments")
        local.locator("#experiment-run").click()
        local.wait_for_selector("#experiment-results", timeout=60000)
        mark("local HTTP gameplay and Learn")
        browser.close()
finally:
    for server in (static_server, local_server):
        server.shutdown()
(ARTIFACTS / "results.json").write_text(json.dumps(report, indent=2) + "\n")
print(json.dumps(report, indent=2))
