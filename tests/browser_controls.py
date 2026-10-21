"""Board stability, rules accessibility and reset regressions on the static worker build."""

import functools
import json
import os
import subprocess
import sys
import threading
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

from playwright.sync_api import sync_playwright

ROOT = Path(__file__).resolve().parents[1]
SITE = ROOT / ".demo-build/controls-site"
OUT = ROOT / ".demo-build/table-qa-artifacts"
OUT.mkdir(parents=True, exist_ok=True)
subprocess.run(
    [sys.executable, "tools/build_table.py", "--output", str(SITE / "gym-hnef")],
    cwd=ROOT,
    check=True,
)


class QuietHandler(SimpleHTTPRequestHandler):
    def log_message(self, *args):
        pass


server = ThreadingHTTPServer(("127.0.0.1", 0), functools.partial(QuietHandler, directory=str(SITE)))
threading.Thread(target=server.serve_forever, daemon=True).start()
report = []


def wait(page, condition="live && !busy && !animating"):
    page.wait_for_function("() => " + condition, timeout=60000)


def move(page):
    action = page.evaluate("view.legal[0]")
    for r, c in (action["source"], action["destination"]):
        page.locator(f'#board [data-row="{r}"][data-col="{c}"]').click()
    wait(page)


try:
    with sync_playwright() as p:
        browser = p.chromium.launch(
            executable_path=os.environ.get("CHROMIUM") or None, args=["--no-sandbox"]
        )
        for width, height, profile in [
            (w, h, ruleset)
            for ruleset in ["mini-v1", "copenhagen-experimental-v2"]
            for w, h in [(1440, 900), (1280, 900), (768, 1024), (390, 844), (320, 568)]
        ]:
            page = browser.new_page(viewport={"width": width, "height": height})
            page.add_init_script(
                """localStorage.setItem('hnef-table-v1', JSON.stringify({
              config:{ruleset:PROFILE,mode:'local',policy:'mcts',seed:42},actions:[]
            }));""".replace("PROFILE", json.dumps(profile))
            )
            page.goto(f"http://127.0.0.1:{server.server_port}/gym-hnef/")
            # Record every frame across startup, errors, cancellation and recovery.
            page.evaluate("""() => {
              window.boxes=[];
              function sample() {
                const r=document.querySelector('.arena').getBoundingClientRect();
                boxes.push([r.x+scrollX,r.y+scrollY,r.width,r.height]);
                window.sampler=requestAnimationFrame(sample);
              }
              sample();
            }""")
            wait(page)
            page.evaluate("document.fonts.ready")
            assert page.locator("#board .square").count() == (25 if profile == "mini-v1" else 121)
            baseline = page.locator(".arena").bounding_box()
            page.locator("#rules-button").click()
            assert page.locator("#close-rules").evaluate("(e)=>e===document.activeElement")
            page.locator("#expand-rules").focus()
            page.keyboard.press("Enter")
            assert page.locator("#expand-rules").get_attribute("aria-expanded") == "true"
            assert page.locator("#rules-details").is_visible()
            page.keyboard.press("Shift+Tab")
            assert page.locator("#close-rules").evaluate("(e)=>e===document.activeElement")
            page.keyboard.press("Shift+Tab")
            assert page.locator("#rules-done").evaluate("(e)=>e===document.activeElement")
            for _ in range(8):
                page.keyboard.press("Tab")
                assert page.evaluate("document.activeElement.closest('#rules-dialog')!==null")
            page.locator("#expand-rules").click()
            assert page.locator("#rules-details").is_hidden()
            page.keyboard.press("Escape")
            assert page.locator("#rules-button").evaluate("(e)=>e===document.activeElement")
            page.locator("#rules-button").click()
            page.locator("#close-rules").click()
            # Real worker cancellation before queued calculation completes, followed by retry.
            page.evaluate("() => {request('ai');}")
            page.locator("#cancel-request").click()
            wait(page, "!busy && !!retryAction")
            assert page.evaluate("live.ply") == 0
            page.locator("#retry").click()
            wait(page)
            assert page.evaluate("live.ply") == 1
            # Long failures flow below the anchored board without clipping or horizontal overflow.
            page.evaluate(
                "showError('The engine stopped. Your saved game is unchanged. '.repeat(4))"
            )
            assert page.locator(".arena").bounding_box() == baseline
            page.evaluate("showError('')")
            page.evaluate("review(0)")
            wait(page)
            page.locator("#live").click()
            # New game preserves settings and does not discard a game until Play is confirmed.
            settings = page.evaluate("JSON.stringify(config)")
            saved = page.evaluate("localStorage.getItem('hnef-table-v1')")
            page.locator("#bottom-new-game").click()
            page.once("dialog", lambda dialog: dialog.dismiss())
            page.locator("#setup-form button[type=submit]").click()
            assert page.evaluate("localStorage.getItem('hnef-table-v1')") == saved
            page.once("dialog", lambda dialog: dialog.accept())
            page.locator("#setup-form button[type=submit]").click()
            wait(page, "live.ply===0 && !busy")
            assert page.evaluate("JSON.stringify(config)") == settings
            assert (
                page.evaluate("JSON.parse(localStorage.getItem('hnef-table-v1')).actions.length")
                == 0
            )
            assert page.evaluate("!reviewing() && !retryAction")
            # Starting setup during a calculation invalidates it; no late move survives reset.
            page.evaluate("() => {request('ai');}")
            page.locator("#bottom-new-game").click()
            page.locator("#setup-form button[type=submit]").click()
            wait(page, "live.ply===0 && !busy")
            page.wait_for_timeout(800)
            assert page.evaluate("live.ply") == 0
            # Dismissing setup after cancellation retains the current game and permits play.
            move(page)
            saved = page.evaluate("localStorage.getItem('hnef-table-v1')")
            before = page.evaluate("JSON.stringify(live)")
            page.evaluate("() => {request('ai');}")
            page.locator("#bottom-new-game").click()
            page.keyboard.press("Escape")
            wait(page)
            page.wait_for_timeout(800)
            assert page.evaluate("JSON.stringify(live)") == before
            assert page.evaluate("localStorage.getItem('hnef-table-v1')") == saved
            assert page.evaluate("!retryAction && !busy")
            move(page)
            assert page.evaluate("live.ply") == 2
            page.emulate_media(reduced_motion="reduce")
            page.evaluate("busy=true; render()")
            assert (
                page.locator("#status").evaluate("(e)=>getComputedStyle(e).animationName") == "none"
            )
            page.evaluate("busy=false; render()")
            # Check each supported rules profile against its selected copy.
            for ruleset in ["copenhagen-experimental-v1", "copenhagen-experimental-v2", "mini-v1"]:
                page.evaluate("(r)=>{config.ruleset=r; openRules()}", ruleset)
                page.locator("#expand-rules").click()
                text = page.locator("#rules-dialog").inner_text()
                assert ("any edge" in text) == (ruleset == "mini-v1")
                if ruleset.endswith("v1") and ruleset != "mini-v1":
                    assert "does not detect protected escape positions" in text
                page.locator("#rules-done").click()
            page.evaluate("(r)=>{config.ruleset=r}", profile)
            # Terminal-state result controls and rematch use the same reset path.
            page.evaluate("view.outcome={winner:'attackers',reason:'king_captured'}; render()")
            page.locator("#result-button").click()
            page.locator("#rematch").click()
            wait(page)
            assert page.evaluate("live.ply===0 && !view.outcome")
            # Scroll from bottom controls is user-driven, so compare document coordinates.
            boxes = page.evaluate("cancelAnimationFrame(sampler); boxes")
            (OUT / f"bounds-{profile}-{width}.json").write_text(
                json.dumps({"baseline": baseline, "bounds": boxes, "frames": len(boxes)})
            )
            assert all(
                all(
                    abs(b[i] - baseline[key]) < 0.1
                    for i, key in enumerate(["x", "y", "width", "height"])
                )
                for b in boxes
            )
            page.evaluate("scrollTo(0,0)")
            assert page.locator(".arena").bounding_box() == baseline
            assert page.evaluate("document.documentElement.scrollWidth <= innerWidth")
            page.screenshot(path=str(OUT / f"controls-{profile}-{width}.png"), full_page=True)
            page.locator("#rules-button").click()
            page.screenshot(path=str(OUT / f"rules-{profile}-{width}.png"), full_page=True)
            report.append(
                {
                    "viewport": [width, height],
                    "ruleset": profile,
                    "board": baseline,
                    "checks": (
                        "cancel/retry, review, rules focus, reset, setup dismissal, reduced motion"
                    ),
                }
            )
            print(f"PASS: {profile} at {width}×{height}", flush=True)
            page.close()
        for width, height in [(1440, 900), (1280, 900), (768, 1024), (390, 844), (320, 568)]:
            page = browser.new_page(viewport={"width": width, "height": height})
            errors = []
            page.on("pageerror", lambda error, errors=errors: errors.append(str(error)))
            page.goto(f"http://127.0.0.1:{server.server_port}/gym-hnef/learn/")
            page.wait_for_function("document.body.classList.contains('enhanced')")
            for topic in ["history", "playing", "tactics", "resources", "experiments"]:
                if width < 600:
                    page.locator("#topic-select").select_option(topic)
                else:
                    page.locator(f'.lesson-nav a[href="learn/#{topic}"]').click()
                page.locator(f"#{topic}").wait_for(state="visible")
                if topic == "tactics":
                    for demo in page.locator(".tactic-demo").all():
                        button = demo.locator("button")
                        initial = demo.locator("img").count()
                        button.focus()
                        page.keyboard.press("Enter")
                        assert button.get_attribute("aria-pressed") == "true"
                        assert demo.locator("img").count() == initial - 1
                        page.keyboard.press("Enter")
                        assert demo.locator("img").count() == initial
                    page.emulate_media(reduced_motion="reduce")
                assert page.evaluate("document.documentElement.scrollWidth <= innerWidth")
                page.screenshot(path=str(OUT / f"learn-{topic}-{width}.png"), full_page=True)
            assert not errors, errors
            report.append({"learn_viewport": [width, height], "page_errors": errors})
            page.close()
        page = browser.new_page(java_script_enabled=False)
        page.goto(f"http://127.0.0.1:{server.server_port}/gym-hnef/learn/")
        assert page.locator("#history").is_visible()
        assert page.locator("#tactics details").count() == 4
        page.locator("#tactics details summary").first.click()
        assert "captures d4" in page.locator("#tactics details").first.inner_text()
        page.close()
        browser.close()
finally:
    server.shutdown()
(OUT / "controls.json").write_text(json.dumps(report, indent=2) + "\n")
print(json.dumps(report, indent=2))
