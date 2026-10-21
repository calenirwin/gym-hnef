"""Opt-in runtime probe; requires Playwright + Chromium, never deploys anything."""

import functools
import hashlib
import json
import os
import platform
import shutil
import sys
import threading
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from time import perf_counter

from playwright.sync_api import sync_playwright

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
from hnef.engine import Ruleset  # noqa: E402
from hnef.replay import random_game  # noqa: E402
from hnef.web import game_response  # noqa: E402
from hnef.web.analysis import analyze_response  # noqa: E402

DEST = ROOT / ".demo-build/pyodide-gate"
DEST.mkdir(parents=True, exist_ok=True)
lock = json.loads((ROOT / "web/runtime-lock.json").read_text())
for name, digest in lock["sha256"].items():
    assert hashlib.sha256((ROOT / ".demo-build/runtime" / name).read_bytes()).hexdigest() == digest
shutil.copytree(ROOT / ".demo-build/runtime", DEST / "runtime", dirs_exist_ok=True)
for name in ("index.html", "worker.mjs"):
    shutil.copyfile(Path(__file__).parent / name, DEST / name)
sources = list((ROOT / "src/hnef/engine").glob("*.py"))
sources += [
    ROOT / "src/hnef" / n
    for n in (
        "__init__.py",
        "actions.py",
        "policies.py",
        "replay.py",
        "web/__init__.py",
        "web/analysis.py",
    )
]
files = {str(p.relative_to(ROOT / "src")): p.read_text() for p in sources}
(DEST / "engine.json").write_text(json.dumps(files))
fixtures = []


def add(name, payload, kind="game"):
    started = perf_counter()
    try:
        value = {"result": (analyze_response if kind == "analyze" else game_response)(payload)}
    except (ValueError, TypeError, KeyError) as error:
        value = {"error": {"type": type(error).__name__, "message": str(error)}}
    fixtures.append(
        {
            "name": name,
            "payload": json.loads(json.dumps(payload)),
            "kind": kind,
            "expected": json.loads(json.dumps(value)),
            "native_ms": (perf_counter() - started) * 1000,
        }
    )


for rules in Ruleset:
    opening = game_response({"ruleset": rules})
    add(f"{rules}:opening", {"ruleset": rules})
    add(
        f"{rules}:move",
        {"ruleset": rules, "operation": "move", "action": opening["legal"][0]["action"]},
    )
    record = json.loads(random_game(42, max_plies=12, ruleset=rules).to_json())
    add(f"{rules}:import", {"operation": "import", "replay": record})
    add(f"{rules}:restore", {"ruleset": rules, "actions": record["actions"]})
    for policy in ("random", "heuristic", "mcts"):
        add(
            f"{rules}:{policy}", {"ruleset": rules, "operation": "ai", "policy": policy, "seed": 42}
        )
    add(f"{rules}:analyze", {"ruleset": rules, "iterations": 8, "rollout_plies": 0}, "analyze")
record = json.loads(
    random_game(3, max_plies=4, ruleset=Ruleset.COPENHAGEN_V2)
    .adjudicate_draw("Agreed dead position")
    .to_json()
)
add("adjudicated draw", {"operation": "import", "replay": record})
record["state_keys"][0] = "tampered"
add("tampered replay", {"operation": "import", "replay": record})
for name, payload, kind in [
    ("invalid ruleset", {"ruleset": "invalid"}, "game"),
    ("invalid action", {"operation": "move", "action": -1}, "game"),
    ("too many plies", {"actions": [0] * 501}, "game"),
    ("invalid seed", {"operation": "ai", "seed": -1}, "game"),
    ("malformed replay", {"operation": "import", "replay": {}}, "game"),
    ("invalid analyze budget", {"iterations": 100000}, "analyze"),
    ("non-object request", [], "game"),
]:
    add(name, payload, kind)
(DEST / "fixtures.json").write_text(json.dumps(fixtures))


class QuietHandler(SimpleHTTPRequestHandler):
    def log_message(self, *args):
        pass


server = ThreadingHTTPServer(("127.0.0.1", 0), functools.partial(QuietHandler, directory=str(DEST)))
threading.Thread(target=server.serve_forever, daemon=True).start()
report = {
    "native_python": sys.version,
    "platform": platform.platform(),
    "runtime": lock["version"],
    "engine_files": len(files),
    "engine_bytes": (DEST / "engine.json").stat().st_size,
    "source_sha256": {n: hashlib.sha256(s.encode()).hexdigest() for n, s in files.items()},
    "parity": [],
    "benchmarks": [],
}
try:
    with sync_playwright() as p:
        browser = p.chromium.launch(
            executable_path=os.environ.get("CHROMIUM", "/usr/bin/chromium"),
            headless=True,
            args=["--no-sandbox"],
        )
        page = browser.new_page(viewport={"width": 390, "height": 844})
        requests = []
        page.on("request", lambda req: requests.append(req.url))
        page.goto(f"http://127.0.0.1:{server.server_port}")
        report["browser"] = browser.version
        report["init"] = page.evaluate("() => rpc('init',{})")
        print("INIT", json.dumps(report["init"]), flush=True)
        for fixture in fixtures:
            result = page.evaluate("f => rpc(f.kind,f.payload)", fixture)
            actual = {k: v for k, v in result.items() if k in ("result", "error")}
            assert actual == fixture["expected"], fixture["name"]
            report["parity"].append(
                {
                    "name": fixture["name"],
                    "native_ms": fixture["native_ms"],
                    "worker_ms": result["elapsed"],
                }
            )
        print("PARITY", len(fixtures), "passed", flush=True)
        # Independent native timings and exact response equality for each benchmark.
        workloads = [
            ("opening", "game", {"ruleset": Ruleset.COPENHAGEN_V2}),
            (
                "move",
                "game",
                {
                    "ruleset": Ruleset.COPENHAGEN_V2,
                    "operation": "move",
                    "action": game_response({"ruleset": Ruleset.COPENHAGEN_V2})["legal"][0][
                        "action"
                    ],
                },
            ),
            (
                "mcts",
                "game",
                {"ruleset": Ruleset.COPENHAGEN_V2, "operation": "ai", "policy": "mcts", "seed": 42},
            ),
            (
                "learn-max",
                "analyze",
                {"ruleset": Ruleset.COPENHAGEN_V2, "iterations": 64, "rollout_plies": 48},
            ),
        ]
        cdp = page.context.new_cdp_session(page)
        for rate in (1, 4):
            cdp.send("Emulation.setCPUThrottlingRate", {"rate": rate})
            for name, kind, payload in workloads:
                values = []
                native = []
                gaps = []
                for _ in range(3):
                    start = perf_counter()
                    expected = (analyze_response if kind == "analyze" else game_response)(payload)
                    native.append((perf_counter() - start) * 1000)
                    page.evaluate("() => {frames=[]}")
                    result = page.evaluate(
                        "x => rpc(x.kind,x.payload)", {"kind": kind, "payload": payload}
                    )
                    assert result["result"] == json.loads(json.dumps(expected)), (name, rate)
                    values.append(result["elapsed"])
                    frames = page.evaluate("frames")
                    gaps.extend(b - a for a, b in zip(frames, frames[1:], strict=False))
                entry = {
                    "workload": name,
                    "cpu_throttle": rate,
                    "worker_ms": values,
                    "native_ms": native,
                    "frame_gap_max_ms": max(gaps) if gaps else None,
                    "frame_gap_p95_ms": sorted(gaps)[int(len(gaps) * 0.95)] if gaps else None,
                }
                report["benchmarks"].append(entry)
                print("BENCH", json.dumps(entry), flush=True)
        cdp.send("Emulation.setCPUThrottlingRate", {"rate": 1})
        # Deadline is consumer-only: synchronous Python keeps running until terminated.
        outcome = page.evaluate("""async () => {
      const count=responses.length;
      let error='';
      try {
        await rpc('analyze',
          {ruleset:'copenhagen-experimental-v2',iterations:64,rollout_plies:48},10)
      } catch(e){error=e.message}
      return {error,count};
    }""")
        assert outcome["error"] == "Probe deadline exceeded"
        page.wait_for_function(
            "count => responses.length>count", arg=outcome["count"], timeout=60000
        )
        report["deadline"] = {"error": outcome["error"], "late_reply_observed": True}
        report["restart"] = page.evaluate("""async () => {
      const saved={ruleset:'copenhagen-experimental-v2',actions:[]};
      const opening=await rpc('game',saved);
      saved.actions=[opening.result.legal[0].action];
      const first=await rpc('game',saved);
      localStorage.setItem('gate-save',JSON.stringify(first.result.replay));
      const pending=rpc('analyze',{...saved,iterations:64,rollout_plies:48}).catch(e=>e.message);
      await new Promise(resolve=>setTimeout(resolve,20));
      restart(); const cancelled=await pending; await rpc('init',{});
      const replay=JSON.parse(localStorage.getItem('gate-save'));
      const restored=await rpc('game',{operation:'import',replay});
      return {cancelled,identical:JSON.stringify(first.result)===JSON.stringify(restored.result)};
    }""")
        assert report["restart"] == {"cancelled": "Worker restarted", "identical": True}
        # Real failure at startup rejects, rather than leaving promises pending.
        page.route("**/engine.json", lambda route: route.abort())
        report["init_failure"] = page.evaluate(
            "async () => {restart();try {await rpc('init',{});return 'unexpected success'}"
            " catch(e){return e.message}}"
        )
        assert "fetch" in report["init_failure"].lower()
        report["request_paths"] = sorted(
            set(url.split(f":{server.server_port}")[-1] for url in requests)
        )
        report["external_requests"] = [
            url for url in requests if not url.startswith(f"http://127.0.0.1:{server.server_port}")
        ]
        assert not report["external_requests"]
        browser.close()
finally:
    server.shutdown()
(DEST / "results.json").write_text(json.dumps(report, indent=2) + "\n")
print("PASS", DEST / "results.json", flush=True)
