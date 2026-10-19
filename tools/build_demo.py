"""Build a portable static site from the actual engine and an evaluated model export."""

import argparse
import json
import shutil
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=Path, default=ROOT / "web/baseline/mini-ac-random-s42.json")
    parser.add_argument("--report", type=Path, default=ROOT / "web/baseline/evaluation.json")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    model, report = json.loads(args.model.read_text()), json.loads(args.report.read_text())
    if model["checkpoint_sha256"] != report["checkpoint_sha256"]:
        raise ValueError("Model and evaluation identities do not match")
    args.output.mkdir(parents=True, exist_ok=True)
    runtime = ROOT / ".demo-build/runtime"
    if not (runtime / "release.json").exists():
        raise ValueError("Run tools/fetch_browser_runtime.py before building the demo")
    if args.output.resolve() in (ROOT.resolve(), (ROOT / "web").resolve()):
        raise ValueError("Output must be a dedicated build directory")
    if args.output.exists():
        shutil.rmtree(args.output)
    args.output.mkdir(parents=True)
    shutil.copytree(runtime, args.output / "runtime", dirs_exist_ok=True)
    for name in ("index.html", "style.css", "app.mjs", "worker.js", "inference.mjs", "NOTICE.txt"):
        shutil.copyfile(ROOT / "web" / name, args.output / name)
    sources = list((ROOT / "src/hnef/engine").glob("*.py"))
    sources += [
        ROOT / "src/hnef" / name
        for name in ("__init__.py", "actions.py", "browser.py", "policies.py", "replay.py")
    ]
    files = {str(path.relative_to(ROOT / "src")): path.read_text() for path in sources}
    (args.output / "engine.json").write_text(json.dumps({"files": files}))
    shutil.copyfile(args.model, args.output / "model.json")
    shutil.copyfile(ROOT / "docs/mini-rules.md", args.output / "mini-rules.md")
    card = {
        "status": "experimental-baseline",
        "algorithm": "episodic-ac-v1",
        "ruleset": model["ruleset"],
        "checkpoint_sha256": model["checkpoint_sha256"],
        "updates": model["update"],
        "features": model["features"],
        "evaluation": report,
        "limitations": [
            "Trained against random play; not an expert or a strength claim.",
            "Browser and PyTorch logits are parity-tested; sampling RNGs differ.",
            "Games are local to the tab. Refresh starts a new game.",
            "Initial loading downloads the bundled Python WebAssembly runtime.",
        ],
    }
    (args.output / "model-card.json").write_text(json.dumps(card, indent=2) + "\n")
    shutil.copyfile(ROOT / "web/runtime-lock.json", args.output / "runtime-lock.json")
    print(args.output)


if __name__ == "__main__":
    main()
