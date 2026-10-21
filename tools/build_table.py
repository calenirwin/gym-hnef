"""Package the original playing table for static hosting; never publish it."""

import argparse
import hashlib
import json
import shutil
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
WEB = ROOT / "src/hnef/web"


def build(output):
    output = Path(output).resolve()
    # Require an ignored artifact directory. Never erase a checkout or source directory.
    allowed = (ROOT / ".demo-build").resolve()
    if (
        allowed not in output.parents
        or output == allowed / "runtime"
        or allowed / "runtime" in output.parents
    ):
        raise ValueError("Output must be a dedicated directory below .demo-build (not runtime)")
    runtime = ROOT / ".demo-build/runtime"
    lock = json.loads((ROOT / "web/runtime-lock.json").read_text())
    for name, digest in lock["sha256"].items():
        if hashlib.sha256((runtime / name).read_bytes()).hexdigest() != digest:
            raise ValueError(f"Runtime integrity check failed: {name}")
    for name in ("LICENSE-PYODIDE.txt", "LICENSE-PYTHON.txt"):
        if not (runtime / name).is_file():
            raise ValueError(f"Missing runtime licence: {name}")
    if output.exists():
        shutil.rmtree(output)
    output.mkdir(parents=True)
    shutil.copytree(runtime, output / "runtime")
    shutil.copytree(WEB / "assets", output / "assets")
    # Source stays root-relative for the local HTTP server. A static document base
    # makes both the table and /learn/ resolve the same root assets at any subpath.
    for name, destination, base in (
        ("index.html", output / "index.html", "./"),
        ("learn.html", output / "learn/index.html", "../"),
    ):
        html = (WEB / name).read_text()
        html = html.replace("<head>", f'<head>\n  <base href="{base}">')
        html = html.replace('data-mode="http"', 'data-mode="worker"')
        html = html.replace('href="/learn"', 'href="learn/"')
        html = html.replace('href="/learn#', 'href="learn/#')
        html = html.replace('href="/"', 'href="./"')
        html = html.replace('href="/', 'href="').replace('src="/', 'src="')
        # Hash links must stay on Learn, rather than resolve to the table's base.
        if name == "learn.html":
            html = html.replace('href="#', 'href="learn/#')
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_text(html)
    for name in (
        "app.js",
        "motion.js",
        "learn.js",
        "learn-math.js",
        "transport.js",
        "table-worker.mjs",
    ):
        shutil.copyfile(WEB / name, output / name)
    for name in ("style.css", "learn.css"):
        css = (WEB / name).read_text().replace("url('/assets/", "url('assets/")
        css = css.replace('url("/assets/', 'url("assets/')
        (output / name).write_text(css)
    sources = list((ROOT / "src/hnef/engine").glob("*.py"))
    sources += [
        ROOT / "src/hnef" / name
        for name in (
            "__init__.py",
            "actions.py",
            "policies.py",
            "replay.py",
            "web/__init__.py",
            "web/analysis.py",
        )
    ]
    files = {str(path.relative_to(ROOT / "src")): path.read_text() for path in sources}
    (output / "engine.json").write_text(json.dumps({"files": files}))
    (output / "runtime-lock.json").write_text(json.dumps(lock, indent=2) + "\n")
    # Include attribution files without adding unrelated Mini/model artefacts.
    (output / "NOTICE.txt").write_text(
        "Hnef original playing table. Canonical Python rules executed with Pyodide.\n"
        "Runtime licences: runtime/LICENSE-PYODIDE.txt and runtime/LICENSE-PYTHON.txt.\n"
        "Font licences: assets/fonts/Cormorant-LICENSE.txt and assets/fonts/Inter-LICENSE.txt.\n"
        "Copenhagen is experimental: conservative fort recognition, "
        "no automatic dead-position draws.\n"
    )
    return output


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    print(build(parser.parse_args().output))
