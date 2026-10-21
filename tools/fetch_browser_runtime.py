"""Vendor the official pinned Pyodide release into an ignored build directory."""

import hashlib
import json
import urllib.request
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

VERSION = "314.0.7"
DESTINATION = Path(__file__).resolve().parents[1] / ".demo-build/runtime"
FILES = (
    "pyodide.mjs",
    "pyodide.asm.mjs",
    "pyodide.asm.wasm",
    "python_stdlib.zip",
    "pyodide-lock.json",
)
LOCK = json.loads((Path(__file__).resolve().parents[1] / "web/runtime-lock.json").read_text())


def fetch(name):
    path = DESTINATION / name
    if not path.exists():
        temporary = path.with_suffix(path.suffix + ".tmp")
        urllib.request.urlretrieve(
            f"https://cdn.jsdelivr.net/pyodide/v{VERSION}/full/{name}", temporary
        )
        temporary.replace(path)
    digest = hashlib.sha256(path.read_bytes()).hexdigest()
    if digest != LOCK["sha256"][name]:
        raise ValueError(f"Pinned runtime checksum mismatch: {name}")
    return name, digest


if __name__ == "__main__":
    DESTINATION.mkdir(parents=True, exist_ok=True)
    with ThreadPoolExecutor(max_workers=5) as executor:
        checksums = dict(executor.map(fetch, FILES))
    licenses = {
        "LICENSE-PYODIDE.txt": f"https://raw.githubusercontent.com/pyodide/pyodide/{VERSION}/LICENSE",
        "LICENSE-PYTHON.txt": "https://raw.githubusercontent.com/python/cpython/v3.14.2/LICENSE",
    }
    for name, url in licenses.items():
        if not (DESTINATION / name).exists():
            urllib.request.urlretrieve(url, DESTINATION / name)
    (DESTINATION / "release.json").write_text(
        json.dumps({"version": VERSION, "sha256": checksums}, indent=2)
    )
    print(DESTINATION)
