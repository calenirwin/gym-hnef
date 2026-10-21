"""Build the original table at the root and preserve Mini at /mini/, without deploying."""

import argparse
import json
import subprocess
import sys
from pathlib import Path

from build_table import ROOT, build


def release(output):
    output = build(output)
    subprocess.run(
        [sys.executable, str(ROOT / "tools/build_demo.py"), "--output", str(output / "mini")],
        check=True,
    )
    sha = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip()
    (output / "deployment.json").write_text(
        json.dumps({"source_sha": sha, "playing_table": "./", "mini": "mini/"}, indent=2) + "\n"
    )
    return output


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    print(release(parser.parse_args().output))
