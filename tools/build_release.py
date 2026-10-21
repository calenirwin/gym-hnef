"""Build only the original playing table at the root, without deploying."""

import argparse
import json
import subprocess
from pathlib import Path

from build_table import ROOT, build


def release(output):
    output = build(output)
    sha = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip()
    (output / "deployment.json").write_text(
        json.dumps({"source_sha": sha, "playing_table": "./"}, indent=2) + "\n"
    )
    return output


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    print(release(parser.parse_args().output))
