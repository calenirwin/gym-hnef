"""Local run catalog and content-addressed release bundles."""

import hashlib
import json
import re
import subprocess
from datetime import UTC, datetime
from pathlib import Path
from uuid import uuid4


def now():
    return datetime.now(UTC).isoformat()


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def write_json(path, data):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{uuid4().hex}.tmp")
    try:
        temporary.write_text(json.dumps(data, indent=2, allow_nan=False) + "\n")
        temporary.replace(path)
    finally:
        temporary.unlink(missing_ok=True)


def provenance():
    root = Path(__file__).resolve().parents[3]

    def git(*args):
        result = subprocess.run(
            ["git", "-C", str(root), *args], capture_output=True, text=True, check=False
        )
        return result.stdout.strip() if result.returncode == 0 else None

    files = sorted((root / "src" / "hnef").rglob("*.py"))
    files += [root / name for name in ("pyproject.toml", "uv.lock") if (root / name).exists()]
    digest = hashlib.sha256()
    for path in files:
        digest.update(str(path.relative_to(root)).encode())
        digest.update(path.read_bytes())
    return {
        "git_commit": git("rev-parse", "HEAD"),
        "git_dirty": bool(git("status", "--porcelain")),
        "source_sha256": digest.hexdigest(),
    }


def promote(checkpoint, report, models, alias="candidate"):
    """Export a tested checkpoint without changing any serving process."""
    from .network import checkpoint_model

    if not re.fullmatch(r"[a-z][a-z0-9-]{0,40}", alias):
        raise ValueError("Alias must be a short lowercase name")
    checkpoint, report, models = Path(checkpoint), Path(report), Path(models)
    _, data = checkpoint_model(checkpoint)
    evaluation = json.loads(report.read_text())
    digest = sha256(checkpoint)
    if evaluation.get("checkpoint_sha256") != digest or not evaluation.get("matchups"):
        raise ValueError("Evaluation report does not belong to this checkpoint")
    model_id = f"mini-ac-{digest}"
    destination = models / model_id
    destination.mkdir(parents=True, exist_ok=True)
    # Content identity is the checkpoint hash; evaluation reports have their own identity.
    target = destination / "checkpoint.pt"
    if target.exists() and sha256(target) != digest:
        raise ValueError("Existing release checkpoint was modified")
    if not target.exists():
        temporary = destination / f".{uuid4().hex}.tmp"
        temporary.write_bytes(checkpoint.read_bytes())
        temporary.replace(target)
    evaluation_name = f"evaluation-{sha256(report)}.json"
    (destination / evaluation_name).write_bytes(report.read_bytes())
    manifest = {
        "schema": 1,
        "model_id": model_id,
        "checkpoint_sha256": digest,
        "ruleset": data["ruleset"],
        "features": data["features"],
        "update": data["update"],
        "training": data["config"],
        "source": data["source"],
        "evaluation": evaluation_name,
        "exported_at": now(),
        "status": "experimental-baseline",
    }
    write_json(destination / "manifest.json", manifest)
    write_json(models / f"{alias}.json", {"model_id": model_id, "checkpoint_sha256": digest})
    return destination


def resolve_model(models, alias):
    if not re.fullmatch(r"[a-z][a-z0-9-]{0,40}", alias):
        raise ValueError("Invalid model alias")
    models = Path(models)
    reference = json.loads((models / f"{alias}.json").read_text())
    model_id = reference["model_id"]
    if not re.fullmatch(r"mini-ac-[a-f0-9]{64}", model_id):
        raise ValueError("Invalid model identity")
    path = models / model_id / "checkpoint.pt"
    if sha256(path) != reference["checkpoint_sha256"]:
        raise ValueError("Model checksum mismatch")
    return path, model_id
