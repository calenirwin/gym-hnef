import json
import shutil
import subprocess
from dataclasses import replace
from pathlib import Path

import pytest

torch = pytest.importorskip("torch")

from hnef import browser  # noqa: E402
from hnef.engine import Outcome, Player, apply_move, initial_state, legal_moves  # noqa: E402
from hnef.learning.artifacts import promote, resolve_model, sha256  # noqa: E402
from hnef.learning.cli import main  # noqa: E402
from hnef.learning.evaluate import evaluate_checkpoint  # noqa: E402
from hnef.learning.network import ActorCritic, NeuralPolicy, checkpoint_model, encode  # noqa: E402
from hnef.learning.train import TrainConfig, terminal_targets, train  # noqa: E402


def test_masks_history_and_mover_return_perspective():
    torch.set_num_threads(1)
    state = initial_state()
    repeated = replace(state, history=(state.history[0],) * 2)
    boards, history, lengths, masks = encode([state, repeated])
    assert lengths.tolist() == [1, 2]
    assert boards[0, -1] != boards[1, -1]
    assert history.shape == (2, 2, 32)
    logits, values = ActorCritic()(boards, history, lengths, masks)
    assert torch.isneginf(logits[~masks]).all()
    assert torch.isfinite(logits[masks]).all()
    assert (values.abs() <= 1).all()
    successor = apply_move(state, legal_moves(state)[0])
    outcome = Outcome(Player.DEFENDERS, "king_escaped")
    assert terminal_targets([state, successor], outcome) == [-1.0, 1.0]
    assert terminal_targets([state], None) == []
    assert terminal_targets([state], Outcome(None, "repetition")) == [0.0]
    with pytest.raises(ValueError, match="terminal"):
        NeuralPolicy(ActorCritic()).choose_move(replace(state, outcome=Outcome(None, "repetition")))


def test_browser_feature_and_transition_parity():
    browser.command('{"kind":"reset"}')
    for _ in range(8):
        features = json.loads(browser.command('{"kind":"features"}'))
        boards, histories, lengths, masks = encode([browser.state])
        torch.testing.assert_close(torch.tensor(features["board"]), boards[0])
        torch.testing.assert_close(torch.tensor(features["history"]), histories[0, : lengths[0]])
        assert features["mask"] == masks[0].tolist()
        before = browser.state
        if before.outcome:
            break
        move = legal_moves(before)[0]
        action = browser.codec.encode(move)
        result = json.loads(browser.command(json.dumps({"kind": "step", "action": action})))
        assert result["key"] == apply_move(before, move).key
    from hnef.replay import Replay

    Replay.from_json(browser.command('{"kind":"replay"}')).verify()


def test_training_resumes_exactly_and_updates_weights(tmp_path):
    config = TrainConfig(seed=12, episodes_per_update=4)
    uninterrupted = train(tmp_path / "full", updates=3, config=config)
    train(tmp_path / "split", updates=1, config=config)
    resumed = train(tmp_path / "split", updates=2, resume=True)
    full_model, full_data = checkpoint_model(uninterrupted)
    split_model, split_data = checkpoint_model(resumed)
    initial, _ = checkpoint_model(tmp_path / "full/checkpoint-000000.pt")
    assert full_data["update"] == split_data["update"] == 3
    for key, tensor in full_model.state_dict().items():
        assert torch.equal(tensor, split_model.state_dict()[key])
    assert any(
        not torch.equal(tensor, initial.state_dict()[key])
        for key, tensor in full_model.state_dict().items()
    )
    assert torch.equal(full_data["sampling_rng"], split_data["sampling_rng"])
    for n in (1, 2, 3):
        name = f"metrics/{n:06d}.json"
        assert (tmp_path / "full" / name).read_text() == (tmp_path / "split" / name).read_text()
    with pytest.raises(FileExistsError):
        train(tmp_path / "full", updates=1)


def test_truncations_are_not_draw_targets(tmp_path):
    path = train(
        tmp_path / "cutoff", updates=1, config=TrainConfig(episodes_per_update=2, max_plies=1)
    )
    trained, _ = checkpoint_model(path)
    initial, _ = checkpoint_model(path.parent / "checkpoint-000000.pt")
    assert all(
        torch.equal(value, initial.state_dict()[key]) for key, value in trained.state_dict().items()
    )
    metrics = json.loads((path.parent / "metrics/000001.json").read_text())
    assert metrics["results"] == {"truncated": 2}
    assert metrics["training_positions"] == 0 and metrics["skipped_update"]


def test_evaluation_release_and_export_roundtrip(tmp_path):
    checkpoint = train(tmp_path / "run", updates=1, config=TrainConfig(episodes_per_update=2))
    output = tmp_path / "evaluation"
    report = evaluate_checkpoint(checkpoint, output, games=2, opponents=("random",), max_plies=40)
    assert report["checkpoint_sha256"] == sha256(checkpoint)
    assert [row["candidate_side"] for row in report["matchups"]] == ["attackers", "defenders"]
    assert all(sum(row["results"].values()) == 2 for row in report["matchups"])
    assert len(list((output / "replays").glob("*.json"))) == 4
    release = promote(checkpoint, output / "report.json", tmp_path / "models")
    resolved, identity = resolve_model(tmp_path / "models", "candidate")
    assert release / "checkpoint.pt" == resolved
    assert identity.startswith("mini-ac-")
    with pytest.raises(ValueError, match="does not belong"):
        promote(
            checkpoint.parent / "checkpoint-000000.pt", output / "report.json", tmp_path / "models"
        )
    resolved.write_bytes(b"corrupted")
    with pytest.raises(ValueError, match="checksum"):
        resolve_model(tmp_path / "models", "candidate")
    export = tmp_path / "network.json"
    main(["export-web", str(checkpoint), "--output", str(export)])
    data = json.loads(export.read_text())
    assert data["ruleset"] == "mini-v1" and len(data["weights"]["actor.bias"]) == 200


def test_javascript_inference_matches_pytorch(tmp_path):
    node = shutil.which("node")
    if not node:
        pytest.skip("Node is needed for browser inference parity")
    checkpoint = train(tmp_path / "run", updates=1, config=TrainConfig(episodes_per_update=2))
    model, _ = checkpoint_model(checkpoint)
    export = tmp_path / "model.json"
    main(["export-web", str(checkpoint), "--output", str(export)])
    browser.command('{"kind":"reset"}')
    examples = []
    for _ in range(12):
        if browser.state.outcome:
            break
        features = json.loads(browser.command('{"kind":"features"}'))
        with torch.inference_mode():
            logits, values = model(*encode([browser.state]))
        examples.append(
            {
                "features": features,
                "logits": [
                    v if valid else None
                    for v, valid in zip(logits[0].tolist(), features["mask"], strict=True)
                ],
                "value": values[0].item(),
            }
        )
        browser.command(
            json.dumps(
                {"kind": "step", "action": browser.codec.encode(legal_moves(browser.state)[0])}
            )
        )
    script = Path(__file__).resolve().parents[1] / "web/check-inference.mjs"
    result = subprocess.run(
        [node, str(script), str(export)],
        input=json.dumps(examples),
        text=True,
        capture_output=True,
        check=False,
    )
    assert result.returncode == 0, result.stderr


@pytest.mark.parametrize(
    "kwargs",
    [
        {"seed": -1},
        {"max_plies": 0},
        {"learning_rate": float("nan")},
        {"opponents": ()},
        {"opponents": ("bogus",)},
    ],
)
def test_invalid_training_config(kwargs):
    with pytest.raises(ValueError):
        TrainConfig(**kwargs)
