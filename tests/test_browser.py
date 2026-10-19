"""The shipped browser bridge must preserve replay and repetition state through undo."""

import json

import pytest

from hnef import browser
from hnef.replay import Replay


def command(kind, **data):
    return json.loads(browser.command(json.dumps({"kind": kind, **data})))


def test_undo_restores_exact_state_and_can_branch():
    opening = command("reset")
    first = command("step", action=opening["legal"][0]["id"])
    command("reference", policy="heuristic", seed=42)
    restored = command("undo", ply=0)
    assert restored == opening
    alternate = command("step", action=opening["legal"][-1]["id"])
    assert alternate["key"] != first["key"]
    replay = Replay.from_json(browser.command('{"kind":"replay"}'))
    assert len(replay.actions) == 1
    assert replay.verify().key == alternate["key"]


def test_local_undo_one_move_preserves_history_and_last_move():
    opening = command("reset")
    first = command("step", action=opening["legal"][0]["id"])
    second = command("step", action=first["legal"][0]["id"])
    assert second["player"] == "attackers"
    assert second["last_move"] is not None
    assert command("undo", ply=1) == first
    assert Replay.from_json(browser.command('{"kind":"replay"}')).verify().key == first["key"]


@pytest.mark.parametrize("ply", [-1, 1, 2, True, 0.5])
def test_invalid_undo_is_atomic(ply):
    opening = command("reset")
    before = command("step", action=opening["legal"][0]["id"])
    with pytest.raises(ValueError, match="Undo"):
        command("undo", ply=ply)
    assert json.loads(json.dumps(browser.snapshot())) == before
