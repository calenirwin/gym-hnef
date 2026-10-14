import json
import os
import subprocess
import sys
from pathlib import Path

import pytest

from hnef.actions import ActionCodec
from hnef.cli import main
from hnef.engine import Move, initial_state, legal_moves
from hnef.replay import Replay, random_game


@pytest.mark.parametrize("size,count", [(5, 200), (9, 1296), (11, 2420)])
def test_codec_is_bijective_and_geometric(size, count):
    codec = ActionCodec(size)
    assert len(codec.moves) == len(set(codec.moves)) == count
    for index, move in enumerate(codec.moves):
        assert codec.encode(move) == index
        assert codec.decode(index) == move
        assert move.source != move.destination
        assert move.source[0] == move.destination[0] or move.source[1] == move.destination[1]


def test_codec_order_is_stable():
    codec = ActionCodec(5)
    assert codec.decode(0) == Move((0, 0), (0, 1))
    assert codec.decode(7) == Move((0, 0), (4, 0))
    assert codec.decode(199) == Move((4, 4), (4, 3))


@pytest.mark.parametrize("action", [-1, 200, True, 1.5, "1"])
def test_codec_rejects_invalid_ids(action):
    with pytest.raises(ValueError):
        ActionCodec(5).decode(action)


@pytest.mark.parametrize("size", [0, 1, -1, True, 5.0])
def test_codec_rejects_invalid_size(size):
    with pytest.raises(ValueError):
        ActionCodec(size)


def test_codec_rejects_diagonal_and_out_of_bounds():
    codec = ActionCodec(5)
    for move in (Move((0, 0), (1, 1)), Move((0, 0), (0, 5))):
        with pytest.raises(ValueError):
            codec.encode(move)


def test_action_mask_exactly_matches_legal_moves():
    state = initial_state()
    codec = ActionCodec(5)
    assert {codec.decode(i) for i, valid in enumerate(codec.mask(state)) if valid} == set(
        legal_moves(state)
    )
    with pytest.raises(ValueError):
        ActionCodec(11).mask(state)


def test_seeded_game_and_json_roundtrip():
    record = random_game(42)
    assert record == random_game(42)
    assert Replay.from_json(record.to_json()) == record
    assert record.verify().outcome is not None
    assert not any(ActionCodec(5).mask(record.verify()))


def test_collection_limit_is_not_a_draw():
    record = random_game(0, max_plies=1)
    state = record.verify()
    assert state.ply == 1
    assert state.outcome is None
    assert legal_moves(state)


@pytest.mark.parametrize("value", [0, -1, True, 1.5])
def test_invalid_ply_limit(value):
    with pytest.raises(ValueError):
        random_game(0, value)


@pytest.mark.parametrize("change", ["version", "ruleset", "actions", "state_keys", "opening"])
def test_replay_detects_corrupt_data(change):
    data = json.loads(random_game(2).to_json())
    if change == "version":
        data["version"] = True
    elif change == "ruleset":
        data["ruleset"] = "copenhagen"
    elif change == "actions":
        data["actions"][0] = -1
    elif change == "state_keys":
        data["state_keys"][1] = "bad"
    else:
        data["state_keys"][0] = "bad"
    with pytest.raises(ValueError):
        Replay.from_json(json.dumps(data))


@pytest.mark.parametrize("text", ["[]", "null", "{}", "not json"])
def test_replay_rejects_invalid_format(text):
    with pytest.raises(ValueError):
        Replay.from_json(text)


def test_replay_rejects_missing_position_and_illegal_move():
    with pytest.raises(ValueError):
        Replay((0,), (initial_state().key,))
    record = Replay((0,), (initial_state().key, "bad"))
    with pytest.raises(ValueError, match="Illegal move"):
        record.verify()


def test_state_key_is_stable_across_python_processes():
    env = dict(os.environ, PYTHONPATH=str(Path(__file__).resolve().parents[1] / "src"))
    for hash_seed in ("1", "2"):
        output = subprocess.check_output(
            [
                sys.executable,
                "-c",
                "from hnef.engine import initial_state; print(initial_state().key)",
            ],
            env=dict(env, PYTHONHASHSEED=hash_seed),
            text=True,
        )
        assert output.strip() == initial_state().key


def test_cli_save_replay_and_benchmark(tmp_path, capsys):
    path = tmp_path / "game.json"
    main(["play", "--seed", "42", "--output", str(path)])
    expected = capsys.readouterr().out
    main(["replay", str(path)])
    assert capsys.readouterr().out == expected
    main(["benchmark", "--games", "3", "--max-plies", "1"])
    report = json.loads(capsys.readouterr().out)
    assert report == {"games": 3, "plies": 3, "results": {"truncated": 3}}


def test_cli_reports_invalid_replay_without_traceback(tmp_path, capsys):
    path = tmp_path / "bad.json"
    path.write_text("{}")
    with pytest.raises(SystemExit) as error:
        main(["replay", str(path)])
    assert error.value.code == 2
    assert "Unsupported replay" in capsys.readouterr().err
