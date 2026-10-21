"""Browser service contracts: canonical rules, complete history, portable replays."""

import json

import pytest

from hnef.engine import Ruleset, apply_move, initial_state, legal_moves
from hnef.replay import Replay, random_game
from hnef.web import game_response


@pytest.mark.parametrize("ruleset", list(Ruleset))
def test_browser_legal_moves_and_move_match_engine(ruleset):
    opening = game_response({"ruleset": ruleset})
    expected = initial_state(ruleset)
    assert opening["board"] == expected.board
    assert len(opening["legal"]) == len(legal_moves(expected))
    action = opening["legal"][0]["action"]
    result = game_response({"ruleset": ruleset, "operation": "move", "action": action})
    assert result["key"] == apply_move(expected, legal_moves(expected)[0]).key
    assert Replay.from_json(json.dumps(result["replay"])).verify().key == result["key"]
    # Every request reconstructs the same complete state, including repetition context.
    assert game_response({"ruleset": ruleset, "actions": [action]}) == result


@pytest.mark.parametrize("policy", ["random", "heuristic", "mcts"])
def test_browser_bot_retry_is_deterministic_and_replayable(policy):
    payload = {"operation": "ai", "policy": policy, "seed": 17}
    first = game_response(payload)
    assert game_response(payload) == first
    second = game_response({**payload, "actions": first["actions"]})
    assert second["ply"] == 2
    assert Replay.from_json(json.dumps(second["replay"])).verify().key == second["key"]


def test_browser_completed_game_and_capture_journal():
    record = random_game(42)
    expected = record.verify()
    assert expected.outcome is not None
    result = game_response({"actions": list(record.actions)})
    assert result["key"] == expected.key
    assert result["outcome"]["winner"] == expected.outcome.winner
    assert not result["legal"]
    assert not result["truncated"]
    captures = sum(len(move["captured"]) for move in result["moves"])
    assert captures == 13 - sum(piece != "." for row in expected.board for piece in row)
    with pytest.raises(ValueError, match="ended"):
        game_response({"actions": list(record.actions), "operation": "ai"})


def test_browser_review_prefix_and_branch_after_undo():
    record = random_game(42)
    prefix = list(record.actions[:4])
    result = game_response({"actions": prefix})
    branch = game_response({"actions": prefix, "operation": "ai", "policy": "heuristic"})
    assert result["ply"] == 4
    assert branch["actions"][:4] == prefix
    assert branch["replay"]["state_keys"][:5] == result["replay"]["state_keys"]
    assert Replay.from_json(json.dumps(branch["replay"])).verify().ply == 5


@pytest.mark.parametrize(
    "payload",
    [
        [],
        {"ruleset": "full-copenhagen"},
        {"actions": "bad"},
        {"actions": [0] * 501},
        {"actions": [True]},
        {"actions": [-1]},
        {"actions": [100000]},
        {"operation": "move", "action": 0},
        {"operation": "move"},
        {"operation": "oops"},
        {"operation": "ai", "policy": "unknown"},
        {"operation": "ai", "seed": -1},
        {"operation": "ai", "seed": True},
        {"operation": "ai", "seed": 2**32},
    ],
)
def test_browser_rejects_malformed_or_illegal_requests(payload):
    with pytest.raises(ValueError):
        game_response(payload)


def test_browser_collection_cap_is_not_a_draw(monkeypatch):
    monkeypatch.setattr("hnef.web.MAX_PLIES", 2)
    actions = list(random_game(42).actions[:2])
    result = game_response({"actions": actions})
    assert result["truncated"]
    assert result["outcome"] is None
    assert result["legal"] == []
    with pytest.raises(ValueError, match="ended"):
        game_response({"actions": actions, "operation": "ai"})


def test_browser_terminal_result_precedes_collection_cap(monkeypatch):
    actions = list(random_game(42).actions)
    monkeypatch.setattr("hnef.web.MAX_PLIES", len(actions))
    result = game_response({"actions": actions})
    assert result["outcome"] is not None
    assert not result["truncated"]


def http_response(path, payload=None, origin="http://127.0.0.1:8000"):
    """Exercise the real HTTP parser and handler without requiring a listening socket."""
    from io import BytesIO
    from types import SimpleNamespace

    from hnef.web import GameHandler

    body = b"" if payload is None else json.dumps(payload).encode()
    method = "GET" if payload is None else "POST"
    request = (
        f"{method} {path} HTTP/1.1\r\nHost: 127.0.0.1:8000\r\n"
        f"Origin: {origin}\r\nContent-Length: {len(body)}\r\n\r\n"
    ).encode() + body
    output = bytearray()
    connection = SimpleNamespace(makefile=lambda *args: BytesIO(request), sendall=output.extend)
    GameHandler(connection, ("127.0.0.1", 12345), SimpleNamespace(server_port=8000))
    headers, body = bytes(output).split(b"\r\n\r\n", 1)
    return headers.decode(), body


def test_browser_replay_download_has_attachment_and_verifies():
    from urllib.parse import urlencode

    actions = list(random_game(42).actions)
    query = urlencode({"ruleset": "mini-v1", "actions": json.dumps(actions)})
    headers, body = http_response("/api/replay?" + query)
    assert "200 OK" in headers
    assert 'Content-Disposition: attachment; filename="hnef-mini-v1-ply' in headers
    assert Replay.from_json(body.decode()).actions == tuple(actions)


@pytest.mark.parametrize(
    "path",
    [
        "/",
        "/learn",
        "/learn/",
        "/learn.js",
        "/learn-math.js",
        "/learn.css",
        "/app.js",
        "/motion.js",
        "/style.css",
        "/assets/attacker.png",
        "/assets/defender.png",
        "/assets/king.png",
        "/assets/board-flourish.png",
        "/assets/albion-footer.png",
        "/assets/fonts/cormorant.woff2",
        "/assets/fonts/inter.woff2",
    ],
)
def test_browser_assets_serve(path):
    headers, body = http_response(path)
    assert "200 OK" in headers
    assert len(body) > 100
    assert "Content-Security-Policy:" in headers


def test_browser_http_errors_and_origin_boundary():
    headers, body = http_response("/api/game", {"operation": "move", "action": 0})
    assert "400 Bad Request" in headers
    assert "error" in json.loads(body)
    headers, _ = http_response("/api/game", {}, origin="https://unrelated.example")
    assert "403 Forbidden" in headers
    headers, _ = http_response("/../engine/rules.py")
    assert "404 Not Found" in headers
    headers, _ = http_response("/api/replay?actions=not-json")
    assert "400 Bad Request" in headers


def test_browser_http_game_response_matches_canonical_state():
    headers, body = http_response("/api/game", {"operation": "ai", "policy": "heuristic"})
    assert "200 OK" in headers
    result = json.loads(body)
    assert Replay.from_json(json.dumps(result["replay"])).verify().key == result["key"]


@pytest.mark.parametrize("ruleset", list(Ruleset))
def test_browser_import_preserves_canonical_replay(ruleset):
    replay = random_game(42, max_plies=8, ruleset=ruleset)
    result = game_response({"operation": "import", "replay": json.loads(replay.to_json())})
    assert result["key"] == replay.verify().key
    assert result["replay"] == json.loads(replay.to_json())
    assert result["actions"] == list(replay.actions)
    assert result["ruleset"] == ruleset


def test_browser_import_preserves_external_draw_and_review_prefix():
    replay = random_game(3, max_plies=4, ruleset=Ruleset.COPENHAGEN_V2).adjudicate_draw(
        "Players agree the position is dead."
    )
    result = game_response({"operation": "import", "replay": json.loads(replay.to_json())})
    assert result["outcome"] == {"winner": None, "reason": "adjudicated_draw"}
    assert result["key"] == replay.verify().key
    assert not result["legal"]
    assert result["ply"] == 4
    assert result["replay"] == json.loads(replay.to_json())
    prefix = game_response({"ruleset": replay.ruleset, "actions": result["actions"][:2]})
    assert prefix["outcome"] is None
    assert prefix["ply"] == 2


@pytest.mark.parametrize("corruption", ["opening", "move", "adjudication"])
def test_browser_import_rejects_tampered_history(corruption):
    replay = random_game(3, max_plies=4, ruleset=Ruleset.COPENHAGEN_V2).adjudicate_draw("Agreed")
    record = json.loads(replay.to_json())
    index = {"opening": 0, "move": 2, "adjudication": -1}[corruption]
    record["state_keys"][index] = "tampered"
    headers, body = http_response("/api/game", {"operation": "import", "replay": record})
    assert "400 Bad Request" in headers
    assert "error" in json.loads(body)


def test_browser_import_rejects_oversized_and_malformed_records():
    for record in (None, [], {}, {"actions": []}, {"actions": [0] * 501}):
        with pytest.raises(ValueError):
            game_response({"operation": "import", "replay": record})


@pytest.mark.parametrize("ruleset", list(Ruleset))
def test_analysis_uses_verified_position_and_accounts_for_visits(ruleset):
    from hnef.web.analysis import analyze_response

    result = analyze_response({"ruleset": ruleset, "iterations": 8, "rollout_plies": 0})
    position = result["position"]
    assert position["key"] == initial_state(ruleset).key
    assert position["actions"] == []
    assert result["legal_count"] == len(position["legal"])
    assert all(
        move["action"] in {item["action"] for item in position["legal"]}
        for move in result["decisions"]
    )
    assert sum(child["visits"] for child in result["search"]["children"]) == 8
    assert 0 <= result["search"]["rollout_cutoffs"] <= 8


def test_analysis_matches_table_decisions_without_mutating_history():
    from copy import deepcopy

    from hnef.web.analysis import analyze_response

    payload = {"seed": 17, "actions": list(random_game(42).actions[:2])}
    before = deepcopy(payload)
    result = analyze_response(payload)
    assert analyze_response(payload) == result
    assert payload == before
    assert result["position"]["actions"] == before["actions"]
    for decision in result["decisions"]:
        played = game_response({**payload, "operation": "ai", "policy": decision["policy"]})
        assert played["actions"][-1] == decision["action"]


def test_analysis_import_preserves_history_and_rejects_terminal_games():
    from hnef.web.analysis import analyze_response

    replay = game_response({"actions": list(random_game(42).actions[:2])})["replay"]
    result = analyze_response({"replay": replay, "iterations": 8, "rollout_plies": 0})
    assert result["position"]["replay"] == replay
    with pytest.raises(ValueError, match="unfinished"):
        analyze_response({"replay": json.loads(random_game(42).to_json())})
    replay["state_keys"][-1] = "tampered"
    with pytest.raises(ValueError):
        analyze_response({"replay": replay})


@pytest.mark.parametrize(
    "payload",
    [
        [],
        {"seed": True},
        {"seed": -1},
        {"seed": 2**32},
        {"iterations": True},
        {"iterations": 1000000},
        {"rollout_plies": -1},
        {"rollout_plies": 49},
        {"rollout_plies": False},
        {"actions": [True]},
        {"actions": [0] * 501},
    ],
)
def test_analysis_rejects_invalid_configuration(payload):
    from hnef.web.analysis import analyze_response

    with pytest.raises(ValueError):
        analyze_response(payload)


def test_analysis_http_origin_and_budget_validation():
    payload = {"iterations": 8, "rollout_plies": 0}
    headers, body = http_response("/api/analyze", payload)
    assert "200 OK" in headers
    assert json.loads(body)["iterations"] == 8
    headers, _ = http_response("/api/analyze", payload, origin="https://unrelated.example")
    assert "403 Forbidden" in headers
    headers, _ = http_response("/api/analyze", {"iterations": 100000})
    assert "400 Bad Request" in headers
