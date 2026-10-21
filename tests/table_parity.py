"""Portable CPython/Pyodide fixtures; no alternate rules or production test hooks."""

import json
from dataclasses import asdict

from hnef.engine import GameState, Move, Player, Ruleset, apply_move, legal_moves
from hnef.replay import random_game
from hnef.web import game_response
from hnef.web.analysis import analyze_response


def api_fixtures():
    fixtures = []

    def add(name, payload, kind="game"):
        try:
            result = (analyze_response if kind == "analyze" else game_response)(payload)
            expected = {"ok": True, "data": result}
        except (ValueError, TypeError, KeyError) as error:
            expected = {"ok": False, "data": {"error": str(error)}}
        fixtures.append({"name": name, "kind": kind, "payload": payload, "expected": expected})

    for rules in Ruleset:
        opening = game_response({"ruleset": rules})
        add(f"{rules}:opening", {"ruleset": rules})
        add(
            f"{rules}:move",
            {"ruleset": rules, "operation": "move", "action": opening["legal"][0]["action"]},
        )
        record = json.loads(random_game(42, max_plies=12, ruleset=rules).to_json())
        add(f"{rules}:import", {"operation": "import", "replay": record})
        add(f"{rules}:restore", {"ruleset": rules, "actions": record["actions"]})
        for policy in ("random", "heuristic", "mcts"):
            add(
                f"{rules}:{policy}",
                {"ruleset": rules, "operation": "ai", "policy": policy, "seed": 42},
            )
        add(f"{rules}:analysis", {"ruleset": rules, "iterations": 8, "rollout_plies": 0}, "analyze")
    record = json.loads(
        random_game(3, max_plies=4, ruleset=Ruleset.COPENHAGEN_V2)
        .adjudicate_draw("Agreed dead position")
        .to_json()
    )
    add("adjudicated draw", {"operation": "import", "replay": record})
    tampered = json.loads(json.dumps(record))
    tampered["state_keys"][0] = "tampered"
    add("tampered replay", {"operation": "import", "replay": tampered})
    # This deterministic history reaches the actual API limit without a terminal outcome.
    long_record = json.loads(random_game(0, max_plies=500, ruleset=Ruleset.COPENHAGEN_V2).to_json())
    assert len(long_record["actions"]) == 500
    for length in (100, 499, 500):
        add(
            f"history:{length}",
            {"ruleset": Ruleset.COPENHAGEN_V2, "actions": long_record["actions"][:length]},
        )
    add(
        "limit prevents AI",
        {"ruleset": Ruleset.COPENHAGEN_V2, "actions": long_record["actions"], "operation": "ai"},
    )
    for name, payload, kind in (
        ("invalid ruleset", {"ruleset": "invalid"}, "game"),
        ("invalid action", {"operation": "move", "action": -1}, "game"),
        ("too many plies", {"actions": [0] * 501}, "game"),
        ("invalid seed", {"operation": "ai", "seed": -1}, "game"),
        ("malformed replay", {"operation": "import", "replay": {}}, "game"),
        ("invalid budget", {"iterations": 100000}, "analyze"),
        ("non-object", [], "game"),
    ):
        add(name, payload, kind)
    return json.loads(json.dumps(fixtures))


def tactical_results(forts):
    results = {}

    def position(pieces, rules, player=Player.ATTACKERS, turns=0):
        pieces = dict(pieces)
        if "K" not in pieces.values():
            pieces[(5, 5)] = "K"
        board = [["."] * 11 for _ in range(11)]
        for square, piece in pieces.items():
            r, c = rotate(square, turns)
            board[r][c] = piece
        return GameState(tuple("".join(row) for row in board), player, ruleset=rules)

    def rotate(square, turns):
        for _ in range(turns):
            square = square[1], 10 - square[0]
        return square

    def record(name, state, moves):
        for move in moves:
            state = apply_move(state, move)
        results[name] = {
            "board": state.board,
            "key": state.key,
            "player": state.player,
            "outcome": asdict(state.outcome) if state.outcome else None,
            "legal": [asdict(move) for move in legal_moves(state)],
        }
        return state

    for rules in (Ruleset.COPENHAGEN, Ruleset.COPENHAGEN_V2):
        for turns in range(4):
            # Published repository fixtures, rotated to exercise all four edges.
            pieces = {(0, c): "D" for c in (2, 3, 4)}
            pieces.update({(1, c): "A" for c in (2, 3, 4)})
            pieces.update({(0, 1): "A", (2, 5): "A"})
            result = record(
                f"{rules}:shieldwall:{turns}",
                position(pieces, rules, turns=turns),
                [Move(rotate((2, 5), turns), rotate((0, 5), turns))],
            )
            assert all(result.at(rotate((0, c), turns)) == "." for c in (2, 3, 4))
            record(
                f"{rules}:ordinary:{turns}",
                position({(0, 1): "D", (2, 2): "A"}, rules, turns=turns),
                [Move(rotate((2, 2), turns), rotate((0, 2), turns))],
            )
            record(
                f"{rules}:escape:{turns}",
                position({(0, 3): "K", (7, 7): "A"}, rules, Player.DEFENDERS, turns),
                [Move(rotate((0, 3), turns), rotate((0, 0), turns))],
            )
        pieces = {
            (r, c): "A" for r in range(3, 8) for c in range(3, 8) if r in (3, 7) or c in (3, 7)
        }
        pieces.update({(4, 4): "D", (1, 5): "A"})
        del pieces[(3, 5)]
        record(f"{rules}:encircled", position(pieces, rules), [Move((1, 5), (3, 5))])
        record(
            f"{rules}:king-capture",
            position({(5, 5): "K", (4, 5): "A", (5, 4): "A", (5, 6): "A", (7, 5): "A"}, rules),
            [Move((7, 5), (6, 5))],
        )
        cycle = [
            Move((1, 1), (1, 2)),
            Move((9, 9), (9, 8)),
            Move((1, 2), (1, 1)),
            Move((9, 8), (9, 9)),
        ]
        record(f"{rules}:repetition", position({(1, 1): "A", (9, 9): "D"}, rules), cycle * 2)
        for fixture in forts:
            for turns in range(4):
                pieces = {
                    (r, c): p
                    for r, row in enumerate(fixture["board"])
                    for c, p in enumerate(row)
                    if p != "."
                }
                source, destination = map(tuple, fixture["closing_move"])
                del pieces[destination]
                pieces[source] = "D"
                pieces[(0, 5)] = "A"
                state = record(
                    f"{rules}:{fixture['name']}:{turns}",
                    position(pieces, rules, Player.DEFENDERS, turns),
                    [Move(rotate(source, turns), rotate(destination, turns))],
                )
                assert (state.outcome is not None) == (rules is Ruleset.COPENHAGEN_V2)
    return json.loads(json.dumps(results))
