"""Local browser game backed by the canonical Python engine."""

import json
from functools import lru_cache
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import parse_qs, urlsplit

from hnef.actions import ActionCodec
from hnef.engine import Ruleset, adjudicate_draw, apply_move, initial_state, legal_moves
from hnef.policies import POLICY_NAMES, make_policy
from hnef.replay import Replay

MAX_PLIES = 500
ASSETS = Path(__file__).parent


@lru_cache(maxsize=3)
def codec_for(ruleset):
    return ActionCodec(ruleset.size)


def game_response(payload):
    """Reconstruct history before every operation; the browser never adjudicates."""
    if not isinstance(payload, dict):
        raise ValueError("Expected a game object")
    imported = None
    operation = payload.get("operation", "state")
    if operation == "import":
        record = payload.get("replay")
        if not isinstance(record, dict) or not isinstance(record.get("actions"), list):
            raise ValueError("Choose a Hnef replay JSON file")
        if len(record["actions"]) > MAX_PLIES:
            raise ValueError("Games are limited to 500 plies")
        imported = Replay.from_json(json.dumps(record))
        payload = {"ruleset": imported.ruleset, "actions": list(imported.actions)}
        operation = "state"
    ruleset = Ruleset(payload.get("ruleset", "mini-v1"))
    actions = payload.get("actions", [])
    if not isinstance(actions, list) or len(actions) > MAX_PLIES:
        raise ValueError("Games are limited to 500 plies")
    actions = list(actions)
    codec = codec_for(ruleset)
    state = initial_state(ruleset)
    keys, moves = [state.key], []

    def advance(action):
        nonlocal state
        move = codec.decode(action)
        before = state
        state = apply_move(state, move)
        captured = [
            [r, c]
            for r in range(ruleset.size)
            for c in range(ruleset.size)
            if before.board[r][c] != "." and state.board[r][c] == "." and (r, c) != move.source
        ]
        moves.append(
            {
                "source": move.source,
                "destination": move.destination,
                "player": before.player,
                "piece": before.at(move.source),
                "captured": captured,
            }
        )
        keys.append(state.key)

    for action in actions:
        advance(action)
    if operation not in ("state", "move", "ai"):
        raise ValueError("Unknown game operation")
    if operation != "state":
        if state.outcome or state.ply >= MAX_PLIES:
            raise ValueError("This game has ended")
        if operation == "ai":
            policy = payload.get("policy", "heuristic")
            seed = payload.get("seed", 42)
            if policy not in POLICY_NAMES or type(seed) is not int or not 0 <= seed < 2**32:
                raise ValueError("Invalid policy or seed")
            # Bounded interactive budget; deterministic across retries and undo.
            bot = make_policy(policy, seed * 1009 + state.ply, iterations=32, rollout_plies=24)
            action = codec.encode(bot.choose_move(state))
        else:
            action = payload.get("action")
        advance(action)
        actions.append(action)
    if imported and imported.adjudication_note is not None:
        state = adjudicate_draw(state)
        keys.append(state.key)
    replay = imported or Replay(tuple(actions), tuple(keys), ruleset)
    return {
        "board": state.board,
        "player": state.player,
        "ply": state.ply,
        "ruleset": ruleset.value,
        "key": state.key,
        "outcome": None
        if state.outcome is None
        else {
            "winner": state.outcome.winner,
            "reason": state.outcome.reason,
        },
        "truncated": state.outcome is None and state.ply >= MAX_PLIES,
        "legal": [
            {"action": codec.encode(move), "source": move.source, "destination": move.destination}
            for move in legal_moves(state)
        ]
        if state.ply < MAX_PLIES
        else [],
        "moves": moves,
        "actions": actions,
        "replay": json.loads(replay.to_json()),
    }


class GameHandler(BaseHTTPRequestHandler):
    def log_message(self, format, *args):
        pass

    def respond(self, status, data, content_type="application/json", filename=None, cache=False):
        if not isinstance(data, bytes):
            data = json.dumps(data).encode()
        self.send_response(status)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(len(data)))
        self.send_header("Cache-Control", "public, max-age=3600" if cache else "no-store")
        if filename:
            self.send_header("Content-Disposition", f'attachment; filename="{filename}"')
        self.send_header("X-Content-Type-Options", "nosniff")
        self.send_header(
            "Content-Security-Policy",
            "default-src 'self'; style-src 'self'; "
            "script-src 'self'; img-src 'self' data:; frame-ancestors 'none'",
        )
        self.end_headers()
        self.wfile.write(data)

    def do_GET(self):
        if urlsplit(self.path).path == "/api/replay":
            try:
                if len(self.path) > 16384:
                    raise ValueError("Replay request is too large")
                query = parse_qs(urlsplit(self.path).query)
                result = game_response(
                    {
                        "ruleset": query.get("ruleset", ["mini-v1"])[0],
                        "actions": json.loads(query.get("actions", ["[]"])[0]),
                    }
                )
                filename = f"hnef-{result['ruleset']}-ply{result['ply']}.json"
                self.respond(200, result["replay"], filename=filename)
            except (ValueError, TypeError, KeyError) as error:
                self.respond(400, {"error": str(error)})
            return
        assets = {
            "/": ("index.html", "text/html; charset=utf-8"),
            "/learn": ("learn.html", "text/html; charset=utf-8"),
            "/learn/": ("learn.html", "text/html; charset=utf-8"),
            "/learn-math.js": ("learn-math.js", "text/javascript; charset=utf-8"),
            "/learn.js": ("learn.js", "text/javascript; charset=utf-8"),
            "/learn.css": ("learn.css", "text/css; charset=utf-8"),
            "/transport.js": ("transport.js", "text/javascript; charset=utf-8"),
            "/app.js": ("app.js", "text/javascript; charset=utf-8"),
            "/motion.js": ("motion.js", "text/javascript; charset=utf-8"),
            "/style.css": ("style.css", "text/css; charset=utf-8"),
        }
        for piece in ("attacker", "defender", "king", "board-flourish", "albion-footer"):
            assets[f"/assets/{piece}.png"] = (f"assets/{piece}.png", "image/png")
        for font in ("cormorant", "inter"):
            assets[f"/assets/fonts/{font}.woff2"] = (f"assets/fonts/{font}.woff2", "font/woff2")
        asset = assets.get(self.path.split("?")[0])
        if asset is None:
            self.respond(404, {"error": "Not found"})
            return
        self.respond(
            200, (ASSETS / asset[0]).read_bytes(), asset[1], cache=asset[0].startswith("assets/")
        )

    def do_POST(self):
        if self.path not in ("/api/game", "/api/analyze"):
            self.respond(404, {"error": "Not found"})
            return
        host = self.headers.get("Host", "")
        allowed = {f"127.0.0.1:{self.server.server_port}", f"localhost:{self.server.server_port}"}
        if host not in allowed or self.headers.get("Origin", f"http://{host}") != f"http://{host}":
            self.respond(403, {"error": "Use the local game address"})
            return
        try:
            length = int(self.headers.get("Content-Length", "0"))
            if not 0 < length <= 131072:
                raise ValueError("Invalid request size")
            payload = json.loads(self.rfile.read(length))
            if self.path == "/api/analyze":
                from .analysis import analyze_response

                self.respond(200, analyze_response(payload))
            else:
                self.respond(200, game_response(payload))
        except (ValueError, TypeError, KeyError) as error:
            self.respond(400, {"error": str(error)})


def serve(port=8000):
    with ThreadingHTTPServer(("127.0.0.1", port), GameHandler) as server:
        print(f"Hnef is ready at http://127.0.0.1:{server.server_port}", flush=True)
        try:
            server.serve_forever()
        except KeyboardInterrupt:
            pass
