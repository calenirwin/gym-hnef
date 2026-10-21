"""Reproducible policy games using the existing verified replay format."""

from random import Random

from .actions import ActionCodec
from .engine import Player, Ruleset, apply_move, initial_state
from .policies import Policy, make_policy
from .replay import Replay, random_game


def play_game(
    attackers: Policy,
    defenders: Policy,
    *,
    max_plies: int = 500,
    ruleset: Ruleset = Ruleset.MINI,
) -> Replay:
    if type(max_plies) is not int or max_plies < 1:
        raise ValueError("max_plies must be a positive integer")
    state = initial_state(ruleset)
    codec = ActionCodec(state.ruleset.size)
    policies = {Player.ATTACKERS: attackers, Player.DEFENDERS: defenders}
    actions, keys = [], [state.key]
    while state.outcome is None and state.ply < max_plies:
        move = policies[state.player].choose_move(state)
        state = apply_move(state, move)
        actions.append(codec.encode(move))
        keys.append(state.key)
    return Replay(tuple(actions), tuple(keys), state.ruleset)


def policy_game(
    seed: int,
    max_plies: int = 500,
    *,
    attackers: str = "random",
    defenders: str = "random",
    ruleset: Ruleset = Ruleset.MINI,
    iterations: int = 64,
    rollout_plies: int = 48,
) -> Replay:
    """Independent player RNG streams; random/random retains its original sequence."""
    if attackers == defenders == "random":
        return random_game(seed, max_plies, ruleset=ruleset)
    rng = Random(seed)
    policies = [
        make_policy(name, rng.getrandbits(64), iterations=iterations, rollout_plies=rollout_plies)
        for name in (attackers, defenders)
    ]
    return play_game(*policies, max_plies=max_plies, ruleset=ruleset)
