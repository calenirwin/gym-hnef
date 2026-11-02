"""Held-out, authored rule examples; unresolved rulings never become win labels."""

import json
from dataclasses import asdict
from pathlib import Path

import pytest

from hnef.engine import GameState, Move, Player, Ruleset, apply_move

CORPUS = json.loads(
    (Path(__file__).parent / "fixtures/copenhagen-heldout-v1/cases.json").read_text()
)["cases"]


@pytest.mark.parametrize(
    "case", [c for c in CORPUS if c["kind"] == "transition"], ids=lambda c: c["id"]
)
def test_heldout_rule_transition(case):
    state = GameState(
        tuple(case["board"]), Player(case["player"]), ruleset=Ruleset(case["ruleset"])
    )
    successor = apply_move(state, Move(*case["move"]))
    assert (asdict(successor.outcome) if successor.outcome else None) == case["expected_outcome"]
    for r, c, piece in case["expected_cells"]:
        assert successor.at((r, c)) == piece
    if case.get("adjudication_status") == "unknown":
        assert successor.outcome is None
        assert case["question"]


def test_corpus_sources_splits_and_unknowns_are_explicit():
    assert len({c["id"] for c in CORPUS}) == len(CORPUS)
    for case in CORPUS:
        assert case["source"].startswith("https://aagenielsen.dk/") and case["rule"]
        if case["kind"] == "research-question":
            assert case["expected_adjudication"] == "unknown"
            assert case["board"] is None and case["limitation"]
        else:
            assert case["split"] == "held-out" and case["origin"]
