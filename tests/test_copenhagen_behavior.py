"""The sampled tactical-prior policy and its actor gradient must agree."""

from pathlib import Path

import pytest

np = pytest.importorskip("numpy")

from hnef.learning.onpolicy import (  # noqa: E402
    behavior_prior,
    behavior_probs,
    collect,
    features,
    gradients,
    starts,
    train,
)
from hnef.learning.plumbing import LinearModel, load  # noqa: E402

DATA = Path(__file__).parent / "fixtures/copenhagen-training-defender-staged-v1/cases.json"


@pytest.fixture(autouse=True)
def caps(monkeypatch):
    for key in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS"):
        monkeypatch.setenv(key, "1")


def test_fixed_prior_masks_and_finite_difference():
    state = starts(DATA)[0][16]
    model = LinearModel()
    x, mask = features(state)
    p, value = model.forward(x[None], mask[None])
    prior = behavior_prior(state, "immediate-win-prior4-v1")
    q = behavior_probs(p[0], prior)
    wins = np.flatnonzero((prior == 4) & mask)
    assert len(wins) > 0
    assert q[~mask].sum() == 0
    assert q.sum() == pytest.approx(1)
    action = int(wins[0])
    assert q[action] > p[0, action]
    sample = (x, mask, action, float(value[0]), 1.0, prior)
    actor, _, _ = gradients(model, [sample])
    # Bias affects every logit; verify selected, unselected legal and masked entries.
    for col in (action, int(np.flatnonzero(mask & (prior == 1))[0]), int(np.flatnonzero(~mask)[0])):
        losses = []
        for sign in (1, -1):
            model.actor[-1, col] = sign * 1e-6
            losses.append(gradients(model, [sample])[2]["loss"])
        model.actor[-1, col] = 0
        assert (losses[0] - losses[1]) / 2e-6 == pytest.approx(actor[-1, col], abs=1e-8)
    with pytest.raises(ValueError, match="prior"):
        behavior_probs(p[0], np.zeros_like(prior))


def test_collection_records_actual_behavior_probability():
    state = starts(DATA)[0][16]
    model = LinearModel()
    samples, trace = collect(
        model, state, np.random.default_rng(17), 23, 2, 0.99, behavior="immediate-win-prior4-v1"
    )
    assert len(samples) == len(trace["behavior_probabilities"])
    for sample, actual in zip(samples, trace["behavior_probabilities"], strict=True):
        x, mask, action, _, _, prior = sample
        p, _ = model.forward(x[None], mask[None])
        assert actual == behavior_probs(p[0], prior)[action]
    assert trace["behavior"] == "immediate-win-prior4-v1"


def test_behavior_resume_is_exact_and_cannot_switch(tmp_path):
    cfg = dict(
        seed=17,
        horizon=2,
        behavior="immediate-win-prior4-v1",
        curriculum="broad64-v1",
        reference_id="mixed-random-heuristic-v1",
    )
    train(tmp_path / "full", DATA, episodes=4, **cfg)
    train(tmp_path / "first", DATA, episodes=2, **cfg)
    train(
        tmp_path / "second", DATA, episodes=2, resume=tmp_path / "first/checkpoint-0002.npz", **cfg
    )
    a, am, ar = load(tmp_path / "full/checkpoint-0004.npz", {})
    b, bm, br = load(tmp_path / "second/checkpoint-0004.npz", {})
    assert np.array_equal(a.actor, b.actor) and np.array_equal(a.critic, b.critic)
    assert am == bm and ar.bit_generator.state == br.bit_generator.state
    cfg["behavior"] = "policy-v1"
    with pytest.raises(ValueError, match="contract"):
        train(
            tmp_path / "bad", DATA, episodes=1, resume=tmp_path / "first/checkpoint-0002.npz", **cfg
        )
