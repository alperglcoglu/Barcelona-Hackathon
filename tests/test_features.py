"""Feature tests on small synthetic arrays."""

import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def test_width_evidence_points_to_the_wider_team():
    from possession import features
    n = 100
    players = np.full((n, 22, 3), np.nan)
    rng = np.random.default_rng(0)
    players[:50, :11, 1] = rng.uniform(5, 63, (50, 11))      # home wide
    players[:50, 11:, 1] = rng.uniform(25, 43, (50, 11))     # away compact
    players[50:, :11, 1] = rng.uniform(25, 43, (50, 11))
    players[50:, 11:, 1] = rng.uniform(5, 63, (50, 11))
    players[:, :, 0] = 50.0
    team = np.array([0] * 11 + [1] * 11, dtype=np.int8)
    evidence = features.width_evidence(players, team)
    assert (evidence[:20] > 0.5).all() and (evidence[-20:] < -0.5).all()
    players[:, 12:, :] = np.nan                               # one away player left: no evidence
    assert (features.width_evidence(players, team) == 0).all()
