"""Turn Smart Tagging phases into weak per-frame evidence about which team has the ball.

Every phase names the team performing it. For an attacking phase (Build Up, Progression, Ball in
the Box, ...) that team has the ball. For a defending phase (Defending in ... Third, Defensive
Transition) that team does not, so the evidence goes to the other side.

This is deliberately weak. The tagging is automatic, its intervals are phases rather than
possessions, and a single phase can cover a turnover. The weight per frame is set so that a whole
phase can tip a frame the tracking is unsure about, but cannot override clear tracking evidence on
its own. Throw-ins are tagged Team = N/A and contribute nothing.
"""

import numpy as np

from . import config

# Params
ATTACKING_CODES = {
    "BUILD UP", "PROGRESSION", "CREATING CHANCES", "ATTACKING TRANSITION", "LONG BALL",
    "BALL IN FINAL THIRD", "BALL IN THE BOX", "PLAYERS IN THE BOX",
}
DEFENDING_CODES = {
    "DEFENDING IN ATTACKING THIRD", "DEFENDING IN MIDDLE THIRD", "DEFENDING IN DEFENSIVE THIRD",
    "DEFENSIVE TRANSITION",
}


def team_prior(events, n_frames, weight=None):
    """Per-frame evidence in the same units as features.team_evidence: + home, - away.

    Overlapping phases add up and the sum is clipped to [-weight, weight], so ten overlapping tags
    of one team count no more than one.
    """
    weight = config.TAG_TEAM_WEIGHT if weight is None else weight
    prior = np.zeros(n_frames)
    for event in events:
        side = event["side"]
        if side is None:
            continue
        if event["code"] in ATTACKING_CODES:
            sign = 1.0 if side == "H" else -1.0
        elif event["code"] in DEFENDING_CODES:
            sign = -1.0 if side == "H" else 1.0
        else:
            continue
        start = max(0, event["start_frame"])
        end = min(n_frames, event["end_frame"])
        prior[start:end] += sign
    return np.clip(prior, -1.0, 1.0) * weight
