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
    """Per-frame evidence in the units of the weighted team evidence: + home, - away.

    Overlapping phases add up and the sum is clipped to [-weight, weight], so ten overlapping tags
    of one team count no more than one. The default weight is TAG_TEAM_SHARE of TEAM_WEIGHT, the
    most a frame of clear tracking control can give. Being relative, it stays weak whatever
    scale the tuning picks for the tracking evidence.
    """
    weight = config.TAG_TEAM_SHARE * config.TEAM_WEIGHT if weight is None else weight
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


def transition_cue(events, n_frames):
    """(n_frames, 2) hand-over cue for home / away from Smart Tagging's ATTACKING TRANSITION tags.

    A transition begins with a change of possession to the tagged team. On BAR-ATH, half of these
    tags have a team switch by the model 0.8-5.1 s (p10-p90) after their start, ten times the rate
    at a random frame (possession.diagnose_tagging, no labels used). Tag starts run early, like the
    kickoff tags that open about 3.5 s before the kickoff. So the tagged team gets a cue of 1 from
    TRANSITION_CUE_FROM to TRANSITION_CUE_TO frames after the start, and methods.decode lowers the
    cost of switching to it there exactly as for the ball cue (features.handover_cue). It only
    makes a switch cheaper: the tracking evidence still has to show it.

    DEFENSIVE TRANSITION tags are the same moments seen from the other team (identical counts and
    timing), so they are not counted twice.
    """
    cue = np.zeros((n_frames, 2))
    for event in events:
        if event["code"] != "ATTACKING TRANSITION" or event["side"] is None:
            continue
        start = max(0, event["start_frame"] + config.TRANSITION_CUE_FROM)
        end = min(n_frames, event["start_frame"] + config.TRANSITION_CUE_TO)
        cue[start:end, 0 if event["side"] == "H" else 1] = 1.0
    return cue
