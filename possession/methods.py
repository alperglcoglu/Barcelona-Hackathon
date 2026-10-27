"""Labelling methods: tracking arrays in, one label per frame out.

Each method takes the dict from io_atd.load() and returns a numpy array of single-character labels
('H', 'A', 'D', 'X'), one per tactical-video frame. pipeline.py picks one by name.

`baseline` is the organisers' reference, re-implemented on arrays. It exists for one reason: if it
does not reproduce evaluation/results/baseline_smooth50.csv exactly, something in our loading or
alignment is wrong, and every method built on the same arrays would inherit the error. Run the
parity check (pipeline.py --check-baseline) once after downloading the data.
"""

import sys

import numpy as np

from . import config, features, hmm, learned, tagging_prior

sys.path.insert(0, config.ROOT)
from games import GAME, half_video_blocks  # noqa: E402

HOME, AWAY, NOBODY = 0, 1, -1


def nearest_team(ball, players, team):
    """Per frame, the team of the tracked player nearest the ball. NOBODY if no ball or no player.

    Distances are in metres on both axes, so the pitch's aspect ratio does not bias the choice
    near the touchlines. Ties go to the lowest slot index, the same order the reference loop
    visits them in.
    """
    diff = players[:, :, :2] - ball[:, None, :]
    distance = np.einsum("fsk,fsk->fs", diff, diff)
    distance = np.where(np.isnan(distance), np.inf, distance)
    slot = np.argmin(distance, axis=1)
    seen = np.isfinite(distance[np.arange(len(slot)), slot])
    return np.where(seen, team[slot], NOBODY).astype(np.int8)


def majority_smooth(codes, window):
    """Centred majority vote over +/- window frames; D where neither team appears. Ties go home.

    Cumulative sums make the cost independent of the window size.
    """
    if window <= 0:
        return np.where(codes == HOME, "H", np.where(codes == AWAY, "A", "D"))
    home = np.concatenate(([0], np.cumsum(codes == HOME)))
    away = np.concatenate(([0], np.cumsum(codes == AWAY)))
    index = np.arange(len(codes))
    low = np.maximum(0, index - window)
    high = np.minimum(len(codes), index + window + 1)
    n_home, n_away = home[high] - home[low], away[high] - away[low]
    labels = np.full(len(codes), "D", dtype="<U1")
    seen = (n_home + n_away) > 0
    labels[seen] = np.where(n_home[seen] >= n_away[seen], "H", "A")
    return labels


def baseline(data, window=config.BASELINE_WINDOW):
    """Nearest player to the ball, then a majority vote. The published floor at window 50."""
    return majority_smooth(nearest_team(data["ball"], data["players"], data["team"]), window)


def frame_scores(data, events=None):
    """(n_frames, 3) per-frame scores for H, A and D, the input to the state model.

    H and A share the team evidence with opposite signs, and D gets the dead-ball log-odds. A
    frame with no information scores 0 everywhere and so never causes a change of state.
    """
    ball = features.interpolate_ball(data["ball"])
    team = config.TEAM_WEIGHT * features.team_evidence(ball, data["players"], data["team"])
    if events:
        team = team + tagging_prior.team_prior(events, len(team))
    dead = config.DEAD_WEIGHT * features.dead_evidence(ball, data["players"], data["pitch"])
    return np.stack([team / 2.0, -team / 2.0, dead], axis=1)


def decode(scores, halves):
    cost = hmm.transition_costs(config.SWITCH_COST, config.TO_DEAD_COST, config.FROM_DEAD_COST)
    labels = np.full(len(scores), "X", dtype="<U1")
    for first, last in halves:
        path = hmm.viterbi(scores[first:last + 1], cost)
        labels[first:last + 1] = hmm.LETTERS[path]
    return labels


def _default_halves(halves):
    if halves is None:
        return [(first, last) for _, first, last in half_video_blocks(GAME)]
    return halves


def state_model(data, events=None, halves=None):
    """Evidence per frame, then Viterbi over H / A / D separately in each half. X outside the halves.

    Each half is decoded on its own, because nothing carries over halftime and the scorer treats
    the halves separately as well. Frames outside both halves are never graded, so they get X.
    `halves` is [(first, last)] video frames, BAR-ATH's published halves by default. Other games
    (tuning/) pass their own.
    """
    return decode(frame_scores(data, events), _default_halves(halves))


def tv_dead_evidence(shown, covered, labels, graded):
    """Per-frame dead-vs-alive log-likelihood ratio from the TV feed, calibrated without labels.

    shown: the broadcast was showing that tactical frame live (possession/broadcast.py).
    The two rates it needs, how often the feed is away from live play during a stoppage and
    during play, are estimated on the whole graded match against `labels`, the tracking-only
    state model's own output. No ground truth is read. Those labels are noisy, which blurs the
    two rates towards each other, so the ratio is if anything an underestimate.
    Returns (evidence, (p_hidden_dead, p_hidden_alive)); 0 outside the broadcast's coverage.
    """
    use = covered & graded & np.isin(labels, ("H", "A", "D"))
    dead = labels == "D"
    if not (use & dead).any() or not (use & ~dead).any():
        return np.zeros(len(shown)), (np.nan, np.nan)
    hidden_dead = float(np.clip(np.mean(~shown[use & dead]), 0.01, 0.99))
    hidden_alive = float(np.clip(np.mean(~shown[use & ~dead]), 0.01, 0.99))
    evidence = np.zeros(len(shown))
    evidence[covered & ~shown] = np.log(hidden_dead / hidden_alive)
    evidence[covered & shown] = np.log((1 - hidden_dead) / (1 - hidden_alive))
    return evidence, (hidden_dead, hidden_alive)


def state_model_tv(data, events=None, halves=None, tv=None):
    """state_model plus the TV feed's dead-ball cue (docs/TECHNICAL.md, section 7).

    Two passes: the tracking-only model labels the match, those labels calibrate the TV evidence
    (tv_dead_evidence), and the Viterbi runs again with it added to the dead-ball score.
    tv is the dict from broadcast.load(); without it this is exactly state_model.
    """
    halves = _default_halves(halves)
    scores = frame_scores(data, events)
    labels = decode(scores, halves)
    if tv is None:
        return labels
    graded = np.zeros(len(scores), dtype=bool)
    for first, last in halves:
        graded[first:last + 1] = True
    evidence, rates = tv_dead_evidence(tv["shown"], tv["covered"], labels, graded)
    print("  TV cue: feed away from live play on {:.1%} of model-dead and {:.1%} of model-alive "
          "frames".format(*rates))
    scores[:, 2] += config.TV_DEAD_WEIGHT * evidence
    return decode(scores, halves)


def learned_model(data, events=None, halves=None, model=None):
    """Like state_model, but the per-frame evidence comes from the learned models in learned.py."""
    halves = _default_halves(halves)
    scores = learned.frame_scores(data, halves, model)
    if events:
        prior = tagging_prior.team_prior(events, len(scores))
        scores[:, 0] += prior / 2.0
        scores[:, 1] -= prior / 2.0
    return decode(scores, halves)


METHODS = {
    "baseline": baseline,
    "state_model": state_model,
    "learned": learned_model,
    "state_model_tv": state_model_tv,
}

# Methods that also read the Smart Tagging. pipeline.py loads it only for these.
USES_TAGGING = {"state_model", "learned", "state_model_tv"}
# Methods that also read the broadcast alignment (possession/broadcast.py).
USES_BROADCAST = {"state_model_tv"}
