"""Learned per-frame evidence: two small logistic models trained on other games.

The hand-written evidence in features.py looks at one thing per question: how close each team is
to the ball, and how fast the players run. It goes silent whenever the ball is lost, which on
BAR-ATH is 41% of the graded frames. These models add **collective** signals that never depend on
seeing the ball, and they learn how much each signal is worth instead of us guessing:

  - the team in possession spreads out to make the pitch big, and the defending team gets compact
    (team spread);
  - the team in possession moves toward the goal it attacks, and the defending team drops back
    (centroid velocity along the attacking direction);
  - after a loss, the team that lost the ball sprints back (team mean speed, per team);
  - a dead ball shows twenty-two slow players, a still ball, or a ball off the pitch.

"Ball missing" is deliberately *not* a dead-ball feature. In the sample games the ball is mostly
missing because it is out of play, so a model would learn "missing means dead". In the automatic
BAR-ATH tracking the ball goes missing in open play all the time (occlusions, duels, high balls), so
that rule would not transfer.

Both are plain logistic regressions over the features below, fitted by tuning/train.py on
Metrica's public sample games (never on BAR-ATH) and stored as coefficients in
possession/models/emission.json. The pipeline therefore needs no training step and no external
data to run: the JSON file is the trained model. A logistic model outputs log-odds directly, which
is exactly the unit the state model adds up (hmm.py).

Every feature is either a ball feature multiplied by "ball visible" (so it is 0 when the ball is
missing), or a collective feature that is always defined. The visibility flag itself is a feature,
so the models can learn how much less to trust a frame without the ball.
"""

import json
import os

import numpy as np

from . import config, features

# Params
MODEL_PATH = os.path.join(config.HERE, "models", "emission.json")
FPS = 25
VELOCITY_SMOOTH = 25     # +/- frames for centroid velocities
SPREAD_SMOOTH = 12       # +/- frames for team spread
CLIP_DISTANCE = 15.0     # metres; beyond this a distance difference says nothing more

TEAM_FEATURES = (
    "closeness_diff",      # features.team_evidence without the speed discount: + = home closer
    "nearest_diff",        # d_away - d_home, clipped: + = home closer
    "contested",           # both teams within 2 m of the ball: a duel, not control
    "ball_attack_x",       # where the ball is along the home attacking direction, -1..1
    "spread_diff",         # home spread - away spread (m)
    "advance_diff",        # home centroid speed toward the goal it attacks - away's (m/s)
    "speed_diff",          # home mean speed - away mean speed (m/s)
)
DEAD_FEATURES = (
    "mean_speed",          # all players, smoothed over 1 s (m/s)
    "speed_below_pivot",   # max(0, 1.6 - mean_speed): slow players, isolated from fast ones
    "ball_out",            # visible and beyond a line
    "ball_still",          # visible and slower than 1 m/s
    "ball_nearest",        # distance from the visible ball to the nearest player, clipped
    "n_tracked",           # players tracked, / 22
    "spread_all",          # spread of all players (m): stoppages are not always compact, but
                           # kick-offs and set pieces have a shape of their own
)


def _smooth(values, width):
    return features.nan_moving_mean(values, width)


def _team_stats(players, team, side):
    """Per frame: centroid x, spread (RMS distance to the centroid), mean speed. NaN if nobody."""
    mine = players[:, team == side]
    ok = ~np.isnan(mine[:, :, 0])
    n = ok.sum(axis=1)
    with np.errstate(invalid="ignore", divide="ignore"):
        cx = np.where(ok, mine[:, :, 0], 0).sum(1) / n
        cy = np.where(ok, mine[:, :, 1], 0).sum(1) / n
        spread = np.sqrt((np.where(ok, (mine[:, :, 0] - cx[:, None]) ** 2
                                   + (mine[:, :, 1] - cy[:, None]) ** 2, 0)).sum(1) / n)
        speed_ok = ok & ~np.isnan(mine[:, :, 2])
        speed = np.where(speed_ok, mine[:, :, 2], 0).sum(1) / speed_ok.sum(1)
    nan = n == 0
    return [np.where(nan, np.nan, v) for v in (cx, spread, speed)]


def attack_sign(players, team, halves, n_frames):
    """+1 on frames where home attacks toward x = length, -1 where it attacks toward x = 0.

    Read off the data: over a half, the team whose players stand on average closer to x = 0 is
    defending that goal. No fixed constant, and no knowledge of the game is needed.
    """
    sign = np.zeros(n_frames)
    home_x, _, _ = _team_stats(players, team, 0)
    away_x, _, _ = _team_stats(players, team, 1)
    for first, last in halves:
        h = np.nanmean(home_x[first:last + 1])
        a = np.nanmean(away_x[first:last + 1])
        sign[first:last + 1] = 1.0 if h < a else -1.0
    return sign


def design(data, halves):
    """(team_X, dead_X): feature matrices (n_frames, len(TEAM_FEATURES) / len(DEAD_FEATURES))."""
    ball = features.interpolate_ball(data["ball"])
    players, team = data["players"], data["team"]
    length, width = float(data["pitch"][0]), float(data["pitch"][1])
    n = len(ball)
    visible = ~np.isnan(ball[:, 0])

    d_home, d_away = features.team_distances(ball, players, team)
    sigma2 = 2.0 * config.CONTROL_SIGMA ** 2
    close_h = np.where(np.isnan(d_home), 0.0, np.exp(-np.nan_to_num(d_home) ** 2 / sigma2))
    close_a = np.where(np.isnan(d_away), 0.0, np.exp(-np.nan_to_num(d_away) ** 2 / sigma2))
    nearest_diff = np.clip(np.nan_to_num(d_away - d_home), -CLIP_DISTANCE, CLIP_DISTANCE)
    with np.errstate(invalid="ignore"):
        contested = visible & (d_home < 2.0) & (d_away < 2.0)

    sign = attack_sign(players, team, halves, n)
    ball_attack_x = np.where(visible, sign * (np.nan_to_num(ball[:, 0]) / length * 2 - 1), 0.0)

    hx, hspread, hspeed = _team_stats(players, team, 0)
    ax, aspread, aspeed = _team_stats(players, team, 1)
    advance_h = np.gradient(np.nan_to_num(_smooth(hx, VELOCITY_SMOOTH))) * FPS * sign
    advance_a = -np.gradient(np.nan_to_num(_smooth(ax, VELOCITY_SMOOTH))) * FPS * sign
    team_X = np.stack([
        (close_h - close_a) * visible,
        nearest_diff * visible,
        contested.astype(float),
        ball_attack_x,
        np.nan_to_num(_smooth(hspread, SPREAD_SMOOTH) - _smooth(aspread, SPREAD_SMOOTH)),
        np.clip(advance_h - advance_a, -5, 5),
        np.nan_to_num(_smooth(hspeed, SPREAD_SMOOTH) - _smooth(aspeed, SPREAD_SMOOTH)),
    ], axis=1)

    speeds = players[:, :, 2]
    seen = ~np.isnan(speeds)
    with np.errstate(invalid="ignore", divide="ignore"):
        mean_speed = np.where(seen, speeds, 0).sum(1) / seen.sum(1)
    mean_speed = np.nan_to_num(_smooth(mean_speed, config.SPEED_SMOOTH), nan=0.0)
    margin = config.OUT_MARGIN
    with np.errstate(invalid="ignore"):
        out = visible & ((ball[:, 0] < -margin) | (ball[:, 0] > length + margin)
                         | (ball[:, 1] < -margin) | (ball[:, 1] > width + margin))
    ball_speed = features.ball_speed(ball)
    with np.errstate(invalid="ignore"):
        still = visible & (np.nan_to_num(ball_speed, nan=99.0) < 1.0)
    nearest = np.fmin(d_home, d_away)
    all_x = np.where(seen | ~np.isnan(players[:, :, 0]), players[:, :, 0], np.nan)
    ok = ~np.isnan(all_x)
    n_tracked = ok.sum(1)
    with np.errstate(invalid="ignore", divide="ignore"):
        cx = np.where(ok, all_x, 0).sum(1) / n_tracked
        cy = np.where(ok, players[:, :, 1], 0).sum(1) / n_tracked
        spread_all = np.sqrt(np.where(ok, (all_x - cx[:, None]) ** 2
                                      + (players[:, :, 1] - cy[:, None]) ** 2, 0).sum(1)
                             / n_tracked)
    dead_X = np.stack([
        mean_speed,
        np.maximum(0.0, 1.6 - mean_speed),
        out.astype(float),
        still.astype(float),
        np.where(visible, np.clip(np.nan_to_num(nearest, nan=CLIP_DISTANCE), 0, CLIP_DISTANCE), 0),
        n_tracked / 22.0,
        np.nan_to_num(_smooth(spread_all, SPREAD_SMOOTH)),
    ], axis=1)
    return team_X, dead_X


# -- fitting and applying ------------------------------------------------------------------------

def fit_logistic(X, y, l2=1.0, iterations=50):
    """Logistic regression by Newton's method (IRLS) with an L2 penalty. numpy only.

    Features are standardised first, and the returned coefficients undo that, so the model is
    stored in raw feature units. y is 0/1.
    """
    mean, std = X.mean(0), X.std(0)
    std[std == 0] = 1.0
    Z = np.c_[np.ones(len(X)), (X - mean) / std]
    w = np.zeros(Z.shape[1])
    penalty = np.r_[0.0, np.full(Z.shape[1] - 1, l2)]
    for _ in range(iterations):
        p = 1.0 / (1.0 + np.exp(-np.clip(Z @ w, -30, 30)))
        gradient = Z.T @ (p - y) + penalty * w
        hessian = (Z * (p * (1 - p))[:, None]).T @ Z + np.diag(penalty)
        step = np.linalg.solve(hessian, gradient)
        w -= step
        if np.abs(step).max() < 1e-8:
            break
    coef = w[1:] / std
    return {"intercept": float(w[0] - coef @ mean), "coef": coef.tolist()}


def log_odds(model, X):
    return model["intercept"] + X @ np.asarray(model["coef"])


def load_model(path=MODEL_PATH):
    with open(path) as handle:
        return json.load(handle)


def frame_scores(data, halves, model=None):
    """(n_frames, 3) scores for H, A, D from the learned models, ready for hmm.viterbi.

    The team model's log-odds of home vs away goes half to H and half (negated) to A, as in
    methods.frame_scores. The dead model's log-odds goes to D. Both are clipped so a single frame
    can never be decisive on its own; that is the state model's job.
    """
    model = model or load_model()
    team_X, dead_X = design(data, halves)
    clip = config.LEARNED_CLIP
    team = np.clip(log_odds(model["team"], team_X), -clip, clip) * config.TEAM_WEIGHT
    dead = np.clip(log_odds(model["dead"], dead_X), -clip, clip) * config.DEAD_WEIGHT
    return np.stack([team / 2.0, -team / 2.0, dead], axis=1)
