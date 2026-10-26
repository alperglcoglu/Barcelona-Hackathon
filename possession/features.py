"""Per-frame evidence computed from the tracking arrays.

Two questions are asked of every frame, and they are answered separately because the tracking
answers them with different signals:

  1. **Is the ball alive?** A dead ball shows up as the ball outside the lines, or, much more often
     because the ball is frequently lost, as twenty-odd players who have stopped running.
     -> `dead_evidence`, a log-odds score: positive means dead, negative means alive.

  2. **If it is alive, which team controls it?** Control means a player is close to the ball and
     the ball is not just flying past him. -> `team_evidence` in [-1, 1]: +1 is clear home
     control, -1 is clear away control, and 0 means nobody controls it (a ball in flight, a
     contested ball, or a missing ball).

A frame with no evidence is not given a guess. It is left at 0 so the state model (hmm.py) keeps
whatever state it was already in. That is exactly the challenge's own rule: a loose or contested
ball stays with the team that last had it (CHALLENGE.md, "Ball in play").

All distances are in metres, speeds in m/s and times in frames at 25 fps.
"""

import numpy as np

from . import config

FPS = 25


def interpolate_ball(ball, max_gap=None):
    """Fill short gaps in the ball track by straight-line interpolation. Long gaps stay NaN.

    The ball is missing on 41% of graded frames, but a large share of those losses are a handful of
    frames long (a player's leg in front of the ball, a blur). Over a gap shorter than about half a
    second the ball cannot have gone far, so a straight line is a better guess than nothing. Over
    longer gaps a straight line invents passes that never happened, so they are left missing.
    """
    max_gap = config.BALL_MAX_GAP if max_gap is None else max_gap
    out = ball.copy()
    valid = ~np.isnan(ball[:, 0])
    if valid.sum() < 2:
        return out
    index = np.arange(len(ball))
    prev = np.maximum.accumulate(np.where(valid, index, -1))
    nxt = np.minimum.accumulate(np.where(valid, index, len(ball))[::-1])[::-1]
    gap = nxt - prev - 1
    fill = ~valid & (prev >= 0) & (nxt < len(ball)) & (gap <= max_gap)
    if fill.any():
        p, n = prev[fill], nxt[fill]
        t = ((index[fill] - p) / (n - p))[:, None]
        out[fill] = ball[p] * (1 - t) + ball[n] * t
    return out


def nan_moving_mean(values, half_width):
    """Centred moving mean over +/- half_width frames that ignores NaN. NaN where all are NaN."""
    ok = ~np.isnan(values)
    total = np.concatenate(([0.0], np.cumsum(np.where(ok, values, 0.0))))
    count = np.concatenate(([0], np.cumsum(ok)))
    index = np.arange(len(values))
    low = np.maximum(0, index - half_width)
    high = np.minimum(len(values), index + half_width + 1)
    n = count[high] - count[low]
    with np.errstate(invalid="ignore", divide="ignore"):
        return np.where(n > 0, (total[high] - total[low]) / np.maximum(n, 1), np.nan)


def ball_speed(ball):
    """Ball speed in m/s from positions, smoothed over a few frames so tracking jitter is not speed."""
    smooth = np.stack([nan_moving_mean(ball[:, k], config.BALL_SPEED_SMOOTH) for k in (0, 1)], 1)
    step = np.full(len(ball), np.nan)
    step[1:] = np.hypot(*(smooth[1:] - smooth[:-1]).T) * FPS
    return step


def team_distances(ball, players, team):
    """(d_home, d_away): distance from the ball to each team's nearest tracked player. NaN if unknown."""
    distance = np.hypot(players[:, :, 0] - ball[:, None, 0], players[:, :, 1] - ball[:, None, 1])
    distance = np.where(np.isnan(distance), np.inf, distance)
    out = []
    for side in (0, 1):
        d = distance[:, team == side].min(axis=1)
        out.append(np.where(np.isfinite(d), d, np.nan))
    return out[0], out[1]


def team_evidence(ball, players, team):
    """Per-frame evidence of control in [-1, 1]: + for home, - for away, 0 for nobody.

    Each team gets a "closeness" score exp(-d^2 / 2 sigma^2) from its nearest player, so a player
    0.5 m from the ball scores about 0.95, one at 2 m about 0.41, and one at 4 m practically 0. The
    evidence is the difference between the two teams' scores. This has the right behaviour in the
    three situations that matter:

      - one player on the ball, nobody else near   -> close to +/-1 (control)
      - players of both teams on the ball          -> close to 0 (a duel decides nothing yet)
      - nobody near the ball                       -> 0 (in flight; the last possessor keeps it)

    A fast ball is discounted, because a ball passing a player at 20 m/s is a pass going by or a
    deflection, not control. CHALLENGE.md explicitly says a touch without control and a deflection
    do not change possession.
    """
    d_home, d_away = team_distances(ball, players, team)
    sigma2 = 2.0 * config.CONTROL_SIGMA ** 2
    close_home = np.where(np.isnan(d_home), 0.0, np.exp(-np.nan_to_num(d_home) ** 2 / sigma2))
    close_away = np.where(np.isnan(d_away), 0.0, np.exp(-np.nan_to_num(d_away) ** 2 / sigma2))
    speed = np.nan_to_num(ball_speed(ball), nan=0.0)
    calm = 1.0 / (1.0 + (speed / config.CONTROL_BALL_SPEED) ** 2)
    evidence = (close_home - close_away) * calm
    if config.TOUCH_WEIGHT > 0:
        evidence = evidence + config.TOUCH_WEIGHT * touch_evidence(ball, d_home, d_away)
    return np.where(np.isnan(ball[:, 0]), 0.0, evidence)


def touch_evidence(ball, d_home, d_away):
    """+1 / -1 on frames where a home / away player touches the ball, 0 elsewhere.

    The idea comes from the possession-zone algorithm of Vidal-Codina et al. (2022), as described
    in the DataBallPy documentation: a player gains the ball when it is inside their zone and its
    velocity changes, because a ball that changes direction or speed next to a player was played
    by that player. Distance alone cannot tell a ball rolling past a player from one the player
    stops or redirects.

    Here: the ball velocity is averaged over TOUCH_SPAN frames before and after each frame, and a
    touch is a local maximum of the change |v_after - v_before| above TOUCH_DV m/s. The touch goes
    to the team whose nearest player is within TOUCH_RADIUS, if the other team's nearest player is
    farther away. The evidence is spread over +/- TOUCH_SPAN frames, the window it was seen in. No
    touch is read next to a ball gap.
    """
    n, k = len(ball), config.TOUCH_SPAN
    if n <= 2 * k + 1:
        return np.zeros(n)
    smooth = np.stack([nan_moving_mean(ball[:, i], config.BALL_SPEED_SMOOTH) for i in (0, 1)], 1)
    velocity = np.full((n, 2), np.nan)
    velocity[1:] = (smooth[1:] - smooth[:-1]) * FPS
    before = np.stack([nan_moving_mean(velocity[:, i], k // 2) for i in (0, 1)], 1)
    change = np.full(n, np.nan)
    change[k:n - k] = np.hypot(*(before[2 * k:] - before[:n - 2 * k]).T)
    # Only where the whole window saw the ball: averages cut short by a gap or by the start of the
    # data shift the estimated velocity and would fake a touch every time the ball reappears.
    reach = k + config.BALL_SPEED_SMOOTH + 1
    seen = nan_moving_mean((~np.isnan(ball[:, 0])).astype(float), reach)
    index = np.arange(n)
    whole = (seen == 1.0) & (index >= reach) & (index < n - reach)
    change = np.where(whole, np.nan_to_num(change, nan=0.0), 0.0)
    peak = np.zeros(n, dtype=bool)
    peak[1:-1] = (change[1:-1] >= change[:-2]) & (change[1:-1] > change[2:])
    peak &= change > config.TOUCH_DV
    home = np.nan_to_num(d_home, nan=np.inf)
    away = np.nan_to_num(d_away, nan=np.inf)
    side = np.where((home < config.TOUCH_RADIUS) & (home < away), 1.0,
                    np.where((away < config.TOUCH_RADIUS) & (away < home), -1.0, 0.0))
    touch = np.where(peak, side, 0.0)
    spread = np.convolve(touch, np.ones(2 * k + 1), mode="same")
    return np.clip(spread, -1.0, 1.0)


def dead_evidence(ball, players, pitch):
    """Per-frame log-odds that the ball is dead (positive) rather than alive (negative).

    Three signals are used, and each covers another's blind spot:

      - **The ball is outside the pitch.** It is decisive when it is visible, but the ball is often
        lost exactly when it goes out.
      - **The players have slowed down.** During a stoppage twenty-two players walk, and this signal
        never depends on the ball being seen. It is averaged over a second so that a single slow
        frame in open play means nothing. The score is linear in the speed: every m/s below the
        pivot speed counts toward dead, every m/s above it toward alive.
      - **The ball has been missing for a long time.** See long_gap.
    """
    length, width = float(pitch[0]), float(pitch[1])
    margin = config.OUT_MARGIN
    x, y = ball[:, 0], ball[:, 1]
    with np.errstate(invalid="ignore"):
        out = (x < -margin) | (x > length + margin) | (y < -margin) | (y > width + margin)

    speeds = players[:, :, 2]
    seen = ~np.isnan(speeds)
    n_seen = seen.sum(axis=1)
    with np.errstate(invalid="ignore", divide="ignore"):
        mean_speed = np.where(n_seen > 0, np.where(seen, speeds, 0.0).sum(axis=1) / n_seen, np.nan)
    mean_speed = nan_moving_mean(mean_speed, config.SPEED_SMOOTH)
    slow = config.DEAD_SPEED_WEIGHT * (config.DEAD_SPEED_PIVOT - mean_speed)
    # No players tracked at all (camera off the pitch): no speed signal, treat as neutral-dead.
    slow = np.where(np.isnan(slow), config.DEAD_NO_PLAYERS, slow)
    return np.clip(slow, -config.DEAD_SPEED_CLIP, config.DEAD_SPEED_CLIP) \
        + np.where(out, config.DEAD_OUT_SCORE, 0.0) \
        + np.where(long_gap(ball, config.DEAD_GAP_MIN), config.DEAD_GAP_SCORE, 0.0)


def long_gap(ball, min_length):
    """True on frames inside a run of missing ball at least min_length frames long.

    The automatic tracker loses the ball for long stretches mostly while it is out of play: on the
    whole BAR-ATH match the ball is missing in only 150 gaps with a median of 268 frames (about 11 s),
    and in the sample games over 80% of dead-ball frames have no ball. A short gap is an occlusion
    and says nothing. A long one is evidence of a stoppage. Pass the ball after interpolation, so the
    short gaps are already filled.
    """
    missing = np.isnan(ball[:, 0]).astype(np.int8)
    edges = np.diff(np.concatenate(([0], missing, [0])))
    starts, ends = np.flatnonzero(edges == 1), np.flatnonzero(edges == -1)
    out = np.zeros(len(ball), dtype=bool)
    for start, end in zip(starts, ends):
        if end - start >= min_length:
            out[start:end] = True
    return out
