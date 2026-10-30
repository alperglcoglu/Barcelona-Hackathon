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

import warnings

import numpy as np

from . import config

FPS = 25
TYPICAL_MIN_PLAYERS = 15   # frames with this many players tracked define the typical pace
BALL_VELOCITY_SPAN = 2     # frames each side for ball and player velocities in pitch control


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


def arrival_times(ball, players, team, chunk=20000):
    """(t_home, t_away): each team's earliest arrival, in seconds, at the ball's projected point.

    The simple form of Spearman's model ("Beyond Expected Goals", 2018): the ball is projected
    PC_LOOKAHEAD seconds ahead along its own velocity; each player keeps running with their
    current velocity for PC_REACTION seconds, then runs straight at that point at PC_MAX_SPEED.
    NaN where the ball or the team is missing. Computed in chunks of frames to bound memory.
    """
    n = len(ball)
    t_home, t_away = np.full(n, np.nan), np.full(n, np.nan)
    k = BALL_VELOCITY_SPAN
    ball_smooth = np.stack([nan_moving_mean(ball[:, i], config.BALL_SPEED_SMOOTH) for i in (0, 1)], 1)
    ball_v = np.zeros_like(ball)
    ball_v[k:n - k] = (ball_smooth[2 * k:] - ball_smooth[:-2 * k]) / (2 * k / FPS)
    target = ball + config.PC_LOOKAHEAD * np.nan_to_num(ball_v)
    home = team == 0
    for a in range(0, n, chunk):
        b = min(a + chunk, n)
        lo, hi = max(a - k, 0), min(b + k, n)
        xy = players[lo:hi, :, :2]
        v = np.zeros_like(xy)
        if hi - lo > 2 * k:
            v[k:-k] = (xy[2 * k:] - xy[:-2 * k]) / (2 * k / FPS)
        v = np.nan_to_num(v)[a - lo:a - lo + (b - a)]
        start = xy[a - lo:a - lo + (b - a)] + config.PC_REACTION * v
        dist = np.hypot(target[a:b, None, 0] - start[:, :, 0], target[a:b, None, 1] - start[:, :, 1])
        arrival = config.PC_REACTION + dist / config.PC_MAX_SPEED
        with np.errstate(all="ignore"), warnings.catch_warnings():
            warnings.simplefilter("ignore", RuntimeWarning)
            t_home[a:b] = np.nanmin(arrival[:, home], axis=1)
            t_away[a:b] = np.nanmin(arrival[:, ~home], axis=1)
    return t_home, t_away


def _control(t_home, t_away):
    """2 P(home first) - 1 from the two arrival times, with Spearman's spread PC_SIGMA."""
    scale = np.pi / (np.sqrt(3.0) * config.PC_SIGMA)
    return np.nan_to_num(np.tanh(0.5 * scale * (t_away - t_home)))


def pitch_control_evidence(ball, players, team):
    """Per-frame evidence in [-1, 1] of which team reaches the ball first: + home, - away.

    The nearest player says who stands next to the ball now. Pitch control asks who gets there
    first, which is what decides a loose ball, an interception or a 50-50, often a second before
    anyone is nearest. 0 where the ball or either team is missing. See arrival_times.
    """
    t_home, t_away = arrival_times(ball, players, team)
    return np.where(np.isnan(ball[:, 0]), 0.0, _control(t_home, t_away))


def pitch_control_cue(ball, players, team):
    """(n_frames, 2) hand-over cue in [0, 1] from pitch control: home (0) / away (1) gets there first.

    Used as evidence on every frame, pitch control swings between the teams while the ball is in
    flight (section 15). As a cue it only lowers the switch cost, and only where a team is clearly
    first (control above PC_CUE_THRESHOLD) and can actually get there soon (its earliest arrival
    within PC_CUE_REACH seconds): an interception or a won loose ball, not a long pass in the air.
    """
    t_home, t_away = arrival_times(ball, players, team)
    control = _control(t_home, t_away)
    thr = config.PC_CUE_THRESHOLD
    cue = np.zeros((len(ball), 2))
    with np.errstate(invalid="ignore"):
        cue[:, 0] = np.clip((control - thr) / (1.0 - thr), 0.0, 1.0) * (t_home <= config.PC_CUE_REACH)
        cue[:, 1] = np.clip((-control - thr) / (1.0 - thr), 0.0, 1.0) * (t_away <= config.PC_CUE_REACH)
    cue[np.isnan(ball[:, 0])] = 0.0
    return cue


def handover_cue(ball, players, team):
    """(n_frames, 2) cue in [0, 1] that home (column 0) / away (column 1) takes the ball here.

    The Viterbi charges SWITCH_COST for every change of team. A fixed cost is a fixed amount of
    evidence a hand-over must collect, so a short possession by the other team is erased unless its
    frames add up to twice that. This cue lets methods.decode lower the cost locally, where the ball
    itself shows a team taking it (docs/TECHNICAL.md, section 10). It reads two signals that only
    exist where the ball is seen:

      - **clear control**: the team's nearest player is close to a calm ball and the other team's
        is not: max(0, closeness_team - closeness_other) * calm, the one-sided part of
        team_evidence. That is the nearest vs second-nearest gap between the two teams;
      - **a touch**: the team's player changes the ball's velocity (touch_evidence).

    The cue is the larger of the two, spread over +/- SWITCH_CUE_SPAN frames so a hand-over placed a
    little before or after the touch still gets it. Where the ball is missing the cue is 0 and the
    cost stays full: the ball-free evidence (width) is too slow to place a hand-over.
    """
    d_home, d_away = team_distances(ball, players, team)
    sigma2 = 2.0 * config.CONTROL_SIGMA ** 2
    close_home = np.where(np.isnan(d_home), 0.0, np.exp(-np.nan_to_num(d_home) ** 2 / sigma2))
    close_away = np.where(np.isnan(d_away), 0.0, np.exp(-np.nan_to_num(d_away) ** 2 / sigma2))
    speed = np.nan_to_num(ball_speed(ball), nan=0.0)
    calm = 1.0 / (1.0 + (speed / config.CONTROL_BALL_SPEED) ** 2)
    control = (close_home - close_away) * calm
    touch = touch_evidence(ball, d_home, d_away)
    cue = np.stack([np.maximum(np.maximum(control, 0.0), np.maximum(touch, 0.0)),
                    np.maximum(np.maximum(-control, 0.0), np.maximum(-touch, 0.0))], axis=1)
    cue[np.isnan(ball[:, 0])] = 0.0
    k = config.SWITCH_CUE_SPAN
    if k > 0:
        padded = np.pad(cue, ((k, k), (0, 0)))
        cue = np.max([padded[i:i + len(cue)] for i in range(2 * k + 1)], axis=0)
    return np.clip(cue, 0.0, 1.0)


def width_evidence(players, team):
    """Ball-free team evidence in [-1, 1] from how wide each team stands: + home wider, - away wider.

    The team in possession spreads out across the pitch to open passing lanes, and the defending
    team narrows to close them. This is a textbook principle of play (width in attack, compactness
    in defence) and the idea behind PathCRF-style models that read possession from the players'
    shape alone. It needs no ball, so it speaks on the frames where the ball is lost, which is
    where the ball-based team evidence is silent.

    Width is the standard deviation of a team's y coordinates (across the pitch). The difference
    home minus away is averaged over +/- WIDTH_SMOOTH frames and squashed with tanh(diff /
    WIDTH_SCALE). On the sample games, the sign of the raw difference alone agrees with the
    possessing team on about 82% of alive frames. Frames where either team has fewer than
    WIDTH_MIN_PLAYERS tracked players give 0.
    """
    y = players[:, :, 1]
    widths = []
    for side in (0, 1):
        own = y[:, team == side]
        n = (~np.isnan(own)).sum(axis=1)
        with np.errstate(invalid="ignore", divide="ignore"):
            spread = np.nanstd(np.where(n[:, None] > 0, own, 0.0), axis=1)
        widths.append(np.where(n >= config.WIDTH_MIN_PLAYERS, spread, np.nan))
    diff = nan_moving_mean(widths[0] - widths[1], config.WIDTH_SMOOTH)
    return np.where(np.isnan(diff), 0.0, np.tanh(np.nan_to_num(diff) / config.WIDTH_SCALE))


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


def pace_ratio(players):
    """Players' mean speed, averaged over +/- SPEED_SMOOTH frames, over the match's typical pace.

    Below DEAD_SPEED_RATIO the players have slowed down as in a stoppage (dead_evidence). NaN
    where no player is tracked.
    """
    speeds = players[:, :, 2]
    seen = ~np.isnan(speeds)
    n_seen = seen.sum(axis=1)
    with np.errstate(invalid="ignore", divide="ignore"):
        mean_speed = np.where(n_seen > 0, np.where(seen, speeds, 0.0).sum(axis=1) / n_seen, np.nan)
    mean_speed = nan_moving_mean(mean_speed, config.SPEED_SMOOTH)
    # Relative to this match's own typical pace, so the pivot transfers between matches and
    # trackers. The median needs no labels: it is taken over the frames where most players are
    # tracked, which leaves out the warm-up and half-time when players are off the pitch.
    crowded = (n_seen >= TYPICAL_MIN_PLAYERS) & np.isfinite(mean_speed)
    typical = np.median(mean_speed[crowded]) if crowded.any() else np.nan
    return mean_speed / typical


def dead_evidence(ball, players, pitch, gap_ball=None):
    """Per-frame log-odds that the ball is dead (positive) rather than alive (negative).

    Three signals are used, and each covers another's blind spot:

      - **The ball is outside the pitch.** It is decisive when it is visible, but the ball is often
        lost exactly when it goes out.
      - **The players have slowed down.** During a stoppage twenty-two players walk, and this signal
        never depends on the ball being seen. It is averaged over a second so that a single slow
        frame in open play means nothing. The speed is divided by the match's own median pace,
        because the trackers and the matches differ (BAR-ATH runs about 10% faster than the sample
        games in play and in stoppages). The score is linear in that ratio: below
        DEAD_SPEED_RATIO counts toward dead, above it toward alive.
      - **The ball has been missing for a long time.** See long_gap. gap_ball, if given, is the
        ball track whose gaps count (the tracking alone, when ball also has video fills).
    """
    length, width = float(pitch[0]), float(pitch[1])
    margin = config.OUT_MARGIN
    x, y = ball[:, 0], ball[:, 1]
    with np.errstate(invalid="ignore"):
        out = (x < -margin) | (x > length + margin) | (y < -margin) | (y > width + margin)

    slow = config.DEAD_SPEED_WEIGHT * (config.DEAD_SPEED_RATIO - pace_ratio(players))
    # No players tracked at all (camera off the pitch): no speed signal, treat as neutral-dead.
    slow = np.where(np.isnan(slow), config.DEAD_NO_PLAYERS, slow)
    return np.clip(slow, -config.DEAD_SPEED_CLIP, config.DEAD_SPEED_CLIP) \
        + np.where(out, config.DEAD_OUT_SCORE, 0.0) \
        + np.where(long_gap(ball if gap_ball is None else gap_ball, config.DEAD_GAP_MIN),
                   config.DEAD_GAP_SCORE, 0.0)


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
