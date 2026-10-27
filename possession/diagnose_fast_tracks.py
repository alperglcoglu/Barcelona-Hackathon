"""Are some "player" tracks really the ball? Aggregate diagnostics, no labels.

    python -m possession.diagnose_fast_tracks

Prints counts, shares and quantiles only, so the output is safe to share.

Watching the model's output, Alper saw a player dot leave the pitch far faster than anyone can run,
and the ball reappear elsewhere right after. A ball tracked under a player slot would hurt twice:
the nearest-player team evidence would read the ball as a player of that team standing on the
ball, and the ball would look missing while it is in fact tracked. Elite sprinters peak near
10 m/s and footballers near 9-10, so a track moving faster than FAST m/s for FAST_MIN frames is
suspect.

The script reports how often that happens, whether it happens when the ball is missing (the ball
is missing on about 41% of graded frames), how far such a track is from the ball when the ball is
seen, and whether such runs end with the track vanishing.
"""

import numpy as np

from . import io_atd

from games import GAME, half_video_blocks  # noqa: E402

# Params
FPS = 25
FAST = 12.0          # m/s: faster than any footballer
FAST_MIN = 5         # frames: a run must last this long (0.2 s) to count, so jitter does not
SMOOTH = 2           # frames each side for the speed estimate
TELEPORT = 3.0       # metres in one frame (75 m/s): an identity swap or a re-detection, not motion
QUANTILES = (10, 25, 50, 75, 90)


def q(values, fmt="{:.1f}"):
    values = np.asarray(values, dtype=float)
    values = values[~np.isnan(values)]
    if len(values) == 0:
        return "n/a"
    return "  ".join(("p{} " + fmt).format(p, v)
                     for p, v in zip(QUANTILES, np.percentile(values, QUANTILES)))


def runs(flags):
    edges = np.diff(np.concatenate(([0], flags.astype(np.int8), [0])))
    return list(zip(np.flatnonzero(edges == 1), np.flatnonzero(edges == -1)))


def smoothed_speed(xy):
    """Speed in m/s of one track, from positions averaged over +/- SMOOTH frames."""
    n = len(xy)
    out = np.full(n, np.nan)
    k = SMOOTH
    if n <= 2 * k + 1:
        return out
    step = np.hypot(*(xy[2 * k:] - xy[:-2 * k]).T) / (2 * k) * FPS
    out[k:n - k] = step
    return out


def main():
    data = io_atd.load(GAME)
    players, team, ball = data["players"], data["team"], data["ball"]
    n, slots = players.shape[:2]
    graded = np.zeros(n, dtype=bool)
    for _, first, last in half_video_blocks(GAME):
        graded[first:last + 1] = True
    ball_seen = ~np.isnan(ball[:, 0])

    fast_any = np.zeros(n, dtype=bool)
    lengths, peak, gone_after, near_ball, teleports = [], [], [], [], 0
    per_team = [0, 0]
    tracked_frames = 0
    for slot in range(slots):
        xy = players[:, slot, :2]
        present = ~np.isnan(xy[:, 0])
        tracked_frames += int((present & graded).sum())
        jump = np.full(n, np.nan)
        jump[1:] = np.hypot(*(xy[1:] - xy[:-1]).T)
        teleports += int(np.sum((jump > TELEPORT) & graded))
        speed = smoothed_speed(xy)
        fast = np.nan_to_num(speed) > FAST
        for a, b in runs(fast & graded):
            if b - a < FAST_MIN:
                continue
            fast_any[a:b] = True
            lengths.append((b - a) / FPS)
            peak.append(np.nanmax(speed[a:b]))
            per_team[int(team[slot])] += 1
            after = present[b:b + FPS]
            gone_after.append(len(after) > 0 and not after.all())
            seen = ball_seen[a:b]
            if seen.any():
                d = np.hypot(*(xy[a:b][seen] - ball[a:b][seen]).T)
                near_ball.append(np.nanmedian(d))

    g = graded
    print("\n== player tracks faster than {:.0f} m/s for at least {} frames (graded frames) ==".format(
        FAST, FAST_MIN))
    print("  runs: {}  (home slots {}, away slots {})".format(len(lengths), *per_team))
    print("  run length (s)       " + q(lengths, "{:.2f}"))
    print("  peak speed (m/s)     " + q(peak))
    print("  frames with at least one such track: {} ({:.2%} of graded)".format(
        int((fast_any & g).sum()), (fast_any & g).sum() / g.sum()))
    print("  ball missing on those frames: {:.1%}   (all graded frames: {:.1%})".format(
        np.mean(~ball_seen[fast_any & g]) if (fast_any & g).any() else float("nan"),
        np.mean(~ball_seen[g])))
    print("  runs where the track vanishes within 1 s after: {:.0%}".format(
        np.mean(gone_after) if gone_after else float("nan")))
    print("  distance from the fast track to the ball, where the ball is seen (m): " + q(near_ball))
    print("\n  one-frame jumps over {:.0f} m (identity swaps / re-detections): {} "
          "({:.3%} of tracked player-frames)".format(TELEPORT, teleports,
                                                   teleports / max(tracked_frames, 1)))

    # Ball gaps: how many have a fast "player" inside them, i.e. a candidate ball track
    gaps = [(a, b) for a, b in runs(~ball_seen & g) if b - a >= FPS]
    with_fast = sum(1 for a, b in gaps if fast_any[a:b].any())
    print("\n  ball gaps of 1 s or more: {}, of which {} ({:.0%}) contain a fast player track".format(
        len(gaps), with_fast, with_fast / max(len(gaps), 1)))


if __name__ == "__main__":
    main()
