"""Put the broadcast recording on the tactical timeline, using the tracking alone.

    python -m possession.broadcast            # prints aggregate numbers only, caches the mapping

Why the broadcast at all
------------------------
The tactical video is one wide camera with no cuts: its homography is valid on 100% of frames, so
it carries no "the director cut away" signal. The broadcast is the TV feed. When play stops, the
director goes to a close-up, the bench, the crowd or a replay, and during live play he mostly stays
on the wide game camera. So "the TV feed was not showing this moment live" is evidence for a dead
ball that none of our tracking features sees, and it does not depend on the ball being tracked.
That matters because our weakest case is a stoppage in which the tracker still sees the ball
(called D only about 10% of the time, see docs/TECHNICAL.md).

A second, independent gain: the broadcast camera follows the ball closely, so its ball track may
see the ball on tactical frames where the tactical track lost it.

Neither is usable until we know, for every broadcast frame, which tactical frame shows the same
moment. DATA.md says no mapping is given. This module builds it without decoding any video.

How the alignment works
-----------------------
Both ATD exports give positions in the same pitch coordinates. If broadcast frame b shows live play
at tactical frame t, the ball and the players are in the same places in both, up to tracking noise.
Both feeds run at 25 fps, so over a stretch of live play t = b + c with a constant offset c. The
offset only changes where the broadcast recording itself was edited (halftime removed, for
example). Replays and close-ups do not change it: the live clock keeps running underneath them.

1. **Anchors (global search).** Take a few hundred 10-second broadcast windows with a valid
   homography and a moving ball. For each, slide the broadcast ball track over the whole tactical
   ball track, on a 5-frame grid, and score each offset by the share of broadcast ball positions
   that land within BALL_INLIER metres of the tactical ball. Keep the best offset when it is clear
   (high score, and no second peak almost as good elsewhere), then refine it to the exact frame.
   This is a brute-force template match; the 5-frame grid keeps it to a few seconds of numpy.
2. **Offsets (voting).** Confident anchors vote. Offsets that agree to within CLUSTER_RADIUS
   frames form a cluster; every cluster with at least CLUSTER_MIN anchors is one piece of the
   piecewise-constant mapping. We expect about two (one per half).
3. **Dense matching.** Cut the whole broadcast into 1-second windows and score a window at an
   offset by how well the broadcast players and ball land on tactical ones of the same class
   (home, away, ball): the share of broadcast points with a same-class tactical point within
   PLAYER_INLIER / BALL_INLIER metres. Two passes:
   - pass 1 tries, on every SURE_STRIDE-th window, each cluster offset plus or minus
     DENSE_SEARCH frames (clock drift), and keeps
     only near-perfect windows (SURE_SCORE). Their offsets give the local offset of each cluster,
     as a median over the sure windows within LOCAL_SPAN frames;
   - pass 2 scores every window only within FINE_SEARCH frames of those local offsets, and accepts
     the best when its share reaches MATCH_MIN_SCORE;
   - ties between neighbouring offsets (players move slowly) go to the offset whose points sit
     closest;
   - a matched stretch shorter than MIN_RUN windows is dropped. A live shot lasts many seconds,
     a lucky match is a lone second.
   Why two passes: trying a hundred offsets per window lets a window that matches nothing (a
   replay) find a lucky one by chance; trying five per cluster does not. Close-ups, crowd shots and
   replays match no live offset, so they come out unmatched, which is exactly what we want.

Holes shorter than one window in the tactical `shown` mask (a frame skipped or repeated where the
offset moves by one between windows) are closed.

The result, per tactical frame: `shown` (the TV feed was showing that moment live) and `bc_ball`
(the broadcast's ball position for that moment, NaN when it has none).

Nothing here uses labels. The printed summary is aggregate only (counts, shares and quantiles),
like possession.diagnose, so it is safe to share. Public-window lines are sanity checks, never
tuning (RULES.md section 2).
"""

import os
import sys

import numpy as np

from . import config, io_atd

sys.path.insert(0, config.ROOT)
from games import GAME, half_video_blocks  # noqa: E402

# Params
FPS = 25
ANCHOR_WINDOW = 250        # frames: 10 s of broadcast per anchor
ANCHOR_STEP = 5            # sample every 5th frame of an anchor, and try offsets on a 5-frame grid
ANCHOR_COUNT = 400         # anchors tried, evenly spread over the broadcast
ANCHOR_MIN_VALID = 0.9     # share of the anchor with a valid broadcast homography
ANCHOR_MIN_BALL = 0.5      # share of anchor samples with a broadcast ball
ANCHOR_MIN_SPREAD = 3.0    # metres: the ball must move, or a still ball matches everywhere
ANCHOR_MIN_SCORE = 0.5     # share of ball samples that must land on the tactical ball
ANCHOR_MAX_AMBIGUITY = 0.7  # second-best peak / best peak must stay below this
AMBIGUITY_EXCLUSION = 250  # frames around the best offset that do not count as a second peak
REFINE = 4                 # +/- frames around the grid optimum, at full resolution
CLUSTER_RADIUS = 50        # frames: anchors this close vote for the same offset
CLUSTER_MIN = 3            # anchors needed to accept an offset
DENSE_WINDOW = 25          # frames: 1-second windows for the dense pass
DENSE_SAMPLES = 5          # frames scored per dense window
DENSE_SEARCH = 25          # +/- frames around each cluster offset (clock drift), pass 1
SURE_SCORE = 0.9           # pass 1 windows this good fix the local offset
SURE_STRIDE = 4            # pass 1 looks at every 4th window; the offset changes slowly
LOCAL_SPAN = 7500          # frames: sure windows within 5 min set a window's local offset
FINE_SEARCH = 2            # +/- frames around the local offset, pass 2
MIN_RUN = 3                # consecutive matched windows (seconds) needed to call a stretch live
BALL_INLIER = 3.0          # metres
PLAYER_INLIER = 3.0        # metres; the tactical nearest-player noise is about 2 m at p50
MATCH_MIN_POINTS = 8       # players + ball seen on the broadcast, over the scored frames
MATCH_MIN_SCORE = 0.6      # share of broadcast points that land on a tactical point
CACHE_VERSION = 1
NO_OFFSET = np.iinfo(np.int32).min
QUANTILES = (10, 25, 50, 75, 90)
PUBLIC_GT = os.path.join(config.ROOT, "evaluation", "gt", "BAR-ATH_possession_gt_public.csv")


def _ball_score(b_ball, t_ball, inlier=BALL_INLIER):
    """Share of broadcast ball samples within `inlier` metres of the tactical ball. NaN never counts."""
    with np.errstate(invalid="ignore"):
        close = np.hypot(*np.moveaxis(b_ball - t_ball, -1, 0)) < inlier
    return close.sum(-1) / max(1, b_ball.shape[-2])


def find_anchor(b0, b_ball, t_ball, window=ANCHOR_WINDOW, step=ANCHOR_STEP):
    """Best offset c (tactical = broadcast + c) for the anchor starting at broadcast frame b0.

    Returns (offset, score, ambiguity) or None when the window has too little ball to try.
    """
    frames = np.arange(b0, b0 + window, step)
    frames = frames[~np.isnan(b_ball[frames, 0])]
    if len(frames) < ANCHOR_MIN_BALL * window / step:
        return None
    if np.max(np.std(b_ball[frames], axis=0)) < ANCHOR_MIN_SPREAD:
        return None

    n_t = len(t_ball)
    offsets = np.arange(-b0, n_t - b0 - window, step)
    if len(offsets) == 0:
        return None
    scores = _ball_score(b_ball[frames][None], t_ball[frames[None, :] + offsets[:, None]])
    best = int(np.argmax(scores))
    far = np.abs(offsets - offsets[best]) > AMBIGUITY_EXCLUSION
    second = scores[far].max() if far.any() else 0.0

    # Exact frame: every broadcast ball frame of the window, around the grid optimum.
    every = np.arange(b0, b0 + window)
    every = every[~np.isnan(b_ball[every, 0])]
    fine = offsets[best] + np.arange(-REFINE, REFINE + 1)
    fine = fine[(every[0] + fine >= 0) & (every[-1] + fine < n_t)]
    fine_scores = _ball_score(b_ball[every][None], t_ball[every[None, :] + fine[:, None]])
    pick = int(np.argmax(fine_scores))
    score = float(fine_scores[pick])
    return int(fine[pick]), score, float(second / max(scores[best], 1e-9))


def cluster_offsets(offsets, radius=CLUSTER_RADIUS, minimum=CLUSTER_MIN):
    """[(centre, n_votes)] for groups of offsets within `radius` of each other, biggest first."""
    groups = []
    for value in np.sort(np.asarray(offsets)):
        if groups and value - groups[-1][-1] <= radius:
            groups[-1].append(value)
        else:
            groups.append([value])
    out = [(int(np.median(group)), len(group)) for group in groups if len(group) >= minimum]
    return sorted(out, key=lambda item: -item[1])


def _point_sets(players, team, ball, frames):
    """Positions (len(frames), n_points, 2) and class (n_points,): team 0/1 for players, 2 = ball."""
    xy = np.concatenate([players[frames, :, :2], ball[frames][:, None, :]], axis=1)
    kind = np.concatenate([team.astype(np.int8), [2]])
    return xy, kind


def match_window(b_frames, offsets, bc, tac):
    """(best offset, score, n points) of one dense window, scoring every candidate offset."""
    b_xy, b_kind = _point_sets(bc["players"], bc["team"], bc["ball"], b_frames)
    seen = ~np.isnan(b_xy[..., 0])
    n_points = int(seen.sum())
    if n_points < MATCH_MIN_POINTS:
        return None, 0.0, n_points
    n_t = len(tac["ball"])
    offsets = offsets[(b_frames[0] + offsets >= 0) & (b_frames[-1] + offsets < n_t)]
    if len(offsets) == 0:
        return None, 0.0, n_points

    # Only the broadcast points actually seen take part: about a dozen per frame, not 52.
    sample, point = np.nonzero(seen)
    b_pts, b_cls = b_xy[sample, point], b_kind[point]              # (P, 2), (P,)
    t_frames = b_frames[None, :] + offsets[:, None]                # (n_off, n_samples)
    t_xy, t_kind = _point_sets(tac["players"], tac["team"], tac["ball"], t_frames.ravel())
    t_xy = t_xy.reshape(len(offsets), len(b_frames), -1, 2)[:, sample]  # (n_off, P, Q, 2)
    diff = b_pts[None, :, None, :] - t_xy
    dist = np.hypot(diff[..., 0], diff[..., 1])                    # (n_off, P, Q)
    same = b_cls[:, None] == t_kind[None, :]
    dist = np.where(same[None] & ~np.isnan(dist), dist, np.inf)
    nearest = dist.min(axis=-1)                                    # (n_off, P)
    limit = np.where(b_cls == 2, BALL_INLIER, PLAYER_INLIER)
    hits = (nearest < limit[None, :]).sum(axis=1)
    # Players move slowly, so neighbouring offsets often tie on hits. Break ties by closeness:
    # the second term is always below 1, so it never outranks one extra hit.
    closeness = np.minimum(nearest, limit[None, :]).sum(axis=1) / (limit.sum() + 1)
    best = int(np.argmax(hits - closeness))
    return int(offsets[best]), float(hits[best] / n_points), n_points


def align(bc, tac, valid=None, anchor_count=ANCHOR_COUNT, verbose=False):
    """Map the broadcast onto the tactical timeline. See the module docstring for the method.

    bc and tac are io_atd.load() dicts. valid is the broadcast homography's valid flag (all True
    if not given). Returns a dict of arrays plus a few summary values; no labels are involved.
    """
    n_b, n_t = len(bc["ball"]), len(tac["ball"])
    valid = np.ones(n_b, dtype=bool) if valid is None else np.asarray(valid, dtype=bool)

    # 1. anchors, in both orientations in case one export is mirrored
    starts = np.linspace(0, n_b - ANCHOR_WINDOW - 1, anchor_count).astype(int)
    starts = [b0 for b0 in starts if valid[b0:b0 + ANCHOR_WINDOW].mean() >= ANCHOR_MIN_VALID]
    pitch = tac["pitch"]
    best_orientation = None
    for mirrored in (False, True):
        b_ball = pitch - bc["ball"] if mirrored else bc["ball"]
        anchors = [(b0,) + found for b0 in starts
                   for found in [find_anchor(b0, b_ball, tac["ball"])] if found]
        confident = [a for a in anchors
                     if a[2] >= ANCHOR_MIN_SCORE and a[3] <= ANCHOR_MAX_AMBIGUITY]
        if verbose:
            print("  orientation {}: {} anchors tried, {} with enough ball, {} confident".format(
                "mirrored" if mirrored else "as-is", len(starts), len(anchors), len(confident)))
        if best_orientation is None or len(confident) > len(best_orientation[1]):
            best_orientation = (mirrored, confident, len(anchors))
    mirrored, confident, n_tried = best_orientation
    if mirrored:
        bc = dict(bc, ball=pitch - bc["ball"],
                  players=np.concatenate([pitch - bc["players"][:, :, :2],
                                          bc["players"][:, :, 2:]], axis=2))

    # 2. offset clusters
    clusters = cluster_offsets([a[1] for a in confident])

    # 3. dense matching, in two passes. Pass 1 tries the wide drift range and keeps only the
    # windows that match almost perfectly; they fix the local offset of each cluster. Pass 2 then
    # scores every window only within +/- FINE_SEARCH of that local offset. Trying ~100 offsets per
    # window lets a non-matching window (a replay) find a lucky one; trying ~10 does not.
    offset = np.full(n_b, NO_OFFSET, dtype=np.int64)
    score = np.zeros(n_b)
    matched = np.zeros(n_b, dtype=bool)
    if clusters:
        wide = np.unique(np.concatenate(
            [centre + np.arange(-DENSE_SEARCH, DENSE_SEARCH + 1) for centre, _ in clusters]))
        windows = [b0 for b0 in range(0, n_b - DENSE_WINDOW + 1, DENSE_WINDOW)
                   if valid[b0:b0 + DENSE_WINDOW].any()]
        samples = {b0: np.linspace(b0, b0 + DENSE_WINDOW - 1, DENSE_SAMPLES).astype(int)
                   for b0 in windows}
        sure = []                                         # (b0, offset) of near-perfect windows
        for b0 in windows[::SURE_STRIDE]:
            best, share, _ = match_window(samples[b0], wide, bc, tac)
            if best is not None and share >= SURE_SCORE:
                sure.append((b0, best))
        sure = np.array(sure, dtype=np.int64).reshape(-1, 2)

        for b0 in windows:
            candidates = []
            for centre, _ in clusters:
                mine = sure[(np.abs(sure[:, 1] - centre) <= DENSE_SEARCH)
                            & (np.abs(sure[:, 0] - b0) <= LOCAL_SPAN)]
                local = int(np.median(mine[:, 1])) if len(mine) else centre
                candidates.append(local + np.arange(-FINE_SEARCH, FINE_SEARCH + 1))
            best, share, _ = match_window(samples[b0], np.unique(np.concatenate(candidates)),
                                          bc, tac)
            if best is None:
                continue
            score[b0:b0 + DENSE_WINDOW] = share
            offset[b0:b0 + DENSE_WINDOW] = best
            matched[b0:b0 + DENSE_WINDOW] = share >= MATCH_MIN_SCORE

        # A live shot lasts many seconds, and a lucky match is a lone second. Keep only runs of
        # at least MIN_RUN matched windows.
        for start, end in _runs(matched):
            if end - start < MIN_RUN * DENSE_WINDOW:
                matched[start:end] = False

    # per tactical frame
    shown = np.zeros(n_t, dtype=bool)
    bc_ball = np.full((n_t, 2), np.nan)
    b_index = np.flatnonzero(matched)
    t_index = b_index + offset[b_index]
    keep = (t_index >= 0) & (t_index < n_t)
    shown[t_index[keep]] = True
    # Small drift corrections between windows skip or repeat a frame or two; close such holes.
    for start, end in _runs(~shown):
        if 0 < start and end < n_t and end - start < DENSE_WINDOW:
            shown[start:end] = True
    bc_ball[t_index[keep]] = bc["ball"][b_index[keep]]
    return {
        "shown": shown, "bc_ball": bc_ball, "matched": matched, "offset": offset,
        "score": score, "mirrored": mirrored, "clusters": clusters,
        "anchors": np.array([a[:4] for a in confident], dtype=float).reshape(-1, 4),
        "n_anchor_tried": n_tried,
    }


def coverage(out, n_t):
    """Tactical frames the broadcast recording spans at all, per offset cluster.

    Outside this (before the broadcast starts, after it ends, in halftime) "not shown" means only
    that there is no recording, so it must carry no evidence.
    """
    covered = np.zeros(n_t, dtype=bool)
    for centre, _ in out["clusters"]:
        frames = np.flatnonzero(out["matched"] & (np.abs(out["offset"] - centre) <= DENSE_SEARCH))
        if len(frames):
            first = max(0, frames[0] + out["offset"][frames[0]])
            last = min(n_t - 1, frames[-1] + out["offset"][frames[-1]])
            covered[first:last + 1] = True
    return covered


# -- running it on the real data -----------------------------------------------------------------

def _cache_path(game):
    return os.path.join(config.CACHE_DIR, "{}_broadcast_align_v{}.npz".format(game, CACHE_VERSION))


def load(game=GAME, data_dir=None, use_cache=True, verbose=False):
    """The alignment for the real match, computed once and then read from data/cache."""
    path = _cache_path(game)
    if use_cache and os.path.isfile(path):
        with np.load(path, allow_pickle=False) as stored:
            out = {key: stored[key] for key in stored.files}
        out["clusters"] = [tuple(int(v) for v in row) for row in out["clusters"]]
        out["mirrored"] = bool(out["mirrored"])
        return out
    sys.path.insert(0, os.path.join(config.ROOT, "starter"))
    import load_homography  # noqa: E402  (the organisers' reader, unmodified)

    tac = io_atd.load(game, data_dir)
    bc = io_atd.load(game, data_dir, source="broadcast")
    _, _, valid = load_homography.load(game, data_dir, source="broadcast")
    out = align(bc, tac, valid=valid, verbose=verbose)
    out["valid"] = valid
    if use_cache:
        os.makedirs(config.CACHE_DIR, exist_ok=True)
        stored = dict(out, clusters=np.array(out["clusters"], dtype=np.int64).reshape(-1, 2))
        np.savez(path, **stored)
    return out


def _quantiles(values):
    values = np.asarray(values, dtype=float)
    values = values[~np.isnan(values)]
    if len(values) == 0:
        return "n/a"
    return "  ".join("p{} {:.2f}".format(q, v)
                     for q, v in zip(QUANTILES, np.percentile(values, QUANTILES)))


def _runs(flags):
    """[(start, end_exclusive)] of the True runs."""
    edges = np.diff(np.concatenate(([0], flags.astype(np.int8), [0])))
    return list(zip(np.flatnonzero(edges == 1), np.flatnonzero(edges == -1)))


def summary(out, tac, labels=None, gt=None):
    """Aggregate lines only: shares, counts and quantiles. Safe to paste into a discussion."""
    lines = []
    n_t = len(tac["ball"])
    graded = np.zeros(n_t, dtype=bool)
    for _, first, last in half_video_blocks(GAME):
        graded[first:last + 1] = True
    matched, valid = out["matched"], out["valid"]

    lines.append("orientation: {}".format("mirrored" if out["mirrored"] else "as-is"))
    lines.append("anchors: {} with enough ball, {} confident".format(
        int(out["n_anchor_tried"]), len(out["anchors"])))
    for centre, votes in out["clusters"]:
        in_cluster = np.abs(out["anchors"][:, 1] - centre) <= CLUSTER_RADIUS
        spread = out["anchors"][in_cluster, 1] - centre
        frames = np.flatnonzero(matched & (np.abs(out["offset"] - centre) <= DENSE_SEARCH))
        span = "broadcast {}..{}".format(frames[0], frames[-1]) if len(frames) else "no frames"
        lines.append("  offset {:+d} frames ({:+.1f} s): {} anchors, spread {}..{}, {}".format(
            centre, centre / FPS, votes, int(spread.min()), int(spread.max()), span))
    if not out["clusters"]:
        lines.append("  NO OFFSET FOUND -- the alignment failed; nothing below is meaningful")
        return lines

    lines.append("dense windows: best-offset score " + _quantiles(out["score"][valid]))
    lines.append("broadcast frames matched: {:.1%} overall, {:.1%} of valid-homography frames, "
                 "{:.1%} of invalid ones".format(matched.mean(), matched[valid].mean(),
                                                 matched[~valid].mean() if (~valid).any() else 0))
    for centre, _ in out["clusters"]:
        near = matched & (np.abs(out["offset"] - centre) <= DENSE_SEARCH)
        if near.any():
            drift = out["offset"][near] - centre
            lines.append("  drift around {:+d}: ".format(centre) + _quantiles(drift))

    shown = out["shown"]
    lines.append("graded tactical frames shown live on TV: {:.1%}".format(shown[graded].mean()))
    gaps = [e - s for s, e in _runs(~shown & graded)]
    lines.append("not-shown stretches in graded time: {}, length (s) ".format(len(gaps))
                 + _quantiles(np.array(gaps) / FPS))

    t_ball, b_ball = tac["ball"], out["bc_ball"]
    both = graded & shown & ~np.isnan(t_ball[:, 0]) & ~np.isnan(b_ball[:, 0])
    lines.append("ball, both tracks visible on {:.1%} of graded frames; distance (m) ".format(
        both.mean() / max(graded.mean(), 1e-9)) + _quantiles(np.hypot(*(t_ball[both] - b_ball[both]).T)))
    t_missing = graded & np.isnan(t_ball[:, 0])
    fill = t_missing & ~np.isnan(b_ball[:, 0])
    lines.append("tactical ball missing on {:.1%} of graded frames; broadcast has it on {:.1%} "
                 "of those".format(t_missing.sum() / graded.sum(), fill.sum() / max(1, t_missing.sum())))

    if labels is not None:
        dead = labels == "D"
        lines.append("frozen model, whole graded match: called D on {:.1%} of shown frames, "
                     "{:.1%} of not-shown frames".format(dead[graded & shown].mean(),
                                                         dead[graded & ~shown].mean()))
    if gt is not None:
        window = gt != "X"
        is_dead = gt == "D"
        visible = ~np.isnan(t_ball[:, 0])
        lines.append("public window (sanity only, not for tuning):")
        lines.append("  shown live on {:.1%} of alive frames, {:.1%} of dead frames".format(
            shown[window & ~is_dead].mean(), shown[window & is_dead].mean()))
        lines.append("  dead frames with the tactical ball visible: {} , not shown on {:.1%}".format(
            int((window & is_dead & visible).sum()),
            (~shown[window & is_dead & visible]).mean() if (window & is_dead & visible).any() else 0))
        n_not = int((window & ~shown).sum())
        if n_not:
            lines.append("  of {} not-shown frames, {:.1%} are dead".format(
                n_not, is_dead[window & ~shown].mean()))
    return lines


def main():
    import argparse

    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--data-dir", help="data bundle location (default: ./data or MTAH_DATA_DIR)")
    parser.add_argument("--no-cache", action="store_true", help="recompute the alignment")
    args = parser.parse_args()

    from . import io_tagging, methods

    print("aligning the broadcast to the tactical timeline (a few minutes the first time) ...")
    out = load(GAME, args.data_dir, use_cache=not args.no_cache, verbose=True)
    tac = io_atd.load(GAME, args.data_dir)
    try:
        events = io_tagging.load(GAME, args.data_dir)
    except IOError:
        events = []
    labels = methods.state_model(tac, events)
    gt = None
    if os.path.isfile(PUBLIC_GT):
        with open(PUBLIC_GT) as handle:
            next(handle)
            gt = np.array([line.rstrip("\r\n").split(",")[1].strip() for line in handle])
    print()
    for line in summary(out, tac, labels, gt):
        print(line)


if __name__ == "__main__":
    main()
