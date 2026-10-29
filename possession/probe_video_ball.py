"""Is the ball visible in the video where the tracking loses it? Aggregate diagnostics, no labels.

    python -m possession.probe_video_ball [--labels evaluation/results/La_Messia.csv]
    python -m possession.probe_video_ball --labels evaluation/results/La_Messia.csv --weights best.pt

Prints counts, shares and quantiles only, so the output is safe to share. It writes no file and no
image.

The ball is missing on 40.7% of graded frames. Short gaps are already bridged
(features.interpolate_ball) and long ones are read as stoppages (features.long_gap), but a long gap
in open play leaves the team evidence silent for seconds. A ball detector run on the tactical video
could fill those frames: the tactical homography is valid on every frame, so a detected pixel maps
straight back to the pitch. Before building that into the model, this script answers two questions
on a sample of frames:

  Part A (tracking only, always runs)
    How the missing frames are spread over gap lengths, and how many of them fall in live play:
    by the labels of a given submission file (the frozen model's own output, not ground truth) or,
    without one, by the players' pace relative to the match's median pace.

  Part B (video, only with --weights; needs requirements-cv.txt)
    Runs a YOLO detector on sampled frames of the tactical video:
      - calibration frames, where the tracking has the ball: how far the detection is from it;
      - gap frames in live play: how often a ball is found on the pitch, and whether the team of
        the nearest player to it agrees with the label.
    Then prints a go / stop line for building the detector into the model.

Weights. Any YOLO model with a class whose name contains "ball" works, for example:
  - football-ball-detection.pt, the ball-only model of github.com/roboflow/sports
    (pip install gdown; gdown -O football-ball-detection.pt
    "https://drive.google.com/uc?id=1isw4wx-MK9h9LMr36VvIWlJD6ppUvw7V", the link in its
    examples/soccer/setup.sh);
  - the "Trained Yolo v5" best.pt of github.com/abdullahtarek/football_analysis (the Google Drive
    link under "Trained Models" in its README; classes ball, goalkeeper, player, referee);
  - a generic COCO model such as yolov8x.pt (class "sports ball"), weakest on a small ball.
All three are public, so the method stays one the organisers can access (RULES.md).
"""

import argparse
import sys

import numpy as np

from . import config, features, io_atd, submission
from .video_ball import MAX_CANDIDATES, NEAR_PLAYER, detect, frame_candidates, select_ball

sys.path.insert(0, config.ROOT)
from games import GAME, half_video_blocks, video_path  # noqa: E402
from starter.load_homography import load as load_homography  # noqa: E402

# Params
FPS = 25
GAP_BINS = (25, 100, 250)          # frames: < 1 s, 1-4 s, 4-10 s, >= 10 s
GAP_BIN_NAMES = ("< 1 s", "1-4 s", "4-10 s", ">= 10 s")
CONF_LEVELS = (0.1, 0.25, 0.5)
DECIDE_CONF = 0.25                 # confidence used for the go / stop line
DECIDE_ERROR = 2.0                 # metres: calibration median error must be below this
DECIDE_FILL = 0.30                 # share of live gap frames that must get an on-pitch ball
RULES = ("top", "on-pitch", "near-player")
QUANTILES = (10, 25, 50, 75, 90)


def q(values, fmt="{:.1f}"):
    values = np.asarray(values, dtype=float)
    values = values[~np.isnan(values)]
    if len(values) == 0:
        return "n/a"
    return "  ".join(("p{} " + fmt).format(p, v)
                     for p, v in zip(QUANTILES, np.percentile(values, QUANTILES)))


def share(flags):
    flags = np.asarray(flags, dtype=bool)
    return "{:.1%}".format(flags.mean()) if len(flags) else "n/a"


def gap_runs(missing):
    """[(start, end)] of each run of True, end exclusive."""
    edges = np.diff(np.concatenate(([0], np.asarray(missing).astype(np.int8), [0])))
    return list(zip(np.flatnonzero(edges == 1), np.flatnonzero(edges == -1)))


def gap_length_bins(missing, bins=GAP_BINS):
    """Per frame: index of the length bin of the gap it sits in, -1 outside gaps."""
    out = np.full(len(missing), -1, dtype=np.int8)
    for a, b in gap_runs(missing):
        out[a:b] = np.searchsorted(bins, b - a, side="right")
    return out


def sample_frames(mask, n, seed=0):
    """Up to n frame indices drawn without replacement where mask is True, sorted."""
    candidates = np.flatnonzero(mask)
    if len(candidates) <= n:
        return candidates
    return np.sort(np.random.default_rng(seed).choice(candidates, n, replace=False))


def part_a(missing, graded, live_labels, live_pace, bins):
    print("\n== ball gaps on graded frames ({} frames) ==".format(int(graded.sum())))
    g = graded
    gaps = [(a, b) for a, b in gap_runs(missing & g)]
    print("  missing: {} frames ({:.1%}) in {} gaps".format(
        int((missing & g).sum()), (missing & g).sum() / max(g.sum(), 1), len(gaps)))
    print("  gap length (s)       " + q([(b - a) / FPS for a, b in gaps], "{:.1f}"))
    print("\n  share of missing frames in live play, by the labels and by the players' pace")
    print("  (the labels are partly circular: the model reads a long gap as a stoppage)")
    print("  {:<9} {:>9} {:>11} {:>12} {:>12}".format(
        "gap", "frames", "of missing", "by labels", "by pace"))
    for k, name in enumerate(GAP_BIN_NAMES):
        sel = missing & g & (bins == k)
        print("  {:<9} {:>9} {:>11} {:>12} {:>12}".format(
            name, int(sel.sum()), "{:.1%}".format(sel.sum() / max((missing & g).sum(), 1)),
            share(live_labels[sel]) if live_labels is not None else "n/a", share(live_pace[sel])))
    sel = missing & g
    print("  {:<9} {:>9} {:>11} {:>12} {:>12}".format(
        "all", int(sel.sum()), "100.0%",
        share(live_labels[sel]) if live_labels is not None else "n/a", share(live_pace[sel])))


def choose(found, frames, matrices, players, pitch, rule, min_conf):
    """(xy, nearest player distance, all candidates' xy) per frame; NaN where the rule picks none."""
    xy = np.full((len(frames), 2), np.nan)
    nearest = np.full(len(frames), np.nan)
    every = []
    for i, frame in enumerate(frames):
        cxy, conf, inside, near = frame_candidates(found.get(frame), matrices[frame],
                                                   players[frame], pitch)
        every.append(cxy)
        j = select_ball(conf, inside, near <= NEAR_PLAYER, rule, min_conf)
        if j >= 0:
            xy[i], nearest[i] = cxy[j], near[j]
    return xy, nearest, every


def part_b(args, data, missing, graded, live, dead, running, bins, labels):
    ball, players, team, pitch = data["ball"], data["players"], data["team"], data["pitch"]
    matrices, _, valid = load_homography(GAME)
    calibration = sample_frames(~missing & graded & valid, args.n, args.seed)
    gap = sample_frames(missing & graded & live & valid, args.n, args.seed + 1)
    dead_gap = sample_frames(missing & graded & dead & valid, args.n, args.seed + 2)
    frames = np.union1d(np.union1d(calibration, gap), dead_gap)
    print("\n== video ball detector on {} frames ({} calibration, {} live gap, {} dead gap) ==".format(
        len(frames), len(calibration), len(gap), len(dead_gap)))
    found, speed = detect(args.weights, video_path(GAME), frames, args.imgsz, min(CONF_LEVELS),
                          args.device)

    print("\n  calibration (tracking has the ball), conf >= {:.2f}:".format(DECIDE_CONF))
    results = {}
    for rule in RULES:
        cal_xy, _, every = choose(found, calibration, matrices, players, pitch, rule, DECIDE_CONF)
        error = np.hypot(*(cal_xy - ball[calibration]).T)
        hit = ~np.isnan(error)
        e = error[hit]
        print("    {:<12} detected {:>6}   error (m) {}   within 1/2/5 m {}/{}/{}".format(
            rule, share(hit), q(e), share(e < 1), share(e < 2), share(e < 5)))
        results[rule] = [float(np.median(e)) if len(e) else float("inf")]
    best = [len(c) > 0 and np.hypot(*(c - ball[f]).T).min() < 2.0
            for c, f in zip(every, calibration)]
    print("    any of the up to {} candidates within 2 m of the tracked ball: {}".format(
        MAX_CANDIDATES, share(best)))

    print("\n  live gap frames (tracking lost the ball), conf >= {:.2f}:".format(DECIDE_CONF))
    for rule in RULES[1:]:
        gap_xy, nearest, _ = choose(found, gap, matrices, players, pitch, rule, DECIDE_CONF)
        use = ~np.isnan(gap_xy[:, 0])
        by_bin = "  ".join("{} {}".format(name, share(use[bins[gap] == k]))
                           for k, name in enumerate(GAP_BIN_NAMES) if (bins[gap] == k).any())
        print("    {:<12} found {:>6}   by gap length: {}".format(rule, share(use), by_bin))
        results[rule].append(float(use.mean()) if len(use) else 0.0)
        if use.any():
            d_home, d_away = features.team_distances(gap_xy[use], players[gap[use]], team)
            close = np.fmin(d_home, d_away) < NEAR_PLAYER
            line = "      nearest player (m) {}   within {:.0f} m {}".format(
                q(nearest[use]), NEAR_PLAYER, share(close))
            if labels is not None:
                near_team = np.where(d_home <= d_away, "H", "A")
                line += "   team agrees with labels {}".format(
                    share(near_team[close] == labels[gap[use]][close]))
            print(line)
    results["top"].append(results["on-pitch"][1])

    print("\n  dead gap frames (tracking lost the ball, labelled dead), conf >= {:.2f}:".format(
        DECIDE_CONF))
    for rule in RULES[1:]:
        dead_xy, _, _ = choose(found, dead_gap, matrices, players, pitch, rule, DECIDE_CONF)
        use = ~np.isnan(dead_xy[:, 0])
        print("    {:<12} found {:>6}   of which the players are running {}   "
              "(running on all sampled dead gap frames {})".format(
                  rule, share(use), share(running[dead_gap][use]), share(running[dead_gap])))

    todo = int((missing & graded).sum())
    print("\n  speed: {:.1f} frames/s -> all {} missing graded frames in about {:.0f} min".format(
        speed, todo, todo / max(speed, 1e-9) / 60))

    print()
    for rule in RULES:
        median_error, fill = results[rule]
        go = median_error < DECIDE_ERROR and fill >= DECIDE_FILL
        print("  DECISION {:<12} calibration median error {:.2f} m (need < {:.0f}), live gaps "
              "filled {:.1%} (need >= {:.0%}) -> {}".format(
                  rule, median_error, DECIDE_ERROR, fill, DECIDE_FILL,
                  "GO: build the detector in" if go else "STOP"))


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--labels", help="a submission CSV (e.g. the submitted La_Messia.csv) to say which "
                        "frames are live; without it the players' pace decides")
    parser.add_argument("--weights", help="YOLO weights; without it only part A runs")
    parser.add_argument("--n", type=int, default=2000, help="frames per sample (default 2000)")
    parser.add_argument("--imgsz", type=int, default=1280, help="YOLO input size (default 1280)")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--device", default="auto", help="auto (GPU if torch sees one), cpu, "
                        "or a CUDA index such as 0")
    args = parser.parse_args(argv)

    data = io_atd.load(GAME)
    ball = features.interpolate_ball(data["ball"])
    n = len(ball)
    graded = np.zeros(n, dtype=bool)
    for _, first, last in half_video_blocks(GAME):
        graded[first:last + 1] = True
    missing = np.isnan(ball[:, 0])
    bins = gap_length_bins(missing)

    with np.errstate(invalid="ignore"):
        running = features.pace_ratio(data["players"]) > config.DEAD_SPEED_RATIO
    labels = live_labels = None
    if args.labels:
        labels = submission.read(args.labels)
        if len(labels) != n:
            raise SystemExit("{}: {} rows, expected {}".format(args.labels, len(labels), n))
        live_labels = np.isin(labels, ("H", "A"))
        live, dead = live_labels, labels == "D"
    else:
        live, dead = running, ~running

    part_a(missing, graded, live_labels, running, bins)
    if args.weights:
        part_b(args, data, missing, graded, live, dead, running, bins, labels)

if __name__ == "__main__":
    main()
