"""Is the ball visible in the video where the tracking loses it? Aggregate diagnostics, no labels.

    python -m possession.probe_video_ball [--labels evaluation/results/team.csv]
    python -m possession.probe_video_ball --labels evaluation/results/team.csv --weights best.pt

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
import os
import sys
import time

import numpy as np

from . import config, features, io_atd, submission

sys.path.insert(0, config.ROOT)
from games import GAME, half_video_blocks, video_path  # noqa: E402
from starter.load_homography import load as load_homography, unproject  # noqa: E402

# Params
FPS = 25
GAP_BINS = (25, 100, 250)          # frames: < 1 s, 1-4 s, 4-10 s, >= 10 s
GAP_BIN_NAMES = ("< 1 s", "1-4 s", "4-10 s", ">= 10 s")
CONF_LEVELS = (0.1, 0.25, 0.5)
DECIDE_CONF = 0.25                 # confidence used for the go / stop line
DECIDE_ERROR = 2.0                 # metres: calibration median error must be below this
DECIDE_FILL = 0.30                 # share of live gap frames that must get an on-pitch ball
ON_PITCH_MARGIN = 2.0              # metres beyond the lines still counted as on the pitch
SEEK_AHEAD = 50                    # frames: closer than this, read forward instead of seeking
PROGRESS = 100                     # frames between progress lines
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


def pace_ratio(players):
    """Players' mean speed, averaged over +/- 1 s, divided by the match's median pace.

    The same quantity features.dead_evidence compares with DEAD_SPEED_RATIO: above the pivot the
    players are running, which is live play far more often than not.
    """
    speeds = players[:, :, 2]
    seen = ~np.isnan(speeds)
    n_seen = seen.sum(axis=1)
    with np.errstate(invalid="ignore", divide="ignore"):
        mean_speed = np.where(n_seen > 0, np.where(seen, speeds, 0.0).sum(axis=1) / n_seen, np.nan)
    mean_speed = features.nan_moving_mean(mean_speed, config.SPEED_SMOOTH)
    crowded = (n_seen >= features.TYPICAL_MIN_PLAYERS) & np.isfinite(mean_speed)
    typical = np.median(mean_speed[crowded]) if crowded.any() else np.nan
    return mean_speed / typical


def pixels_to_metres(matrix, pixel_x, pixel_y, pitch):
    """Image pixels -> pitch metres, through the frame's homography (which maps normalised 0-1
    pitch coordinates to pixels) and the pitch size, the same scale io_atd uses."""
    x, y = unproject(matrix, pixel_x, pixel_y)
    return x * float(pitch[0]), y * float(pitch[1])


def pick_ball(xyxy, conf, cls, ball_ids):
    """(centre_x, centre_y, confidence) of the most confident ball box, or None if there is none."""
    xyxy, conf, cls = np.asarray(xyxy, float).reshape(-1, 4), np.asarray(conf, float), np.asarray(cls)
    keep = np.isin(cls, list(ball_ids))
    if not keep.any():
        return None
    best = np.flatnonzero(keep)[np.argmax(conf[keep])]
    x0, y0, x1, y1 = xyxy[best]
    return (x0 + x1) / 2.0, (y0 + y1) / 2.0, float(conf[best])


def sample_frames(mask, n, seed=0):
    """Up to n frame indices drawn without replacement where mask is True, sorted."""
    candidates = np.flatnonzero(mask)
    if len(candidates) <= n:
        return candidates
    return np.sort(np.random.default_rng(seed).choice(candidates, n, replace=False))


def on_pitch(x, y, pitch, margin=ON_PITCH_MARGIN):
    return (x > -margin) & (x < pitch[0] + margin) & (y > -margin) & (y < pitch[1] + margin)


def read_frames(path, frames):
    """Yield (frame, image) for sorted frame indices, reading forward over short distances."""
    import cv2
    capture = cv2.VideoCapture(path)
    if not capture.isOpened():
        raise IOError("cannot open {}".format(path))
    position = None
    try:
        for frame in frames:
            if position is None or not 0 <= frame - position <= SEEK_AHEAD:
                capture.set(cv2.CAP_PROP_POS_FRAMES, int(frame))
                position = int(frame)
            while position < frame:
                capture.grab()
                position += 1
            ok, image = capture.read()
            position += 1
            if ok:
                yield int(frame), image
    finally:
        capture.release()


def pick_device(requested, cuda_available):
    """"0" (the first GPU) or "cpu" for --device auto; anything else is passed through."""
    if requested != "auto":
        return requested
    return "0" if cuda_available else "cpu"


def report_device(requested):
    """Choose the device and say which one runs, so a CPU-only torch install is noticed at once."""
    import torch
    cuda = torch.cuda.is_available()
    device = pick_device(requested, cuda)
    gpu = torch.cuda.get_device_name(0) if cuda else "none"
    print("  torch {}, CUDA available: {}, GPU: {}, running on: {}".format(
        torch.__version__, cuda, gpu, device))
    if device == "cpu":
        print("  WARNING: running on the CPU, which is slow. With an NVIDIA GPU, install the CUDA "
              "build of torch:\n    pip uninstall -y torch torchvision\n    pip install torch "
              "torchvision --index-url https://download.pytorch.org/whl/cu124")
    return device


def detect(weights, path, frames, imgsz, conf, device="auto"):
    """{frame: (centre_x, centre_y, confidence)} for frames with a ball box, and frames/s.

    Prints a progress line every PROGRESS frames (counts only) and, at the end, how the time splits
    between reading the video and running the model.
    """
    from ultralytics import YOLO
    device = report_device(device)
    model = YOLO(weights)
    ball_ids = [i for i, name in model.names.items() if "ball" in str(name).lower()]
    if not ball_ids:
        raise SystemExit("the model has no class named like 'ball': {}".format(model.names))
    print("  ball classes: {}".format([model.names[i] for i in ball_ids]))
    found, start = {}, time.time()
    reading = predicting = 0.0
    total, done = len(frames), 0
    tick = time.time()
    for frame, image in read_frames(path, frames):
        now = time.time()
        reading += now - tick
        boxes = model.predict(image, imgsz=imgsz, conf=conf, device=device,
                              half=device != "cpu", verbose=False)[0].boxes
        hit = pick_ball(boxes.xyxy.cpu().numpy(), boxes.conf.cpu().numpy(),
                        boxes.cls.cpu().numpy(), ball_ids)
        if hit is not None:
            found[frame] = hit
        tick = time.time()
        predicting += tick - now
        done += 1
        if done % PROGRESS == 0 or done == total:
            rate = done / max(tick - start, 1e-9)
            print("  {}/{} frames, {:.1f} frames/s, about {:.1f} min left".format(
                done, total, rate, (total - done) / rate / 60), flush=True)
    print("  time: reading the video {:.0f} s, running the model {:.0f} s".format(reading, predicting))
    return found, total / max(time.time() - start, 1e-9)


def part_a(missing, graded, live, bins, source):
    print("\n== ball gaps on graded frames ({} frames) ==".format(int(graded.sum())))
    g = graded
    gaps = [(a, b) for a, b in gap_runs(missing & g)]
    print("  missing: {} frames ({:.1%}) in {} gaps".format(
        int((missing & g).sum()), (missing & g).mean() / max(g.mean(), 1e-9), len(gaps)))
    print("  gap length (s)       " + q([(b - a) / FPS for a, b in gaps], "{:.1f}"))
    print("\n  {:<9} {:>9} {:>11} {:>12}   (live by {})".format(
        "gap", "frames", "of missing", "live share", source))
    for k, name in enumerate(GAP_BIN_NAMES):
        sel = missing & g & (bins == k)
        print("  {:<9} {:>9} {:>11} {:>12}".format(
            name, int(sel.sum()), "{:.1%}".format(sel.sum() / max((missing & g).sum(), 1)),
            share(live[sel])))
    upper = missing & g & live
    print("  live missing frames: {} ({:.1%} of graded): the most a video detector could inform".format(
        int(upper.sum()), upper.sum() / max(g.sum(), 1)))


def part_b(args, data, missing, graded, live, bins, labels):
    ball, players, team, pitch = data["ball"], data["players"], data["team"], data["pitch"]
    matrices, _, valid = load_homography(GAME)
    calibration = sample_frames(~missing & graded & valid, args.n, args.seed)
    gap = sample_frames(missing & graded & live & valid, args.n, args.seed + 1)
    frames = np.union1d(calibration, gap)
    print("\n== video ball detector on {} frames ({} calibration, {} live gap) ==".format(
        len(frames), len(calibration), len(gap)))
    found, speed = detect(args.weights, video_path(GAME), frames, args.imgsz, min(CONF_LEVELS),
                          args.device)

    def metres(subset):
        xy = np.full((len(subset), 2), np.nan)
        score = np.zeros(len(subset))
        for i, frame in enumerate(subset):
            if frame in found:
                px, py, score[i] = found[frame]
                xy[i] = pixels_to_metres(matrices[frame], px, py, pitch)
        return xy, score

    cal_xy, cal_score = metres(calibration)
    error = np.hypot(*(cal_xy - ball[calibration]).T)
    print("\n  calibration (tracking has the ball):")
    for level in CONF_LEVELS:
        hit = cal_score >= level
        e = error[hit]
        print("    conf >= {:.2f}: detected {}   error (m) {}   within 1/2/5 m {}/{}/{}".format(
            level, share(hit), q(e), share(e < 1), share(e < 2), share(e < 5)))

    gap_xy, gap_score = metres(gap)
    inside = on_pitch(gap_xy[:, 0], gap_xy[:, 1], pitch)
    print("\n  live gap frames (tracking lost the ball):")
    for level in CONF_LEVELS:
        hit = gap_score >= level
        print("    conf >= {:.2f}: detected {}   on the pitch {}".format(
            level, share(hit), share(hit & inside)))
    use = (gap_score >= DECIDE_CONF) & inside
    print("    on-pitch at conf >= {:.2f}, by gap length:".format(DECIDE_CONF))
    for k, name in enumerate(GAP_BIN_NAMES):
        sel = bins[gap] == k
        print("      {:<9} n {:>5}   {}".format(name, int(sel.sum()), share(use[sel])))

    if use.any():
        d_home, d_away = features.team_distances(gap_xy[use], players[gap[use]], team)
        nearest = np.fmin(d_home, d_away)
        print("    nearest player to the detected ball (m): " + q(nearest))
        if labels is not None:
            near_team = np.where(d_home <= d_away, "H", "A")
            close = nearest < 2.0
            agree = near_team[close] == labels[gap[use]][close]
            print("    nearest player within 2 m: {}; their team agrees with the labels: {}".format(
                share(close), share(agree)))

    remaining = int((missing & graded & live).sum())
    print("\n  speed: {:.1f} frames/s -> all {} live gap frames in about {:.0f} min".format(
        speed, remaining, remaining / max(speed, 1e-9) / 60))

    hit = cal_score >= DECIDE_CONF
    median_error = float(np.nanmedian(error[hit])) if hit.any() else float("inf")
    fill = float(use.mean()) if len(use) else 0.0
    go = median_error < DECIDE_ERROR and fill >= DECIDE_FILL
    print("\n  DECISION: calibration median error {:.2f} m (need < {:.0f}), live gaps filled {:.1%} "
          "(need >= {:.0%}) -> {}".format(median_error, DECIDE_ERROR, fill, DECIDE_FILL,
                                          "GO: build the detector in" if go else "STOP"))


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--labels", help="a submission CSV (e.g. the frozen team.csv) to say which "
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

    labels = None
    if args.labels:
        labels = submission.read(args.labels)
        if len(labels) != n:
            raise SystemExit("{}: {} rows, expected {}".format(args.labels, len(labels), n))
        live, source = np.isin(labels, ("H", "A")), "labels"
    else:
        with np.errstate(invalid="ignore"):
            live, source = pace_ratio(data["players"]) > config.DEAD_SPEED_RATIO, "players' pace"

    part_a(missing, graded, live, bins, source)
    if args.weights:
        part_b(args, data, missing, graded, live, bins, labels)


if __name__ == "__main__":
    main()
