"""Ball positions from the tactical video, for the frames where the tracking has lost the ball.

    python -m possession.video_ball --weights football-ball-detection.pt     # once, about 1-2 h

The automatic tracking has no ball on 40.7% of graded frames. A ball detector (any public YOLO
model with a ball class; docs/TECHNICAL.md, sections 12 and 13) is run on exactly those frames of
the tactical video, and the boxes are cached in data/cache, which is gitignored like all derived
match data. Every candidate box is kept (up to MAX_CANDIDATES per frame), so the selection below
can change without running the detector again. The run saves its progress every SAVE_EVERY frames
and resumes where it stopped.

fill() turns the cache into ball positions in metres, through the tactical homography (valid on
every frame): on each frame the most confident box on the pitch above VIDEO_BALL_CONF, then a
temporal check that drops a point more than VIDEO_BALL_JUMP metres from the median of its
neighbours, because a wrong box (a spare ball, a pitch marking) jumps while the real ball moves
smoothly. Only frames without a tracked ball are filled. methods.frame_scores uses the result when
config.VIDEO_BALL is on.

Only counts are printed; no frame, position or image leaves the machine.
"""

import argparse
import hashlib
import os
import sys
import time

import numpy as np

from . import config, features, io_atd

sys.path.insert(0, config.ROOT)
from games import GAME, half_video_blocks, video_path  # noqa: E402
from starter.load_homography import load as load_homography, unproject  # noqa: E402

# Params
ON_PITCH_MARGIN = 2.0              # metres beyond the lines still counted as on the pitch
SEEK_AHEAD = 50                    # frames: closer than this, read forward instead of seeking
PROGRESS = 1000                    # frames between progress lines
SAVE_EVERY = 5000                  # frames between saves of the cache, so a run can resume
MAX_CANDIDATES = 5                 # ball boxes kept per frame
NEAR_PLAYER = 2.0                  # metres: a ball this close to a player is at someone's feet
DETECT_CONF = 0.1                  # boxes below this are not even stored
IMGSZ = 1280                       # YOLO input size: the ball is a few pixels wide in a wide shot
CACHE_VERSION = 1


def pixels_to_metres(matrix, pixel_x, pixel_y, pitch):
    """Image pixels -> pitch metres, through the frame's homography (which maps normalised 0-1
    pitch coordinates to pixels) and the pitch size, the same scale io_atd uses."""
    x, y = unproject(matrix, pixel_x, pixel_y)
    return x * float(pitch[0]), y * float(pitch[1])


def ball_candidates(xyxy, conf, cls, ball_ids, k=MAX_CANDIDATES):
    """(m, 3) array of [centre_x, centre_y, confidence] for up to k ball boxes, most confident first."""
    xyxy, conf, cls = np.asarray(xyxy, float).reshape(-1, 4), np.asarray(conf, float), np.asarray(cls)
    keep = np.flatnonzero(np.isin(cls, list(ball_ids)))
    keep = keep[np.argsort(-conf[keep], kind="stable")][:k]
    return np.column_stack([(xyxy[keep, 0] + xyxy[keep, 2]) / 2.0,
                            (xyxy[keep, 1] + xyxy[keep, 3]) / 2.0, conf[keep]])


def pick_ball(xyxy, conf, cls, ball_ids):
    """(centre_x, centre_y, confidence) of the most confident ball box, or None if there is none."""
    candidates = ball_candidates(xyxy, conf, cls, ball_ids, k=1)
    return tuple(candidates[0]) if len(candidates) else None


def select_ball(conf, inside, near, rule, min_conf=0.0):
    """Index of the candidate a rule picks, or -1. Candidates are sorted by confidence.

      top          the most confident box, wherever it is
      on-pitch     the most confident box on the pitch
      near-player  the most confident box on the pitch with a player within NEAR_PLAYER metres
    """
    ok = np.asarray(conf) >= min_conf
    if rule == "on-pitch":
        ok &= np.asarray(inside, dtype=bool)
    elif rule == "near-player":
        ok &= np.asarray(inside, dtype=bool) & np.asarray(near, dtype=bool)
    elif rule != "top":
        raise ValueError(rule)
    hits = np.flatnonzero(ok)
    if rule == "top":
        return 0 if len(ok) and ok[0] else -1
    return int(hits[0]) if len(hits) else -1


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
    """{frame: ball_candidates array} for every frame read, and frames/s.

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
        boxes = model.predict(image, imgsz=imgsz, conf=conf, device=device, verbose=False)[0].boxes
        found[frame] = ball_candidates(boxes.xyxy.cpu().numpy(), boxes.conf.cpu().numpy(),
                                       boxes.cls.cpu().numpy(), ball_ids)
        tick = time.time()
        predicting += tick - now
        done += 1
        if done % PROGRESS == 0 or done == total:
            rate = done / max(tick - start, 1e-9)
            print("  {}/{} frames, {:.1f} frames/s, about {:.1f} min left".format(
                done, total, rate, (total - done) / rate / 60), flush=True)
    print("  time: reading the video {:.0f} s, running the model {:.0f} s".format(reading, predicting))
    return found, total / max(time.time() - start, 1e-9)


def frame_candidates(candidates, matrix, frame_players, pitch):
    """Candidates of one frame on the pitch: (xy metres, conf, inside, nearest player distance)."""
    if candidates is None or len(candidates) == 0:
        empty = np.zeros(0)
        return np.zeros((0, 2)), empty, empty.astype(bool), empty
    x, y = pixels_to_metres(matrix, candidates[:, 0], candidates[:, 1], pitch)
    xy = np.column_stack([x, y])
    seen = ~np.isnan(frame_players[:, 0])
    if seen.any():
        d = np.hypot(xy[:, None, 0] - frame_players[None, seen, 0],
                     xy[:, None, 1] - frame_players[None, seen, 1])
        nearest = d.min(axis=1)
    else:
        nearest = np.full(len(xy), np.inf)
    return xy, candidates[:, 2], on_pitch(x, y, pitch), nearest


def cache_path(game=GAME):
    return os.path.join(config.CACHE_DIR, "{}_video_ball_v{}.npz".format(game, CACHE_VERSION))


def frames_to_detect(ball, game=GAME):
    """Graded frames on which the tracking has no ball (after the short-gap bridge)."""
    graded = np.zeros(len(ball), dtype=bool)
    for _, first, last in half_video_blocks(game):
        graded[first:last + 1] = True
    return np.flatnonzero(graded & np.isnan(features.interpolate_ball(ball)[:, 0]))


def save(path, found):
    frames = np.array(sorted(found), dtype=np.int64)
    candidates = np.full((len(frames), MAX_CANDIDATES, 3), np.nan)
    for i, frame in enumerate(frames):
        c = found[frame][:MAX_CANDIDATES]
        candidates[i, :len(c)] = c
    os.makedirs(os.path.dirname(path), exist_ok=True)
    tmp = path + ".tmp.npz"
    np.savez(tmp, frames=frames, candidates=candidates)
    os.replace(tmp, path)


def content_hash(found):
    """sha256 of the cached detections themselves (not of the file), to compare two runs."""
    digest = hashlib.sha256()
    for frame in sorted(found):
        digest.update(np.int64(frame).tobytes())
        digest.update(np.round(np.asarray(found[frame], dtype=np.float64), 3).tobytes())
    return digest.hexdigest()


def load_cache(path):
    """{frame: (m, 3) candidates} from a cache file, empty if there is none."""
    if not os.path.isfile(path):
        return {}
    with np.load(path) as stored:
        frames, candidates = stored["frames"], stored["candidates"]
    return {int(f): c[~np.isnan(c[:, 2])] for f, c in zip(frames, candidates)}


def extract(weights, game=GAME, imgsz=IMGSZ, device="auto"):
    """Run the detector on every frame still missing from the cache, saving as it goes."""
    data = io_atd.load(game)
    path = cache_path(game)
    found = load_cache(path)
    todo = np.array([f for f in frames_to_detect(data["ball"], game) if f not in found])
    print("  {} frames to detect, {} already in the cache".format(len(todo), len(found)))
    for start in range(0, len(todo), SAVE_EVERY):
        chunk = todo[start:start + SAVE_EVERY]
        new, _ = detect(weights, video_path(game), chunk, imgsz, DETECT_CONF, device)
        found.update(new)
        save(path, found)
        print("  saved {} / {} frames".format(min(start + SAVE_EVERY, len(todo)), len(todo)),
              flush=True)
    print("  detections sha256 {}".format(content_hash(found)))
    return path


def fill(ball, players, pitch, matrices, found, conf=None, jump=None, smooth=None):
    """(ball with video positions on its missing frames, mask of the filled frames)."""
    conf = config.VIDEO_BALL_CONF if conf is None else conf
    jump = config.VIDEO_BALL_JUMP if jump is None else jump
    smooth = config.VIDEO_BALL_SMOOTH if smooth is None else smooth
    out = ball.copy()
    video = np.full_like(ball, np.nan)
    for frame, candidates in found.items():
        if not np.isnan(ball[frame, 0]):
            continue
        xy, score, inside, near = frame_candidates(candidates, matrices[frame], players[frame], pitch)
        j = select_ball(score, inside, near <= NEAR_PLAYER, "on-pitch", conf)
        if j >= 0:
            video[frame] = xy[j]
    # A point far from the median of its neighbours (+/- smooth frames, itself excluded) is a box
    # on something else. Isolated points with no neighbour at all are dropped too.
    keep = np.zeros(len(ball), dtype=bool)
    has = np.flatnonzero(~np.isnan(video[:, 0]))
    for frame in has:
        a, b = max(frame - smooth, 0), frame + smooth + 1
        window = np.delete(video[a:b], frame - a, axis=0)
        window = window[~np.isnan(window[:, 0])]
        if len(window) and np.hypot(*(video[frame] - np.median(window, axis=0))) <= jump:
            keep[frame] = True
    out[keep] = video[keep]
    return out, keep


def load_filled(data, game=GAME):
    """fill() on the cached detections, for methods.frame_scores. Fails loudly without a cache."""
    path = cache_path(game)
    if not os.path.isfile(path):
        raise SystemExit("no video ball cache at {}\nrun first:  python -m possession.video_ball "
                         "--weights football-ball-detection.pt".format(path))
    matrices, _, _ = load_homography(game)
    found = load_cache(path)
    ball, filled = fill(features.interpolate_ball(data["ball"]), data["players"], data["pitch"],
                        matrices, found)
    print("  video ball: {} frames filled (detections sha256 {})".format(
        int(filled.sum()), content_hash(found)[:12]))
    return ball, filled


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--weights", required=True, help="YOLO weights with a ball class")
    parser.add_argument("--imgsz", type=int, default=IMGSZ)
    parser.add_argument("--device", default="auto", help="auto (GPU if torch sees one), cpu, "
                        "or a CUDA index such as 0")
    args = parser.parse_args(argv)
    path = extract(args.weights, GAME, args.imgsz, args.device)
    print("wrote {}".format(os.path.relpath(path, config.ROOT)))


if __name__ == "__main__":
    main()
