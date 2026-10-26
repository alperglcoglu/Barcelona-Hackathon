"""Aggregate diagnostics: check that each signal means what we think it means.

    python -m possession.diagnose

Prints summary statistics only: quantiles, rates and counts, never frame-level values. That is
what makes the output safe to paste into a discussion without sharing any extract of the match
data.

Two kinds of numbers come out:

  1. **Signal sanity on the public 5-minute window.** For example, the mean player speed on frames
     the ground truth calls D versus alive, or how often the team evidence points the same way as
     the ground truth. CHALLENGE.md says to use the public window to understand what the labels
     mean, and that is all this does. Nothing here feeds back into config.py automatically, and the
     rules forbid choosing parameters on this window (RULES.md section 2), so a number that looks
     wrong here is a reason to re-read a feature, not to retune a constant until it looks right.
  2. **Full-match shape of a submission.** How many possessions it reports, how long they are, and
     the share of dead ball. These need no ground truth.
"""

import os
import sys

import numpy as np

from . import config, features, io_atd, io_tagging, methods

sys.path.insert(0, config.ROOT)
from games import GAME, half_video_blocks  # noqa: E402

# Params
PUBLIC_GT = os.path.join(config.ROOT, "evaluation", "gt", "BAR-ATH_possession_gt_public.csv")
QUANTILES = (10, 25, 50, 75, 90)
FPS = 25


def read_gt(path=PUBLIC_GT):
    with open(path) as handle:
        next(handle)
        return np.array([line.rstrip("\r\n").split(",")[1].strip() for line in handle])


def quantiles(values):
    values = values[~np.isnan(values)]
    if len(values) == 0:
        return "n/a"
    return "  ".join("p{} {:.2f}".format(q, v) for q, v in zip(QUANTILES,
                                                              np.percentile(values, QUANTILES)))


def runs(labels):
    """[(label, start, end_exclusive)] of maximal runs."""
    change = np.flatnonzero(labels[1:] != labels[:-1]) + 1
    starts = np.r_[0, change]
    ends = np.r_[change, len(labels)]
    return [(labels[s], s, e) for s, e in zip(starts, ends)]


def shape(labels):
    """Possessions, their mean and median length in seconds, and the dead-ball share. Graded halves only."""
    spells, graded = [], []
    for _, first, last in half_video_blocks(GAME):
        half = labels[first:last + 1]
        graded.append(half)
        spells += [(e - s) / FPS for lab, s, e in runs(half) if lab in ("H", "A")]
    graded = np.concatenate(graded)
    dead_share = float(np.mean(graded == "D")) if len(graded) else 0.0
    if not spells:
        return 0, 0.0, 0.0, dead_share
    return len(spells), float(np.mean(spells)), float(np.median(spells)), dead_share


def main():
    data = io_atd.load(GAME)
    gt = read_gt()
    window = gt != "X"
    alive = np.isin(gt, ("H", "A"))

    ball = features.interpolate_ball(data["ball"])
    speeds = data["players"][:, :, 2]
    seen = ~np.isnan(speeds)
    with np.errstate(invalid="ignore", divide="ignore"):
        mean_speed = np.where(seen, speeds, 0).sum(1) / seen.sum(1)
    team = features.team_evidence(ball, data["players"], data["team"])
    dead = features.dead_evidence(ball, data["players"], data["pitch"])

    print("== public window: {} frames, signal sanity (not for tuning) ==".format(window.sum()))
    for name, mask in (("alive", window & alive), ("dead", window & (gt == "D"))):
        print("\n[{}] {} frames".format(name, mask.sum()))
        print("  ball missing (raw)        {:.1%}".format(np.isnan(data["ball"][mask, 0]).mean()))
        print("  ball missing (after fill) {:.1%}".format(np.isnan(ball[mask, 0]).mean()))
        print("  mean player speed         " + quantiles(mean_speed[mask]))
        print("  dead evidence             " + quantiles(dead[mask]))

    # Shape of the tracking noise over the whole graded match. These calibrate the simulated
    # degradation in tuning/sample_data.degrade, which is how the parameters are made to transfer.
    graded = np.zeros(len(gt), dtype=bool)
    for _, first, last in half_video_blocks(GAME):
        graded[first:last + 1] = True
    missing = np.isnan(data["ball"][:, 0]) & graded
    gap_lengths = [e - s for lab, s, e in runs(missing.astype(int)) if lab == 1]
    print("\n== whole graded match: tracking noise (calibrates tuning/sample_data.degrade) ==")
    print("  ball missing on {:.1%} of graded frames, in {} gaps".format(
        missing.sum() / graded.sum(), len(gap_lengths)))
    print("  gap length (frames)       " + quantiles(np.array(gap_lengths, dtype=float)))
    print("  mean gap {:.1f} frames, share of missing frames in gaps > 2 s: {:.1%}".format(
        np.mean(gap_lengths), sum(g for g in gap_lengths if g > 50) / max(1, missing.sum())))
    print("  players tracked per frame " + quantiles(seen[graded].sum(1).astype(float)))

    informative = window & alive & (np.abs(team) > 0.3)
    agree = np.sign(team[informative]) == np.where(gt[informative] == "H", 1, -1)
    print("\nteam evidence |e| > 0.3 on {:.1%} of alive frames, agrees with GT on {:.1%}".format(
        informative.sum() / max(1, (window & alive).sum()), agree.mean() if agree.size else 0))

    try:
        events = io_tagging.load(GAME)
    except IOError:
        events = []
    restarts = [s for lab, s, e in runs(np.where(window, gt, "X"))
                if lab in ("H", "A") and s > 0 and gt[s - 1] == "D"]
    set_pieces = [ev["start_frame"] for ev in events if ev["code"] == "SET PIECES"]
    if restarts and set_pieces:
        offsets = [min(set_pieces, key=lambda f: abs(f - r)) - r for r in restarts]
        print("\nSET PIECES tag start minus true restart, seconds: "
              + ", ".join("{:+.1f}".format(o / FPS) for o in sorted(offsets)))

    print("\n== full match, shape of each method's output (no ground truth needed) ==")
    for name in ("baseline", "state_model", "learned"):
        method = methods.METHODS[name]
        labels = method(data, events) if name in methods.USES_TAGGING else method(data)
        n, mean_s, median_s, dead_share = shape(labels)
        print("  {:<12} {:>5} possessions  mean {:5.1f} s  median {:5.1f} s  dead {:.1%}".format(
            name, n, mean_s, median_s, dead_share))


if __name__ == "__main__":
    main()
