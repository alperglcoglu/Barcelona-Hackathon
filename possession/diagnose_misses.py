"""Why does a submission miss true possessions? Public-window counts, sanity only, no tuning.

    python -m possession.diagnose_misses                                   # the submitted file
    python -m possession.diagnose_misses evaluation/results/a.csv evaluation/results/b.csv

Prints counts only, so the output is safe to share.

The submitted file matches 8 of the 17 true possessions in the public window at IoU 0.5 and
predicts 10. Two lines of work could recover the others, and they fix different errors:

  - a better **team** signal (pitch control) helps where a true possession is absorbed by the
    other team, or where two possessions of the same team are merged across a short spell of the
    other team that the output never shows (a missed switch);
  - a better **dead-ball** signal helps where two possessions of the same team are merged across
    a stoppage the output does not mark (a missed pause), or where a possession is called dead.

Each true possession that is not matched at IoU 0.5 is put in one of these classes:

  absorbed      most of its frames carry the other team
  called dead   most of its frames are D
  merged        most of its frames carry its own team, inside one predicted possession that also
                covers another true possession of that team. What separated the two in the truth
                (other team only, or a D) is counted too
  boundary      most of its frames carry its own team, but the predicted possession starts or ends
                too far away for IoU 0.5, without a merge

tuning/segmentation.py's breakdown (interruptions and pauses under 5 s) is printed alongside.
The window has 17 possessions, so every count is a handful: this decides where to look next, not
any parameter.
"""

import argparse
import os
import sys

import numpy as np

from . import config, submission
from .diagnose import read_gt

sys.path.insert(0, config.ROOT)
from tuning import metrics, segmentation  # noqa: E402

# Params
MATCH_IOU = 0.5
CLASSES = ("absorbed", "called dead", "merged", "boundary")


def classify(gt, pred):
    """(counts per class, separators of merged possessions, number matched) for one file."""
    runs = metrics.graded_runs(gt)
    filled = metrics.fill(pred, runs)
    truth = metrics.spells(gt, runs)
    guess = metrics.spells(filled, runs)
    counts = dict.fromkeys(CLASSES, 0)
    separators = {"other team": 0, "dead ball": 0}
    matched = 0
    for i, (s, e, k) in enumerate(truth):
        best, best_iou = None, 0.0
        for ps, pe, pk in guess:
            if pk == k and pe > s and ps < e:
                iou = (min(e, pe) - max(s, ps)) / float(max(e, pe) - min(s, ps))
                if iou > best_iou:
                    best, best_iou = (ps, pe), iou
        if best_iou >= MATCH_IOU:
            matched += 1
            continue
        seg = filled[s:e]
        other = "A" if k == "H" else "H"
        if np.mean(seg == other) > 0.5:
            counts["absorbed"] += 1
        elif np.mean(seg == "D") > 0.5:
            counts["called dead"] += 1
        else:
            ps, pe = best if best is not None else (s, e)
            neighbours = [j for j, (ts, te, tk) in enumerate(truth)
                          if j != i and tk == k and ts < pe and te > ps]
            if neighbours:
                counts["merged"] += 1
                j = min(neighbours, key=lambda j: abs(j - i))
                a, b = sorted((i, j))
                gap = gt[truth[a][1]:truth[b][0]]
                separators["dead ball" if np.any(gap == "D") else "other team"] += 1
            else:
                counts["boundary"] += 1
    return counts, separators, matched, len(truth), len(guess)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("labels", nargs="*", help="submission CSVs (default: the submitted file)")
    args = parser.parse_args(argv)
    paths = args.labels or [os.path.join(config.RESULTS_DIR, "{}.csv".format(config.TEAM_NAME))]
    gt = read_gt()
    print("== public window: why true possessions are missed (sanity, not for tuning) ==")
    for path in paths:
        pred = submission.read(path)
        counts, separators, matched, n_true, n_pred = classify(gt, pred)
        r = segmentation.breakdown(gt, pred)
        print("\n{}".format(os.path.basename(path)))
        print("  true {}  predicted {}  matched at IoU {:.1f}: {}".format(
            n_true, n_pred, MATCH_IOU, matched))
        print("  not matched: " + "  ".join("{} {}".format(c, counts[c]) for c in CLASSES))
        print("  merged across: other team only {}  a dead ball {}".format(
            separators["other team"], separators["dead ball"]))
        print("  short (< {:.0f} s) other-team spells missed {} of {}, short pauses skipped {} of {}"
              .format(segmentation.SHORT, r["interrupt_missed"], r["interrupts"],
                      r["pause_skipped"], r["pauses"]))
        print("  predicted changes {}, spurious {}; boundary error median start {:.1f} s, "
              "end {:.1f} s".format(r["pred_changes"], r["spurious"], r["start"], r["end"]))


if __name__ == "__main__":
    main()
