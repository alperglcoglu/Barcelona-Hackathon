"""spellF1 and macroF1, re-implemented for games the official scorer does not know.

evaluation/possession_metrics.py is the specification and decides the competition. It is tied to
BAR-ATH's frame count, so for the sample games we restate the same definitions here: X forward-filled
inside each graded run, then back-filled at its head; possessions as maximal H/A runs inside runs;
greedy one-to-one matching by temporal IoU, IoU-descending, ties by index; F1 averaged over IoU
0.3 to 0.7. test_metrics.py checks the two agree on the official baselines and public ground truth.
"""

import numpy as np

# Params
IOUS = (0.3, 0.4, 0.5, 0.6, 0.7)


def graded_runs(gt):
    graded = (gt != "X").astype(np.int8)
    edges = np.diff(np.concatenate(([0], graded, [0])))
    return list(zip(np.flatnonzero(edges == 1), np.flatnonzero(edges == -1)))


def fill(pred, runs):
    out = pred.copy()
    for start, stop in runs:
        window = out[start:stop]
        known = np.flatnonzero(window != "X")
        if not len(known):
            continue
        carry = np.maximum.accumulate(np.where(window != "X", np.arange(len(window)), -1))
        carry[carry < 0] = known[0]
        out[start:stop] = window[carry]
    return out


def spells(labels, runs):
    out = []
    for start, stop in runs:
        window = labels[start:stop]
        breaks = np.flatnonzero(window[1:] != window[:-1]) + 1
        bounds = np.concatenate(([0], breaks, [len(window)]))
        for a, b in zip(bounds[:-1], bounds[1:]):
            if window[a] in ("H", "A"):
                out.append((start + int(a), start + int(b), window[a]))
    return out


def spell_f1(gt, pred):
    runs = graded_runs(gt)
    truth, guess = spells(gt, runs), spells(fill(pred, runs), runs)
    # Overlaps via a sweep, not all pairs: both lists are sorted and non-overlapping in time.
    overlaps, j0 = [], 0
    for i, (gs, ge, gk) in enumerate(truth):
        while j0 < len(guess) and guess[j0][1] <= gs:
            j0 += 1
        j = j0
        while j < len(guess) and guess[j][0] < ge:
            ps, pe, pk = guess[j]
            if pk == gk:
                overlaps.append(((min(ge, pe) - max(gs, ps)) / float(max(ge, pe) - min(gs, ps)), i, j))
            j += 1
    overlaps.sort(key=lambda o: (-o[0], o[1], o[2]))
    f1s = []
    for threshold in IOUS:
        used_gt, used_pred = set(), set()
        for iou, i, j in overlaps:
            if iou < threshold:
                break
            if i in used_gt or j in used_pred:
                continue
            used_gt.add(i)
            used_pred.add(j)
        tp = len(used_gt)
        p = tp / len(guess) if guess else 0.0
        r = tp / len(truth) if truth else 0.0
        f1s.append(2 * p * r / (p + r) if p + r else 0.0)
    return float(np.mean(f1s)), len(truth), len(guess)


def macro_f1(gt, pred):
    graded = gt != "X"
    g, p = gt[graded], pred[graded]
    scores = []
    for k in ("H", "A", "D"):
        tp = np.sum((g == k) & (p == k))
        fp = np.sum((g != k) & (p == k))
        fn = np.sum((g == k) & (p != k))
        scores.append(2 * tp / (2 * tp + fp + fn) if tp else 0.0)
    return float(np.mean(scores))
