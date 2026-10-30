"""Which possessions does the state model merge? Segmentation diagnostics on the tuning games.

    python -m tuning.segmentation                  # Metrica's three games
    python -m tuning.segmentation --games all      # and SkillCorner's ten
    python -m tuning.segmentation --sweep SWITCH_COST=2,3,4,5.12,7

BAR-ATH's public window has 17 possessions and the model predicts 10: it undersegments. spellF1
alone does not say how. This script breaks the errors down on the tuning games (never BAR-ATH),
each degraded as in tuning/search.py, with possession/config.py as it is:

  - **true / pred**: possessions (maximal H/A runs, after the scorer's X fill) in truth and output.
  - **missed**: true possessions that no predicted possession of the same team overlaps at all.
  - **interrupt missed**: short true possessions (< SHORT seconds) between two of the other team
    (H -> A -> H with nothing between), where the output keeps the outer team on most of the frames.
    This is the merge the Viterbi switch cost causes directly.
  - **pause skipped**: short true dead spells (< SHORT seconds) between two possessions of the same
    team (H -> D -> H), where the output has no D frame. The two possessions become one.
  - **spurious**: predicted changes of label with no true change within TOLERANCE seconds. The
    control: lowering the switch cost buys recall with these.
  - **start / end**: median absolute error, in seconds, of the start and end of each true
    possession against its best-overlapping predicted possession of the same team.

With --sweep, one decode parameter is varied and the summary is printed for each value, so the
trade-off between merged and spurious possessions is visible. Selection is still by spellF1.
"""

import argparse

import numpy as np

from possession import methods
from . import metrics, search

# Params
FPS = 25
SHORT = 5.0       # seconds; a "short" interruption or pause
TOLERANCE = 2.0   # seconds; a predicted change this close to a true one is not spurious


def runs_of(labels):
    """[(start, stop, letter)] maximal runs of one letter."""
    breaks = np.flatnonzero(labels[1:] != labels[:-1]) + 1
    bounds = np.concatenate(([0], breaks, [len(labels)]))
    return [(int(a), int(b), labels[a]) for a, b in zip(bounds[:-1], bounds[1:])]


def changes(labels, runs):
    """Frames where the label changes, inside graded runs."""
    out = []
    for start, stop in runs:
        window = labels[start:stop]
        out.extend(start + np.flatnonzero(window[1:] != window[:-1]) + 1)
    return np.array(sorted(out), dtype=int)


def breakdown(gt, pred):
    """The counts described in the module docstring for one game."""
    graded = metrics.graded_runs(gt)
    filled = metrics.fill(pred, graded)
    truth = metrics.spells(gt, graded)
    guess = metrics.spells(filled, graded)
    short = int(SHORT * FPS)
    tol = int(TOLERANCE * FPS)

    missed, starts, ends = 0, [], []
    for s, e, k in truth:
        best, best_iou = None, 0.0
        for ps, pe, pk in guess:
            if pk != k or pe <= s or ps >= e:
                continue
            iou = (min(e, pe) - max(s, ps)) / float(max(e, pe) - min(s, ps))
            if iou > best_iou:
                best, best_iou = (ps, pe), iou
        if best is None:
            missed += 1
        else:
            starts.append(abs(best[0] - s) / FPS)
            ends.append(abs(best[1] - e) / FPS)

    interrupts = interrupts_missed = pauses = pauses_skipped = 0
    for start, stop in graded:
        rows = runs_of(gt[start:stop])
        for (_, _, before), (a, b, mid), (_, _, after) in zip(rows, rows[1:], rows[2:]):
            if b - a >= short or before != after or before not in ("H", "A"):
                continue
            seg = filled[start + a:start + b]
            if mid in ("H", "A") and mid != before:
                interrupts += 1
                interrupts_missed += np.mean(seg == before) > 0.5
            elif mid == "D":
                pauses += 1
                pauses_skipped += not np.any(seg == "D")

    true_changes = changes(gt, graded)
    pred_changes = changes(filled, graded)
    if len(true_changes):
        offset = pred_changes[:, None] - true_changes[None, :]
        nearest = offset[np.arange(len(pred_changes)), np.abs(offset).argmin(axis=1)]
        near = np.abs(nearest) <= tol
        latency = float(np.median(nearest[near]) / FPS) if near.any() else np.nan
    else:
        near = np.zeros(len(pred_changes), dtype=bool)
        latency = np.nan
    return {
        "spellF1": metrics.spell_f1(gt, pred)[0],
        "true": len(truth), "pred": len(guess), "missed": missed,
        "interrupts": interrupts, "interrupt_missed": int(interrupts_missed),
        "pauses": pauses, "pause_skipped": int(pauses_skipped),
        "pred_changes": len(pred_changes), "spurious": int((~near).sum()), "latency": latency,
        "start": float(np.median(starts)) if starts else np.nan,
        "end": float(np.median(ends)) if ends else np.nan,
    }


HEADER = "{:<11} {:>7} {:>5} {:>5} {:>6} {:>11} {:>11} {:>12} {:>6} {:>6}".format(
    "game", "spellF1", "true", "pred", "missed", "interrupt", "pause", "spurious/chg",
    "start", "end")


def row(name, r):
    return "{:<11} {:7.3f} {:5.0f} {:5.0f} {:6.0f} {:>5.0f}/{:<5.0f} {:>5.0f}/{:<5.0f} " \
           "{:>5.0f}/{:<6.0f} {:6.1f} {:6.1f}".format(
               name, r["spellF1"], r["true"], r["pred"], r["missed"], r["interrupt_missed"],
               r["interrupts"], r["pause_skipped"], r["pauses"], r["spurious"],
               r["pred_changes"], r["start"], r["end"])


def evaluate(scores, cases, params):
    """{game: breakdown averaged over the degradation seeds} for one set of decode parameters."""
    per_game = {}
    with search.Patched(params):
        for raw, (name, _, _, halves, gt) in zip(scores, cases):
            pred = methods.decode(methods.weigh(raw), halves)
            per_game.setdefault(name, []).append(breakdown(gt, pred))
    return {name: {k: float(np.mean([r[k] for r in rows])) for k in rows[0]}
            for name, rows in per_game.items()}


def summary(result):
    keys = result[next(iter(result))].keys()
    return {k: float(np.nanmean([r[k] for r in result.values()]))
            if k in ("spellF1", "start", "end", "latency")
            else float(np.sum([r[k] for r in result.values()])) for k in keys}


def main():
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--games", choices=("metrica", "all"), default="metrica")
    parser.add_argument("--sweep", help="NAME=v1,v2,... one parameter to vary (decode or evidence)")
    args = parser.parse_args()

    cases = search.degraded_set(search.load_games(args.games))
    scores = search.raw_scores("state_model", {}, cases)
    print("columns: interrupt = short H->A->H missed / total, pause = short H->D->H skipped / "
          "total,\n         spurious/chg = predicted changes with no true change within "
          "{:.0f} s / all predicted changes,\n         start, end = median |error| in seconds. "
          "Mean over {} degradations.".format(TOLERANCE, len(search.SCENARIOS)))

    result = evaluate(scores, cases, {})
    print("\n" + HEADER)
    for name in sorted(result):
        print(row(name, result[name]))
    print(row("total", summary(result)))

    if args.sweep:
        key, values = args.sweep.split("=")
        print("\nsweep {} (spellF1 is the mean over games, the counts are totals)".format(key))
        print(HEADER.replace("game       ", "{:<11}".format(key)))
        evidence = key not in search.DECODE_SPACE
        for value in values.split(","):
            params = {key: float(value)}
            res = evaluate(search.raw_scores("state_model", params, cases) if evidence else scores,
                           cases, params)
            print(row(value, summary(res)), flush=True)


if __name__ == "__main__":
    main()
