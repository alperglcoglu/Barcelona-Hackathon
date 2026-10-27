"""Where does the TV cue call a live ball dead? Aggregate diagnostics for methods.state_model_tv.

    python -m possession.diagnose_tv

Prints counts, shares and quantiles only, like possession.diagnose, so the output is safe to share.

Two hypotheses are checked for the alive frames that state_model_tv calls D:

  1. **Short cutaways during play.** The director sometimes leaves the live picture for a second or
     two while play goes on (a close-up of a player, a coach). If short not-shown stretches are
     rarely stoppages, the TV evidence should depend on the stretch length.
  2. **Lag at the edges.** The director cuts away some time after the whistle and may keep a
     replay running after the restart. Then not-shown stretches are shifted against the stoppages,
     and the cue should be shifted back.

The first block needs no labels: it compares the not-shown stretches with the tracking-only
model's own D spells over the whole match. That is what any fix would be calibrated on. The
second block reads the public window as a sanity check only (RULES.md section 2).
"""

import numpy as np

from . import broadcast, diagnose, io_atd, io_tagging, methods

from games import GAME, half_video_blocks  # noqa: E402  (methods put the root on sys.path)

# Params
FPS = 25
LENGTH_BINS = (0, 3, 8, 20, 10 ** 6)   # seconds; edges of the not-shown stretch length bins
NEAR = 5 * FPS                         # frames: "near" a restart or a stoppage start
QUANTILES = (10, 25, 50, 75, 90)


def spells(flags):
    """[(start, end_exclusive)] of the True runs."""
    edges = np.diff(np.concatenate(([0], flags.astype(np.int8), [0])))
    return list(zip(np.flatnonzero(edges == 1), np.flatnonzero(edges == -1)))


def q(values):
    values = np.asarray(values, dtype=float)
    if len(values) == 0:
        return "n/a"
    return "  ".join("p{} {:+.1f}".format(p, v)
                     for p, v in zip(QUANTILES, np.percentile(values, QUANTILES)))


def edge_lags(hidden_spells, dead_spells):
    """Seconds from each dead spell's start and end to the overlapping hidden stretch's.

    Positive start lag: the feed left live play after the stoppage began. Positive end lag: the
    feed came back after the stoppage ended (the replay ran into play).
    """
    starts, ends = [], []
    for d0, d1 in dead_spells:
        overlap = [(h0, h1) for h0, h1 in hidden_spells if h0 < d1 and h1 > d0]
        if overlap:
            starts.append((overlap[0][0] - d0) / FPS)
            ends.append((overlap[-1][1] - d1) / FPS)
    return starts, ends


def length_bin(length_frames):
    seconds = length_frames / FPS
    for low, high in zip(LENGTH_BINS[:-1], LENGTH_BINS[1:]):
        if low <= seconds < high:
            return "{}-{} s".format(low, high) if high < 10 ** 6 else ">= {} s".format(low)
    return "?"


def main():
    data = io_atd.load(GAME)
    try:
        events = io_tagging.load(GAME)
    except IOError:
        events = []
    tv = broadcast.load(GAME)
    n = len(data["ball"])
    tv["covered"] = broadcast.coverage(tv, n)
    plain = methods.state_model(data, events)
    with_tv = methods.state_model_tv(data, events, tv=tv)

    graded = np.zeros(n, dtype=bool)
    for _, first, last in half_video_blocks(GAME):
        graded[first:last + 1] = True
    scope = graded & tv["covered"]
    hidden = scope & ~tv["shown"]
    hidden_spells = spells(hidden)

    # per frame: the length of the not-shown stretch it sits in
    stretch_len = np.zeros(n, dtype=int)
    for h0, h1 in hidden_spells:
        stretch_len[h0:h1] = h1 - h0

    print("\n== whole graded match, no labels: not-shown stretches vs the tracking-only model ==")
    print("  {:<10} {:>6} {:>8}  {:>18}  {:>14}".format(
        "length", "count", "frames", "tracking-only D", "with TV, D"))
    for low, high in zip(LENGTH_BINS[:-1], LENGTH_BINS[1:]):
        inside = hidden & (stretch_len >= low * FPS) & (stretch_len < high * FPS)
        count = sum(1 for h0, h1 in hidden_spells if low * FPS <= h1 - h0 < high * FPS)
        if not inside.any():
            continue
        print("  {:<10} {:>6} {:>8}  {:>17.1%}  {:>13.1%}".format(
            length_bin(low * FPS), count, int(inside.sum()),
            np.mean(plain[inside] == "D"), np.mean(with_tv[inside] == "D")))
    shown = scope & tv["shown"]
    print("  {:<10} {:>6} {:>8}  {:>17.1%}  {:>13.1%}".format(
        "shown", "", int(shown.sum()), np.mean(plain[shown] == "D"), np.mean(with_tv[shown] == "D")))

    plain_dead = [s for s in spells(scope & (plain == "D")) if s[1] - s[0] >= 2 * FPS]
    starts, ends = edge_lags(hidden_spells, plain_dead)
    print("\n  tracking-only D spells >= 2 s: {}, of which {} overlap a not-shown stretch".format(
        len(plain_dead), len(starts)))
    print("  feed leaves live minus D start (s)  " + q(starts))
    print("  feed returns minus D end (s)        " + q(ends))

    tv_dead = spells(scope & (with_tv == "D"))
    lengths = np.array([s[1] - s[0] for s in tv_dead]) / FPS
    no_plain = sum(1 for d0, d1 in tv_dead if not np.any(plain[d0:d1] == "D"))
    print("\n  state_model_tv D spells: {}, length (s) {}".format(len(tv_dead), q(lengths)))
    print("  of which {} contain no tracking-only D frame (created by the TV cue alone)".format(
        no_plain))

    gt = diagnose.read_gt()
    window = gt != "X"
    alive = window & np.isin(gt, ("H", "A"))
    wrong = alive & (with_tv == "D")
    print("\n== public window, sanity only: alive frames that state_model_tv calls D ==")
    print("  {} of {} alive frames (tracking-only model: {})".format(
        int(wrong.sum()), int(alive.sum()), int((alive & (plain == "D")).sum())))
    if wrong.any():
        print("  not shown live: {:.1%}".format(np.mean(hidden[wrong])))
        for low, high in zip(LENGTH_BINS[:-1], LENGTH_BINS[1:]):
            inside = wrong & hidden & (stretch_len >= low * FPS) & (stretch_len < high * FPS)
            if inside.any():
                print("    in a not-shown stretch of {:<9}: {:.1%}".format(
                    length_bin(low * FPS), inside.sum() / wrong.sum()))
        gt_dead = spells(window & (gt == "D"))
        after_restart = np.zeros(n, dtype=bool)
        before_stop = np.zeros(n, dtype=bool)
        for d0, d1 in gt_dead:
            after_restart[d1:d1 + NEAR] = True
            before_stop[max(0, d0 - NEAR):d0] = True
        print("  within 5 s after a true restart:    {:.1%}".format(np.mean(after_restart[wrong])))
        print("  within 5 s before a true stoppage:  {:.1%}".format(np.mean(before_stop[wrong])))
        spurious = np.zeros(n, dtype=bool)
        for d0, d1 in spells(window & (with_tv == "D")):
            if not np.any(gt[d0:d1] == "D"):
                spurious[d0:d1] = True
        print("  in a predicted D spell with no true D frame: {:.1%}".format(np.mean(spurious[wrong])))

        starts, ends = edge_lags(spells(window & hidden), gt_dead)
        print("  true D spells: {}; feed leaves live minus D start (s) {}".format(len(gt_dead), q(starts)))
        print("                     feed returns minus D end (s)       {}".format(q(ends)))
        restarts = [d1 for d0, d1 in gt_dead if d1 < n and window[d1]]
        pred_dead = window & (with_tv == "D")
        lags = []
        for r in restarts:
            back = r
            while back < n and pred_dead[back]:
                back += 1
            fwd = r
            while fwd > 0 and not pred_dead[fwd - 1] and r - fwd < 10 * FPS:
                fwd -= 1
            lags.append((back - r) / FPS if back > r else -(r - fwd) / FPS)
        print("  predicted restart minus true restart (s) " + q(lags))


if __name__ == "__main__":
    main()
