"""What does the Smart Tagging say that the model does not use yet? Aggregate diagnostics.

    python -m possession.diagnose_tagging

Prints counts, shares and quantiles only, like possession.diagnose, so the output is safe to share.

The model reads the tagging as a weak team prior (tagging_prior.py): an attacking phase means its
team has the ball, a defending phase means the other team does. Three other uses are possible and
each needs one number before it is worth building:

  1. **Agreement.** Inside each code, how often does the model's label agree with the team the
     code implies? A code that agrees strongly where the model is unsure is worth more weight.
  2. **Transitions as turnover hints.** ATTACKING / DEFENSIVE TRANSITION begin with a change of
     possession. If their starts sit close to the model's own team switches, a switch near one
     can be made cheaper (as the ball cue does in features.handover_cue).
  3. **Set pieces as restarts.** Every set piece ends a dead ball. If a type's start or end sits
     at a steady offset from the model's restarts, that type can place restarts.

The whole-match block compares the tags with the model's own labels (no ground truth, the same
calibration idea as the TV cue). The public-window block compares them with the truth and is a
sanity check only (RULES.md section 2): five minutes hold only a handful of tags.
"""

import numpy as np

from . import broadcast, diagnose, io_atd, io_tagging, methods, tagging_prior

from games import GAME, half_video_blocks  # noqa: E402  (methods put the root on sys.path)

# Params
FPS = 25
NEAR = 3 * FPS                       # frames: "near" a switch or a restart
SEARCH = 15 * FPS                    # frames: look this far for the nearest one
QUANTILES = (10, 25, 50, 75, 90)


def q(values):
    values = np.asarray(values, dtype=float)
    if len(values) == 0:
        return "n/a"
    return "  ".join("p{} {:+.1f}".format(p, v)
                     for p, v in zip(QUANTILES, np.percentile(values, QUANTILES)))


def nearest(frame, points):
    """Signed seconds from frame to the nearest point within SEARCH, or None."""
    if len(points) == 0:
        return None
    k = np.searchsorted(points, frame)
    best = None
    for j in (k - 1, k):
        if 0 <= j < len(points) and abs(points[j] - frame) <= SEARCH:
            if best is None or abs(points[j] - frame) < abs(best):
                best = points[j] - frame
    return None if best is None else best / FPS


def team_switches(labels, graded):
    """Frames where the possessing team changes (H <-> A, across D too), inside graded frames."""
    alive = graded & np.isin(labels, ("H", "A"))
    idx = np.flatnonzero(alive)
    if len(idx) < 2:
        return np.zeros(0, dtype=int)
    seq = labels[idx]
    return idx[1:][seq[1:] != seq[:-1]]


def restarts(labels, graded):
    """Frames where a D spell ends in an alive label, inside graded frames."""
    prev, cur = labels[:-1], labels[1:]
    hit = (prev == "D") & np.isin(cur, ("H", "A")) & graded[1:]
    return np.flatnonzero(hit) + 1


def implied_side(event):
    side = event["side"]
    if side is None:
        return None
    if event["code"] in tagging_prior.ATTACKING_CODES:
        return side
    if event["code"] in tagging_prior.DEFENDING_CODES:
        return "A" if side == "H" else "H"
    return None


def report(events, labels, graded, title):
    print("\n== {} ==".format(title))
    inside = [e for e in events if graded[min(max(e["start_frame"], 0), len(graded) - 1)]]

    print("\n  1. agreement: model label vs the team the code implies (alive frames inside)")
    print("  {:<30} {:>4} {:>8} {:>9} {:>7}".format("code", "n", "len p50", "agree", "dead"))
    for code in sorted({e["code"] for e in inside}):
        group = [e for e in inside if e["code"] == code]
        lengths = [(e["end_frame"] - e["start_frame"]) / FPS for e in group]
        agree = total = dead = frames = 0
        for e in group:
            seg = labels[max(e["start_frame"], 0):e["end_frame"]]
            seg = seg[graded[max(e["start_frame"], 0):e["end_frame"]]]
            frames += len(seg)
            dead += int(np.sum(seg == "D"))
            side = implied_side(e)
            if side is not None:
                alive = seg[np.isin(seg, ("H", "A"))]
                agree += int(np.sum(alive == side))
                total += len(alive)
        print("  {:<30} {:>4} {:>7.1f}s {:>9} {:>6.0%}".format(
            code[:30], len(group), np.median(lengths),
            "{:.0%}".format(agree / total) if total else "-", dead / frames if frames else 0))

    switches = team_switches(labels, graded)
    print("\n  2. transitions: seconds from the tag's start to the nearest team switch "
          "(+ = switch after)")
    for code in ("ATTACKING TRANSITION", "DEFENSIVE TRANSITION", "LONG BALL"):
        offsets = [nearest(e["start_frame"], switches) for e in inside if e["code"] == code]
        found = [o for o in offsets if o is not None]
        if offsets:
            print("  {:<22} n {:>3}  switch within 15 s {:>4.0%}  within 3 s {:>4.0%}  {}".format(
                code, len(offsets), len(found) / len(offsets),
                np.mean([abs(o) <= NEAR / FPS for o in found]) if found else 0, q(found)))
    # base rate: how often a random frame has a switch within 3 s, to compare with the above
    if len(switches):
        rng = np.random.default_rng(0)
        probe = rng.choice(np.flatnonzero(graded), 2000)
        base = np.mean([(o is not None and abs(o) <= NEAR / FPS)
                        for o in (nearest(p, switches) for p in probe)])
        print("  {:<22} a random graded frame has a switch within 3 s {:.0%}".format(
            "(base rate)", base))

    starts = restarts(labels, graded)
    print("\n  3. set pieces: seconds from the tag's start / end to the nearest restart "
          "(+ = restart after)")
    types = sorted({(e["tags"].get("Type") or "?") for e in inside if e["code"] == "SET PIECES"})
    for kind in types:
        group = [e for e in inside if e["code"] == "SET PIECES" and (e["tags"].get("Type") or "?") == kind]
        s = [o for o in (nearest(e["start_frame"], starts) for e in group) if o is not None]
        t = [o for o in (nearest(e["end_frame"], starts) for e in group) if o is not None]
        print("  {:<22} n {:>3}  found {:>3}".format(kind[:22], len(group), len(s)))
        print("      from start  " + q(s))
        print("      from end    " + q(t))


def main():
    data = io_atd.load(GAME)
    events = io_tagging.load(GAME)
    tv = broadcast.load(GAME)
    n = len(data["ball"])
    tv["covered"] = broadcast.coverage(tv, n)
    labels = methods.state_model_tv(data, events, tv=tv)

    graded = np.zeros(n, dtype=bool)
    for _, first, last in half_video_blocks(GAME):
        graded[first:last + 1] = True
    report(events, labels, graded, "whole graded match, no labels: tags vs the model's output")

    gt = diagnose.read_gt()
    window = gt != "X"
    report(events, gt, window, "public window, sanity only: tags vs the truth")


if __name__ == "__main__":
    main()
