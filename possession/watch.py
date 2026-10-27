"""Watch the model at work: list the moments worth a look, and animate them with its labels.

    python -m possession.watch --list                      # moments to look at, frames only
    python -m possession.watch --start 30000 --end 30500   # writes evaluation/watch/30000.html

For looking with your own eyes, on your own machine. The list prints frame numbers and times only,
so it is safe to share. The animations draw the match data, so they are not: keep them local and
describe what you see in words (DATA.md, terms of use).

**What the list contains.** Three kinds of moment where the model is most likely wrong:

  - **missed transition?** a Smart Tagging ATTACKING TRANSITION with no team switch by the model
    within 15 s. On BAR-ATH half of these tags have a model switch 1-5 s after their start, ten
    times the base rate (possession.diagnose_tagging), so one without any is a turnover the model
    may have missed;
  - **short possession**: a predicted possession under 4 s. On the tuning games only one in five
    of those is right (docs/TECHNICAL.md section 11);
  - **long possession**: a predicted possession over 40 s, which often hides a merge.

**What the animation shows.** Players (home red, away blue), the ball (white, gone where the
tracking lost it) and, in the title: the frame, the model's label, the team evidence (+ home,
- away), the dead-ball evidence (+ dead), whether the TV feed shows live play, and the Smart
Tagging codes active on that frame. Look for what the model gets wrong and why: a duel, a
deflection, a long ball, a stoppage it misses. Rules: looking is allowed, labelling or correcting
frames by hand is not (RULES.md).
"""

import argparse
import os

import numpy as np

from . import broadcast, config, io_atd, io_tagging, methods

from games import GAME, half_video_blocks  # noqa: E402  (methods put the root on sys.path)

# Params
FPS = 25
SHORT = 4 * FPS
LONG = 40 * FPS
TRANSITION_SEARCH = 15 * FPS
HOME_COLOUR, AWAY_COLOUR, BALL_COLOUR = "#d62728", "#1f77b4", "#ffffff"
OUT_DIR = os.path.join(config.ROOT, "evaluation", "watch")


def run_model(data, events):
    tv = broadcast.load(GAME)
    tv["covered"] = broadcast.coverage(tv, len(data["ball"]))
    labels = methods.state_model_tv(data, events, tv=tv)
    return labels, tv


def spells(labels):
    breaks = np.flatnonzero(labels[1:] != labels[:-1]) + 1
    bounds = np.concatenate(([0], breaks, [len(labels)]))
    return [(int(a), int(b), labels[a]) for a, b in zip(bounds[:-1], bounds[1:])]


def clock(frame):
    seconds = frame / FPS
    return "{:d}:{:04.1f}".format(int(seconds // 60), seconds % 60)


def moments(labels, events):
    """[(kind, first, last)] frames worth a look, in time order."""
    graded = np.zeros(len(labels), dtype=bool)
    for _, first, last in half_video_blocks(GAME):
        graded[first:last + 1] = True
    out = []
    for a, b, label in spells(labels):
        if label in ("H", "A") and graded[a]:
            if b - a < SHORT:
                out.append(("short possession", a, b))
            elif b - a > LONG:
                out.append(("long possession", a, b))
    alive = np.flatnonzero(graded & np.isin(labels, ("H", "A")))
    seq = labels[alive]
    switches = alive[1:][seq[1:] != seq[:-1]]
    for event in events:
        if event["code"] != "ATTACKING TRANSITION" or not graded[min(event["start_frame"],
                                                                    len(graded) - 1)]:
            continue
        start = event["start_frame"]
        if not np.any(np.abs(switches - start) <= TRANSITION_SEARCH):
            out.append(("missed transition?", start, event["end_frame"]))
    return sorted(out, key=lambda m: m[1])


def animate(data, labels, scores, tv, events, first, last, path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.animation import FuncAnimation

    import sys
    sys.path.insert(0, os.path.join(config.ROOT, "starter"))
    from utils import draw_pitch

    length, width = data["pitch"]
    figure, ax = plt.subplots(figsize=(10.5, 7.2))
    draw_pitch(ax, length, width)
    home = ax.scatter([], [], s=90, color=HOME_COLOUR, edgecolors="black", linewidths=0.5, zorder=3)
    away = ax.scatter([], [], s=90, color=AWAY_COLOUR, edgecolors="black", linewidths=0.5, zorder=3)
    ball = ax.scatter([], [], s=70, color=BALL_COLOUR, edgecolors="black", linewidths=1.2, zorder=4)
    title = ax.set_title("", fontsize=9, family="monospace", loc="left")
    team, players = data["team"], data["players"]

    def active_codes(frame):
        return ", ".join(sorted({e["code"].title() for e in events
                                 if e["start_frame"] <= frame < e["end_frame"]}))[:90]

    def update(step):
        t = first + step
        xy = players[t, :, :2]
        seen = ~np.isnan(xy[:, 0])
        home.set_offsets(xy[seen & (team == 0)].reshape(-1, 2))
        away.set_offsets(xy[seen & (team == 1)].reshape(-1, 2))
        b = data["ball"][t]
        ball.set_offsets(b.reshape(1, 2) if not np.isnan(b[0]) else np.empty((0, 2)))
        shown = "live" if tv["shown"][t] else ("away" if tv["covered"][t] else "n/a")
        title.set_text(
            "frame {}  {}   model {}   team {:+.2f}   dead {:+.2f}   TV {}{}\n{}".format(
                t, clock(t), labels[t], scores[t, 0] - scores[t, 1], scores[t, 2], shown,
                "   no ball" if np.isnan(b[0]) else "", active_codes(t)))
        return home, away, ball, title

    animation = FuncAnimation(figure, update, frames=last - first, interval=1000 / FPS)
    with open(path, "w") as handle:
        handle.write(animation.to_jshtml(fps=FPS))
    plt.close(figure)


def main():
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--list", action="store_true", help="print the moments worth a look")
    parser.add_argument("--start", type=int, help="first frame to animate")
    parser.add_argument("--end", type=int, help="frame after the last one (at most 60 s later)")
    parser.add_argument("--pad", type=int, default=3 * FPS, help="frames added on both sides")
    args = parser.parse_args()

    data = io_atd.load(GAME)
    events = io_tagging.load(GAME)
    labels, tv = run_model(data, events)

    if args.list or args.start is None:
        found = moments(labels, events)
        print("\n{:<20} {:>8} {:>8} {:>9} {:>7}".format("kind", "start", "end", "clock", "length"))
        for kind, a, b in found:
            print("{:<20} {:>8} {:>8} {:>9} {:>6.1f}s".format(kind, a, b, clock(a), (b - a) / FPS))
        counts = {}
        for kind, _, _ in found:
            counts[kind] = counts.get(kind, 0) + 1
        print("\n" + "   ".join("{} {}".format(k, v) for k, v in sorted(counts.items())))
        print("animate one with: python -m possession.watch --start <start> --end <end>")
        return

    end = args.end if args.end is not None else args.start + 10 * FPS
    first = max(0, args.start - args.pad)
    last = min(len(labels), end + args.pad, first + 60 * FPS)
    scores = methods.frame_scores(data, events)
    os.makedirs(OUT_DIR, exist_ok=True)
    path = os.path.join(OUT_DIR, "{}.html".format(args.start))
    animate(data, labels, scores, tv, events, first, last, path)
    print("wrote {} ({} frames, {:.0f} s); open it in a browser".format(path, last - first,
                                                                       (last - first) / FPS))


if __name__ == "__main__":
    main()
