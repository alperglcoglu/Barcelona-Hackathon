"""Grid over evidence and decode parameters on the tuning games, with a leave-one-game-out check.

    python -m tuning.sweep_grid --games all --evidence PC_CUE_WEIGHT=0,0.5,1 \\
        --decode SWITCH_RELIEF=0.4,0.6

Every other parameter stays as in possession/config.py. The frame scores are computed once per
evidence combination and decoded for every decode combination, like tuning/search.py. For each
combination it prints the mean spellF1 and macroF1 over the games, and from
tuning/segmentation.py the short other-team spells missed and the spurious changes (totals).

Leave-one-game-out: for each game, the combination that is best on the others is scored on it.
The mean of those held-out scores is the honest estimate of what choosing on the grid is worth;
compare it with the same figure for config.py as it is (the first row of the grid when the
current values are in it).
"""

import argparse
import itertools

import numpy as np

from . import search, segmentation


def parse(items):
    grid = {}
    for item in items or []:
        key, values = item.split("=")
        grid[key] = [float(v) for v in values.split(",")]
    return grid


def combos(grid):
    keys = sorted(grid)
    return [dict(zip(keys, values)) for values in itertools.product(*(grid[k] for k in keys))]


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--games", choices=("metrica", "all"), default="all")
    parser.add_argument("--evidence", action="append", help="NAME=v1,v2,... (frame scores)")
    parser.add_argument("--decode", action="append", help="NAME=v1,v2,... (Viterbi only)")
    args = parser.parse_args(argv)

    cases = search.degraded_set(search.load_games(args.games))
    rows = []
    for evidence in combos(parse(args.evidence)):
        scores = search.raw_scores("state_model", evidence, cases)
        for decode in combos(parse(args.decode)):
            params = dict(evidence, **decode)
            result = segmentation.evaluate(scores, cases, params)
            total = segmentation.summary(result)
            macro = search.decode_score(scores, params, cases)[1]
            rows.append((params, result, total, macro))
            print("{}  spellF1 {:.4f}  macroF1 {:.3f}  short spells missed {:.0f}/{:.0f}  "
                  "spurious {:.0f}  latency {:+.2f} s".format(
                      "  ".join("{} {:g}".format(k, v) for k, v in params.items()),
                      total["spellF1"], macro, total["interrupt_missed"], total["interrupts"],
                      total["spurious"], total["latency"]), flush=True)

    games = sorted(rows[0][1])
    held = []
    for game in games:
        others = [g for g in games if g != game]
        params, result, _, _ = max(rows, key=lambda r: np.mean([r[1][g]["spellF1"] for g in others]))
        held.append(result[game]["spellF1"])
        print("  held out {:<10} {:.3f}  with {}".format(
            game, result[game]["spellF1"], {k: round(v, 3) for k, v in params.items()}))
    best = max(rows, key=lambda r: r[2]["spellF1"])
    print("best on all games: {}  spellF1 {:.4f}".format(
        {k: round(v, 3) for k, v in best[0].items()}, best[2]["spellF1"]))
    print("mean held-out spellF1 {:.4f}".format(np.mean(held)))


if __name__ == "__main__":
    main()
