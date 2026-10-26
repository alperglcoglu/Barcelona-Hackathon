"""Fit the learned evidence models (possession/learned.py) on Metrica's public sample games.

    python -m tuning.train              # leave-one-game-out report, then fit on all games and save
    python -m tuning.train --report     # report only, do not overwrite the saved model

Training frames come from the degraded sample games (tuning/search.py, same seeds), never from
BAR-ATH. The team model is fitted on alive frames (y = home has the ball). The dead model is fitted
on every graded frame (y = dead ball).

The report compares the hand-written evidence (methods.state_model) and the learned evidence
(methods.learned) under the same transition penalties, on games the learned model has not seen.
"""

import argparse
import json
import os

import numpy as np

from possession import config, learned, methods
from . import metrics, search

# Params
L2 = 1.0
SUBSAMPLE = 5          # every 5th frame: neighbouring frames are near duplicates


def fit(cases):
    team_X, team_y, dead_X, dead_y = [], [], [], []
    for _, _, data, halves, gt in cases:
        tX, dX = learned.design(data, halves)
        rows = np.arange(0, len(gt), SUBSAMPLE)
        graded = rows[gt[rows] != "X"]
        alive = graded[np.isin(gt[graded], ("H", "A"))]
        team_X.append(tX[alive])
        team_y.append(gt[alive] == "H")
        dead_X.append(dX[graded])
        dead_y.append(gt[graded] == "D")
    return {
        "team": dict(learned.fit_logistic(np.concatenate(team_X), np.concatenate(team_y), L2),
                     features=list(learned.TEAM_FEATURES)),
        "dead": dict(learned.fit_logistic(np.concatenate(dead_X), np.concatenate(dead_y), L2),
                     features=list(learned.DEAD_FEATURES)),
        "trained_on": "Metrica sample games 1-3, degraded by tuning/sample_data.degrade, seeds {}"
                      .format(list(search.SEEDS)),
    }


def evaluate(cases, model):
    out = {}
    for name, _, data, halves, gt in cases:
        for method in ("state_model", "learned"):
            if method == "learned":
                pred = methods.learned_model(data, halves=halves, model=model)
            else:
                pred = methods.state_model(data, halves=halves)
            out.setdefault((name, method), []).append(
                (metrics.spell_f1(gt, pred)[0], metrics.macro_f1(gt, pred)))
    return {k: np.mean(v, axis=0) for k, v in out.items()}


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--report", action="store_true")
    args = parser.parse_args()

    cases = search.degraded_set(search.load_games())
    names = sorted({c[0] for c in cases})
    print("leave-one-game-out (model fitted on the other games):")
    print("  {:<8} {:>18} {:>18}".format("held out", "hand spell/macro", "learned spell/macro"))
    totals = {"state_model": [], "learned": []}
    for game in names:
        model = fit([c for c in cases if c[0] != game])
        result = evaluate([c for c in cases if c[0] == game], model)
        hand, mine = result[(game, "state_model")], result[(game, "learned")]
        totals["state_model"].append(hand)
        totals["learned"].append(mine)
        print("  {:<8} {:>9.3f} / {:.3f} {:>10.3f} / {:.3f}".format(game, *hand, *mine))
    print("  {:<8} {:>9.3f} / {:.3f} {:>10.3f} / {:.3f}".format(
        "mean", *np.mean(totals["state_model"], 0), *np.mean(totals["learned"], 0)))

    if args.report:
        return
    model = fit(cases)
    os.makedirs(os.path.dirname(learned.MODEL_PATH), exist_ok=True)
    with open(learned.MODEL_PATH, "w") as handle:
        json.dump(model, handle, indent=2)
    print("\nsaved {}".format(os.path.relpath(learned.MODEL_PATH, config.ROOT)))
    for part in ("team", "dead"):
        print("  {} model:".format(part))
        for name, value in zip(model[part]["features"], model[part]["coef"]):
            print("    {:<18} {:+.3f}".format(name, value))


if __name__ == "__main__":
    main()
