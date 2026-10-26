"""Choose the state model's parameters on Metrica's public sample games, never on BAR-ATH.

    bash tuning/fetch_sample_data.sh            # once
    python -m tuning.search                     # random search, prints the best configurations
    python -m tuning.search --evaluate          # score the values currently in possession/config.py

Protocol:

  - Each of the three sample games is degraded (sample_data.degrade) with fixed seeds so it looks
    like automatic tracking, and scored against labels derived from its event data.
  - A random search draws parameter sets from the ranges in SPACE and scores each one on all three
    games. The objective is mean spellF1, the competition's ranking metric.
  - **Leave-one-game-out**: for each game, the configuration that is best on the *other two* is
    scored on the held-out one. That is the honest estimate of how a tuned configuration transfers
    to an unseen match, and it is what the technical document reports.

Everything is seeded and deterministic.
"""

import argparse
import os
import pickle

import numpy as np

from possession import config, methods
from . import metrics, sample_data

# Params
CACHE = os.path.join(config.CACHE_DIR, "sample_games_v1.pkl")
SEEDS = (0, 1)              # two independent degradations per game
N_TRIALS = 300
SEARCH_SEED = 2027

# Parameter -> (low, high, scale). Log scale for strictly positive sizes.
SPACE = {
    "SWITCH_COST": (1.0, 20.0, "log"),
    "TO_DEAD_COST": (5.0, 80.0, "log"),
    "FROM_DEAD_COST": (5.0, 80.0, "log"),
    "CONTROL_SIGMA": (0.6, 3.0, "log"),
    "CONTROL_BALL_SPEED": (3.0, 30.0, "log"),
    "DEAD_SPEED_PIVOT": (1.0, 2.2, "lin"),
    "DEAD_SPEED_WEIGHT": (0.2, 3.0, "log"),
    "DEAD_SPEED_CLIP": (0.3, 3.0, "log"),
    "TEAM_WEIGHT": (0.3, 3.0, "log"),
    "BALL_MAX_GAP": (0, 50, "int"),
}


def load_games():
    """{name: (data, halves, gt)} for the three sample games, cached after the first parse."""
    if os.path.isfile(CACHE):
        with open(CACHE, "rb") as handle:
            return pickle.load(handle)
    games = {
        "game1": sample_data.load_csv_game(1),
        "game2": sample_data.load_csv_game(2),
        "game3": sample_data.load_epts_game(),
    }
    os.makedirs(config.CACHE_DIR, exist_ok=True)
    with open(CACHE, "wb") as handle:
        pickle.dump(games, handle)
    return games


def degraded_set(games):
    """[(name, seed, data, halves, gt)] with every game degraded once per seed."""
    out = []
    for name, (data, halves, gt) in sorted(games.items()):
        for seed in SEEDS:
            out.append((name, seed, sample_data.degrade(data, np.random.default_rng(seed)),
                        halves, gt))
    return out


def score(params, cases):
    """{game: mean spellF1 over seeds} and mean macroF1, with config temporarily set to params."""
    saved = {key: getattr(config, key) for key in params}
    try:
        for key, value in params.items():
            setattr(config, key, value)
        spell, macro = {}, []
        for name, _, data, halves, gt in cases:
            pred = methods.state_model(data, halves=halves)
            spell.setdefault(name, []).append(metrics.spell_f1(gt, pred)[0])
            macro.append(metrics.macro_f1(gt, pred))
        return {k: float(np.mean(v)) for k, v in spell.items()}, float(np.mean(macro))
    finally:
        for key, value in saved.items():
            setattr(config, key, value)


def draw(rng):
    params = {}
    for key, (low, high, scale) in SPACE.items():
        if scale == "log":
            params[key] = float(np.exp(rng.uniform(np.log(low), np.log(high))))
        elif scale == "int":
            params[key] = int(rng.integers(low, high + 1))
        else:
            params[key] = float(rng.uniform(low, high))
    return params


def current():
    return {key: getattr(config, key) for key in SPACE}


def show(params):
    return "  ".join("{}={:.3g}".format(k, v) for k, v in params.items())


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--evaluate", action="store_true", help="score config.py as it is")
    parser.add_argument("--trials", type=int, default=N_TRIALS)
    args = parser.parse_args()

    cases = degraded_set(load_games())
    base = score(current(), cases)
    print("config.py      spellF1 {}  mean {:.3f}  macroF1 {:.3f}".format(
        {k: round(v, 3) for k, v in base[0].items()}, np.mean(list(base[0].values())), base[1]))
    if args.evaluate:
        return

    rng = np.random.default_rng(SEARCH_SEED)
    trials = [(current(), base)]
    for i in range(args.trials):
        params = draw(rng)
        trials.append((params, score(params, cases)))
        if (i + 1) % 25 == 0:
            best = max(trials, key=lambda t: np.mean(list(t[1][0].values())))
            print("{:>4} trials, best mean spellF1 {:.3f}".format(
                i + 1, np.mean(list(best[1][0].values()))))

    games = sorted(base[0])
    ranked = sorted(trials, key=lambda t: -np.mean(list(t[1][0].values())))
    print("\ntop 5 on all games:")
    for params, (spell, macro) in ranked[:5]:
        print("  mean {:.3f}  macro {:.3f}  {}".format(np.mean(list(spell.values())), macro,
                                                       {k: round(v, 3) for k, v in spell.items()}))
        print("    " + show(params))

    print("\nleave-one-game-out (best on the other two, scored on the held-out game):")
    held = []
    for game in games:
        others = [g for g in games if g != game]
        params, (spell, _) = max(trials, key=lambda t: np.mean([t[1][0][g] for g in others]))
        held.append(spell[game])
        print("  {}  {:.3f}   with {}".format(game, spell[game], show(params)))
    print("  mean held-out spellF1 {:.3f}".format(np.mean(held)))


if __name__ == "__main__":
    main()
