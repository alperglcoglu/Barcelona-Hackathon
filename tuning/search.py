"""Choose the state model's parameters on Metrica's public sample games, never on BAR-ATH.

    bash tuning/fetch_sample_data.sh                  # once
    python -m tuning.search                           # hand-written evidence (methods.state_model)
    python -m tuning.search --method learned          # learned evidence (methods.learned)
    python -m tuning.search --evaluate                # score possession/config.py as it is
    python -m tuning.search --outer 40 --refine 40    # random search, then a local search

Protocol:

  - Each of the three sample games is degraded twice (sample_data.degrade, SCENARIOS) so it
    looks like automatic tracking, and scored against labels derived from its event data.
  - Random search in two levels. An outer draw sets the parameters that change the evidence itself
    (EVIDENCE_SPACE), and the per-frame scores are computed once for it. Several inner draws then
    set the weights and transition penalties (DECODE_SPACE), which only need a new Viterbi pass.
    The objective is mean spellF1, the competition's ranking metric.
  - **Leave-one-game-out**: for each game, the configuration that is best on the other two is
    scored on the held-out one. That is the honest estimate of how a tuned configuration transfers
    to an unseen match, and it is what the technical document reports.

Everything is seeded and deterministic.
"""

import argparse
import os
import pickle

import numpy as np

from possession import config, learned, methods
from . import metrics, sample_data

# Params
CACHE = os.path.join(config.CACHE_DIR, "sample_games_v2.pkl")
# Every game is degraded once per scenario: (seed, share of dead-time ball gaps filled). The two
# scenarios bracket how often the BAR-ATH tracker sees the ball during stoppages (see
# sample_data.show_dead_ball). Changing degrade() changes every score below.
SCENARIOS = ((0, 0.0), (1, 0.5))
SEEDS = tuple(seed for seed, _ in SCENARIOS)
N_OUTER = 30
N_INNER = 12
N_REFINE = 0
REFINE_SCALE = 0.3
SEARCH_SEED = 2027

# Parameter -> (low, high, scale). Log scale for strictly positive sizes.
EVIDENCE_SPACE = {
    "state_model": {
        "CONTROL_SIGMA": (0.6, 3.0, "log"),
        "CONTROL_BALL_SPEED": (3.0, 30.0, "log"),
        "DEAD_SPEED_RATIO": (0.4, 1.2, "lin"),
        "DEAD_SPEED_WEIGHT": (0.3, 6.0, "log"),
        "DEAD_SPEED_CLIP": (0.3, 3.0, "log"),
        "BALL_MAX_GAP": (0, 50, "int"),
        "DEAD_GAP_MIN": (12, 300, "int"),
        "DEAD_GAP_SCORE": (0.0, 3.0, "lin"),
        "TOUCH_WEIGHT": (0.0, 1.5, "lin"),
        "TOUCH_DV": (2.0, 8.0, "log"),
        "TOUCH_RADIUS": (0.8, 3.0, "log"),
        "WIDTH_WEIGHT": (0.0, 0.4, "lin"),
        "WIDTH_SCALE": (1.0, 10.0, "log"),
    },
    "learned": {
        "CONTROL_SIGMA": (0.6, 3.0, "log"),
        "BALL_MAX_GAP": (0, 50, "int"),
        "LEARNED_CLIP": (0.5, 6.0, "log"),
    },
}
DECODE_SPACE = {
    "SWITCH_COST": (1.0, 25.0, "log"),
    "TO_DEAD_COST": (3.0, 80.0, "log"),
    "FROM_DEAD_COST": (3.0, 80.0, "log"),
    "TEAM_WEIGHT": (0.3, 4.0, "log"),
    "DEAD_WEIGHT": (0.3, 4.0, "log"),
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
    """[(name, seed, data, halves, gt)] with every game degraded once per scenario."""
    out = []
    for name, (data, halves, gt) in sorted(games.items()):
        for seed, dead_visible in SCENARIOS:
            degraded = sample_data.degrade(data, np.random.default_rng(seed), labels=gt,
                                           dead_visible=dead_visible)
            out.append((name, seed, degraded, halves, gt))
    return out


class Patched:
    """Temporarily set attributes of possession.config."""

    def __init__(self, params):
        self.params = params

    def __enter__(self):
        self.saved = {key: getattr(config, key) for key in self.params}
        for key, value in self.params.items():
            setattr(config, key, value)

    def __exit__(self, *exc):
        for key, value in self.saved.items():
            setattr(config, key, value)


def raw_scores(method, params, cases, model=None):
    """Per-case unweighted frame scores (TEAM_WEIGHT = DEAD_WEIGHT = 1) for one evidence draw."""
    with Patched(dict(params, TEAM_WEIGHT=1.0, DEAD_WEIGHT=1.0)):
        if method == "learned":
            return [learned.frame_scores(data, halves, model) for _, _, data, halves, _ in cases]
        return [methods.frame_scores(data) for _, _, data, _, _ in cases]


def decode_score(scores, params, cases):
    """{game: mean spellF1 over seeds} and mean macroF1 for one set of decode parameters."""
    with Patched(params):
        spell, macro = {}, []
        for raw, (name, _, _, halves, gt) in zip(scores, cases):
            weighted = raw * np.array([config.TEAM_WEIGHT, config.TEAM_WEIGHT, config.DEAD_WEIGHT])
            pred = methods.decode(weighted, halves)
            spell.setdefault(name, []).append(metrics.spell_f1(gt, pred)[0])
            macro.append(metrics.macro_f1(gt, pred))
    return {k: float(np.mean(v)) for k, v in spell.items()}, float(np.mean(macro))


def draw(space, rng):
    params = {}
    for key, (low, high, scale) in space.items():
        if scale == "log":
            params[key] = float(np.exp(rng.uniform(np.log(low), np.log(high))))
        elif scale == "int":
            params[key] = int(rng.integers(low, high + 1))
        else:
            params[key] = float(rng.uniform(low, high))
    return params


def perturb(params, space, rng, scale=REFINE_SCALE):
    """A random neighbour of params: each value moved by about `scale` (log or linear), in bounds."""
    out = {}
    for key, (low, high, kind) in space.items():
        value = params[key]
        if kind == "lin":
            value = value + rng.normal(0, scale * (high - low) / 2)
        else:
            value = float(np.exp(np.log(max(value, 1e-6)) + rng.normal(0, scale)))
        value = min(max(value, low), high)
        out[key] = int(round(value)) if kind == "int" else value
    return out


def current(space):
    return {key: getattr(config, key) for key in space}


def show(params):
    return "  ".join("{}={:.3g}".format(k, v) for k, v in params.items())


def mean_of(result, games=None):
    spell = result[0]
    return float(np.mean([spell[g] for g in (games or sorted(spell))]))


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--method", default="state_model", choices=sorted(EVIDENCE_SPACE))
    parser.add_argument("--evaluate", action="store_true", help="score config.py as it is")
    parser.add_argument("--outer", type=int, default=N_OUTER)
    parser.add_argument("--inner", type=int, default=N_INNER)
    parser.add_argument("--refine", type=int, default=N_REFINE,
                        help="extra evidence draws around the best so far (local search)")
    args = parser.parse_args()

    cases = degraded_set(load_games())
    space = EVIDENCE_SPACE[args.method]
    start = (current(space), current(DECODE_SPACE))
    scores = raw_scores(args.method, start[0], cases)
    base = decode_score(scores, start[1], cases)
    print("config.py  {}  spellF1 {}  mean {:.3f}  macroF1 {:.3f}".format(
        args.method, {k: round(v, 3) for k, v in base[0].items()}, mean_of(base), base[1]))
    if args.evaluate:
        return

    rng = np.random.default_rng(SEARCH_SEED)
    trials = [(dict(start[0], **start[1]), base)]
    for i in range(args.outer):
        evidence = draw(space, rng)
        scores = raw_scores(args.method, evidence, cases)
        for _ in range(args.inner):
            decode = draw(DECODE_SPACE, rng)
            trials.append((dict(evidence, **decode), decode_score(scores, decode, cases)))
        best = max(trials, key=lambda t: mean_of(t[1]))
        print("{:>3}/{} evidence draws, best mean spellF1 {:.3f}".format(
            i + 1, args.outer, mean_of(best[1])), flush=True)
    # Local refinement: random search alone is thin in 13 dimensions, so the second phase draws
    # neighbours of the best configuration so far, for both the evidence and the decode parameters.
    for i in range(args.refine):
        best = max(trials, key=lambda t: mean_of(t[1]))[0]
        evidence = perturb(best, space, rng)
        scores = raw_scores(args.method, evidence, cases)
        for _ in range(args.inner):
            decode = perturb(best, DECODE_SPACE, rng)
            trials.append((dict(evidence, **decode), decode_score(scores, decode, cases)))
        print("{:>3}/{} refinement draws, best mean spellF1 {:.3f}".format(
            i + 1, args.refine, mean_of(max(trials, key=lambda t: mean_of(t[1]))[1])), flush=True)

    games = sorted(base[0])
    ranked = sorted(trials, key=lambda t: -mean_of(t[1]))
    print("\ntop 5 on all games:")
    for params, result in ranked[:5]:
        print("  mean {:.3f}  macro {:.3f}  {}".format(
            mean_of(result), result[1], {k: round(v, 3) for k, v in result[0].items()}))
        print("    " + show(params))

    print("\nleave-one-game-out (best on the other two, scored on the held-out game):")
    held = []
    for game in games:
        others = [g for g in games if g != game]
        params, result = max(trials, key=lambda t: mean_of(t[1], others))
        held.append(result[0][game])
        print("  {}  {:.3f}   with {}".format(game, result[0][game], show(params)))
    print("  mean held-out spellF1 {:.3f}".format(np.mean(held)))


if __name__ == "__main__":
    main()
