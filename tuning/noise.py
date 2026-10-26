"""Print the same unlabeled tracking-noise statistics as possession/diagnose.py, for the sample games.

    python -m tuning.noise

One block per game, clean and degraded (tuning/search.py's seeds). Comparing them with the
"whole graded match" block of `python -m possession.diagnose` shows whether
sample_data.degrade makes the sample tracking look like the BAR-ATH tracking.
"""

import numpy as np

from possession.diagnose import tracking_noise
from . import sample_data, search


def main():
    for name, (data, halves, gt) in sorted(search.load_games().items()):
        graded = gt != "X"
        print("== {} clean ==".format(name))
        print("\n".join(tracking_noise(data, graded)))
        degraded = sample_data.degrade(data, np.random.default_rng(search.SEEDS[0]))
        print("== {} degraded (seed {}) ==".format(name, search.SEEDS[0]))
        print("\n".join(tracking_noise(degraded, graded)))


if __name__ == "__main__":
    main()
