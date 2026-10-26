"""Labelling methods: tracking arrays in, one label per frame out.

Each method takes the dict from io_atd.load() and returns a numpy array of single-character labels
('H', 'A', 'D', 'X'), one per tactical-video frame. pipeline.py picks one by name.

`baseline` is the organisers' reference, re-implemented on arrays. It exists for one reason: if it
does not reproduce evaluation/results/baseline_smooth50.csv exactly, something in our loading or
alignment is wrong, and every method built on the same arrays would inherit the error. Run the
parity check (pipeline.py --check-baseline) once after downloading the data.
"""

import numpy as np

from . import config

HOME, AWAY, NOBODY = 0, 1, -1


def nearest_team(ball, players, team):
    """Per frame, the team of the tracked player nearest the ball. NOBODY if no ball or no player.

    Distances are in metres on both axes, so the pitch's aspect ratio does not bias the choice
    near the touchlines. Ties go to the lowest slot index, the same order the reference loop
    visits them in.
    """
    diff = players[:, :, :2] - ball[:, None, :]
    distance = np.einsum("fsk,fsk->fs", diff, diff)
    distance = np.where(np.isnan(distance), np.inf, distance)
    slot = np.argmin(distance, axis=1)
    seen = np.isfinite(distance[np.arange(len(slot)), slot])
    return np.where(seen, team[slot], NOBODY).astype(np.int8)


def majority_smooth(codes, window):
    """Centred majority vote over +/- window frames; D where neither team appears. Ties go home.

    Cumulative sums make the cost independent of the window size.
    """
    if window <= 0:
        return np.where(codes == HOME, "H", np.where(codes == AWAY, "A", "D"))
    home = np.concatenate(([0], np.cumsum(codes == HOME)))
    away = np.concatenate(([0], np.cumsum(codes == AWAY)))
    index = np.arange(len(codes))
    low = np.maximum(0, index - window)
    high = np.minimum(len(codes), index + window + 1)
    n_home, n_away = home[high] - home[low], away[high] - away[low]
    labels = np.full(len(codes), "D", dtype="<U1")
    seen = (n_home + n_away) > 0
    labels[seen] = np.where(n_home[seen] >= n_away[seen], "H", "A")
    return labels


def baseline(data, window=config.BASELINE_WINDOW):
    """Nearest player to the ball, then a majority vote. The published floor at window 50."""
    return majority_smooth(nearest_team(data["ball"], data["players"], data["team"]), window)


METHODS = {
    "baseline": baseline,
}
