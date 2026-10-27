"""SkillCorner's open A-League matches as extra tuning games (other matches, allowed by RULES.md).

    bash tuning/fetch_skillcorner.sh      # once; about 900 MB into data/external/skillcorner/

https://github.com/SkillCorner/opendata publishes twenty 2024-25 A-League matches with broadcast
tracking (players and ball, 10 fps) and, on every frame, the team in possession. Ten of them
(MATCHES) next to Metrica's three make the tuning less tied to three games and the
leave-one-game-out estimate less noisy.

**Labels.** Each frame carries `possession.group`: "home team", "away team" or none. A frame with a
team is that team's possession (H / A). A frame with none is dead ball (D). That reading was checked
on match 1886347 before it was used: the players' mean speed is 1.1 m/s on frames with no team and
2.3 m/s on frames with one, the same ratio as stoppages against play in Metrica's games (about 1.0
and 2.0), and runs with no team last a median 17 s, the length of stoppages. Like Metrica's
event-derived labels, these are not the competition's definition, only close to it.

**Tracking.** Positions are metres from the centre spot; they are moved to the corner origin used
everywhere else. The players are SkillCorner's "extrapolated" positions, which keep players outside
the camera view on the pitch, closer to a wide tactical camera than the detected-only subset. The
ball is kept only where SkillCorner detected it. Everything is resampled from 10 fps to 25 fps by
linear interpolation (the ball only between two detected samples 0.1 s apart), so every frame-based
constant means the same thing as on BAR-ATH. Labels take the nearest 10 fps frame.

The result is the same (data, halves, gt) triple as sample_data, so the tuning code runs unchanged
and degrade() then adds the same noise as to the other games.
"""

import json
import os

import numpy as np

from . import sample_data

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
DATA_DIR = os.environ.get("SKILLCORNER_DIR") or os.path.join(ROOT, "data", "external",
                                                               "skillcorner")

# Params
# The first ten of the twenty published matches, by id. Ten keeps a full search under an hour.
MATCHES = (1874553, 1886347, 1899585, 1925299, 1927964,
           1953632, 1959846, 1986691, 1996435, 1996436)
SOURCE_FPS = 10
FPS = 25
GROUP_LABEL = {"home team": "H", "away team": "A"}


def match_ids():
    """The matches present on disk, in MATCHES order."""
    return [m for m in MATCHES if os.path.isfile(os.path.join(DATA_DIR, "{}.jsonl".format(m)))]


def _resample(values, valid, n_out):
    """Linear interpolation of (n_src, ...) values from 10 fps to 25 fps, NaN unless both ends valid."""
    t = np.arange(n_out) * SOURCE_FPS / FPS
    low = np.minimum(np.floor(t).astype(int), len(values) - 1)
    high = np.minimum(low + 1, len(values) - 1)
    w = (t - low).reshape((-1,) + (1,) * (values.ndim - 1))
    out = values[low] * (1 - w) + values[high] * w
    ok = valid[low] & valid[high]
    return np.where(ok.reshape(ok.shape + (1,) * (out.ndim - ok.ndim)), out, np.nan), \
        np.rint(t).astype(int).clip(0, len(values) - 1)


def load_game(match_id):
    """(data, halves, gt) for one SkillCorner match, in the format of sample_data's games."""
    with open(os.path.join(DATA_DIR, "{}_match.json".format(match_id))) as handle:
        meta = json.load(handle)
    length, width = float(meta["pitch_length"]), float(meta["pitch_width"])
    home_id = meta["home_team"]["id"]
    slot_of = {p["id"]: k for k, p in enumerate(meta["players"])}
    team = np.array([0 if p["team_id"] == home_id else 1 for p in meta["players"]], dtype=np.int8)

    frames = []
    with open(os.path.join(DATA_DIR, "{}.jsonl".format(match_id))) as handle:
        for line in handle:
            frames.append(json.loads(line))
    n_src = len(frames)
    ball = np.full((n_src, 2), np.nan)
    players = np.full((n_src, len(team), 2), np.nan)
    labels = np.full(n_src, "X", dtype="<U1")
    period = np.zeros(n_src, dtype=int)
    for i, frame in enumerate(frames):
        if not frame["period"]:
            continue
        period[i] = frame["period"]
        labels[i] = GROUP_LABEL.get(frame["possession"]["group"], "D")
        b = frame["ball_data"]
        if b["is_detected"] and b["x"] is not None:
            ball[i] = (b["x"] + length / 2, b["y"] + width / 2)
        for p in frame["player_data"]:
            slot = slot_of.get(p["player_id"])
            if slot is not None and p["x"] is not None:
                players[i, slot] = (p["x"] + length / 2, p["y"] + width / 2)

    n_out = int((n_src - 1) * FPS / SOURCE_FPS) + 1
    ball25, nearest = _resample(ball, ~np.isnan(ball[:, 0]), n_out)
    xy25, _ = _resample(players, ~np.isnan(players[:, :, 0]), n_out)
    players25 = np.concatenate([xy25, np.full(xy25.shape[:2] + (1,), np.nan)], axis=2)
    players25 = sample_data.add_speeds(players25)
    gt = labels[nearest]
    halves = []
    for half in (1, 2):
        rows = np.flatnonzero(period[nearest] == half)
        if len(rows):
            halves.append((int(rows[0]), int(rows[-1])))
    data = {"ball": ball25, "players": players25, "team": team,
            "pitch": np.array([length, width])}
    return data, halves, gt
