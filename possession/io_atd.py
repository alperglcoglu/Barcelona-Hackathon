"""Read the provided automatic tracking (ATD) into dense numpy arrays.

The raw file is Metrica's FIFA/EPTS text export, one line per video frame:

    <frame_id>:<x,y,s>;<x,y,s>;...;<x,y,s>:<ball group>

with one `x,y,s` group per track slot (46 on the tactical recording) and `NaN` for anything not
observed. Coordinates are normalised to 0-1 and speed is in m/s (DATA.md). frame_id counts from 1,
submissions from 0, so record `frame_id` lands in row `frame_id - 1` (starter/load_atd.py).

Why not kloppy: it builds one Python object per frame and per player, which costs minutes and
gigabytes for 199,319 frames. Everything downstream is array maths, so we parse straight into
arrays once and cache them. The parse takes about a minute; loading the cache takes a second.

What comes out (`load`):

    ball      (n_frames, 2)          float64, metres, NaN where the ball is not tracked
    players   (n_frames, n_slots, 3) float64, x and y in metres and speed in m/s, NaN if absent
    team      (n_slots,)             int8, 0 = home (H), 1 = away (A)
    pitch     (length, width)        metres, from games.py

Rows that have no record in the file stay NaN, so a frame index is always a submission row.
"""

import os
import sys
import xml.etree.ElementTree as ET

import numpy as np

from . import config

sys.path.insert(0, config.ROOT)
from games import GAME, GAMES, require, source_frames  # noqa: E402

# Params
FIELD_SEP = ":"
GROUP_SEP = ";"
VALUE_SEP = ","
HOME, AWAY = 0, 1
# Bump when the cached layout changes, so a stale cache is rebuilt instead of misread.
CACHE_VERSION = 1


def team_of_each_track(meta_path):
    """Array over track slots: 0 if the slot belongs to the home team, 1 if away.

    The first <Team> in the metadata is the home side (the FIFA/EPTS convention, and Barcelona for
    this match), and every <Player> -- really a track slot, not a person -- names its team. This is
    the same rule as starter/baseline.py, restated here so the package has no dependency on the
    starter scripts.
    """
    root = ET.parse(meta_path).getroot()
    team_ids = [team.get("id") for team in root.iter("Team")]
    if len(team_ids) != 2:
        raise ValueError("{}: expected 2 teams, found {}".format(meta_path, len(team_ids)))
    slots = [player.get("teamId") for player in root.iter("Player")]
    if not slots:
        raise ValueError("{}: no Player entries".format(meta_path))
    return np.array([HOME if team_id == team_ids[0] else AWAY for team_id in slots],
                    dtype=np.int8)


def _floats(text, n):
    """The first n comma-separated numbers of a group, NaN-padded. 'NaN' parses as float('nan')."""
    values = text.split(VALUE_SEP)[:n]
    out = [float(value) for value in values]
    return out + [float("nan")] * (n - len(out))


def parse_raw(raw_path, n_frames, n_slots, pitch):
    """Stream the raw file into (ball, players) arrays in metres. See the module docstring."""
    length, width = pitch
    ball = np.full((n_frames, 2), np.nan, dtype=np.float64)
    players = np.full((n_frames, n_slots, 3), np.nan, dtype=np.float64)

    with open(raw_path) as handle:
        for line in handle:
            parts = line.rstrip("\n").split(FIELD_SEP)
            if len(parts) != 3:
                continue
            row = int(parts[0]) - 1
            if not 0 <= row < n_frames:
                continue

            groups = parts[1].split(GROUP_SEP)
            flat = parts[1].replace(GROUP_SEP, VALUE_SEP).split(VALUE_SEP)
            if len(groups) == n_slots and len(flat) == 3 * n_slots:
                # The common case: every slot present as exactly x,y,s. One conversion per line.
                players[row] = np.array(flat, dtype=np.float64).reshape(n_slots, 3)
            else:
                for slot, group in enumerate(groups[:n_slots]):
                    players[row, slot] = _floats(group, 3)

            ball[row] = _floats(parts[2], 2)

    players[:, :, 0] *= length
    players[:, :, 1] *= width
    ball[:, 0] *= length
    ball[:, 1] *= width
    return ball, players


def _cache_path(game, source, raw_path):
    """Cache file for one raw export. Its size and mtime are in the name, so replacing the data
    bundle (or pointing MTAH_DATA_DIR elsewhere) can never silently reuse arrays from another file."""
    stat = os.stat(raw_path)
    return os.path.join(config.CACHE_DIR, "{}_{}_atd_v{}_{}_{}.npz".format(
        game, source, CACHE_VERSION, stat.st_size, int(stat.st_mtime)))


def load(game=GAME, data_dir=None, source="tactical", use_cache=True):
    """The tracking as arrays (see the module docstring), parsed once and then read from cache."""
    meta_path, raw_path = require(["atd_meta", "atd_raw"], game, data_dir, source)
    cache = _cache_path(game, source, raw_path)
    if use_cache and os.path.isfile(cache):
        with np.load(cache) as stored:
            return {key: stored[key] for key in stored.files}

    team = team_of_each_track(meta_path)
    pitch = np.array(GAMES[game]["pitch"], dtype=np.float64)
    print("parsing {} (once; cached afterwards) ...".format(os.path.basename(raw_path)))
    ball, players = parse_raw(raw_path, source_frames(source, game), len(team), pitch)
    data = {"ball": ball, "players": players, "team": team, "pitch": pitch}

    if use_cache:
        os.makedirs(config.CACHE_DIR, exist_ok=True)
        np.savez(cache, **data)
    return data
