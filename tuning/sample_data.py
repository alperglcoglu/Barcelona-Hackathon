"""Metrica's public sample games, turned into the same arrays and labels as the competition data.

Why this exists: RULES.md forbids tuning on the competition game, including its public five
minutes, but explicitly allows labelled data from *other* games. Metrica publishes three
anonymised professional matches with tracking (players and ball) and hand-made event data:

    https://github.com/metrica-sports/sample-data    (fetch with tuning/fetch_sample_data.sh)

They come from the same provider as the competition data, and game 3 even uses the same FIFA/EPTS
file format. This module loads each game into the dict io_atd.load() returns, so the same methods
run on it unchanged, and derives frame labels (H / A / D) from the events.

**How labels are derived from events** (documented in docs/TECHNICAL.md, section 3):

  - Every on-ball event (pass, carry, recovery, set piece, shot, ball lost) says its team had the
    ball at its start frame. Possession is the team of the most recent such event. Between a loss
    and the other team's recovery the ball stays with the losing team, the same rule the challenge
    uses for loose balls.
  - The ball is dead from a stoppage (ball out, foul, goal, a shot that goes out) until the next
    restart (a set piece, a goal kick, a kick-off), and before each period's kick-off.
  - Challenges, cards and other non-possession events change nothing.

These labels come from human event data, not from corrected tracking like the competition ground
truth, so they are not the same definition. They are close enough to tune penalty sizes and to
compare model variants, which is what they are used for. Nothing is tuned to agree with the
competition window.
"""

import csv
import json
import os
import xml.etree.ElementTree as ET

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
SAMPLE_DIR = os.environ.get("SAMPLE_DATA_DIR") or os.path.join(ROOT, "data", "external",
                                                                "sample-data", "data")

# Params
PITCH = (105.0, 68.0)   # both sample games, per the sample-data README
FPS = 25
SPEED_SMOOTH = 2        # +/- frames of position smoothing before computing speed
JITTER_WIDTH = 15       # frames over which the simulated tracking error drifts
ON_BALL = {"PASS", "CARRY", "RECOVERY", "SET PIECE", "SHOT", "BALL LOST"}
STOPPAGE = {"BALL OUT", "FAULT RECEIVED"}


# -- tracking ----------------------------------------------------------------------------------

def _moving_mean(values, half_width):
    ok = ~np.isnan(values)
    total = np.concatenate(([0.0], np.cumsum(np.where(ok, values, 0.0))))
    count = np.concatenate(([0], np.cumsum(ok)))
    index = np.arange(len(values))
    low, high = np.maximum(0, index - half_width), np.minimum(len(values), index + half_width + 1)
    n = count[high] - count[low]
    with np.errstate(invalid="ignore", divide="ignore"):
        return np.where(n > 0, (total[high] - total[low]) / np.maximum(n, 1), np.nan)


def add_speeds(players):
    """Fill players[:, :, 2] with speed in m/s from positions. The sample data has no speed column."""
    for slot in range(players.shape[1]):
        x = _moving_mean(players[:, slot, 0], SPEED_SMOOTH)
        y = _moving_mean(players[:, slot, 1], SPEED_SMOOTH)
        speed = np.full(len(x), np.nan)
        speed[1:] = np.hypot(np.diff(x), np.diff(y)) * FPS
        players[:, slot, 2] = np.where(np.isnan(players[:, slot, 0]), np.nan, speed)
    return players


def _read_csv_team(path):
    """(period, frame, positions (n, players, 2), ball (n, 2)) from one Metrica CSV team file."""
    with open(path) as handle:
        rows = list(csv.reader(handle))
    header = rows[2]
    player_cols = [i for i, name in enumerate(header) if name.startswith("Player")]
    ball_col = header.index("Ball")
    data = np.array([[float(v) if v not in ("", "NaN") else np.nan for v in row]
                     for row in rows[3:]])
    positions = np.stack([data[:, [c, c + 1]] for c in player_cols], axis=1)
    return data[:, 0].astype(int), data[:, 1].astype(int), positions, data[:, [ball_col, ball_col + 1]]


def load_csv_game(number):
    """Sample games 1 and 2 (Metrica CSV format)."""
    folder = os.path.join(SAMPLE_DIR, "Sample_Game_{}".format(number))
    stem = os.path.join(folder, "Sample_Game_{}_RawTrackingData_".format(number))
    period, frame, home, ball = _read_csv_team(stem + "Home_Team.csv")
    _, frame_away, away, _ = _read_csv_team(stem + "Away_Team.csv")
    assert np.array_equal(frame, frame_away)
    positions = np.concatenate([home, away], axis=1) * PITCH
    players = np.concatenate([positions, np.full(positions.shape[:2] + (1,), np.nan)], axis=2)
    team = np.r_[np.zeros(home.shape[1]), np.ones(away.shape[1])].astype(np.int8)
    events = _csv_events(os.path.join(folder, "Sample_Game_{}_RawEventsData.csv".format(number)))
    return _finish(ball * PITCH, players, team, period, events)


def load_epts_game():
    """Sample game 3 (FIFA/EPTS, the competition's own format, with substitutions)."""
    folder = os.path.join(SAMPLE_DIR, "Sample_Game_3")
    root = ET.parse(os.path.join(folder, "Sample_Game_3_metadata.xml")).getroot()
    team_ids = [team.get("id") for team in root.iter("Team")]
    team_of_player = {p.get("id"): team_ids.index(p.get("teamId")) for p in root.iter("Player")}
    player_of_channel = {c.get("id"): c.get("playerId") for c in root.iter("PlayerChannel")}
    slots = sorted(team_of_player, key=lambda p: (team_of_player[p], p))
    slot_of = {p: i for i, p in enumerate(slots)}

    specs = []   # (first_frame, last_frame, [slot per group])
    for spec in root.iter("DataFormatSpecification"):
        refs = [r.get("playerChannelId") for r in spec.iter("PlayerChannelRef")]
        order = [slot_of[player_of_channel[r]] for r in refs if r.endswith("_x")]
        specs.append((int(spec.get("startFrame")), int(spec.get("endFrame")), order))

    n = max(last for _, last, _ in specs)
    ball = np.full((n, 2), np.nan)
    players = np.full((n, len(slots), 3), np.nan)
    spec_index = 0
    with open(os.path.join(folder, "Sample_Game_3_tracking.txt")) as handle:
        for line in handle:
            frame_id, groups, ball_text = line.rstrip("\n").split(":")
            frame_id = int(frame_id)
            while frame_id > specs[spec_index][1]:
                spec_index += 1
            order = specs[spec_index][2]
            row = frame_id - 1
            for slot, group in zip(order, groups.split(";")):
                players[row, slot, :2] = [float(v) for v in group.split(",")[:2]]
            ball[row] = [float(v) for v in ball_text.split(",")[:2]]
    players[:, :, 0] *= PITCH[0]
    players[:, :, 1] *= PITCH[1]
    ball *= PITCH
    team = np.array([team_of_player[p] for p in slots], dtype=np.int8)

    half_end = {p.findtext("Name"): p.findtext("Value") for p in root.iter("ProviderParameter")}
    first_end = int(half_end["first_half_end"])
    period = np.where(np.arange(1, n + 1) <= first_end, 1, 2)
    events = _json_events(os.path.join(folder, "Sample_Game_3_events.json"), team_ids)
    return _finish(ball, players, team, period, events)


def _finish(ball, players, team, period, events):
    players = add_speeds(players)
    halves = []
    for p in sorted(set(period.tolist())):
        rows = np.flatnonzero(period == p)
        halves.append((int(rows[0]), int(rows[-1])))
    data = {"ball": ball, "players": players, "team": team, "pitch": np.array(PITCH)}
    return data, halves, labels_from_events(events, len(ball), halves)


# -- events -> labels ----------------------------------------------------------------------------

def _csv_events(path):
    out = []
    with open(path) as handle:
        for row in csv.DictReader(handle):
            out.append({"side": {"Home": "H", "Away": "A"}.get(row["Team"]),
                        "type": row["Type"].strip().upper(), "subtype": row["Subtype"].upper(),
                        "start": int(row["Start Frame"]) - 1, "end": int(row["End Frame"]) - 1})
    return out


def _json_events(path, team_ids):
    out = []
    for event in json.load(open(path))["data"]:
        sub = event.get("subtypes")
        if isinstance(sub, dict):
            sub = [sub]
        team = event.get("team") or {}
        out.append({"side": {team_ids[0]: "H", team_ids[1]: "A"}.get(team.get("id")),
                    "type": event["type"]["name"].upper(),
                    "subtype": "-".join(s["name"] for s in (sub or [])).upper(),
                    "start": int(event["start"]["frame"]) - 1,
                    "end": int(event["end"]["frame"]) - 1})
    return out


def _is_restart(event):
    return event["type"] == "SET PIECE" or "GOAL KICK" in event["subtype"]


def _starts_stoppage(event):
    if event["type"] in STOPPAGE:
        return True
    return event["type"] == "SHOT" and ("GOAL" in event["subtype"] or "OUT" in event["subtype"])


def _controlled(spell_events):
    """Did the team in this spell actually control the ball, rather than just touch it?

    CHALLENGE.md: a touch without control, a deflection, or an aerial duel won without the team then
    keeping the ball is not a change of possession. In event terms, a team controlled the ball if it
    made a deliberate action: a pass (completed or intercepted), a carry, a shot or a set piece. A
    spell that holds only a recovery and an immediate loss (a header back, a block, a clearance) does
    not count.
    """
    for event in spell_events:
        if event["type"] in ("PASS", "CARRY", "SHOT", "SET PIECE"):
            return True
        if event["type"] == "BALL LOST" and "INTERCEPTION" in event["subtype"] \
                and "HEAD" not in event["subtype"] and "CLEARANCE" not in event["subtype"]:
            return True
    return False


def labels_from_events(events, n_frames, halves, controlled_only=True):
    """Frame labels from events, by the rules in the module docstring. X outside the periods.

    With controlled_only, spells without control (see _controlled) are given back to the team that
    had the ball before them, which is the challenge's definition of possession. Without it, every
    touch by the other team is a possession, which is closer to raw event data.
    """
    labels = np.full(n_frames, "X", dtype="<U1")
    events = sorted(events, key=lambda e: e["start"])
    for first, last in halves:
        inside = [e for e in events if first <= e["start"] <= last]
        spells = [[first, "D", []]]   # [start_frame, label, events], in time order
        stopped = False              # inside a stoppage, waiting for the restart
        for event in inside:
            if stopped and not _is_restart(event):
                continue                 # nothing restarts play except a restart
            stopped = False
            if event["side"] and event["type"] in ON_BALL:
                if event["side"] != spells[-1][1]:
                    spells.append([event["start"], event["side"], []])
                spells[-1][2].append(event)
            if _starts_stoppage(event):
                spells.append([max(event["end"], event["start"]), "D", []])
                stopped = True

        if controlled_only:
            for i in range(1, len(spells)):
                label, previous = spells[i][1], spells[i - 1][1]
                if label in ("H", "A") and previous in ("H", "A") and previous != label \
                        and not _controlled(spells[i][2]):
                    spells[i][1] = previous

        state = np.full(last - first + 1, "D", dtype="<U1")
        for (start, label, _), nxt in zip(spells, spells[1:] + [[last + 1, None, None]]):
            state[start - first:nxt[0] - first] = label
        labels[first:last + 1] = state
    return labels


# -- degradation -----------------------------------------------------------------------------

def _smooth_noise(rng, shape, std, width=JITTER_WIDTH):
    """Zero-mean noise that drifts over about `width` frames, like a tracker's position error,
    rather than jumping every frame (white noise would turn into several m/s of fake speed)."""
    white = rng.normal(0, 1, shape)
    kernel = np.ones(width) / np.sqrt(width)
    smooth = np.apply_along_axis(lambda v: np.convolve(v, kernel, mode="same"), 0, white)
    return smooth * std


def degrade(data, rng, ball_missing=None, mean_gap=20, jitter=0.3, slot_dropout=0.04):
    """A copy of data made to look like automatic, uncorrected tracking.

    Adds slowly drifting positional error and short dropouts of player tracks, so parameters are not
    tuned on cleaner positions than they will meet.

    **The ball is left as it is by default.** possession/diagnose.py, run on the whole graded
    BAR-ATH match with no ground truth involved, found the ball missing on 40.7% of frames in only
    150 gaps with a median length of 268 frames: long stretches, not short occlusions. The sample
    games already look like that on their own (32-41% missing, 155-169 gaps, median 127-196
    frames), so no extra ball dropout is added. `ball_missing` (a target share) and `mean_gap` can
    still add short bursts, to test robustness to a worse tracker.
    """
    ball = data["ball"].copy()
    players = data["players"].copy()
    n = len(ball)
    already = np.isnan(ball[:, 0]).mean()
    target = max(0.0, ball_missing - already) if ball_missing is not None else 0.0
    if target > 0:
        # Alternating renewal process: missing bursts of mean `mean_gap`, visible runs sized to hit
        # the target share of missing frames.
        mean_visible = mean_gap * (1 - target) / target
        t, missing = 0, np.zeros(n, dtype=bool)
        while t < n:
            t += int(rng.geometric(1.0 / mean_visible))
            gap = int(rng.geometric(1.0 / mean_gap))
            missing[t:t + gap] = True
            t += gap
        ball[missing] = np.nan
    ball += _smooth_noise(rng, ball.shape, jitter)
    players[:, :, :2] += _smooth_noise(rng, players[:, :, :2].shape, jitter)
    drop = rng.random(players.shape[:2]) < slot_dropout / 25.0
    for f, s in zip(*np.nonzero(drop)):
        players[f:f + int(rng.integers(5, 50)), s] = np.nan
    return {**data, "ball": ball, "players": players}
