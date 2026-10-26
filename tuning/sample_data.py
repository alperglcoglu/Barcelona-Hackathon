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
# Simulated tracking error, matched to BAR-ATH's unlabeled statistics (possession/diagnose.py,
# tuning/noise.py): distance from the ball to the nearest player, position jitter, players tracked.
NOISE_SIGMA = 10        # frames: the error drifts smoothly (Gaussian-smoothed), it does not jump
BALL_OFFSET = 0.7       # metres: std of the ball's position error
PLAYER_OFFSET = 0.3     # metres: std of each player's position error
TRACK_LOSS = 0.12       # share of frames on which a player slot is not tracked
TRACK_LOSS_RUN = 100    # frames: mean length of one loss
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

def _smooth_noise(rng, shape, std, sigma=NOISE_SIGMA):
    """Zero-mean noise with standard deviation `std` that drifts over about `sigma` frames.

    White noise is convolved with a Gaussian kernel (via FFT) and rescaled. A tracker's position
    error is like that: it wanders, it does not jump. BAR-ATH's frame-to-frame jitter is under a
    centimetre, so any jumpy noise would be both unrealistic and turned into fake speed.
    """
    n = shape[0]
    t = np.arange(-4 * sigma, 4 * sigma + 1)
    kernel = np.exp(-t ** 2 / (2.0 * sigma ** 2))
    kernel /= np.sqrt((kernel ** 2).sum())
    size = n + len(kernel) - 1
    white = rng.normal(0, 1, (size,) + tuple(shape[1:]))
    padded = np.zeros(size)
    padded[:len(kernel)] = kernel
    spectrum = np.fft.rfft(padded).reshape((-1,) + (1,) * (len(shape) - 1))
    smooth = np.fft.irfft(np.fft.rfft(white, axis=0) * spectrum, n=size, axis=0)
    return smooth[len(kernel) - 1:len(kernel) - 1 + n] * std


def _lose_tracks(rng, players, share, mean_run):
    """Blank each player slot in runs (geometric lengths, mean `mean_run`) on about `share` of frames."""
    out = players.copy()
    n = len(out)
    mean_visible = mean_run * (1 - share) / share
    for slot in range(out.shape[1]):
        t = int(rng.geometric(1.0 / mean_visible))
        while t < n:
            run = int(rng.geometric(1.0 / mean_run))
            out[t:t + run, slot] = np.nan
            t += run + int(rng.geometric(1.0 / mean_visible))
    return out


def show_dead_ball(ball, labels, share, rng):
    """A copy of ball in which about `share` of the ball gaps that fall in dead time are filled.

    In the sample games the ball is almost always missing while play is stopped, so "ball gone for
    a while" is nearly a perfect stoppage detector there. A tracker on a broadcast-style camera can
    also keep seeing the ball during a stoppage (a throw-in being taken, a free kick being set up).
    How often BAR-ATH's tracker does that is unknown, so the tuning sees both worlds: games as they
    are, and games where some dead-time gaps are filled by a straight line from where the ball was
    last seen to where it is seen next (the ball carried to the restart spot).
    """
    out = ball.copy()
    missing = np.isnan(out[:, 0])
    edges = np.diff(np.r_[0, missing.astype(np.int8), 0])
    for start, stop in zip(np.flatnonzero(edges == 1), np.flatnonzero(edges == -1)):
        if start == 0 or stop >= len(out) or np.mean(labels[start:stop] == "D") < 0.5:
            continue
        if rng.random() < share:
            w = np.linspace(0, 1, stop - start + 2)[1:-1, None]
            out[start:stop] = (1 - w) * out[start - 1] + w * out[stop]
    return out


def degrade(data, rng, ball_missing=None, mean_gap=20, ball_offset=BALL_OFFSET,
            player_offset=PLAYER_OFFSET, track_loss=TRACK_LOSS, labels=None, dead_visible=0.0):
    """A copy of data made to look like automatic, uncorrected tracking.

    Every number is matched to BAR-ATH's unlabeled whole-match statistics (no labels involved):

      - **Position error.** Smoothly drifting error, 0.7 m on the ball and 0.3 m on players. On
        BAR-ATH the nearest player to a visible ball is at p10 0.44 / p50 1.91 / p90 7.02 m; in the
        clean sample games it is at p10 0.04 / p50 0.28 m, because their ball positions were
        corrected by hand. Jitter stays under a centimetre on both, as on BAR-ATH.
      - **Lost player tracks.** BAR-ATH tracks a median of 20 of the 22 players (p10 15); the
        sample games track all 22. Each slot is lost on 12% of frames, in runs of about 4 s.
      - **The ball is left as it is by default.** The ball is missing on 40.7% of BAR-ATH's graded
        frames in only 150 long gaps (median 268 frames), and the sample games already look like
        that (32-41% missing, 155-169 gaps). `ball_missing` (a target share) and `mean_gap` can
        still add short bursts, to test robustness to a worse tracker.
      - `dead_visible` > 0 (with the game's `labels`) fills that share of the dead-time ball gaps,
        see show_dead_ball.
    """
    ball = data["ball"].copy()
    if dead_visible > 0:
        ball = show_dead_ball(ball, labels, dead_visible, rng)
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
    ball += _smooth_noise(rng, ball.shape, ball_offset)
    players[:, :, :2] += _smooth_noise(rng, players[:, :, :2].shape, player_offset)
    if track_loss > 0:
        players = _lose_tracks(rng, players, track_loss, TRACK_LOSS_RUN)
    return {**data, "ball": ball, "players": players}
