"""Tests that run without the gated match data, on a small synthetic ATD in the real file format.

    python -m pytest tests

What they pin down is the part that is expensive to get wrong silently: frame alignment
(frame_id 1 -> row 0), the metre conversion, missing-value handling, and that the vectorised
baseline agrees with a plain per-frame loop written the way starter/baseline.py writes it.
"""

import os
import sys

import numpy as np
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from possession import io_atd, methods, submission  # noqa: E402

PITCH = (105.3, 68.0)
META = """<?xml version="1.0"?>
<MainMeta><Metadata><Teams>
  <Team id="T1"><Name>Home FC</Name></Team>
  <Team id="T2"><Name>Away FC</Name></Team>
</Teams><Players>
  <Player id="P1" teamId="T1"/><Player id="P2" teamId="T2"/><Player id="P3" teamId="T1"/>
</Players></Metadata></MainMeta>
"""


def _line(frame_id, players, ball):
    group = lambda p: "NaN,NaN,NaN" if p is None else "{},{},{}".format(*p)  # noqa: E731
    return "{}:{}:{}\n".format(frame_id, ";".join(group(p) for p in players), group(ball))


@pytest.fixture
def atd(tmp_path):
    rng = np.random.default_rng(0)
    lines = []
    for frame_id in range(1, 201):
        players = [tuple(rng.random(2)) + (1.0,) for _ in range(3)]
        if frame_id % 7 == 0:
            players[1] = None                     # a slot the tracker lost
        ball = None if frame_id % 5 == 0 else tuple(rng.random(2)) + (0.0,)
        lines.append(_line(frame_id, players, ball))
    lines.append("garbage line without separators\n")
    raw = tmp_path / "atd.txt"
    raw.write_text("".join(lines))
    meta = tmp_path / "atd.xml"
    meta.write_text(META)
    return raw, meta, lines


def _reference_nearest(lines, team, n_frames):
    """The per-frame loop from starter/baseline.py, restated independently for comparison."""
    out = np.full(n_frames, methods.NOBODY, dtype=np.int8)
    for line in lines:
        parts = line.split(":")
        if len(parts) != 3 or parts[2][0] == "N":
            continue
        bx, by = (float(v) for v in parts[2].split(",")[:2])
        best, best_team = None, methods.NOBODY
        for slot, group in enumerate(parts[1].split(";")):
            if group[0] == "N":
                continue
            x, y = (float(v) for v in group.split(",")[:2])
            d = ((x - bx) * PITCH[0]) ** 2 + ((y - by) * PITCH[1]) ** 2
            if best is None or d < best:
                best, best_team = d, team[slot]
        out[int(parts[0]) - 1] = best_team
    return out


def test_team_slots(atd):
    _, meta, _ = atd
    assert io_atd.team_of_each_track(str(meta)).tolist() == [0, 1, 0]


def test_parse_alignment_and_units(atd):
    raw, _, lines = atd
    ball, players = io_atd.parse_raw(str(raw), 205, 3, PITCH)
    first = lines[0].split(":")
    x, y = (float(v) for v in first[2].split(",")[:2])
    assert ball[0] == pytest.approx([x * PITCH[0], y * PITCH[1]])   # frame_id 1 -> row 0
    assert np.isnan(ball[4]).all()                                   # frame_id 5: no ball
    assert np.isnan(players[6, 1]).all()                             # frame_id 7: slot lost
    assert np.isnan(ball[200:]).all()                                # rows past the file stay NaN


def test_baseline_matches_reference_loop(atd):
    raw, meta, lines = atd
    team = io_atd.team_of_each_track(str(meta))
    ball, players = io_atd.parse_raw(str(raw), 205, 3, PITCH)
    ours = methods.nearest_team(ball, players, team)
    assert np.array_equal(ours, _reference_nearest(lines[:-1], team, 205))


def test_majority_smooth():
    codes = np.array([0, 0, 1, 0, 0, -1, -1, -1, 1, 1], dtype=np.int8)
    assert "".join(methods.majority_smooth(codes, 0)) == "HHAHHDDDAA"
    assert "".join(methods.majority_smooth(codes, 1)) == "HHHHHHDAAA"


def test_submission_roundtrip(tmp_path):
    labels = np.array(list("XHHADX"))
    assert submission.check(labels, 6) is None
    assert submission.check(labels, 7) is not None
    assert submission.check(np.array(list("XHQ")), 3) is not None
    path = submission.write(str(tmp_path / "s.csv"), labels)
    assert open(path).readline() == "frame,label\n"
    assert np.array_equal(submission.read(path), labels)


# -- state model -----------------------------------------------------------------------------

from possession import features, hmm, tagging_prior  # noqa: E402

# Fixed costs, so these tests pin down the decoder's behaviour rather than the tuned values.
COST = hmm.transition_costs(10.0, 25.0, 25.0)


def _scores(team, dead):
    team, dead = np.asarray(team, float), np.asarray(dead, float)
    return np.stack([team / 2, -team / 2, dead], axis=1)


def _decode(team, dead):
    return "".join(hmm.LETTERS[hmm.viterbi(_scores(team, dead), COST)])


def test_viterbi_ignores_flicker_and_places_switch_on_time():
    team = np.r_[np.ones(100), -np.ones(100)]
    team[40:45] = -1                                  # 5 frames of the other team: a touch
    out = _decode(team, -np.ones(200))
    assert out == "H" * 100 + "A" * 100


def test_viterbi_keeps_short_clear_possession():
    team = np.r_[np.ones(100), -np.ones(30), np.ones(100)]
    out = _decode(team, -np.ones(230))
    assert out == "H" * 100 + "A" * 30 + "H" * 100


def test_no_evidence_keeps_last_possessor():
    team = np.r_[np.ones(50), np.zeros(300), -np.ones(50)]   # ball lost / in flight in between
    out = _decode(team, -np.ones(400))
    assert out[:50] == "H" * 50 and out[-40:] == "A" * 40 and set(out[50:300]) == {"H"}


def test_dead_spell():
    team = np.r_[np.ones(100), np.zeros(200), -np.ones(100)]
    dead = np.r_[-np.ones(100), np.ones(200), -np.ones(100)]
    out = _decode(team, dead)
    assert out == "H" * 100 + "D" * 200 + "A" * 100


def test_interpolate_ball_bridges_short_gaps_only():
    ball = np.full((40, 2), np.nan)
    ball[0], ball[5], ball[30] = (0, 0), (5, 10), (5, 10)
    out = features.interpolate_ball(ball, max_gap=12)
    assert out[2] == pytest.approx([2, 4])           # 4-frame gap filled linearly
    assert np.isnan(out[10]).all()                    # 24-frame gap left missing
    assert np.isnan(out[35]).all()                    # after the last sighting: no extrapolation


def test_team_evidence_sign_and_flight():
    team = np.array([0, 1], dtype=np.int8)
    ball = np.array([[10.0, 10.0], [10.0, 10.0], [50.0, 30.0], [np.nan, np.nan]])
    players = np.array([
        [[10.3, 10.0, 1.0], [14.0, 10.0, 1.0]],      # home on the ball
        [[14.0, 10.0, 1.0], [10.2, 10.0, 1.0]],      # away on the ball
        [[10.0, 10.0, 1.0], [80.0, 10.0, 1.0]],      # ball far from everyone
        [[10.0, 10.0, 1.0], [80.0, 10.0, 1.0]],      # ball missing
    ])
    e = features.team_evidence(ball, players, team)
    assert e[0] > 0.5 and e[1] < -0.5 and abs(e[2]) < 1e-6 and e[3] == 0


def test_dead_evidence_out_of_play_and_slow_players():
    pitch = np.array(PITCH)
    n = 60
    ball = np.tile([50.0, 30.0], (n, 1))
    ball[50:] = (-3.0, 30.0)                          # over the goal line
    players = np.zeros((n, 4, 3))
    players[:, :, 2] = np.r_[np.full(30, 4.0), np.full(30, 0.5)][:, None]
    g = features.dead_evidence(ball, players, pitch)
    assert g[5] < 0 < g[45] < g[55]


def test_tagging_prior_signs():
    events = [{"code": "BUILD UP", "side": "H", "start_frame": 0, "end_frame": 5},
              {"code": "DEFENDING IN MIDDLE THIRD", "side": "H", "start_frame": 5, "end_frame": 10},
              {"code": "SET PIECES", "side": None, "start_frame": 10, "end_frame": 12}]
    prior = tagging_prior.team_prior(events, 12, weight=1.0)
    assert (prior[:5] == 1).all() and (prior[5:10] == -1).all() and (prior[10:] == 0).all()
