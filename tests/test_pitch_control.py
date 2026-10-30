"""Pitch control evidence on small synthetic scenes."""

import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

TEAM = np.array([0, 1], dtype=np.int8)


def _scene(n, ball, home, away):
    players = np.zeros((n, 2, 3))
    players[:, 0, :2], players[:, 1, :2] = home, away
    return np.asarray(ball, dtype=float), players


def test_the_team_closer_in_time_gets_the_evidence():
    from possession import features
    n = 50
    ball = np.tile([50.0, 30.0], (n, 1))
    b, p = _scene(n, ball, (50.5, 30.0), (60.0, 30.0))
    assert (features.pitch_control_evidence(b, p, TEAM)[5:-5] > 0.9).all()
    b, p = _scene(n, ball, (45.0, 30.0), (55.0, 30.0))
    assert np.allclose(features.pitch_control_evidence(b, p, TEAM)[5:-5], 0.0, atol=1e-9)


def test_a_moving_ball_goes_to_the_player_it_runs_to():
    from possession import features
    n = 50
    t = np.arange(n) / 25.0
    ball = np.c_[50.0 + 15.0 * t, np.full(n, 30.0)]          # a pass at 15 m/s towards +x
    home = np.c_[np.full(n, 49.0), np.full(n, 30.0)]          # 1 m behind the ball, now nearest
    away = np.c_[np.full(n, 60.0), np.full(n, 30.0)]          # ahead, where the ball is going
    b, p = _scene(n, ball, home, away)
    assert (features.pitch_control_evidence(b, p, TEAM)[10:-10] < 0).all()


def test_no_ball_or_no_team_gives_no_evidence():
    from possession import features
    n = 20
    ball = np.tile([50.0, 30.0], (n, 1))
    ball[5:10] = np.nan
    b, p = _scene(n, ball, (50.5, 30.0), (60.0, 30.0))
    p[12:, 1, :] = np.nan                                     # the away team is gone
    e = features.pitch_control_evidence(b, p, TEAM)
    assert (e[5:10] == 0).all() and (e[12:] == 0).all()


def test_cue_marks_a_team_that_is_clearly_first_to_a_ball_it_can_reach():
    from possession import config, features
    n = 50
    ball = np.tile([50.0, 30.0], (n, 1))
    b, p = _scene(n, ball, (51.0, 30.0), (60.0, 30.0))        # home 1 m away, away 10 m
    cue = features.pitch_control_cue(b, p, TEAM)
    assert (cue[5:-5, 0] > 0.5).all() and (cue[:, 1] == 0).all()
    b, p = _scene(n, ball, (70.0, 30.0), (80.0, 30.0))        # home first, but 20 m away
    assert (features.pitch_control_cue(b, p, TEAM) == 0).all()
    assert config.PC_CUE_WEIGHT == 0.0                         # off unless switched on
