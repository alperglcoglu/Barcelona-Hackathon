"""Tests of the video ball helpers and its use in the model, on small synthetic arrays.

No video and no CV packages are needed.
"""

import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

PITCH = np.array([105.0, 68.0])
MATRIX = np.array([[1500.0, 120.0, 210.0], [15.0, 700.0, 300.0], [0.02, 0.3, 1.0]])


def _pixels(x, y):
    from starter.load_homography import project
    return project(MATRIX, np.asarray(x) / PITCH[0], np.asarray(y) / PITCH[1])


def test_pixels_to_metres_inverts_the_homography():
    from possession import video_ball as probe
    from starter.load_homography import project
    matrix = np.array([[1500.0, 120.0, 210.0], [15.0, 700.0, 300.0], [0.02, 0.3, 1.0]])
    pitch = (105.0, 68.0)
    x, y = np.array([0.0, 30.0, 52.5, 104.0]), np.array([0.0, 10.0, 34.0, 60.0])
    px, py = project(matrix, x / pitch[0], y / pitch[1])
    back_x, back_y = probe.pixels_to_metres(matrix, px, py, pitch)
    assert np.allclose(back_x, x) and np.allclose(back_y, y)



def test_pick_ball_takes_the_most_confident_ball_box():
    from possession import video_ball as probe
    xyxy = [[0, 0, 10, 10], [100, 100, 110, 120], [200, 200, 204, 204]]
    conf = [0.9, 0.4, 0.6]
    cls = [2, 0, 0]                                        # class 2 is a player, 0 the ball
    x, y, score = probe.pick_ball(xyxy, conf, cls, ball_ids=[0])
    assert (x, y, score) == (202.0, 202.0, 0.6)
    assert probe.pick_ball(xyxy, conf, cls, ball_ids=[5]) is None
    assert probe.pick_ball(np.zeros((0, 4)), [], [], ball_ids=[0]) is None



def test_pick_device_prefers_the_gpu_only_when_asked_for_auto():
    from possession import video_ball as probe
    assert probe.pick_device("auto", True) == "0"
    assert probe.pick_device("auto", False) == "cpu"
    assert probe.pick_device("cpu", True) == "cpu"
    assert probe.pick_device("1", True) == "1"



def test_ball_candidates_keeps_the_most_confident_ball_boxes():
    from possession import video_ball as probe
    xyxy = [[0, 0, 2, 2], [10, 10, 12, 12], [20, 20, 22, 22], [30, 30, 32, 32]]
    conf = [0.3, 0.9, 0.5, 0.95]
    cls = [0, 0, 0, 1]                                     # the last box is not a ball
    c = probe.ball_candidates(xyxy, conf, cls, ball_ids=[0], k=2)
    assert c.shape == (2, 3)
    assert np.allclose(c[:, 2], [0.9, 0.5]) and np.allclose(c[0, :2], [11, 11])
    assert probe.ball_candidates(xyxy, conf, cls, ball_ids=[7]).shape == (0, 3)



def test_select_ball_rules():
    from possession import video_ball as probe
    conf = np.array([0.9, 0.6, 0.4, 0.2])                  # sorted, as ball_candidates returns them
    inside = np.array([False, True, True, True])
    near = np.array([False, False, True, True])
    assert probe.select_ball(conf, inside, near, "top") == 0
    assert probe.select_ball(conf, inside, near, "on-pitch") == 1
    assert probe.select_ball(conf, inside, near, "near-player") == 2
    assert probe.select_ball(conf, inside, near, "near-player", min_conf=0.5) == -1
    assert probe.select_ball(conf[:0], inside[:0], near[:0], "top") == -1


def test_fill_writes_only_missing_frames_and_drops_wrong_boxes():
    from possession import video_ball
    n = 40
    ball = np.full((n, 2), np.nan)
    ball[:5] = (30.0, 30.0)                                 # tracked: must not be overwritten
    players = np.full((n, 2, 3), np.nan)
    matrices = np.repeat(MATRIX[None], n, axis=0)
    found = {}
    for f in range(n):
        x, y = 30.0 + 0.2 * f, 30.0                         # the real ball rolls along
        px, py = _pixels(x, y)
        boxes = [(px, py, 0.6)]
        if f == 20:                                         # a jump to another object
            boxes = [_pixels(80.0, 10.0) + (0.9,)]
        if f in (10, 11):                                   # a confident spare ball off the pitch
            boxes = [_pixels(110.0, 30.0) + (0.95,)] + boxes
        if f == 30:                                         # too unsure
            boxes = [(px, py, 0.1)]
        found[f] = np.array(boxes, dtype=float)
    out, filled = video_ball.fill(ball, players, PITCH, matrices, found, conf=0.25, jump=3.0,
                                  smooth=5)
    assert not filled[:5].any() and np.allclose(out[:5], 30.0)
    assert filled[10] and abs(out[10, 0] - 32.0) < 1e-6          # the on-pitch box, not the spare
    assert not filled[20] and np.isnan(out[20, 0])               # the jump is dropped
    assert not filled[30]
    assert filled[5:20].all() and filled[21:30].all()


def test_dead_evidence_gap_ball_defaults_to_ball():
    from possession import features
    rng = np.random.default_rng(1)
    n = 200
    ball = rng.uniform(10, 60, (n, 2))
    ball[50:120] = np.nan
    players = np.concatenate([rng.uniform(0, 60, (n, 22, 2)), rng.uniform(0, 5, (n, 22, 1))], 2)
    a = features.dead_evidence(ball, players, PITCH)
    assert np.allclose(a, features.dead_evidence(ball, players, PITCH, gap_ball=ball))
    filled = ball.copy()
    filled[50:120] = (30.0, 30.0)
    b = features.dead_evidence(filled, players, PITCH)
    c = features.dead_evidence(filled, players, PITCH, gap_ball=ball)
    assert (c[60:110] > b[60:110]).all()                         # the tracking gap still counts


def test_frame_scores_ignore_video_ball_when_switched_off():
    from possession import config, methods
    rng = np.random.default_rng(2)
    n = 300
    ball = rng.uniform(10, 60, (n, 2))
    ball[100:200] = np.nan
    players = np.concatenate([rng.uniform(0, 60, (n, 22, 2)), rng.uniform(0, 5, (n, 22, 1))], 2)
    data = {"ball": ball, "players": players, "team": np.array([0] * 11 + [1] * 11, np.int8),
            "pitch": PITCH}
    saved = config.VIDEO_BALL
    try:
        config.VIDEO_BALL = 0
        base = methods.frame_scores(data)
        video = ball.copy()
        video[100:200] = players[100:200, 0, :2]                 # the video puts it at a home foot
        data.update(video_ball=video, video_filled=np.isnan(ball[:, 0]))
        assert np.array_equal(base, methods.frame_scores(data))
        config.VIDEO_BALL = 1
        on = methods.frame_scores(data)
    finally:
        config.VIDEO_BALL = saved
    assert on[100:200, 0].mean() > base[100:200, 0].mean()       # home evidence where it was silent
    assert on[120:180, 2].mean() < base[120:180, 2].mean()       # and the gap no longer reads dead


def test_gap_pace_keeps_a_stoppage_with_the_ball_in_view():
    from possession import config, methods
    rng = np.random.default_rng(4)
    n = 400
    ball = rng.uniform(10, 60, (n, 2))
    ball[100:300] = np.nan                                       # one long tracking gap
    players = np.concatenate([rng.uniform(0, 60, (n, 22, 2)), np.full((n, 22, 1), 3.0)], 2)
    players[100:200, :, 2] = 0.5                                 # first half of the gap: walking
    video = ball.copy()
    video[100:300] = (30.0, 30.0)                                # the video sees the ball throughout
    data = {"ball": ball, "players": players, "team": np.array([0] * 11 + [1] * 11, np.int8),
            "pitch": PITCH, "video_ball": video, "video_filled": np.isnan(ball[:, 0])}
    saved = config.VIDEO_BALL, config.VIDEO_BALL_GAP_PACE
    try:
        config.VIDEO_BALL = 1
        config.VIDEO_BALL_GAP_PACE = 0
        plain = methods.frame_scores(data)[:, 2]
        config.VIDEO_BALL_GAP_PACE = 1
        paced = methods.frame_scores(data)[:, 2]
    finally:
        config.VIDEO_BALL, config.VIDEO_BALL_GAP_PACE = saved
    assert (paced[130:170] > plain[130:170]).all()               # walking: the gap still reads dead
    assert np.allclose(paced[230:270], plain[230:270])           # running: the video ends the gap
