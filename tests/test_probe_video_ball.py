"""Tests of the video-ball probe's helpers on small synthetic arrays. No video, no CV packages."""

import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def test_pixels_to_metres_inverts_the_homography():
    from possession import probe_video_ball as probe
    from starter.load_homography import project
    matrix = np.array([[1500.0, 120.0, 210.0], [15.0, 700.0, 300.0], [0.02, 0.3, 1.0]])
    pitch = (105.0, 68.0)
    x, y = np.array([0.0, 30.0, 52.5, 104.0]), np.array([0.0, 10.0, 34.0, 60.0])
    px, py = project(matrix, x / pitch[0], y / pitch[1])
    back_x, back_y = probe.pixels_to_metres(matrix, px, py, pitch)
    assert np.allclose(back_x, x) and np.allclose(back_y, y)


def test_gap_runs_and_length_bins():
    from possession import probe_video_ball as probe
    missing = np.zeros(400, dtype=bool)
    missing[10:15] = True        # 5 frames: < 1 s
    missing[50:150] = True       # 100 frames: 4-10 s (bins are right-closed at the edges)
    missing[200:400] = True      # 200 frames, runs to the end
    assert probe.gap_runs(missing) == [(10, 15), (50, 150), (200, 400)]
    bins = probe.gap_length_bins(missing)
    assert (bins[:10] == -1).all() and (bins[10:15] == 0).all()
    assert (bins[50:150] == 2).all() and (bins[200:] == 2).all()


def test_pick_ball_takes_the_most_confident_ball_box():
    from possession import probe_video_ball as probe
    xyxy = [[0, 0, 10, 10], [100, 100, 110, 120], [200, 200, 204, 204]]
    conf = [0.9, 0.4, 0.6]
    cls = [2, 0, 0]                                        # class 2 is a player, 0 the ball
    x, y, score = probe.pick_ball(xyxy, conf, cls, ball_ids=[0])
    assert (x, y, score) == (202.0, 202.0, 0.6)
    assert probe.pick_ball(xyxy, conf, cls, ball_ids=[5]) is None
    assert probe.pick_ball(np.zeros((0, 4)), [], [], ball_ids=[0]) is None


def test_sample_frames_is_deterministic_and_inside_the_mask():
    from possession import probe_video_ball as probe
    mask = np.zeros(10000, dtype=bool)
    mask[::7] = True
    a, b = probe.sample_frames(mask, 300, seed=3), probe.sample_frames(mask, 300, seed=3)
    assert np.array_equal(a, b) and len(a) == 300 and mask[a].all()
    assert (np.diff(a) > 0).all()
    assert len(probe.sample_frames(mask, 10 ** 6)) == mask.sum()


def test_pick_device_prefers_the_gpu_only_when_asked_for_auto():
    from possession import probe_video_ball as probe
    assert probe.pick_device("auto", True) == "0"
    assert probe.pick_device("auto", False) == "cpu"
    assert probe.pick_device("cpu", True) == "cpu"
    assert probe.pick_device("1", True) == "1"
