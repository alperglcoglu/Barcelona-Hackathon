"""Tests of the video-ball probe's helpers on small synthetic arrays. No video, no CV packages."""

import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


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


def test_sample_frames_is_deterministic_and_inside_the_mask():
    from possession import probe_video_ball as probe
    mask = np.zeros(10000, dtype=bool)
    mask[::7] = True
    a, b = probe.sample_frames(mask, 300, seed=3), probe.sample_frames(mask, 300, seed=3)
    assert np.array_equal(a, b) and len(a) == 300 and mask[a].all()
    assert (np.diff(a) > 0).all()
    assert len(probe.sample_frames(mask, 10 ** 6)) == mask.sum()
