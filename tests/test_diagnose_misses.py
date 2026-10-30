"""The miss classes on a hand-made window. No match data needed."""

import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def test_classify_sorts_missed_possessions_by_cause():
    from possession import diagnose_misses as dm
    x, h, a, d = ["X"] * 10, ["H"] * 100, ["A"] * 100, ["D"] * 20
    gt = np.array(x + h + d + h + a + ["H"] * 300 + d + a + x)
    pred = np.array(x + ["H"] * 220 + ["H"] * 100 + ["H"] * 300 + d + ["D"] * 100 + x)
    counts, separators, matched, n_true, n_pred = dm.classify(gt, pred)
    assert (n_true, n_pred) == (5, 1)
    assert matched == 0
    assert counts["absorbed"] == 1                     # the first A spell, shown as H
    assert counts["called dead"] == 1                  # the last A spell, shown as D
    assert counts["merged"] == 3 and separators["dead ball"] >= 1
