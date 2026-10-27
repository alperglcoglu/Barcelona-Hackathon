"""The broadcast alignment on synthetic tracking, where the true mapping is known by construction."""

import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from possession import broadcast  # noqa: E402

PITCH = np.array([105.3, 68.0])


def _match(n_frames, rng):
    """A tactical recording: a wandering ball and 22 players drifting after it."""
    # AR(1) velocity: about 4 m/s typical, changing direction over a few seconds
    kicks = rng.normal(0, 0.03, (n_frames, 2))
    velocity = np.zeros((n_frames, 2))
    for f in range(1, n_frames):
        velocity[f] = 0.98 * velocity[f - 1] + kicks[f]
    ball = PITCH / 2 + np.cumsum(velocity, axis=0)
    ball = np.abs((ball + PITCH) % (2 * PITCH) - PITCH)          # bounce inside the pitch
    home = rng.uniform(-20, 20, (11, 2))
    away = rng.uniform(-20, 20, (11, 2))
    base = np.concatenate([home, away])
    wobble = np.cumsum(rng.normal(0, 0.05, (n_frames, 22, 2)), axis=0)
    players = np.clip(ball[:, None, :] + base[None] + wobble, 0, PITCH)
    players = np.concatenate([players, np.full((n_frames, 22, 1), 2.0)], axis=2)
    team = np.array([0] * 11 + [1] * 11, dtype=np.int8)
    ball[rng.random(n_frames) < 0.2] = np.nan                    # the tracker loses the ball
    return {"ball": ball, "players": players, "team": team, "pitch": PITCH}


def _broadcast(tac, pieces, rng, n_slots=26):
    """Concatenate (tactical_start, length, kind) pieces into a broadcast recording.

    kind "live" copies the moment with noise, keeping only players near the ball (the TV camera is
    zoomed in); "closeup" has no tracking at all.
    """
    balls, players, valid = [], [], []
    for start, length, kind in pieces:
        ball = np.full((length, 2), np.nan)
        slots = np.full((length, n_slots, 3), np.nan)
        if kind == "live":
            frames = np.arange(start, start + length)
            ball = tac["ball"][frames] + rng.normal(0, 0.5, (length, 2))
            ball[rng.random(length) < 0.3] = np.nan
            near = np.hypot(*np.moveaxis(tac["players"][frames, :, :2]
                                         - np.nan_to_num(tac["ball"][frames])[:, None], -1, 0)) < 30
            for k in range(22):
                xy = tac["players"][frames, k, :2] + rng.normal(0, 0.8, (length, 2))
                slots[near[:, k], k, :2] = xy[near[:, k]]
                slots[near[:, k], k, 2] = 2.0
        balls.append(ball)
        players.append(slots)
        valid.append(np.full(length, kind == "live"))
    team = np.array([0] * 11 + [1] * 11 + [0, 1, 0, 1], dtype=np.int8)
    return ({"ball": np.concatenate(balls), "players": np.concatenate(players), "team": team,
             "pitch": PITCH}, np.concatenate(valid))


def test_recovers_two_offsets_and_skips_replay_and_closeup():
    rng = np.random.default_rng(1)
    tac = _match(9000, rng)
    pieces = [(1000, 3000, "live"),        # broadcast 0..2999     -> tactical 1000..3999 (+1000)
              (0, 300, "closeup"),         # broadcast 3000..3299  -> no tracking at all
              (3500, 300, "live"),         # broadcast 3300..3599  -> a replay of earlier play
              (5600, 3000, "live")]        # broadcast 3600..6599  -> tactical 5600..8599 (+2000)
    bc, valid = _broadcast(tac, pieces, rng)
    out = broadcast.align(bc, tac, valid=valid, anchor_count=60)

    assert sorted(c for c, _ in out["clusters"]) == [1000, 2000]
    assert not out["mirrored"]

    shown = out["shown"]
    assert shown[1100:3900].mean() > 0.95
    assert shown[5700:8500].mean() > 0.95
    assert shown[4100:5500].mean() == 0              # never on TV
    # the replay (broadcast 3300..3599, tactical 3500..3799) must not create a second mapping
    matched = out["matched"]
    assert matched[3000:3300].mean() == 0             # close-up
    assert matched[3300:3600].mean() == 0             # replay of earlier play


def test_detects_a_mirrored_export():
    rng = np.random.default_rng(2)
    tac = _match(6000, rng)
    bc, valid = _broadcast(tac, [(500, 4000, "live")], rng)
    bc["ball"] = PITCH - bc["ball"]
    bc["players"][:, :, :2] = PITCH - bc["players"][:, :, :2]
    out = broadcast.align(bc, tac, valid=valid, anchor_count=30)
    assert out["mirrored"]
    assert [c for c, _ in out["clusters"]] == [500]
    assert out["shown"][600:4400].mean() > 0.95


def test_cluster_offsets_votes():
    assert broadcast.cluster_offsets([10, 12, 11, 500, 900, 905, 902, 901]) == [(901, 4), (11, 3)]


def test_tv_evidence_is_calibrated_from_model_labels():
    from possession import methods
    labels = np.array(list("HHHHDDDDAAAAX"))
    shown = np.array([1, 1, 1, 0, 0, 0, 0, 1, 1, 1, 1, 1, 0], dtype=bool)
    covered = np.ones(13, dtype=bool)
    covered[-1] = False
    graded = np.ones(13, dtype=bool)
    evidence, (hidden_dead, hidden_alive) = methods.tv_dead_evidence(shown, covered, labels, graded)
    assert hidden_dead == 0.75 and hidden_alive == 0.125
    assert np.isclose(evidence[3], np.log(0.75 / 0.125))      # hidden -> towards dead
    assert np.isclose(evidence[0], np.log(0.25 / 0.875))      # shown -> towards alive
    assert evidence[-1] == 0                                  # no broadcast here: no evidence


def test_coverage_spans_matched_frames_only():
    matched = np.zeros(100, dtype=bool)
    matched[10:40] = True
    offset = np.full(100, broadcast.NO_OFFSET, dtype=np.int64)
    offset[10:40] = 50
    out = {"clusters": [(50, 5)], "matched": matched, "offset": offset}
    covered = broadcast.coverage(out, 200)
    assert covered[60:90].all() and not covered[:60].any() and not covered[90:].any()


def test_tv_lag_and_shift():
    from possession import methods
    n = 4000
    labels = np.full(n, "H")
    hidden = np.zeros(n, dtype=bool)
    for start in range(100, 3800, 300):                     # 13 stoppages of 4 s
        labels[start:start + 100] = "D"
        hidden[start + 75:start + 175] = True                 # the feed leaves 3 s late
    graded = np.ones(n, dtype=bool)
    lag = methods.tv_lag(hidden, labels, graded)
    assert lag == 75
    shifted = methods.shift_earlier(hidden, lag)
    assert shifted[100:200].all() and not shifted[200:400].any()


def test_binned_evidence_weighs_long_stretches_more():
    from possession import methods
    n = 6000
    labels = np.full(n, "H")
    hidden = np.zeros(n, dtype=bool)
    for start in range(0, 6000, 1000):
        hidden[start:start + 50] = True                       # 2 s cutaways in play
        labels[start + 300:start + 900] = "D"
        hidden[start + 300:start + 900] = True                # 24 s stoppages off air
    covered = graded = np.ones(n, dtype=bool)
    evidence, table = methods.tv_binned_evidence(hidden, covered, labels, graded)
    assert evidence[400] > 2 and evidence[10] < 0 and evidence[100] < 0
