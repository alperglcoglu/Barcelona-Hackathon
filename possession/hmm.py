"""The state model: turn noisy per-frame evidence into a small number of clean possessions.

Three states, H, A and D. Every frame contributes a score to each state (from features.py), and
every change of state costs a fixed penalty. Viterbi finds the single label sequence with the
highest total, which is the sequence that explains the evidence best while changing state as
little as possible.

Why this and not a smoothing window:

  - **A window delays every transition it smooths.** A majority vote over +/- 2 s only flips once
    the new team has half the window. Viterbi puts each change exactly where the evidence for the
    new state starts to outweigh the old, because it sees the whole sequence. spellF1 punishes
    late transitions hard: one second late costs about 0.35.
  - **A penalty is a minimum amount of evidence, not a minimum duration.** A short but unambiguous
    spell (a player clearly on the ball for 20 frames) can still win, and a long run of weak or
    contested frames cannot. A fixed window cannot tell those two apart.
  - **No evidence means no change.** Frames with zero evidence (missing ball, ball in flight) add
    nothing to any state, so the model simply stays where it was. That is the challenge's own
    rule for loose and contested balls.

The penalties are in units of "frames of full evidence". With SWITCH = 10, a hand-over that sticks
needs the equivalent of 10 frames (0.4 s) of the other team clearly controlling a calm ball. A
short spell by the other team in the middle of a possession costs two switches, so it survives
only with more than 2 x SWITCH (20 frames, 0.8 s) of evidence. A touch or a deflection does not.
"""

import numpy as np

H, A, D = 0, 1, 2
LETTERS = np.array(["H", "A", "D"])


def transition_costs(switch, to_dead, from_dead):
    """3x3 matrix, cost[i, j] of moving from state i to state j. Staying is free."""
    return np.array([[0.0, switch, to_dead],
                     [switch, 0.0, to_dead],
                     [from_dead, from_dead, 0.0]])


def viterbi(scores, cost, initial=None):
    """Best state path for scores (n_frames, 3) under transition costs (3, 3).

    Plain Python over lists: 3 states make the inner work tiny, and this is several times faster
    than per-frame numpy calls. About a second per half.
    """
    n = len(scores)
    if n == 0:
        return np.zeros(0, dtype=np.int8)
    rows = scores.tolist()
    c = cost.tolist()
    total = list(rows[0]) if initial is None else [rows[0][k] - initial[k] for k in range(3)]
    back_rows = [[0, 0, 0]] * n
    for t in range(1, n):
        row = rows[t]
        new = [0.0, 0.0, 0.0]
        pointers = [0, 0, 0]
        for j in range(3):
            best_i, best = 0, total[0] - c[0][j]
            for i in (1, 2):
                value = total[i] - c[i][j]
                if value > best:
                    best_i, best = i, value
            new[j] = best + row[j]
            pointers[j] = best_i
        total = new
        back_rows[t] = pointers
    back = np.array(back_rows, dtype=np.int8)

    path = np.zeros(n, dtype=np.int8)
    path[-1] = int(np.argmax(total))
    for t in range(n - 1, 0, -1):
        path[t - 1] = back[t, path[t]]
    return path
