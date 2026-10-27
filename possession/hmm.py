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

The penalties are in the same units as the weighted per-frame evidence. With the tuned values in
config.py (SWITCH_COST 5.5, TEAM_WEIGHT 0.445) a hand-over that sticks needs about 12 frames of
the other team clearly controlling a calm ball, or about 6 where the ball shows it taking the ball
(SWITCH_RELIEF). A possession shorter than 3 s also pays SHORT_SPELL_COST
(viterbi_min_duration). A short spell by the other team in the middle of a
possession costs two switches, so it survives only with about twice that. A touch or a deflection
does not.
"""

import numpy as np

H, A, D = 0, 1, 2
LETTERS = np.array(["H", "A", "D"])


def transition_costs(switch, to_dead, from_dead):
    """3x3 matrix, cost[i, j] of moving from state i to state j. Staying is free."""
    return np.array([[0.0, switch, to_dead],
                     [switch, 0.0, to_dead],
                     [from_dead, from_dead, 0.0]])


def viterbi(scores, cost, initial=None, switch_scale=None):
    """Best state path for scores (n_frames, 3) under transition costs (3, 3).

    `switch_scale` (n_frames, 2), optional, multiplies the H <-> A cost of entering H (column 0)
    or A (column 1) at each frame: the evidence-dependent switch cost of methods.decode. A value
    below 1 makes a hand-over cheaper exactly where the ball evidence shows the new team taking it.

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
    scale = None if switch_scale is None else switch_scale.tolist()
    switch = c[0][1]
    for t in range(1, n):
        row = rows[t]
        if scale is not None:
            c[1][0] = switch * scale[t][0]
            c[0][1] = switch * scale[t][1]
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


def viterbi_min_duration(scores, cost, min_len, short_cost, switch_scale=None):
    """viterbi() plus a soft minimum duration: a possession (H or A spell) shorter than min_len
    frames pays short_cost once, when it ends. A spell still running at the end of the half pays
    nothing. D spells are unchanged.

    This is the duration part of a semi-Markov model, kept to what the errors call for: on the
    tuning games most predicted possessions under 4 s are wrong (docs/TECHNICAL.md section 11),
    while a hard minimum would also delete the short true ones the evidence does show.

    Exact, and still linear in the number of frames. Each team gets a "young" state (spell shorter
    than min_len so far) next to the usual "mature" one. A young spell's value at frame t is
    K[s] + C[t], where s is the frame it started, C the running sum of that team's scores and
    K[s] = (value on entering at s) - C[s]. So the best young spell to leave from is a maximum of
    K over the last min_len - 1 start frames, kept with a monotonic queue, and a young spell turns
    mature when it reaches min_len frames. The path is rebuilt from the start frames.
    """
    from collections import deque

    n = len(scores)
    if n == 0:
        return np.zeros(0, dtype=np.int8)
    w = max(1, int(min_len) - 1)          # young spells span 1 .. w frames
    rows = scores.tolist()
    c = cost.tolist()
    switch = c[0][1]
    scale = None if switch_scale is None else switch_scale.tolist()
    neg = -1e18

    # per team X in (0, 1): mature value, running sum C, keys K, and the window queue
    mature = [rows[0][0], rows[0][1]]
    dead = rows[0][2]
    csum = [[0.0] * n, [0.0] * n]
    keys = [[neg] * n, [neg] * n]
    queue = [deque(), deque()]
    csum[0][0], csum[1][0] = rows[0][0], rows[0][1]
    # back pointers
    m_from = [[0] * n, [0] * n]          # mature at t: 0 = mature at t-1, 1 = young started t-w
    enter_from = [[0] * n, [0] * n]      # young started at t: 0 = other mature, 1 = other young, 2 = D
    enter_start = [[0] * n, [0] * n]     # ... and the start frame of that other young spell
    d_from = [0] * n                     # D at t: 0 = D, 1 = H mature, 2 = H young, 3 = A mature, 4 = A young
    d_start = [0] * n

    for t in range(1, n):
        row = rows[t]
        if scale is not None:
            into = (switch * scale[t][0], switch * scale[t][1])
        else:
            into = (switch, switch)
        # leave values at t-1 per team: best of mature and young (young pays short_cost)
        exits = []
        for x in (0, 1):
            q = queue[x]
            while q and q[0] < t - w:
                q.popleft()
            young_best = keys[x][q[0]] + csum[x][t - 1] - short_cost if q else neg
            if young_best > mature[x]:
                exits.append((young_best, 1, q[0]))
            else:
                exits.append((mature[x], 0, 0))
        new_mature = [0.0, 0.0]
        for x in (0, 1):
            o = 1 - x
            # entering x at t
            from_other = exits[o][0] - into[x]
            from_dead = dead - c[2][x]
            if from_other >= from_dead:
                enter, enter_from[x][t], enter_start[x][t] = from_other, exits[o][1], exits[o][2]
            else:
                enter, enter_from[x][t] = from_dead, 2
            enter += row[x]
            csum[x][t] = csum[x][t - 1] + row[x]
            keys[x][t] = enter - csum[x][t]
            # mature at t: stay mature, or a young spell that started t - w reaches min_len frames
            s = t - w
            grown = keys[x][s] + csum[x][t - 1] if s >= 0 and keys[x][s] > neg / 2 else neg
            if grown > mature[x]:
                new_mature[x], m_from[x][t] = grown + row[x], 1
            else:
                new_mature[x], m_from[x][t] = mature[x] + row[x], 0
        # D at t
        best, d_from[t] = dead, 0
        for x, base in ((0, 1), (1, 3)):
            value = exits[x][0] - c[x][2]
            if value > best:
                best, d_from[t], d_start[t] = value, base + exits[x][1], exits[x][2]
        new_dead = best + row[2]
        # push today's young starts into the queues (after reading them for t's exits)
        for x in (0, 1):
            q = queue[x]
            while q and keys[x][q[-1]] <= keys[x][t]:
                q.pop()
            q.append(t)
        mature, dead = new_mature, new_dead

    # final state: mature, young (no penalty at the end) or D
    t = n - 1
    candidates = [(mature[0], ("m", 0, 0)), (mature[1], ("m", 1, 0)), (dead, ("d", 2, 0))]
    for x in (0, 1):
        q = queue[x]
        while q and q[0] < t - w + 1:
            q.popleft()
        if q:
            candidates.append((keys[x][q[0]] + csum[x][t], ("y", x, q[0])))
    state = max(candidates, key=lambda v: v[0])[1]

    path = np.zeros(n, dtype=np.int8)
    while t >= 0:
        kind, x, start = state
        if kind == "y":
            path[start:t + 1] = x
            if start == 0:
                break
            src = enter_from[x][start]
            t = start - 1
            state = (("m", 1 - x, 0) if src == 0 else ("y", 1 - x, enter_start[x][start])
                     if src == 1 else ("d", 2, 0))
        elif kind == "m":
            path[t] = x
            if t == 0:
                break
            if m_from[x][t] == 1:
                start = t - w
                path[start:t] = x
                state = ("y", x, start)
                t -= 1
                # the young spell covers start .. t; continue from its entry
                kind, x, start = state
                path[start:t + 1] = x
                if start == 0:
                    break
                src = enter_from[x][start]
                t = start - 1
                state = (("m", 1 - x, 0) if src == 0 else ("y", 1 - x, enter_start[x][start])
                         if src == 1 else ("d", 2, 0))
            else:
                t -= 1
        else:
            path[t] = 2
            if t == 0:
                break
            src = d_from[t]
            t -= 1
            state = {0: ("d", 2, 0), 1: ("m", 0, 0), 2: ("y", 0, d_start[t + 1]),
                     3: ("m", 1, 0), 4: ("y", 1, d_start[t + 1])}[src]
    return path
