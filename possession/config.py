"""Every tunable number in the pipeline, with the reason for its value.

Keeping them here, rather than as defaults scattered across functions, is what makes the technical
document honest: a reader can see in one file what was chosen, and the rules require that nothing
here is fitted to the public five minutes of BAR-ATH (RULES.md, section 2). Values are set from
football reasoning or tuned on *other* games; the comment on each one says which.
"""

import os

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)

# Params
# Parsed tracking is cached here as .npz so the slow text parse runs once. It lives under data/,
# which is gitignored, because a frame-level export of the tracking is derived data the terms do
# not let us publish (RULES.md, "Do not publish derived data that would reconstruct it").
CACHE_DIR = os.path.join(ROOT, "data", "cache")

# Where submissions are written by default. evaluation/results/*.csv is gitignored apart from the
# organisers' baselines, so nothing we write there is committed by accident.
RESULTS_DIR = os.path.join(ROOT, "evaluation", "results")

# The team name the submission form asks the file to be named after (SUBMISSION.md). Change it
# once the team is registered.
TEAM_NAME = "team"

# -- baseline (the organisers' reference, reproduced for a parity check) ----------------------
# Half-width of the majority-vote window, in frames. 50 is the published floor, baseline_smooth50,
# spellF1 0.2433 on the full match. It is here only so `--method baseline` regenerates that exact
# file; it is not a value we chose.
BASELINE_WINDOW = 50

# -- state model (methods.state_model) ----------------------------------------------------------
# These are starting values chosen from football reasoning. Step 3 of the plan tunes them on
# *other* games; nothing here is fitted to BAR-ATH. Distances are metres, speeds m/s, times frames.

# Ball gaps up to this long are bridged with a straight line. 12 frames is about half a second, too
# short for the ball to change hands unseen in most situations.
BALL_MAX_GAP = 12
# Ball positions are averaged over +/- this many frames before differencing them into a speed.
BALL_SPEED_SMOOTH = 2

# Control: a player's "closeness" to the ball is exp(-d^2 / 2 sigma^2). 1.2 m gives about 0.7 at
# 1 m (on the ball) and about 0.06 at 3 m (marking distance, not control).
CONTROL_SIGMA = 1.2
# A ball moving at this speed counts half as much as a ball at rest. 8 m/s is a firm pass. A ball
# going past at 20 m/s counts about 14% as much, which keeps deflections from flipping possession.
CONTROL_BALL_SPEED = 8.0

# Dead ball: a pivot on the team's mean speed. Above the pivot counts toward alive, below it toward
# dead. During open play outfield players average roughly 2-3 m/s, while walking during a stoppage
# is about 1 m/s.
SPEED_SMOOTH = 25          # +/- 1 s moving mean of the players' mean speed
DEAD_SPEED_PIVOT = 1.6     # m/s
DEAD_SPEED_WEIGHT = 0.6    # log-odds per m/s below the pivot
DEAD_SPEED_CLIP = 1.0      # one frame of slow players is never more than 1 unit of evidence
DEAD_NO_PLAYERS = 1.0      # no players tracked at all (camera off the pitch): lean dead
OUT_MARGIN = 0.5           # metres beyond a line before the ball counts as out (tracking noise)
DEAD_OUT_SCORE = 3.0       # ball visibly out of play: strong, per frame

# Weak team evidence from the Smart Tagging phases, per frame, in the same units as the tracking
# evidence. 0.15 means a 4-second phase is worth 15 frames of clear tracking, enough to tip an
# ambiguous stretch and nowhere near enough to override clear control.
TAG_TEAM_WEIGHT = 0.15

# Transition penalties, in frames of full evidence (see hmm.py).
# A possession by the other team in the middle of ours needs two switches, so it is only kept when
# it brings more than 2 x SWITCH_COST of evidence: 20 frames (0.8 s) of clear control at 10.
SWITCH_COST = 10.0         # H <-> A
TO_DEAD_COST = 25.0        # alive -> D
FROM_DEAD_COST = 25.0      # D -> alive
# Scale of the team evidence relative to the dead evidence. 1.0 means one frame of clear control
# weighs the same as one frame of clearly slow players.
TEAM_WEIGHT = 1.0

# -- learned evidence (methods.learned, possession/learned.py) ---------------------------------
# A single frame's log-odds is clipped to this, so no one frame can force a change of state alone.
LEARNED_CLIP = 4.0
# Scale of the dead-ball log-odds relative to the team log-odds (TEAM_WEIGHT scales the latter).
DEAD_WEIGHT = 1.0
