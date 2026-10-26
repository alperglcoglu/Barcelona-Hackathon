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
# Values marked [tuned] come from tuning/search.py on Metrica's public sample games (other matches,
# degraded to look like automatic tracking). Its best configuration over all three games is used
# here, rounded. Nothing is fitted to BAR-ATH. Values marked [fixed] were set by reasoning and not
# searched. Distances are metres, speeds m/s, times frames.

# [tuned] Ball gaps up to this long are bridged with a straight line (4 frames is 0.16 s). Longer
# gaps stay missing: in this data they are mostly stoppages (see DEAD_GAP_MIN).
BALL_MAX_GAP = 4
# [fixed] Ball positions are averaged over +/- this many frames before differencing into a speed.
BALL_SPEED_SMOOTH = 2

# [tuned] Control: a player's "closeness" to the ball is exp(-d^2 / 2 sigma^2). At 0.74 m that gives
# about 0.4 at 1 m (on the ball) and practically 0 at 2.5 m (marking distance, not control).
CONTROL_SIGMA = 0.74
# [tuned] A ball at this speed counts half as much as a ball at rest. The tuned value is high, so
# only very fast balls (long passes, shots) are discounted much.
CONTROL_BALL_SPEED = 26.0

# Dead ball: a pivot on the players' mean speed. Above the pivot counts toward alive, below it
# toward dead. In the sample games the median is about 2 m/s in play and about 1 m/s in stoppages.
SPEED_SMOOTH = 25          # [fixed] +/- 1 s moving mean of the players' mean speed
DEAD_SPEED_PIVOT = 1.03    # [tuned] m/s
DEAD_SPEED_WEIGHT = 1.43   # [tuned] log-odds per m/s below the pivot
DEAD_SPEED_CLIP = 0.89     # [tuned] cap on one frame's speed evidence, either way
DEAD_NO_PLAYERS = 1.0      # [fixed] no players tracked at all (camera off the pitch): lean dead
OUT_MARGIN = 0.5           # [fixed] metres beyond a line before the ball counts as out
DEAD_OUT_SCORE = 3.0       # [fixed] ball visibly out of play: strong, per frame
DEAD_GAP_MIN = 30          # [tuned] a ball gap at least this long (frames) counts as a stoppage
DEAD_GAP_SCORE = 2.94      # [tuned] dead-ball evidence per frame inside such a gap

# [fixed] Weak team evidence from the Smart Tagging phases, per frame, added after TEAM_WEIGHT. The
# sample games have no Smart Tagging, so this cannot be tuned. 0.6 makes a 4-second phase worth
# about 15 frames of clear tracking control: enough to tip an ambiguous stretch, nowhere near enough
# to override clear control.
TAG_TEAM_WEIGHT = 0.6

# [tuned] Transition penalties, in the same units as the weighted evidence (see hmm.py). With
# TEAM_WEIGHT 0.32 a hand-over needs about 9 frames of full control evidence (2.74 / 0.32), and a
# short possession by the other team inside ours needs about 17 (2 x 2.74 / 0.32). Leaving a dead
# ball is expensive (22.0), so a restart is only declared once play has clearly resumed.
SWITCH_COST = 2.74         # H <-> A
TO_DEAD_COST = 4.54        # alive -> D
FROM_DEAD_COST = 22.0      # D -> alive
# [tuned] Scale of the team evidence and of the dead-ball evidence.
TEAM_WEIGHT = 0.32
DEAD_WEIGHT = 0.72

# -- learned evidence (methods.learned, possession/learned.py) ---------------------------------
# A single frame's log-odds is clipped to this, so no one frame can force a change of state alone.
LEARNED_CLIP = 4.0
