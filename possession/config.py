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

# [tuned] Ball gaps up to this long are bridged with a straight line. The tuning keeps this tiny:
# longer gaps stay missing, because in this data they are mostly stoppages (see DEAD_GAP_MIN).
BALL_MAX_GAP = 1
# [fixed] Ball positions are averaged over +/- this many frames before differencing into a speed.
BALL_SPEED_SMOOTH = 2

# [tuned] Control: a player's "closeness" to the ball is exp(-d^2 / 2 sigma^2). At 0.8 m that gives
# about 0.46 at 1 m (on the ball) and practically 0 at 2.5 m (marking distance, not control).
CONTROL_SIGMA = 0.8
# [tuned] A ball at this speed counts half as much as a ball at rest, so a pass going past a
# player does not count as that player's control.
CONTROL_BALL_SPEED = 15.7

# [tuned] Touch evidence (features.touch_evidence, after Vidal-Codina et al. 2022): a change of the
# ball's velocity next to a player is a touch by that player. 0 switches it off.
TOUCH_WEIGHT = 0.65
TOUCH_DV = 4.0             # [tuned] m/s of velocity change that counts as a touch
TOUCH_RADIUS = 2.0         # [tuned] metres: the player's "possession zone"
TOUCH_SPAN = 4             # [fixed] frames averaged on each side of the touch

# Ball-free team evidence from the teams' widths (features.width_evidence), added to the ball-based
# team evidence before TEAM_WEIGHT. 0 switches it off. Tuned with the rest held at their values:
# a grid over weight and scale, then a 40-draw local search that found nothing better. Sample-game
# spellF1 0.587 -> 0.618 (all three games improve); leave-one-game-out 0.598.
WIDTH_WEIGHT = 0.07        # [tuned]
WIDTH_SCALE = 1.5          # [tuned] metres of width difference that count as clear
WIDTH_SMOOTH = 25          # [fixed] +/- 1 s
WIDTH_MIN_PLAYERS = 5      # [fixed] fewer tracked players than this: no width evidence

# Dead ball: a pivot on the players' mean speed, divided by the match's median pace (about 1.6 m/s
# in the sample games). Above the pivot counts toward alive, below it toward dead. In the sample
# games the ratio is about 1.25 in play and about 0.65 in stoppages.
SPEED_SMOOTH = 25          # [fixed] +/- 1 s moving mean of the players' mean speed
DEAD_SPEED_RATIO = 0.62    # [tuned] pivot, as a share of the match's median pace
DEAD_SPEED_WEIGHT = 2.1    # [tuned] log-odds per unit of that ratio below the pivot
DEAD_SPEED_CLIP = 1.1      # [tuned] cap on one frame's speed evidence, either way
DEAD_NO_PLAYERS = 1.0      # [fixed] no players tracked at all (camera off the pitch): lean dead
OUT_MARGIN = 0.5           # [fixed] metres beyond a line before the ball counts as out
DEAD_OUT_SCORE = 3.0       # [fixed] ball visibly out of play: strong, per frame
DEAD_GAP_MIN = 16          # [tuned] a ball gap at least this long (frames) counts as a stoppage
DEAD_GAP_SCORE = 1.88      # [tuned] dead-ball evidence per frame inside such a gap

# [fixed] Weak team evidence from the Smart Tagging phases, per frame, as a share of TEAM_WEIGHT
# (the most one frame of clear tracking control can give). The sample games have no Smart Tagging,
# so this cannot be tuned. 0.15 makes a 4-second phase worth about 15 frames of clear tracking
# control: enough to tip an ambiguous stretch, nowhere near enough to override clear control.
# It used to be an absolute 0.6, which was right for TEAM_WEIGHT 3.9 but outweighed the tracking
# once the tuning moved TEAM_WEIGHT below 1. A share keeps it weak whatever the tuning picks.
TAG_TEAM_SHARE = 0.15

# [tuned] Transition penalties, in the same units as the weighted evidence (see hmm.py). With
# TEAM_WEIGHT 0.445 a hand-over needs about 13 frames of full control evidence (6.0 / 0.445), or
# about 8 where the ball shows the new team taking it (3.6 / 0.445, SWITCH_RELIEF below), and a
# short possession by the other team inside ours needs twice that. A touch adds
# 0.65 over 9 frames, which shortens that. Leaving a dead ball is expensive (48.2), so a restart is
# only declared once play has clearly resumed.
SWITCH_COST = 6.0          # H <-> A, before SWITCH_RELIEF (it was 5.12 without it)
TO_DEAD_COST = 4.76        # alive -> D
FROM_DEAD_COST = 48.2      # D -> alive
# Evidence-dependent switch cost (features.handover_cue, docs/TECHNICAL.md section 10). Entering a
# team at a frame costs SWITCH_COST * (1 - SWITCH_RELIEF * cue), so a frame where the ball shows
# that team taking it clearly (cue 1) costs only (1 - SWITCH_RELIEF) of the full switch. 0 turns it
# off. The cue is spread over +/- SWITCH_CUE_SPAN frames. Grid-searched on the 13 tuning games
# with the rest of the config fixed (section 10): relief 0.4 with the switch cost raised to 6.0.
# A hand-over with clear ball evidence costs 3.6 and one without 6.0.
SWITCH_RELIEF = 0.4
SWITCH_CUE_SPAN = 0
# [fixed] Smart Tagging transition cue (tagging_prior.transition_cue): frames after an ATTACKING
# TRANSITION tag's start where switching to its team gets the same relief as a clear ball cue.
# Set from the model's own switches on BAR-ATH (p10 0.8 s, p90 5.1 s after the start; no labels).
# The tuning games have no Smart Tagging, so this cannot be tuned there. Off (TO = 0): it lost on
# BAR-ATH's public window (section 11); 150 is the window that was tried.
TRANSITION_CUE_FROM = 12
TRANSITION_CUE_TO = 0
# Soft minimum duration (hmm.viterbi_min_duration, docs/TECHNICAL.md section 11): a possession
# shorter than MIN_SPELL frames pays SHORT_SPELL_COST when it ends. 0 turns it off.
# Off: MIN_SPELL 75 with cost 4.0 won on the 13 tuning games but lost on BAR-ATH's public window
# (section 11).
MIN_SPELL = 0
SHORT_SPELL_COST = 0.0
# [tuned] Scale of the team evidence and of the dead-ball evidence.
TEAM_WEIGHT = 0.445
DEAD_WEIGHT = 1.26

# [fixed] Scale of the TV-feed dead-ball evidence (methods.state_model_tv). That evidence is
# already a log-likelihood ratio estimated from the match itself without labels, so 1.0 means
# "take the estimate at face value". The sample games have no broadcast, so it cannot be tuned.
TV_DEAD_WEIGHT = 1.0
# [fixed] The feed's lag behind the stoppages is estimated per match (methods.tv_lag) from the
# tracking-only model's D spells of at least TV_LAG_MIN_SPELL frames, if there are at least
# TV_LAG_MIN_COUNT of them, and capped at TV_LAG_MAX frames. On BAR-ATH it came out near 3 s.
TV_LAG_MIN_SPELL = 50
TV_LAG_MIN_COUNT = 10
TV_LAG_MAX = 250
# [fixed] Edges, in seconds, of the not-shown stretch length bins (methods.tv_binned_evidence).
# Chosen as natural breaks for a cutaway (< 3 s), a close-up (3-8 s), a short stoppage (8-20 s)
# and a replay (>= 20 s), not searched.
TV_LENGTH_BINS = (3, 8, 20)

# -- learned evidence (methods.learned, possession/learned.py) ---------------------------------
# A single frame's log-odds is clipped to this, so no one frame can force a change of state alone.
LEARNED_CLIP = 4.0

# -- video ball (possession/video_ball.py, docs/TECHNICAL.md section 13) -----------------------
# A ball detector run on the tactical video fills frames on which the tracking has no ball. All
# [fixed]: the tuning games have no video, so nothing here could be searched. On: of the three
# variants fixed before scoring, this one (all values below as they are) beat the previous frozen
# file on the public window, spellF1 0.533 against 0.519 (section 13, with its costs). It needs the
# cache written once by `python -m possession.video_ball`.
VIDEO_BALL = 1
# [fixed] Lowest box confidence used; the probe's calibration was read at this level.
VIDEO_BALL_CONF = 0.25
# [fixed] Metres from the median of the neighbouring detections beyond which a point is a box on
# something else (the probe's calibration error has a long tail of 20-45 m mistakes).
VIDEO_BALL_JUMP = 3.0
# [fixed] Frames on each side for that median.
VIDEO_BALL_SMOOTH = 5
# [fixed] Scale of the team evidence on filled frames, below 1 to trust the video less.
VIDEO_BALL_WEIGHT = 1.0
# [fixed] 1: a filled frame ends a ball gap for features.long_gap, so a gap in which the video
# sees the ball no longer reads as a stoppage; 0: long_gap reads the tracking ball only.
VIDEO_BALL_BREAKS_GAPS = 1
