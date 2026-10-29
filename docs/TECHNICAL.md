# Technical document

Frame-level ball possession for More than a Hack 2027 (FC Barcelona vs Athletic Club). The
section headings follow what SUBMISSION.md asks the document to cover.

## 1. Approach

### 1.1 What the metric rewards, and what that implies for the design

`spellF1` treats each possession (a maximal run of `H` or `A`) as a single object. It matches
predicted possessions to true ones one-to-one by temporal IoU and averages the resulting F1 over
IoU thresholds 0.3 to 0.7. Three consequences shape everything below:

1. **Fragmentation is the main failure mode.** One spurious frame of the other team inside a real
   possession splits it in two. Both halves then have a lower IoU with the truth, and the extra
   piece is a false positive. The raw nearest-player rule scores macroF1 0.65 but spellF1 0.05 for
   exactly this reason.
2. **Timing matters to about a third of a second.** A transition one frame late costs nothing,
   eight frames late costs about 0.11, and one second late costs about 0.35 (EVALUATION.md).
3. **Abstaining (`X`) means "repeat the previous label".** It is free inside a possession and
   expensive across a hand-over. Our model never abstains on a graded frame: where it has no
   evidence it keeps the previous state, which is what `X` would mean anyway, and macroF1 does not
   charge it as a miss.

So the problem is treated as **segmentation**, not per-frame classification. The model decides
where possessions start and end, and every change of state has a price.

### 1.2 The model: evidence per frame, decoded by a three-state Viterbi

```
tracking ──► per-frame evidence ──► 3-state Viterbi (H, A, D) ──► labels
              team: who controls        each half decoded
              dead: is play stopped     separately; a change of
              (+ weak Smart Tagging)    state costs a penalty
```

**Evidence of control** (`features.team_evidence`). For each team, the distance from the ball to
its nearest tracked player gives a closeness score `exp(-d² / 2σ²)`. The evidence is home closeness
minus away closeness, in [-1, 1]. It behaves correctly in the three situations that matter:

| situation | evidence |
|---|---|
| one player on the ball, nobody else near | close to ±1 (control) |
| players of both teams on the ball | close to 0 (a duel decides nothing yet) |
| nobody near the ball (a pass in flight) | 0 (the last possessor keeps it) |

A fast ball is discounted by `1 / (1 + (v / v₀)²)`. A ball passing a player at 20 m/s is a pass
going by or a deflection, not control, and CHALLENGE.md says neither changes possession.

**Evidence of a touch** (`features.touch_evidence`). Distance alone cannot tell a ball rolling past
a player from one the player stops or redirects. The possession-zone algorithm of Vidal-Codina et
al. (2022), as described in the DataBallPy documentation, uses exactly that difference: a player
gains the ball when it is inside their zone and its velocity changes. We take the idea, not the
library. The ball velocity is averaged over 4 frames before and after each frame. A touch is a
local maximum of the change `|v_after - v_before|` above `TOUCH_DV` = 4 m/s, and it goes to the
team whose nearest player is within `TOUCH_RADIUS` = 2 m and closer than the other team's. Each
touch adds `TOUCH_WEIGHT` = 0.65 of team evidence over ±4 frames. No touch is read next to a ball
gap, because a velocity averaged across a gap is not a velocity. On the degraded sample games the
touch evidence is worth +0.04 spellF1 with every other parameter held fixed (section 3).

**Evidence of a dead ball** (`features.dead_evidence`) is a log-odds score from three signals that
cover each other's blind spots:

- *The ball is visibly beyond a line* (with a 0.5 m margin for tracking noise). Decisive, but the
  tracking often loses the ball exactly when it goes out.
- *The players have stopped running.* During a stoppage twenty-two players walk. The mean speed of
  all tracked players, averaged over ±1 s, is divided by the match's own median pace (over frames
  with at least 15 players tracked, so warm-up and half-time do not count) and enters linearly
  around a pivot ratio, `DEAD_SPEED_RATIO` = 0.62. This signal never needs the ball, which
  matters because the ball is missing on 41% of graded frames. The ratio is what transfers
  between matches. BAR-ATH runs faster than the sample games: in its public window, used only as a
  sanity check, the players' median speed is 2.27 m/s in play and 1.19 m/s in stoppages, against
  about 2.0 and 1.0 in the sample games. An absolute pivot of 1.0 m/s, tuned on the sample games,
  sat below most BAR-ATH stoppages, so it voted "alive" during them. A relative pivot adapts to
  each match's pace without using its labels. Re-running the local search with the ratio free
  picked 0.63, the same value, and gained nothing measurable (0.590 against 0.587), so the
  previous tuning is kept.
- *The ball has been missing for a long time* (`features.long_gap`). Every frame inside a run of
  at least `DEAD_GAP_MIN` = 16 missing frames (0.6 s) gets `DEAD_GAP_SCORE` of dead evidence. The
  reason is how the automatic tracking loses the ball. On the whole BAR-ATH match (aggregate
  statistics only, no labels, `possession/diagnose.py`) the ball is missing on 40.7% of graded
  frames but in only 150 gaps, with a median length of 268 frames (10.7 s). These are not short
  occlusions in open play. They are mostly the ball leaving the camera's view or being picked up
  for a restart. In the public five minutes, used here only as a sanity check, the ball is missing
  on 6.5% of in-play frames and on 33.6% of dead-ball frames. Metrica's sample games show the same
  pattern naturally (32-41% missing, 155-169 gaps, median 127-196 frames).

**Missing ball** (`features.interpolate_ball`). Gaps up to `BALL_MAX_GAP` = 1 frame are bridged
with a straight line, because the ball cannot travel far unseen in that time. Longer gaps stay
missing and contribute no team evidence, so the team state holds through them, while the long-gap
signal above pushes toward `D`. The search preferred a short bridge: bridging long gaps would
invent ball positions exactly where the ball is out of play.

**Smart Tagging** (`tagging_prior.py`) adds weak team evidence: an attacking phase says its team
has the ball, and a defending phase says the other team does. Its weight is small, so a phase can
tip an ambiguous stretch but never overrides clear tracking. Throw-ins are tagged `N/A` and add
nothing. The weight is set as a share of `TEAM_WEIGHT` (`TAG_TEAM_SHARE` = 0.15), not as an
absolute number. The sample games have no Smart Tagging, so the tuning cannot see it. An absolute
weight that was weak next to one tracking scale became stronger than the tracking itself when the
tuning chose a smaller `TEAM_WEIGHT`, and a noisy phase then decided whole possessions.

**Decoding** (`hmm.py`). Each frame gives a score to each of H, A and D (team evidence split ±½,
dead log-odds to D). Each change of state costs a penalty: `SWITCH` for H↔A, and `TO_DEAD` and
`FROM_DEAD` for entering and leaving a stoppage. Viterbi finds the label sequence with the highest
total score. This is the core design decision, for three reasons:

- *No delay.* A centred smoothing window flips only once the new team holds half the window, so
  every transition comes late. Viterbi sees the whole half and places each change where the
  evidence for the new state starts to outweigh the old one.
- *A penalty is a minimum amount of evidence, not a minimum duration.* A short spell with clear
  control survives. A long run of weak, contested frames does not. A fixed window cannot tell the
  two apart. A short possession by the other team inside ours costs two switches, so it survives
  only with more than `2 × SWITCH` of evidence. A touch or a deflection does not get there.
- *No evidence means no change.* That is the challenge's own rule for loose and contested balls.

The halves are decoded separately, and frames outside the halves are written as `X`.

### 1.3 Alternative tried: learned evidence

`learned.py` replaces the hand-written evidence with two logistic regressions, trained on other
games (section 3). They add collective features that never need the ball: team spread, centroid
velocity along the attacking direction (read off the data per half), and the speed difference
between the teams. With penalties tuned for it, it reached a mean spellF1 of about 0.57 under the
earlier degradation, below the hand-written evidence. Under the current degradation and the
current penalties, leave-one-game-out gives 0.466, against 0.589 for the hand-written evidence.
The collective features do carry signal: the possessing team is more spread out, and the defending team runs faster. But the event-derived labels are coarse, and in the
sample data "ball missing" mostly means "ball out of play", which a learned model exploits and
which would not transfer to BAR-ATH, so we removed it as a feature. We keep the hand-written
evidence as the tracking-only model (`--method state_model`) and `--method learned` as a
documented alternative.

## 2. Assets used, and assets dropped

| asset | used? | how |
|---|---|---|
| tactical ATD (players + ball) | **yes** | all evidence; parsed straight into arrays, no kloppy |
| Smart Tagging | **yes, weakly** | team phases as low-weight evidence |
| homography | no | it maps pitch to pixels; useful only together with the video |
| tactical video | no | out of scope for this version (see limitations) |
| broadcast ATD and homography | experiment, not in the submission yet | aligned to the tactical timeline from the tracking alone (section 7) |
| broadcast video | no | not needed: the homography `valid` flag and the alignment already say when the feed cut away |
| public 5-minute ground truth | **checks only** | pipeline correctness and signal sanity (`possession/diagnose.py`), never tuning |

The loader is checked against the organisers' baseline: `python -m possession.pipeline
--check-baseline` re-implements their nearest-player baseline on our arrays and must reproduce
`baseline_smooth50.csv` frame for frame.

## 3. Training data

**No labels of any kind on the provided game are used**, neither for training nor for tuning.

Parameters are tuned, and the learned evidence trained, on **Metrica Sports' public sample data**
(https://github.com/metrica-sports/sample-data): three anonymised professional matches with
tracking of players and ball and hand-annotated event data by Metrica. They are other games, and
they are public, so the organisers can access them (RULES.md). `tuning/fetch_sample_data.sh`
downloads them.

**Labels.** Frame labels are derived from the events by a fixed rule (`tuning/sample_data.py`):

- every on-ball event (pass, carry, recovery, set piece, shot, ball lost) says its team had the
  ball at that frame, and possession is the team of the most recent one;
- the ball is dead from a stoppage (ball out, foul, goal, a shot that goes out) until the next
  restart (set piece, goal kick or kick-off), and before each kick-off;
- a spell in which the team never makes a deliberate action (no pass, carry, shot or set piece,
  only a recovery and an immediate loss, as with a header back or a block) is given back to the
  team that had the ball before. This is CHALLENGE.md's "touch without control" and "aerial duel
  won without subsequent control".

This definition is close to the competition's but not identical to it. The competition labels
come from corrected tracking, ours from events. The labels are good enough to size penalties and
to compare model variants, and they are not used to claim anything about BAR-ATH.

**Degradation.** The sample tracking is cleaner than the automatic BAR-ATH tracking, so before
tuning each game is degraded (`sample_data.degrade`). Every setting is matched to unlabeled
whole-match statistics of BAR-ATH. `possession/diagnose.py` prints them for BAR-ATH, and
`python -m tuning.noise` prints the same numbers for each sample game, clean and degraded:

| statistic (whole match, no labels) | BAR-ATH | sample, clean | sample, degraded |
|---|---|---|---|
| nearest player to a visible ball, p10 / p50 / p90 (m) | 0.44 / 1.91 / 7.02 | 0.04-0.06 / 0.28-1.27 / 4.9-5.3 | 0.50-0.55 / 1.61-1.90 / 6.2-6.9 |
| ball jitter, p50 (cm) | 0.46 | 0.05-0.07 | 0.40-0.42 |
| player jitter, p50 (cm) | 0.07 | 0.11-0.15 | 0.20-0.22 |
| players tracked per frame, p10 / p50 | 15 / 20 | 22 / 22 | 17 / 19 |
| ball missing, share of graded frames | 40.7%, 150 gaps | 32-41%, 155-169 gaps | unchanged |

Jitter is the distance of a position from the midpoint of its two neighbours.

- *Position error* drifts smoothly: white noise convolved with a Gaussian of 10 frames, 0.7 m on
  the ball and 0.3 m on each player. The BAR-ATH tracker is smooth from frame to frame (jitter
  under a centimetre) but its ball sits 1-2 m from the nearest player much more often than in the
  hand-corrected sample data. An earlier version used noise that jumped from frame to frame. It
  was 20 times too jittery and too small in offset, and with it the tuning chose a control radius
  that was too tight for the real tracking.
- *Lost player tracks*: each slot is blanked on 12% of frames, in runs of about 4 s.
- *The ball is not removed.* An earlier version removed it in short bursts until about 41% of
  frames were missing. The gap statistics showed that the sample games already match BAR-ATH.
- *Two scenarios for the dead ball.* In the sample data the ball is almost always missing during
  a stoppage, so "ball gone" is a near-perfect stoppage detector there. How often the BAR-ATH
  tracker keeps the ball during stoppages is not something we can measure without labels. So each
  game is degraded twice: once as it is, and once with half of its dead-time gaps filled by a
  straight line from where the ball was last seen to where it is seen next
  (`sample_data.show_dead_ball`). The tuning has to work in both.

Calibrating the noise on unlabeled aggregates of the target match is allowed; tuning on its
labels is not, and is not done.

**Protocol.** A seeded search over the parameters in `possession/config.py`, with mean spellF1
over the three games as the objective. It runs in two phases: random draws over the whole space,
then a local search that draws neighbours of the best configuration so far (`--refine`), because
random draws alone are thin in 16 dimensions. Generalisation is estimated **leave-one-game-out**:
the best configuration on two games is scored on the third.

**Results** (spellF1 / macroF1, mean over the two scenarios of each game, calibrated degradation):

| method | game 1 | game 2 | game 3 | mean |
|---|---|---|---|---|
| official baseline, ±50-frame window | 0.259 / 0.655 | 0.276 / 0.674 | 0.289 / 0.703 | 0.275 / 0.677 |
| state model, first tuning (no long-gap, no touch) | 0.512 / 0.784 | 0.534 / 0.769 | 0.486 / 0.814 | 0.511 / 0.789 |
| state model, second tuning (long-gap signal) | 0.584 / 0.832 | 0.575 / 0.800 | 0.521 / 0.850 | 0.560 / 0.827 |
| state model, current tuning, touch switched off | 0.567 / 0.831 | 0.572 / 0.802 | 0.504 / 0.847 | 0.548 / 0.827 |
| **state model, current tuning (`config.py`)** | **0.619** / 0.84 | **0.593** / 0.81 | **0.554** / 0.86 | **0.589** / 0.835 |

Each row is a configuration that was tuned at some point, all scored on the same current
degradation. With the more realistic noise every method scores lower than it did under the old
degradation, so the earlier figures (0.635 and 0.596) are not comparable with these. Two
differences are worth reading. The long-gap signal adds +0.05. The touch evidence adds +0.04
with every other value fixed. Leave-one-game-out, where the configuration is chosen on the other
two games, gave 0.605, 0.602 and 0.530, a **mean held-out spellF1 of about 0.58**, against 0.593
in-sample in the same search run. That run came before a small fix that ignores touches next to a
ball gap and moves the in-sample figure to 0.589.

These numbers are on other games with event-derived labels, so they are not a prediction of the
BAR-ATH score. They do measure what each design choice is worth, under the same noise and the same
metric.

## 4. Possession rules implemented

- **Restarts.** A stoppage ends when the evidence of play (players moving, the ball near a player)
  outweighs `FROM_DEAD`. The first team in control after it starts the new possession.
- **Loose balls and passes in flight.** No evidence, so no change: possession stays with the last
  team in control.
- **Duels and challenges.** When players of both teams are at the ball, their closeness scores
  cancel. A challenge changes possession only if one team then controls the ball for long enough
  to pay the switch penalty.
- **Touches and deflections.** A fast ball counts for less, and a short excursion by the other
  team must pay two switches. Both mechanisms keep a deflection or a single touch from splitting a
  possession. A touch that changes the ball's velocity is evidence for the toucher's team, but one
  touch (0.65 over 9 frames) is far below the switch penalty: it decides a hand-over only together
  with the control that follows it.
- **Dead ball.** The ball beyond a line, players clearly slowing down, or the ball lost by the
  tracking for more than about a second, for long enough to pay `TO_DEAD`. Leaving a stoppage is
  expensive (`FROM_DEAD` = 22), so a restart is declared only once play has clearly resumed. Who restarts is not asked: a dead ball is `D`.

## 5. Limitations

- **Definition gap in the tuning labels.** Event-derived labels are not the corrected-tracking
  labels of the competition, especially for very short possessions.
- **Simulated noise.** The degradation is a model of the automatic tracking, calibrated on the
  aggregate figures in DATA.md (and on `diagnose.py`'s aggregate statistics). It does not
  reproduce identity switches or duplicated tracks.
- **Long ball gaps.** A long gap is read as a likely stoppage. A long gap in open play (the ball
  hidden in a crowd, or the camera lagging) can therefore be labelled `D` wrongly, and a change of
  possession inside it is placed only when the ball reappears.
- **No video.** A ball detector on the tactical video could fill some of the 41% of missing ball
  frames, and that is the clearest next step.

## 6. Reproduction

See [README.md](../README.md#reproduce-the-submission). In short:

```bash
pip install -r requirements-dev.txt
python -m possession.pipeline             # writes evaluation/results/<team>.csv
python evaluation/validate_submission.py evaluation/results/<team>.csv

# optional: redo the tuning and training on the public sample games
bash tuning/fetch_sample_data.sh
python -m tuning.search
python -m tuning.train
```

The run is deterministic. No randomness, network access or state outside `data/` is involved, so
the same data always gives the same file.

## 7. The TV feed as a dead-ball signal

Part of the submission since 27 October 2026: `state_model_tv` is the default method. `possession/broadcast.py` builds it; `python -m
possession.broadcast` prints aggregate numbers only and caches the result under `data/cache/`.

### Why

Our weakest case is a stoppage during which the tracker still sees the ball: on the public window
such frames are called `D` only about 10% of the time. Nothing in the tracking separates them well,
because the players do not always stop and the ball does not always leave the pitch.

The tactical video cannot help here. It is one wide camera without cuts (its homography is valid on
every frame), so there is no "the camera cut away" event to detect in it. The broadcast is the TV
feed. When play stops, the director goes to a close-up, the bench, the crowd or a replay; during
live play he mostly stays on the wide game camera. So *"the TV feed was not showing this moment
live"* is evidence for a dead ball that is independent of all our tracking features, and it does
not need the ball to be tracked. The broadcast ball track is also a second, zoomed-in view of the
ball, which may fill tactical ball gaps.

Neither can be used before every broadcast frame is mapped to the tactical frame that shows the
same moment. The organisers give no such mapping (DATA.md).

### How the alignment works

No video is decoded. Both ATD exports give positions in the same pitch coordinates, so if broadcast
frame `b` shows live play at tactical frame `t`, the ball and the players are in the same places in
both, up to tracking noise. Both run at 25 fps, so during live play `t = b + c` with a constant
offset `c`. The offset only changes where the broadcast recording itself was edited (halftime, for
example). Replays and close-ups do not shift it, because the live clock keeps running under them.

1. **Anchors, a global template search.** About 400 ten-second broadcast windows with a valid
   homography and a moving ball are each slid over the whole tactical ball track, on a 5-frame
   grid. An offset scores the share of broadcast ball positions that fall within 3 m of the
   tactical ball. An anchor is kept when its best score is at least 0.5 and no other peak, more
   than 10 s away, reaches 70% of it. Its offset is then refined to the exact frame. A still ball
   (kick-off spot, placed free kick) matches everywhere, so anchors whose ball barely moves are not
   tried. Both orientations of the broadcast pitch are tried, in case one export is mirrored.
2. **Voting.** Anchor offsets within 2 s of each other form a cluster; each cluster with at least
   3 votes is one piece of the piecewise-constant mapping.
3. **Dense matching.** The whole broadcast is cut into 1-second windows. A window scores, at a
   given offset, the share of broadcast points (home players, away players, ball) that have a
   tactical point of the same class within 3 m. The broadcast camera is zoomed in, so it sees only
   some players; checking broadcast-to-tactical and not the reverse makes that harmless.
   - Pass 1 tries each cluster offset ±1 s (clock drift) and keeps only near-perfect windows
     (score ≥ 0.9). They fix the local offset, as a median over sure windows within 5 minutes.
   - Pass 2 scores every window only within ±2 frames of that local offset and accepts it at a
     score of 0.6 or more.
   - When neighbouring offsets tie on the number of hits (players move slowly), the offset whose
     points sit closest wins.
   - Matched stretches shorter than 3 s are dropped, and holes under 1 s in the tactical `shown`
     mask (a frame skipped where the offset moves by one) are closed.

   The two passes exist because a window that matches nothing (a replay of earlier play) can find
   a lucky offset by chance when it is allowed a hundred tries; with five per cluster it cannot.
   The same reasoning is behind the 3-second minimum: a live shot lasts many seconds.

The output per tactical frame is `shown` (the feed was showing that moment live) and `bc_ball` (the
broadcast's ball position at that moment). The synthetic tests in `tests/test_broadcast.py` build a
broadcast from known pieces (two halves at different offsets, a close-up, a replay, a mirrored
export) and check that the mapping is recovered and that the close-up and the replay stay unmatched.

On a full-size synthetic match (199,319 tactical and 164,488 broadcast frames) the alignment runs
in about 40 s, finds both offsets exactly, and leaves only the close-up and the replay unmatched.

### What decides whether it goes in

The summary prints, with no labels: the offsets found, how much of the broadcast matched, how much
of the graded tactical time was shown live, how far apart the two ball tracks are when both see the
ball, and how much of the tactical ball gap the broadcast fills. It also prints how often the
frozen model already calls `D` on shown and not-shown frames. On the public window it prints, as a
sanity check only, the share of dead and alive frames that were shown live.

The signal is worth adding only if not-shown frames are far more often dead than shown ones. Its
weight cannot be tuned on the sample games, which have no broadcast, so it would be set from these
unlabeled rates, in the same way as the Smart Tagging weight (section 1.2).

### First run on BAR-ATH (aggregates only)

- Two offsets were found, +4,011 and +26,920 frames, one per half, with 78 and 66 anchor votes;
  the other orientation got 8. The dense windows match with a median score of 0.94 and drift by at
  most a few frames.
- 90.8% of valid-homography broadcast frames match a live moment, against 4.0% of invalid ones.
- 62.5% of graded tactical time is shown live. The not-shown stretches number 145, with a median
  of 7 s and a p90 of 39 s: the length of stoppages, not of quick camera changes.
- Without labels: the frozen model calls `D` on 26.3% of shown frames and 78.3% of not-shown ones.
- Public window, sanity only: shown on 90.2% of alive frames and 29.5% of dead ones. The dead
  frames on which the tracker still sees the ball, which the model calls `D` only 10% of the time,
  are not shown on 71.4%.
- The broadcast ball covers only 9% of the tactical ball gaps, so gap filling is left aside.

### How it enters the model (`methods.state_model_tv`)

As one more term in the dead-ball score of the Viterbi, a per-frame log-likelihood ratio:

    not shown:  log( P(hidden | dead) / P(hidden | alive) )
    shown:      log( P(shown | dead)  / P(shown | alive)  )

and 0 where the broadcast recording does not exist (before it starts, halftime, after it ends).

The two rates are estimated on the whole graded match **against the tracking-only model's own
labels**, so no ground truth is read and nothing is tuned on the public window. The model runs
twice: once without the cue, to calibrate it, and once with it. With the run above the rates are
about 0.64 and 0.15, which gives +1.45 per not-shown frame and -0.86 per shown frame. That is the
same order as the long-ball-gap evidence (1.88 per frame) the tuning chose. The model's labels are
noisy, which pulls the two rates towards each other, so this ratio is if anything an underestimate.
`TV_DEAD_WEIGHT` (1.0) scales it; it cannot be tuned because the sample games have no broadcast.

### Result

Public window, as a sanity check (no parameter was chosen on it; `TV_DEAD_WEIGHT` stayed at 1.0):

| | spellF1 | macroF1 | D recall | possessions (truth 17) |
|---|---|---|---|---|
| `state_model` (previous submission) | 0.325 | 0.658 | 40% | 15 |
| `state_model_tv` | 0.413 | 0.741 | 86% | 13 |

On the whole match the calibration found the feed away from live play on 62.2% of model-dead and
12.9% of model-alive frames. The output has 188 possessions and 78,925 `D` frames. The cost is
some alive frames now called `D` (D precision 0.62), which is the next thing to work on.

### Refinement: the feed's lag and the length of a cutaway

`python -m possession.diagnose_tv` looked at where the first version calls a live ball dead.
Without labels, over the whole match:

| not-shown stretch | stretches | tracking-only model calls `D` |
|---|---|---|
| under 3 s | 37 | 53% |
| 3-8 s | 42 | 73% |
| 8-20 s | 33 | 61% |
| 20 s or more | 30 | 88% |
| shown live | | 26% |

and the feed leaves live play a median 3.3 s after a tracking-only stoppage starts. The public
window, read as a sanity check, agrees: the feed leaves 3.0 s after a true stoppage starts and
returns 2.0 s after the true restart, and the first version restarts play exactly when the feed
returns. 30% of its alive-called-`D` frames sit in the 5 s after a true restart.

So the feed is a *delayed* indicator: the director reacts to the whistle and to the restart a few
seconds late. Two refinements follow, both calibrated on the match itself without labels:

1. **Lag** (`methods.tv_lag`). For every tracking-only `D` spell of at least 2 s, the delay from
   its start to the start of the overlapping not-shown stretch; the median is the feed's lag, and
   the cue is moved that many frames earlier. Starts are used and not ends because entering `D` is
   cheap in the Viterbi while leaving it is expensive, so the model's stoppage starts are roughly
   on time and its restarts are late.
2. **Length bins** (`methods.tv_binned_evidence`). One log-ratio per not-shown stretch length bin
   (under 3 s, 3-8 s, 8-20 s, 20 s or more) and one for shown frames, each
   `log P(bin | dead) / P(bin | alive)` against the tracking-only labels. A short cutaway during
   play then counts for less than a long absence. The bin edges are natural breaks, not searched.

`--tv-variant` runs each refinement alone (`shift`, `bins`), both (`full`) or neither (`plain`, the
first version), so their effects can be measured separately.

Public window, as a sanity check (the lag came out at 3.3 s on the whole match):

| variant | spellF1 | macroF1 | D precision / recall | possessions (truth 17) |
|---|---|---|---|---|
| `plain` (first version) | 0.413 | 0.741 | 0.62 / 0.86 | 13 |
| `shift` (**default**) | 0.441 | 0.808 | 0.77 / 0.88 | 12 |
| `bins` | 0.388 | 0.774 | 0.77 / 0.75 | 15 |
| `full` | 0.400 | 0.819 | 0.95 / 0.73 | 13 |

The lag correction is the clear gain: it removes most of the live frames called `D` after a
restart without losing dead frames. The length bins raise D precision but miss more stoppages, and
cost spellF1 alone and together with the lag. `shift` is the default. In fairness: choosing between
`shift` and `full` uses this window, the gap between them is within its noise band (17 possessions),
and both were designed before it was looked at. `shift` is also the simpler of the two, with no
extra constants, which is the reason to prefer it independently of the window.

## 8. Ball-free team evidence: the teams' widths

The ball is missing on 41% of graded frames, and there the ball-based team evidence is silent: the
model can only keep the last team. Which team has the ball also shows in the players' shape. The
team in possession spreads across the pitch to open passing lanes, and the defending team narrows
to close them (width in attack, compactness in defence). PathCRF-style models read possession from
the players alone for the same reason.

Candidates were measured on the sample games, by how often the sign of the home-minus-away
difference agrees with the team in possession on alive frames:

| feature (home minus away) | agrees with the possessing team |
|---|---|
| width (std of y across the pitch) | 81-83% in all three games |
| spread (sqrt of x and y variance) | 71-83% |
| length (std of x) | 58-77% |
| block position (centroid along the home attack direction) | 59-70% |
| forward speed of the centroid | 58-66% |

Width is the most accurate and the most stable across games, and it needs no attacking direction,
so it is the one used (`features.width_evidence`): the difference of the two teams' y standard
deviations, averaged over ±1 s, squashed with `tanh(diff / WIDTH_SCALE)` and added to the team
evidence with weight `WIDTH_WEIGHT`. It is weak per frame on purpose: 0.07 against 1 for a player
on the ball, so it decides only where the ball says nothing for many seconds.

Sample games (tuned the usual way, section 3): spellF1 0.587 → 0.618, with all three games
improving (0.618 → 0.645, 0.591 → 0.647, 0.553 → 0.564), and leave-one-game-out 0.598 against
about 0.58 before.

## 9. A wider tuning set: SkillCorner's open A-League matches

Three tuning games is thin, and Metrica's event-derived labels are coarser than the competition's:
about 230 possessions per match in game 1 and 2, against roughly 320 implied by the 17 possessions
in BAR-ATH's public five minutes. A model tuned only on them merges possessions.

SkillCorner publishes twenty 2024-25 A-League matches with broadcast tracking and, on every frame,
the team in possession (github.com/SkillCorner/opendata). Ten are added (`tuning/skillcorner.py`,
fetched by `tuning/fetch_skillcorner.sh`):

- **Labels.** A frame with a team is `H` or `A`; a frame with none is `D`. Checked before use: the
  players move at 1.1 m/s on no-team frames and 2.3 m/s on team frames, the stoppage/play ratio of
  Metrica's games, and no-team runs last a median 17 s. The matches have about 350 possessions
  each, close to BAR-ATH's granularity.
- **Tracking.** SkillCorner's ball is missing on about 40% of in-play frames, like BAR-ATH (40.7%).
  Positions are moved to a corner origin and resampled from 10 to 25 fps, so every frame-based
  constant keeps its meaning. `degrade()` then adds the same noise as to Metrica's games.
- **Not the DFL data.** The DFL/Sportec open matches were the first choice, but their host
  (figshare) could not be reached from the development environment; SkillCorner's are on GitHub.

`python -m tuning.search --games all --outer 15 --refine 40` then tuned every parameter on all
thirteen games (two degradations each):

| | Metrica (3) | SkillCorner (10) | all 13 | leave-one-game-out |
|---|---|---|---|---|
| previous config (Metrica only) | 0.619 | 0.363 | 0.422 | not tuned on SkillCorner |
| retuned on all 13 | 0.582 | 0.385 | 0.430 | 0.428 |

The retuned configuration traded a little on Metrica's coarse labels for more on the ten matches
that look like BAR-ATH. The main changes were a wider control radius (`CONTROL_SIGMA` 0.8 → 2.39 m,
because the automatic tracking puts the ball 1-2 m from its holder), short ball gaps bridged up to
0.6 s, only gaps of 4 s or more read as stoppages, and a stronger width evidence (0.07 → 0.37).

**Result on BAR-ATH: not adopted.** On the public five minutes (sanity check only, never tuned on)
the retuned configuration scored spellF1 0.446 / macroF1 0.849, against 0.489 / 0.829 for the
frozen one. Team accuracy on alive frames rose (0.818 → 0.862) and so did macroF1, but it predicted
9 possessions instead of 10 (truth: 17), so the spells merged even more and spellF1, the ranking
metric, fell. The rule is that a change must beat the frozen file, so `possession/config.py` keeps
the frozen values. The SkillCorner loader and `--games all` stay as tuning infrastructure.

The lesson: a better frame-level fit does not cure undersegmentation. A search that rewards
spellF1 on SkillCorner matches with BAR-ATH-like granularity, or a model change that makes team
switches cheaper where the evidence supports them, is needed instead of more of the same tuning.

## 10. Undersegmentation: where possessions merge, and an evidence-dependent switch cost

On BAR-ATH's public window the model predicts 10 possessions against 17. Before changing anything,
`python -m tuning.segmentation --games all` breaks the errors down on the 13 tuning games (each
degraded twice, config as frozen; never BAR-ATH):

| | count |
|---|---|
| true / predicted possessions | 4058 / 2584 |
| true possessions no predicted one of the same team touches | 1323 (33%) |
| short (< 5 s) H → A → H interruptions missed | 560 of 831 (67%) |
| short (< 5 s) H → D → H pauses with no D predicted | 62 of 126 (49%) |
| predicted changes with no true change within 2 s | 1506 of 4051 (37%) |
| median start / end error of a found possession | 2.1 s / 2.3 s |

**Sweeping only SWITCH_COST** (2 → 8) moves along one trade-off: at 2 the output has as many
possessions as the truth, but spurious changes rise from 1506 to 2242 and spellF1 falls from
0.422 to 0.393. Even there, 61% of the short interruptions are still missed. 5.12 was already the
best fixed value.

**Why the interruptions are missed.** Inside the missed ones the ball is seen on 99% of frames, so
it is not the missing ball. They last a median 1.7 s, and the summed team evidence over them
points to the *outer* team (median -2.1, against +15.5 inside the interruptions that are found).
The evidence itself does not show them, so no switch cost can recover them without inventing many
false ones. Some of them are likely label artefacts: SkillCorner's possession group and Metrica's
event-derived labels flip on a single contested touch.

**Why the pauses are missed.** Lowering FROM_DEAD_COST (48 → 10) finds 2 more of the 126 and adds
590 spurious changes. On SkillCorner a few seconds with no team in possession is most often a loose
ball, not a stoppage, so these "pauses" are mostly a difference of definition, not an error to fix.

**Evidence-dependent switch cost.** Instead of one cost for every hand-over, entering a team at a
frame costs `SWITCH_COST x (1 - SWITCH_RELIEF x cue)`, where the cue (`features.handover_cue`, in
[0, 1]) is the larger of two signals that need the ball: clear control by the new team (its nearest
player close to a calm ball and the other team's not, the nearest vs second-nearest gap) and a touch
by the new team (`touch_evidence`). Where the ball is missing the cue is 0 and the cost is full.
`hmm.viterbi` takes the per-frame scale of the H <-> A cost (`switch_scale`).

A grid over the cue's spread (0-50 frames), the relief (0.3-0.6) and the base cost (5.12-7), every
other parameter fixed:

| | mean spellF1 (13 games) | games better | predicted | spurious |
|---|---|---|---|---|
| frozen (fixed cost 5.12) | 0.422 | | 2584 | 1506 |
| relief 0.4, cost 6.0, no spread | 0.428 | 12 of 13 | 2708 | 1532 |
| relief 0.6, cost 5.12, spread 50 | 0.404 | 3 of 13 | 3738 | 2067 |

Relief helps only if the base cost goes up with it: hand-overs the ball shows clearly get cheaper
(3.6), the ones it does not show get dearer (6.0). Spreading the cue over a window, or relieving more
than half the cost, adds spurious changes faster than it finds true ones. The gain is small (+0.006)
but consistent across games, and it adds possessions (2708 against 2584) without adding many
spurious changes, which is the direction BAR-ATH needs. Whether it beats the frozen file on BAR-ATH
is decided by the local run, as always.

## 11. Smart Tagging transitions and a soft minimum possession length

### Transition tags as turnover hints

`python -m possession.diagnose_tagging` compares the tags with the model's own labels over the
whole match (no ground truth). Findings on BAR-ATH:

- Tags come in mirrored pairs: every ATTACKING TRANSITION has a DEFENSIVE TRANSITION with the same
  times for the other team, and the same holds for Build Up / Defending in Attacking Third,
  Progression / Defending in Middle Third, Creating Chances / Defending in Defensive Third.
- Team agreement with the model is 74-95% depending on the code (Build Up and Progression highest).
- **Transitions:** for half of the ATTACKING TRANSITION tags, the model switches team within 3 s of
  the tag's start, against 10% for a random frame. The switch comes 0.8-5.1 s after the start
  (p10-p90). Tag starts run early, like the kickoff tags (3.3 s, which matches the organisers'
  notebook).
- **Set pieces** sit 2-7 s before the model's restarts, with a 9 s spread. That is too loose to
  place a restart, so they are not used.

`tagging_prior.transition_cue` gives the tagged team a hand-over cue of 1 from 0.5 s to 6 s after
each ATTACKING TRANSITION start, combined with the ball cue (maximum), so a switch there gets the
same relief as one the ball shows clearly. It never forces a switch, because the tracking evidence
still has to pay for it. The window comes from the model's own switches, not from labels. The
tuning games have no Smart Tagging, so it could not be tuned and is fixed.

### Soft minimum duration (the duration part of a semi-Markov model)

On the 13 tuning games, predicted possessions by length (two degradations each):

| length | true | predicted | predicted and matched at IoU 0.5 |
|---|---|---|---|
| < 2 s | 1614 | 251 | 7% |
| 2-4 s | 1370 | 621 | 22% |
| 4-8 s | 1684 | 1258 | 39% |
| 8-15 s | 1492 | 1325 | 61% |
| 15-30 s | 1312 | 1199 | 71% |
| > 30 s | 644 | 763 | 78% |

Short predicted possessions are mostly wrong, but a hard minimum would also delete the few right
ones and the short true possessions the evidence does show. `hmm.viterbi_min_duration` adds a soft
minimum instead: a possession shorter than `MIN_SPELL` frames pays `SHORT_SPELL_COST` once, when it
ends. It is exact, not a post-filter: each team gets a "young" state next to the usual one, and
the best young spell to leave from is a sliding-window maximum, so decoding stays linear in the
number of frames (0.4 s per half against 0.2 s). A brute-force test over all paths of small
sequences checks that it finds the optimum.

Grid on the 13 games (mean spellF1, current config 0.428):

| MIN_SPELL | SHORT_SPELL_COST 2 | 4 | 8 |
|---|---|---|---|
| 2 s | 0.431 | 0.431 | 0.432 |
| 3 s | 0.432 | 0.432 | 0.433 |
| 4 s | 0.431 | 0.430 | 0.430 |

With 3 s and cost 4 it removes predicted possessions (2708 → 2552). Since BAR-ATH already has too
few, the switch cost and relief were re-searched with it: `SWITCH_COST 5.5`, `SWITCH_RELIEF 0.5`
gives 0.434, better on all 13 games, with 2668 predicted possessions (2708 before) and about the
same number of spurious changes (1538 against 1532).

`python -m possession.pipeline --set NAME=VALUE` overrides a config value for an ablation run, so
both additions can be checked on BAR-ATH separately (`--set TRANSITION_CUE_TO=0`,
`--set MIN_SPELL=0`).

### Result on BAR-ATH: neither is adopted

Alper's local run on the public window (sanity check, never tuned on), against the frozen file
(spellF1 0.519 / macroF1 0.835, 10 possessions):

| | spellF1 | macroF1 | possessions |
|---|---|---|---|
| frozen (PR #14) | 0.519 | 0.835 | 10 |
| + transition cue | 0.504 | 0.829 | 10 |
| + soft minimum duration, cost 5.5, relief 0.5 | 0.483 | 0.829 | 12 |
| both | 0.469 | 0.823 | 12 |

Both lose, so both are switched off in `config.py` (`TRANSITION_CUE_TO = 0`, `MIN_SPELL = 0`) and
the submission stays the frozen one. The code stays, for ablations with `--set`.

The differences are inside the window's noise (17 possessions, one more or less matched moves
spellF1 by about 0.03), and the duration model won on all 13 tuning games. But the rule is that a
change must beat the frozen file where it is checked, and choosing the other way would be tuning on
judgement. The duration variant did add two possessions (12 against 10), the direction BAR-ATH
needs, but they did not match: more possessions alone are not the goal.

## 12. Ball gaps and the video: a feasibility probe

### Why

The ball is missing on 40.7% of graded frames. Two things already handle it: gaps of up to
`BALL_MAX_GAP` frames are bridged with a straight line (section 1; the search preferred a short
bridge), and long gaps count as evidence of a stoppage (`features.long_gap`). Between those, a long
gap in open play leaves the ball-based team evidence silent for seconds, and only the slow width
evidence (section 8) speaks. The ground truth is derived from corrected tracking (CHALLENGE.md), so
it knows where the ball is on those frames. The broadcast tracking fills only 9% of the gaps
(section 7).

The tactical video is the remaining source. Its homography is valid on every frame, so a ball found
in a frame maps straight back to the pitch in metres. That is worth building only if the ball is
actually visible in the video where the tracker loses it, and if a detector places it accurately
enough for the nearest-player evidence (`CONTROL_SIGMA` 0.8 m) to use. `possession/probe_video_ball.py`
measures both before anything enters the model.

### An open football pipeline considered

[abdullahtarek/football_analysis](https://github.com/abdullahtarek/football_analysis) is a popular
open project: YOLO detection of players, referees and the ball, ByteTrack, team assignment by
shirt colour (k-means), camera motion by optical flow, a perspective transform, and team ball
control as "the player nearest to the ball within 70 px, carried forward". Most of it duplicates
what the provided data already gives, better: the tracking has players in metres with their team,
and the homography replaces the camera-motion and perspective steps. Its ball control rule is the
nearest-player rule that the official baseline already implements. The one part the provided data
lacks is a ball detector that works where the tracker has lost the ball, so that is the only part
taken, and only as an idea: any public YOLO weights with a ball class can be used (its `best.pt`,
the ball model of [roboflow/sports](https://github.com/roboflow/sports), or a generic COCO model).

### What the probe measures (aggregates only)

- **Part A, tracking only.** Missing graded frames by gap length (< 1 s, 1-4 s, 4-10 s, >= 10 s),
  and the share of them in live play: by the labels of a given submission file (the frozen model's
  own output, not ground truth) or, without one, by the players' pace against the match's median.
  Live missing frames are the most a video detector could inform.
- **Part B, video** (`--weights`, packages in `requirements-cv.txt`). On a seeded sample of
  frames: on calibration frames, where the tracking has the ball, the distance between detection
  and tracked ball; on live gap frames, how often a ball is found on the pitch, by gap length, how
  far the nearest player is from it and whether that player's team agrees with the labels. The
  detected box centre is mapped to the pitch with `starter/load_homography.unproject`.
- **Decision.** Build the detector into the model only if the calibration median error is below
  2 m and at least 30% of live gap frames get an on-pitch ball at confidence 0.25. Both thresholds
  are fixed in advance and use no labels of BAR-ATH.

The probe changes nothing in the submission: the pipeline does not read it.

### Result

First run on BAR-ATH (Alper's machine, 500 calibration and 500 live gap frames, confidence 0.25,
the most confident box per frame):

| | roboflow/sports ball model | abdullahtarek best.pt |
|---|---|---|
| calibration: ball detected | 86% | 71% |
| calibration: error p25 / p50 / p75 (m) | 0.7 / 2.6 / 23.6 | 1.1 / 10.9 / 35.8 |
| live gaps: ball found on the pitch | 49% | 37% |
| nearest player to that ball within 2 m | 85% | 44% |
| that player's team agrees with the labels | 77% | 54% |

Both were STOP under the fixed thresholds. The roboflow model is clearly better, so the other one
is dropped. Its error is bimodal: about half of the detections are within 1 m of the tracked ball
and most of the rest are 20-45 m away, another ball-like object (a spare ball by the touchline, a
pitch marking). Taking the most confident box anywhere in the frame lets those win. The probe now
keeps up to five candidates per frame and compares three selection rules, all label-free: the most
confident box, the most confident box on the pitch, and the most confident box on the pitch with a
player within 2 m.

Part A also showed that, by the frozen model's labels, only 1.4% of graded frames are live with the
ball missing. That figure is partly circular, because the model reads a long gap as a stoppage. The
probe therefore also gives the live share by the players' pace, and looks for an on-pitch ball at a
player's feet on gap frames the model calls dead.

Second run, roboflow model only, 1000 frames per sample, confidence 0.25:

| rule | calibration detected | error p25 / p50 / p75 (m) | within 2 m | live gaps filled |
|---|---|---|---|---|
| most confident | 85% | 0.6 / 3.1 / 24.1 | 44% | (59%) |
| most confident on the pitch | 75% | 0.4 / 1.4 / 7.5 | 58% | 59% |
| on the pitch, player within 2 m | 49% | 0.5 / 2.2 / 16.9 | 48% | 51% |

Only the on-pitch rule passes (GO). Requiring a nearby player is worse: it prefers less confident
boxes on players' feet and socks over a ball in flight. On the live gap frames the on-pitch ball has
a player within 2 m 84% of the time, and that player's team agrees with the frozen labels 75% of
the time.

Live share of the missing frames, by the frozen labels and by the players' pace:

| gap | share of missing frames | live by labels | live by pace |
|---|---|---|---|
| 4-10 s | 14.5% | 19.8% | 84.0% |
| >= 10 s | 85.4% | 0.5% | 51.4% |
| all | 100% | 3.4% | 56.2% |

The pace threshold (`DEAD_SPEED_RATIO`) is lenient, so the pace figure is an upper bound. But the
gap is large, and the public window agrees in direction: there the model calls `D` on 98% of the
391 live frames without a ball (`python -m possession.diagnose`). Long gaps are the model's main
source of false stoppages, which is what the video ball can address. On gap frames the model calls
dead, an on-pitch ball is found about as often (49%), so a visible ball alone does not mean live
play. The pace evidence and the Viterbi still have to decide that.

## 13. The video ball in the model

`possession/video_ball.py` runs the detector once on every graded frame without a tracked ball
(about 64,600) and caches every candidate box (up to five per frame) in `data/cache`. `fill()` then
picks, on each such frame:
- the most confident box on the pitch at confidence 0.25 or more, the rule that passed the probe;
- the box mapped to metres through the tactical homography;
- the point dropped if it lies more than 3 m from the median of its detections within 5 frames on
  each side (`VIDEO_BALL_JUMP`, `VIDEO_BALL_SMOOTH`), because a box on another object jumps while
  the ball moves smoothly.

Frames with a tracked ball are never changed.

With `VIDEO_BALL = 1`, `methods.frame_scores` uses the filled ball for the team evidence, the touch
and hand-over cues and the dead-ball evidence. Two switches decide how far it reaches:
`VIDEO_BALL_WEIGHT` scales the team evidence on filled frames, and `VIDEO_BALL_BREAKS_GAPS` decides
whether a filled frame ends a ball gap for `long_gap` (1) or the gap is still read from the
tracking alone (0). The tuning games have no video, so every value is fixed in advance, not
searched. The default is off, which leaves the frozen submission unchanged.

Three variants were fixed before any of them was scored, and are checked on the public window
against the frozen file (spellF1 0.519), following the freeze rule:

| variant | settings |
|---|---|
| vb | `VIDEO_BALL=1` |
| vb_team | `VIDEO_BALL=1 VIDEO_BALL_BREAKS_GAPS=0` (team evidence only) |
| vb_half | `VIDEO_BALL=1 VIDEO_BALL_WEIGHT=0.5` |

### Result: adopted

Detections ran once on all 64,641 graded frames without a tracked ball; 27,283 of them were filled
after the selection and the jump check. Public window (sanity check, never tuned on), against the
frozen file:

| | spellF1 | macroF1 | possessions (true 17) | transF1 | full match: D share | full match: possessions |
|---|---|---|---|---|---|---|
| frozen (section 10) | 0.519 | 0.835 | 10 | 0.364 | 49.0% | 182 |
| **vb** | **0.533** | 0.824 | 10 | 0.235 | 39.2% | 203 |
| vb_team | 0.519 | 0.835 | 10 | 0.364 | 49.0% | 182 |
| vb_half | 0.500 | 0.821 | 11 | 0.286 | 39.2% | 201 |

`vb` beats the frozen file on spellF1, so by the freeze rule it becomes the submission
(`VIDEO_BALL = 1`). Reference run (RTX 4070 laptop GPU, torch 2.6.0+cu124, ultralytics 8.4):
the pipeline prints `detections sha256 504c66529e6d` for the detector cache and writes the file
with sha256 `3b610b635edd4ba52e93130be98b25cfc566274d806d59dbfebb242347415b1b`. A rerun whose
detection hash differs comes from a different detector output, not from the pipeline. What the
numbers do and do not say:

- The gain is small and inside the window's noise. It comes from one more possession matched at
  IoU 0.7 (spellF1 at 0.7: 0.296 → 0.370), not from more matched possessions (8 of 17 in both).
- On the full match it moves in the expected direction. The `D` share falls from 49% to 39%,
  near the usual share of dead time in a match, and the number of possessions rises (BAR-ATH had
  too few: 10 predicted against 17 true in the window). That fits the probe: by the players' pace,
  most ball gaps are live play the old model called dead.
- It has costs. In the window, true `D` frames called `A` rise from 77 to 181 (`D` recall
  0.879 → 0.808), and transition timing gets worse (transF1 at ±25 frames 0.364 → 0.235). The
  likely cause is a ball the video sees on the pitch during a set piece: the filled frames end the
  ball gap, and with it the long-gap dead-ball evidence. macroF1, the tie-break, falls by 0.011.
- `vb_team` changes nothing in the window. The effect comes from breaking long gaps, not from the
  team evidence on the filled frames.
- Three variants were scored and the best one taken. They were fixed in advance and are all
  reported here, but picking the best of three on 17 possessions is a mild selection effect.

Not tried yet: ending a ball gap only where the players are also running, which would keep a
stoppage with a visible ball dead.
