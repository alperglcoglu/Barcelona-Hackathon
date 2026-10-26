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

**Evidence of a dead ball** (`features.dead_evidence`) is a log-odds score from two signals that
cover each other's blind spots:

- *The ball is visibly beyond a line* (with a 0.5 m margin for tracking noise). Decisive, but the
  tracking often loses the ball exactly when it goes out.
- *The players have stopped running.* During a stoppage twenty-two players walk. The mean speed of
  all tracked players, averaged over ±1 s, enters linearly around a pivot speed. This signal never
  needs the ball, which matters because the ball is missing on 41% of graded frames.

**Missing ball** (`features.interpolate_ball`). Gaps up to about one second are bridged with a
straight line, because the ball cannot travel far unseen in that time. Longer gaps stay missing
and contribute no team evidence, so the model holds its state through them.

**Smart Tagging** (`tagging_prior.py`) adds weak team evidence: an attacking phase says its team
has the ball, and a defending phase says the other team does. Its weight is small, so a phase can
tip an ambiguous stretch but never overrides clear tracking. Throw-ins are tagged `N/A` and add
nothing.

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
between the teams. With penalties tuned for it, it reached a mean spellF1 of about
0.57, below the hand-written evidence (0.60), and that figure is even flattering, because the
model had seen every game. The collective features do carry signal: the possessing team is more
spread out, and the defending team runs faster. But the event-derived labels are coarse, and in the
sample data "ball missing" mostly means "ball out of play", which a learned model exploits and
which would not transfer to BAR-ATH, so we removed it as a feature. We keep the hand-written
evidence as the submission (`--method state_model`, the default) and `--method learned` as a
documented alternative.

## 2. Assets used, and assets dropped

| asset | used? | how |
|---|---|---|
| tactical ATD (players + ball) | **yes** | all evidence; parsed straight into arrays, no kloppy |
| Smart Tagging | **yes, weakly** | team phases as low-weight evidence |
| homography | no | it maps pitch to pixels; useful only together with the video |
| tactical video | no | out of scope for this version (see limitations) |
| broadcast video, ATD and homography | no | not synchronised to tactical; mapping it back is a project of its own |
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
tuning each game is degraded (`sample_data.degrade`): ball removed in bursts until about 41% of
frames are missing, slowly drifting position error, and short player-track dropouts. Without this,
the parameters would be tuned on cleaner data than they meet.

**Protocol.** A seeded random search over the parameters in `possession/config.py`, with mean
spellF1 over the three games as the objective. Generalisation is estimated **leave-one-game-out**:
the best configuration on two games is scored on the third.

**Results** (spellF1 / macroF1, mean over two degradations of each game):

| method | game 1 | game 2 | game 3 | mean |
|---|---|---|---|---|
| official baseline, ±50-frame window | 0.348 / 0.744 | 0.362 / 0.769 | 0.348 / 0.803 | 0.353 / 0.772 |
| hand evidence, untuned starting values | 0.558 / 0.806 | 0.581 / 0.769 | 0.473 / 0.818 | 0.537 / 0.798 |
| **hand evidence, tuned (`config.py`)** | **0.606** / 0.825 | **0.631** / 0.815 | **0.552** / 0.826 | **0.596** / 0.822 |

Leave-one-game-out, where the configuration is chosen on the other two games, gives 0.607, 0.578
and 0.554, a **mean held-out spellF1 of 0.580**. The small gap to the in-sample 0.596 says that the
tuned values are not overfitted to particular games.

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
  possession.
- **Dead ball.** The ball beyond a line, or players clearly slowing down, for long enough to pay
  `TO_DEAD`. Who restarts is not asked: a dead ball is `D`.

## 5. Limitations

- **Definition gap in the tuning labels.** Event-derived labels are not the corrected-tracking
  labels of the competition, especially for very short possessions.
- **Simulated noise.** The degradation is a model of the automatic tracking, calibrated on the
  aggregate figures in DATA.md (and on `diagnose.py`'s aggregate statistics). It does not
  reproduce identity switches or duplicated tracks.
- **Long ball gaps.** During gaps of several seconds the model holds its state, and a change of
  possession inside the gap is placed when the ball reappears, which is late.
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
