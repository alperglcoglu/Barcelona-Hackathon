# Technical document

> Work in progress. The section headings follow what SUBMISSION.md asks the document to cover.
> Each section is filled in when the corresponding part of the method is built.

## 1. Approach

*To be written in step 2, the state model.*

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
   expensive across a hand-over.

The method is therefore built as a **segmentation** problem, not as per-frame classification:
decide where possessions start and end, and charge every change of state a cost.

### 1.2 Pipeline (step 1: infrastructure)

- `io_atd.py` parses the raw FIFA/EPTS export straight into arrays (ball `(N, 2)`, players
  `(N, 46, 3)` in metres) and caches them. kloppy is not used, because one Python object per
  player per frame is too slow for 199,319 frames.
- `methods.baseline` re-implements the organisers' floor (nearest player, then a ±50-frame
  majority vote) on those arrays. `--check-baseline` compares it frame by frame with the shipped
  `baseline_smooth50.csv`, so any loading or alignment error shows up before anything is built on
  top.

## 2. Assets used, and assets dropped

*To be written.*

## 3. Training data

No labels of any kind on the provided game are used. *External data, if any, to be listed here.*

## 4. Possession rules implemented

*To be written: loose balls, challenges, deflections and restarts.*

## 5. Limitations

*To be written.*

## 6. Reproduction

See [README.md](../README.md#reproduce-the-submission).
