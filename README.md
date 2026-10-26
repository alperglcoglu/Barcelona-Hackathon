# Barcelona Hackathon: frame-level ball possession

Our entry to **More than a Hack 2027, Phase 1** (Metrica Sports, FC Barcelona and the Mobile World
Capital Barcelona). For each of the 199,319 frames of the tactical video of FC Barcelona vs
Athletic Club, the pipeline outputs `H` (home in possession), `A` (away in possession), `D` (dead
ball) or `X` (no answer). Entries are ranked on `spellF1`, with `macroF1` as the tie-break.

- How the method works and why: [docs/TECHNICAL.md](docs/TECHNICAL.md)
- The organisers' challenge documents are kept unchanged at the repository root:
  [ORGANISERS_README.md](ORGANISERS_README.md), [CHALLENGE.md](CHALLENGE.md), [DATA.md](DATA.md),
  [EVALUATION.md](EVALUATION.md), [RULES.md](RULES.md), [SUBMISSION.md](SUBMISSION.md) and
  [FAQ.md](FAQ.md).

## Reproduce the submission

```bash
python -m venv .venv && source .venv/bin/activate     # Windows: .venv\Scripts\activate
pip install -r requirements-dev.txt             # Python 3.9-3.14; no compiler needed

# unpack the gated match data into ./data (see data/README.md), or set MTAH_DATA_DIR
python -m possession.pipeline --check-baseline         # one-off: loader reproduces the official floor
python -m possession.pipeline                          # writes evaluation/results/<team>.csv
python evaluation/validate_submission.py evaluation/results/<team>.csv
python evaluation/run_evaluation.py                    # score on the public 5-minute window
```

The run is deterministic, so the same data always gives the same file. The first run parses the
tracking once into `data/cache/`, which is gitignored like everything else under `data/`.

## Repository layout

```
possession/             our code
  config.py             every tunable number, with the reason for its value
  io_atd.py             raw tracking -> numpy arrays (metres), cached
  io_tagging.py         Smart Tagging -> intervals on the video-frame axis
  features.py           per-frame evidence: who controls the ball, is play stopped
  tagging_prior.py      weak team evidence from Smart Tagging phases
  hmm.py                three-state (H/A/D) Viterbi with transition penalties
  learned.py            alternative: logistic evidence trained on other games
  methods.py            baseline (official floor), state_model (default), learned
  submission.py         write and check the frame,label CSV
  pipeline.py           the single command that regenerates the submission
  diagnose.py           aggregate-only signal checks on the real data
tuning/                 parameter search and training on Metrica's public sample games
tests/                  run without the match data: python -m pytest tests
docs/TECHNICAL.md       the technical document submitted with the entry
evaluation/, starter/, games.py
                        the organisers' scorer and tooling, unmodified
```

## Data and rules

The match data is not in this repository and must never be committed: `.gitignore` excludes
`data/` and every submission CSV. Following the competition terms, no match data, and no extract
of it, is sent to any cloud-hosted service or LLM. The code runs locally. Labels are produced only
by the algorithm. The public 5-minute ground truth is used to check that the pipeline is correct,
never to train or tune.

## Licence

MIT. `evaluation/`, `starter/`, `games.py` and the organisers' documents are © Metrica Sports
under the MIT licence in [LICENSE](LICENSE), copied from
[metrica-sports/more-than-a-hack-2027](https://github.com/metrica-sports/more-than-a-hack-2027)
at commit `f8f2629`. The match data is licensed separately and is not covered.
