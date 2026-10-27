"""Regenerate a submission from the provided data. This is the command the technical document cites.

    python -m possession.pipeline                         # default method, file named after the team
    python -m possession.pipeline --method baseline --out evaluation/results/baseline_check.csv
    python -m possession.pipeline --check-baseline        # parity with the organisers' floor

The run is deterministic: no randomness, no network, no state outside data/. The same data in gives
the same file out, which is what RULES.md section 3 requires.
"""

import argparse
import hashlib
import os
import sys

import numpy as np

from . import config, io_atd, io_tagging, methods, submission

sys.path.insert(0, config.ROOT)
from games import GAME, source_frames  # noqa: E402

# Params
DEFAULT_METHOD = "state_model"
BASELINE_REFERENCE = os.path.join(config.RESULTS_DIR, "baseline_smooth50.csv")


def summarise(labels):
    """Label counts and possession count -- the first numbers worth reading on any new file."""
    counts = {label: int(np.sum(labels == label)) for label in submission.VALID_LABELS}
    alive = np.isin(labels, ("H", "A"))
    starts = alive & np.concatenate(([True], labels[1:] != labels[:-1]))
    return counts, int(np.sum(starts))


def check_baseline(data):
    """Does our loader reproduce the organisers' baseline_smooth50.csv frame for frame?"""
    ours = methods.baseline(data, window=50)
    theirs = submission.read(BASELINE_REFERENCE)
    differ = np.flatnonzero(ours != theirs)
    if len(differ) == 0:
        print("PARITY OK  baseline matches {} on all {} frames".format(
            os.path.basename(BASELINE_REFERENCE), len(ours)))
        return True
    print("PARITY FAIL  {} frames differ, first at frame {}".format(len(differ), differ[0]))
    return False


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--method", default=DEFAULT_METHOD, choices=sorted(methods.METHODS))
    parser.add_argument("--out", help="output CSV (default: evaluation/results/<team>.csv)")
    parser.add_argument("--data-dir", help="data bundle location (default: ./data or MTAH_DATA_DIR)")
    parser.add_argument("--no-cache", action="store_true", help="re-parse the tracking")
    parser.add_argument("--no-tagging", action="store_true",
                        help="ignore the Smart Tagging (ablation: tracking only)")
    parser.add_argument("--check-baseline", action="store_true",
                        help="only verify the loader against the organisers' baseline file")
    args = parser.parse_args(argv)

    data = io_atd.load(GAME, args.data_dir, use_cache=not args.no_cache)
    if args.check_baseline:
        return 0 if check_baseline(data) else 1

    method = methods.METHODS[args.method]
    if args.method in methods.USES_TAGGING and not args.no_tagging:
        labels = method(data, io_tagging.load(GAME, args.data_dir))
    else:
        labels = method(data)
    reason = submission.check(labels, source_frames("tactical", GAME))
    if reason:
        print("FAIL  {}".format(reason))
        return 1

    out = args.out or os.path.join(config.RESULTS_DIR, "{}.csv".format(config.TEAM_NAME))
    submission.write(out, labels)
    counts, n_possessions = summarise(labels)
    print("wrote {}".format(os.path.relpath(out, config.ROOT)))
    print("  " + "  ".join("{} {}".format(k, v) for k, v in counts.items()))
    print("  {} possessions".format(n_possessions))
    with open(out, "rb") as handle:
        print("  sha256 {}".format(hashlib.sha256(handle.read()).hexdigest()))
    print("\nnext:  python evaluation/validate_submission.py {}".format(out))
    print("       python evaluation/run_evaluation.py")
    return 0


if __name__ == "__main__":
    sys.exit(main())
