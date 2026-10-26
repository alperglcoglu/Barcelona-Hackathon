"""Write a submission and check it against the contract in SUBMISSION.md.

The check here is a fast first line of defence that needs nothing but numpy. It does not replace
evaluation/validate_submission.py, which runs the official scorer's own checks; pipeline.py prints
the command to run it on the file it just wrote.
"""

import os

import numpy as np

# Params
VALID_LABELS = ("H", "A", "D", "X")


def check(labels, n_frames):
    """None if the labels satisfy the contract, otherwise a one-line reason."""
    if len(labels) != n_frames:
        return "expected {} rows, got {}".format(n_frames, len(labels))
    bad = ~np.isin(labels, VALID_LABELS)
    if bad.any():
        return "invalid label {!r} at frame {}".format(labels[bad][0], int(np.argmax(bad)))
    return None


def write(path, labels):
    """A frame,label CSV with frame 0..n-1 in order, no index column."""
    folder = os.path.dirname(path)
    if folder:
        os.makedirs(folder, exist_ok=True)
    with open(path, "w", newline="\n") as handle:
        handle.write("frame,label\n")
        handle.writelines("{},{}\n".format(frame, label) for frame, label in enumerate(labels))
    return path


def read(path):
    """Labels of a frame,label CSV as an array, in frame order. Used for parity checks."""
    with open(path) as handle:
        next(handle)
        return np.array([line.rstrip("\r\n").split(",")[1].strip().upper() for line in handle])
