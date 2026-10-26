"""Frame-level ball possession for More than a Hack 2027 (BAR-ATH).

The package turns the provided automatic tracking (and, from step 2 on, the Smart Tagging) into one
H / A / D / X label per tactical-video frame. Layout:

    config.py      every tunable number, in one place, each with the reason for its value
    io_atd.py      the raw FIFA/EPTS tracking -> dense numpy arrays, cached to data/cache/
    io_tagging.py  the Smart Tagging XML -> intervals on the video-frame axis
    methods.py     labelling methods; `baseline` reproduces the organisers' floor exactly
    submission.py  write the frame,label CSV and check it against the submission contract
    pipeline.py    the single command that regenerates a submission from the data

    python -m possession.pipeline --method baseline --out evaluation/results/baseline_check.csv
"""
