"""Read the Smart Tagging XML as intervals on the video-frame axis.

Smart Tagging is Metrica's automatic tactical annotation: 638 phases and set pieces, each with a
code, a start and end in seconds from the first tactical-video frame, and tags such as Team and
Type. `round(seconds * 25)` is a video frame and nothing else needs correcting (DATA.md, verified
by the organisers against both kickoffs).

It is evidence, not an answer: it is automatic, it has false positives and negatives, every
throw-in is tagged Team = N/A, and its intervals are phases rather than possessions. This module
only reads it; how much to trust it is decided where it is used.
"""

import sys
import xml.etree.ElementTree as ET

from . import config

sys.path.insert(0, config.ROOT)
from games import GAME, GAMES, require  # noqa: E402

# Params
# The XML names clubs; the submission wants a side. Spelling as it appears in the export
# ("Atletic", not "Athletic"), matched case-insensitively.
TEAM_SIDE = {"barcelona": "H", "atletic club": "A"}


def load(game=GAME, data_dir=None):
    """[{code, start_frame, end_frame, side, tags}] in file order.

    `code` is upper-cased and stripped, tag values are kept as written. `side` is H, A or None
    (for N/A or an unknown team). end_frame is exclusive, like every interval in this package.
    """
    fps = GAMES[game]["fps"]
    path = require(["smart_tagging"], game, data_dir)[0]
    out = []
    for instance in ET.parse(path).iter("instance"):
        tags = {label.findtext("group"): label.findtext("text")
                for label in instance.findall("label")}
        team = (tags.get("Team") or "").strip().lower()
        out.append({
            "code": (instance.findtext("code") or "").strip().upper(),
            "start_frame": int(round(float(instance.findtext("start")) * fps)),
            "end_frame": int(round(float(instance.findtext("end")) * fps)),
            "side": TEAM_SIDE.get(team),
            "tags": tags,
        })
    return out
