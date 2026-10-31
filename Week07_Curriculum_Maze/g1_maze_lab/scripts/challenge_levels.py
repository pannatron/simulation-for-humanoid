"""Challenge 7.5 levels, shared by challenge.py, challenge_runner.py and challenge_server.py (no simulator imports).

    7      7 x 7, start bottom-left, goal top-right          practice
    9      9 x 9, same corners, more hazards (slalom spikes)  practice
    final  10 x 10, start and goal random from the seed      the real run (instructor only until the day)

Spike beds are deadly on every level. Practise random start / goal on 7 or 9 with ``--random-ends``.
"""

import os

LEVELS = {
    "7": {"task": "G1-Maze-Hard", "cells": 7, "time": 300, "random_ends": False},
    "9": {"task": "G1-Maze-Hard9", "cells": 9, "time": 480, "random_ends": False},
}
_FINAL = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "src", "g1_maze_lab", "tasks", "challenge_final")
if os.path.isdir(_FINAL):
    LEVELS["final"] = {"task": "G1-Maze-Final", "cells": 10, "time": 600, "random_ends": True}
DEFAULT_LEVEL = "7"


def level_info(name):
    if name not in LEVELS:
        raise SystemExit(f"unknown level {name!r}: use one of {', '.join(LEVELS)}"
                         + (" (the final level comes with the challenge day)" if name == "final" else ""))
    return LEVELS[name]
