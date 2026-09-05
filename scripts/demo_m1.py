"""Launch the 3-D orrery.

    python scripts/demo_m1.py                       # interactive
    python scripts/demo_m1.py --date 2020-12-21     # start on a date
    python scripts/demo_m1.py --frames 3 --backend openGL_mock   # wiring smoke test

Scene units are astronomical units. Positions are exactly where the planets
were; the spheres are drawn hundreds of times too large, because at true scale
the Earth is about a pixel across when its whole orbit is in frame. The
exaggeration factor sits on screen next to the sliders that set it.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from orrery.times import jd  # noqa: E402
from orrery.view import Orrery  # noqa: E402


def parse_date(text: str) -> float:
    parts = [int(p) for p in text.split("-")]
    if len(parts) != 3:
        raise argparse.ArgumentTypeError("date must be YYYY-MM-DD")
    return jd(*parts)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--date", type=parse_date, default=None, help="YYYY-MM-DD")
    parser.add_argument(
        "--frames", type=int, default=None, help="run N frames then exit"
    )
    parser.add_argument("--screenshot", default=None, help="write a PNG on exit")
    parser.add_argument(
        "--view", default="inner", choices=("inner", "planets", "all"),
        help="opening framing",
    )
    parser.add_argument(
        "--backend",
        default="",
        help="polyscope backend; openGL_mock runs with no display",
    )
    args = parser.parse_args()

    orrery = Orrery(args.date, view=args.view)
    orrery.run(frames=args.frames, screenshot=args.screenshot, backend=args.backend)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
