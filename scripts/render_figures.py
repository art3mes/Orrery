"""Regenerate the 3-D figures in ``docs/images`` from the current code.

    python scripts/render_figures.py                    # all of them
    python scripts/render_figures.py saturn-in-orbit    # just these

A window opens while it runs. ``earth-from-the-sun.png`` comes from
``demo_m5.py --body earth --from-sun --screenshot``, and the charts and the
eclipse map come from their own demos.
"""

from __future__ import annotations

import argparse
import sys
from dataclasses import dataclass
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from orrery.kepler import position  # noqa: E402
from orrery.times import jd  # noqa: E402
from orrery.view import Orrery  # noqa: E402

IMAGES = Path(__file__).resolve().parents[1] / "docs" / "images"

SEP_3 = jd(2026, 9, 3)
SEP_6 = jd(2026, 9, 6)


@dataclass(frozen=True)
class Figure:
    date: float
    view: str = "all"
    focus: str | None = None  # one body, close up
    from_sun: bool = True
    planet_size: float | None = None  # exaggeration; None keeps the preset's
    reach: float | None = None  # camera distance in au; None keeps the preset's
    target: str | None = None  # what the camera looks at; None is the Sun
    lift: float = 0.5  # how far above the ecliptic, as a fraction of reach


FIGURES = {
    "inner-system": Figure(SEP_3, view="inner"),
    "system-wide": Figure(SEP_6, planet_size=1200.0, reach=14.0, lift=0.55),
    "system-inner": Figure(SEP_6, planet_size=700.0, reach=4.5),
    "saturn-in-orbit": Figure(
        SEP_6, planet_size=900.0, reach=3.2, target="saturn", lift=0.35
    ),
    "jupiter-in-orbit": Figure(
        SEP_6, planet_size=900.0, reach=2.4, target="jupiter", lift=0.3
    ),
    "follow-mars": Figure(SEP_6, view="inner", focus="mars"),
    "rings-jupiter": Figure(SEP_6, view="inner", focus="jupiter", from_sun=False),
    "rings-saturn": Figure(SEP_6, view="inner", focus="saturn", from_sun=False),
    "rings-uranus": Figure(SEP_6, view="inner", focus="uranus", from_sun=False),
}


def aim(figure: Figure) -> None:
    """Sit out and to the side of the target, lifted off the ecliptic."""
    import polyscope as ps

    up = np.array([0.0, 0.0, 1.0])
    if figure.target is None:
        centre = np.zeros(3)
        outward = np.array([0.0, -1.0, 0.0])
    else:
        centre = np.asarray(position(figure.target, figure.date), dtype=float)
        outward = centre / np.linalg.norm(centre)
    side = np.cross(outward, up)
    side /= np.linalg.norm(side)

    eye = centre + figure.reach * (0.55 * outward + 0.75 * side + figure.lift * up)
    ps.look_at(tuple(eye), tuple(centre))


def render(name: str) -> None:
    import polyscope as ps

    figure = FIGURES[name]
    ps.remove_all_structures()

    orrery = Orrery(figure.date, view=figure.view)
    orrery.build()
    ps.set_user_callback(orrery.gui)

    if figure.planet_size is not None:
        orrery.planet_exaggeration = figure.planet_size
        orrery._place_globes()
    if figure.focus is not None:
        orrery.focus_from_sun = figure.from_sun
        orrery.focus_on(figure.focus)
    if figure.reach is not None:
        aim(figure)

    # Taken from inside the callback, a few frames in, so the orbit lines have
    # had a frame to settle to the camera's distance.
    orrery._screenshot_path = str(IMAGES / f"{name}.png")
    orrery._screenshot_frame = 4
    ps.show(forFrames=7)
    print(f"wrote docs/images/{name}.png")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("names", nargs="*", metavar="NAME", help=", ".join(FIGURES))
    args = parser.parse_args()
    unknown = sorted(set(args.names) - set(FIGURES))
    if unknown:
        parser.error(f"unknown figure: {', '.join(unknown)}")

    import polyscope as ps

    ps.init()
    ps.set_window_size(1280, 720)
    for name in args.names or FIGURES:
        render(name)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
