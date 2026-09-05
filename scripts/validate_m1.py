"""M1 gate: is the picture the right picture?

M0 established that the positions are good to tens or hundreds of arcseconds.
M1 adds a rendering layer on top, and a rendering layer can be wrong in ways the
positions cannot: an orbit ring drawn from the wrong epoch's elements, a trail
that silently extrapolates past the table, a planet drawn somewhere its own
orbit does not go. None of those look wrong on screen. All of them are
checkable without opening a window, which is why ``scene.py`` holds the geometry
and ``view.py`` holds nothing but wiring.

Four gates:

1. **The rings are the orbits.** Every planet sits on the ellipse drawn for it,
   to within that polyline's own sagitta, and every trail ends exactly at the
   planet and never reaches outside the element table.
2. **The great conjunction of 2020.** Jupiter and Saturn's closest approach,
   found by the same code on this model and on DE440.
3. **The Venus transits of 2004 and 2012.** Venus has to actually land on the
   solar disc, and leave it at about the right time.
4. **Mars oppositions, 1990-2050.** Every one found, none invented, each within
   a day of where DE440 puts it.

Events are computed from DE440 by the *same* finder, never against dates copied
out of an almanac. An almanac disagreement cannot tell you whether the model or
the definition is at fault -- published opposition dates use apparent geocentric
right ascension, this uses maximum elongation, and the two differ by hours all
on their own.

Usage::

    python scripts/validate_m1.py
    python scripts/validate_m1.py --offline    # require cached ground truth
"""

from __future__ import annotations

import argparse
import sys
import warnings
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from orrery import elements, events, frames, kepler, scene, times, truth  # noqa: E402

# --- tolerances -------------------------------------------------------------

# The separation curve at a conjunction is nearly flat, so a small position
# error moves the minimum by hours. A day is the resolution this model can
# honestly claim; see the note in demo_m0.py.
CONJUNCTION_TOLERANCE_DAYS = 1.0

# Same argument, applied to the elongation maximum at opposition.
OPPOSITION_TOLERANCE_DAYS = 1.0

# Venus crosses the Sun's disc in about six hours. Contact times are much more
# forgiving than they sound, because the model's error is mostly along-track and
# shared between the two bodies.
CONTACT_TOLERANCE_HOURS = 1.0


@dataclass
class Check:
    name: str
    passed: bool
    lines: list[str] = field(default_factory=list)


def our_positions(bodies, jd) -> dict[str, np.ndarray]:
    return {b: kepler.position(b, jd) for b in bodies}


def reference_positions(bodies, jd, *, offline: bool) -> dict[str, np.ndarray]:
    """DE440 positions.

    Returned in the equatorial frame while ours are ecliptic. That is harmless
    here and deliberately left alone: every quantity this script compares is an
    angle between vectors or a date, and both are invariant under the rotation
    between the two frames. Rotating one side would only add a step that could
    itself be wrong.
    """
    fetch = truth.cached_only if offline else truth.heliocentric_equatorial
    return fetch(bodies, jd)


# --- gate 1: the rings are the orbits ---------------------------------------


def _point_to_polyline(point: np.ndarray, loop: np.ndarray) -> float:
    """Shortest distance from *point* to a closed polyline, in au."""
    starts = loop
    ends = np.roll(loop, -1, axis=0)
    segment = ends - starts
    length_squared = np.sum(segment**2, axis=-1)
    t = np.clip(
        np.sum((point - starts) * segment, axis=-1) / length_squared, 0.0, 1.0
    )
    closest = starts + t[:, None] * segment
    return float(np.min(frames.norm(point - closest)))


def _max_sagitta(body: str, jd: float, samples: int) -> float:
    """How far the drawn polyline cuts inside the true ellipse, at most.

    Sampling the same ellipse at twice the resolution puts a true curve point at
    every chord midpoint; the gap between them is the sagitta, and it is the
    floor on how close a point on the curve can be to the polyline.
    """
    fine = kepler.ellipse(body, jd, samples=2 * samples)
    vertices, midpoints = fine[0::2], fine[1::2]
    chord_midpoints = 0.5 * (vertices + np.roll(vertices, -1, axis=0))
    return float(np.max(frames.norm(midpoints - chord_midpoints)))


def gate_orbit_geometry(samples: int = 512) -> Check:
    # 2049, not 2050: the table ends on 2050-01-01, and a mid-2050 date is
    # already extrapolation. The warnings-as-errors block below catches that,
    # which is how this line came to say 2049.
    dates = [times.jd(y, 6, 15) for y in (1850, 1920, 1985, 2026, 2049)]
    worst_ratio, worst_body = 0.0, ""
    worst_trail_gap = 0.0

    with warnings.catch_warnings():
        # A trail that reached outside the element table would warn; here that
        # must be an error, since on screen it would just be a slightly wrong
        # curve.
        warnings.simplefilter("error", RuntimeWarning)
        for jd in dates:
            for body in elements.ORDER:
                loop = scene.orbit_loop(body, jd, samples)
                here = kepler.position(body, jd)

                gap = _point_to_polyline(here, loop)
                allowed = _max_sagitta(body, jd, samples)
                ratio = gap / allowed
                if ratio > worst_ratio:
                    worst_ratio, worst_body = ratio, body

                span = scene.trail_span_days(body, scene.DEFAULT_TRAIL_FRACTION)
                path = scene.trail(body, jd, span, samples=200)
                worst_trail_gap = max(
                    worst_trail_gap, float(frames.norm(path[-1] - here))
                )

    # Not zero: the trail's last sample and the planet's position take slightly
    # different routes through the same arithmetic (array broadcast versus
    # scalar), so they agree to the last few bits rather than exactly. 1e-9 au
    # is 150 m -- still far tighter than anything physical, and immune to that.
    passed = worst_ratio <= 1.05 and worst_trail_gap < 1e-9
    return Check(
        "orbit rings and trails",
        passed,
        [
            f"planet-to-ring gap, worst case {worst_ratio:.3f} x the polyline's own"
            f" sagitta ({worst_body})",
            f"trail endpoint vs planet, worst case {worst_trail_gap:.2e} au",
            "no trail sampled outside the element table's 1800-2050 window",
        ],
    )


# --- gate 2: the great conjunction ------------------------------------------


def _closest_approach(a: str, b: str, jd: np.ndarray, pos) -> tuple[float, float]:
    separation = events.separation_from_earth(pos[a], pos[b], pos["embary"])
    found = events.find_extrema(jd, separation, kind="min")
    if not found:
        return float("nan"), float("nan")
    return min(found, key=lambda pair: pair[1])


def gate_great_conjunction(*, offline: bool) -> Check:
    jd = np.arange(times.jd(2020, 12, 1), times.jd(2021, 1, 10), 0.005)
    ours = our_positions(["embary", "jupiter", "saturn"], jd)
    theirs = reference_positions(["embary", "jupiter", "saturn"], jd, offline=offline)

    our_when, our_sep = _closest_approach("jupiter", "saturn", jd, ours)
    their_when, their_sep = _closest_approach("jupiter", "saturn", jd, theirs)
    offset_hours = (our_when - their_when) * 24.0

    passed = abs(offset_hours) < CONJUNCTION_TOLERANCE_DAYS * 24.0
    return Check(
        "Jupiter-Saturn conjunction 2020",
        passed,
        [
            f"ours    {times.isoformat(our_when)}   {our_sep / 60:.3f} arcmin",
            f"DE440   {times.isoformat(their_when)}   {their_sep / 60:.3f} arcmin",
            f"offset  {offset_hours:+.1f} h   (allowed"
            f" {CONJUNCTION_TOLERANCE_DAYS * 24:.0f} h)",
        ],
    )


# --- gate 3: the Venus transits ---------------------------------------------


def _transit(jd: np.ndarray, pos) -> tuple[float, float, list[float]]:
    """Minimum separation from the Sun's centre, the disc radius there, contacts."""
    separation = events.separation_from_earth(pos["venus"], np.zeros(3), pos["embary"])
    disc = events.solar_radius_arcsec(pos["embary"])
    index = int(np.argmin(separation))
    contacts = events.find_crossings(jd, separation - disc, 0.0)
    return float(separation[index]), float(disc[index]), contacts


def gate_venus_transit(*, offline: bool) -> Check:
    lines: list[str] = []
    passed = True

    for year, month, day in ((2004, 6, 8), (2012, 6, 6)):
        jd = np.arange(times.jd(year, month, day) - 1.0, times.jd(year, month, day) + 1.0, 0.001)
        ours = our_positions(["embary", "venus"], jd)
        theirs = reference_positions(["embary", "venus"], jd, offline=offline)

        our_min, our_disc, our_contacts = _transit(jd, ours)
        their_min, their_disc, their_contacts = _transit(jd, theirs)

        on_disc = our_min < our_disc
        lines.append(
            f"{year}: closest to Sun centre {our_min:.0f}\" against a"
            f" {our_disc:.0f}\" disc"
            f"  ->  {'on the disc' if on_disc else 'MISSED THE SUN'}"
        )
        lines.append(f"      DE440 {their_min:.0f}\" against {their_disc:.0f}\"")

        if len(our_contacts) == len(their_contacts) == 2:
            drift = [(a - b) * 24.0 for a, b in zip(our_contacts, their_contacts)]
            lines.append(
                f"      contacts {drift[0]:+.2f} h, {drift[1]:+.2f} h vs DE440"
                f"   (allowed {CONTACT_TOLERANCE_HOURS:.0f} h)"
            )
            if max(abs(d) for d in drift) > CONTACT_TOLERANCE_HOURS:
                passed = False
        else:
            lines.append(
                f"      contact count {len(our_contacts)} vs DE440"
                f" {len(their_contacts)}"
            )
            passed = False

        passed = passed and on_disc

    return Check("Venus transits 2004 and 2012", passed, lines)


# --- gate 4: Mars oppositions -----------------------------------------------


def gate_mars_oppositions(*, offline: bool) -> Check:
    jd = np.arange(times.jd(1990, 1, 1), times.jd(2050, 1, 1), 1.0)
    ours = our_positions(["embary", "mars"], jd)
    theirs = reference_positions(["embary", "mars"], jd, offline=offline)

    def oppositions(pos):
        angle = events.elongation_deg(pos["mars"], pos["embary"])
        return [when for when, _ in events.find_extrema(jd, angle, kind="max", threshold=150.0)]

    ours_found, theirs_found = oppositions(ours), oppositions(theirs)

    if len(ours_found) != len(theirs_found):
        return Check(
            "Mars oppositions 1990-2050",
            False,
            [f"found {len(ours_found)}, DE440 found {len(theirs_found)}"],
        )

    drift = np.array(ours_found) - np.array(theirs_found)
    worst = int(np.argmax(np.abs(drift)))
    passed = np.max(np.abs(drift)) < OPPOSITION_TOLERANCE_DAYS

    return Check(
        "Mars oppositions 1990-2050",
        passed,
        [
            f"{len(ours_found)} oppositions, all matched to DE440",
            f"worst  {times.isoformat(ours_found[worst])[:10]}"
            f"  {drift[worst] * 24:+.1f} h",
            f"rms    {np.sqrt(np.mean((drift * 24) ** 2)):.1f} h"
            f"   (allowed {OPPOSITION_TOLERANCE_DAYS * 24:.0f} h)",
        ],
    )


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--offline", action="store_true", help="require cached truth")
    args = parser.parse_args()

    checks = [
        gate_orbit_geometry(),
        gate_great_conjunction(offline=args.offline),
        gate_venus_transit(offline=args.offline),
        gate_mars_oppositions(offline=args.offline),
    ]

    print("\nM1 -- the 3-D scene, checked without opening a window\n")
    for check in checks:
        print(f"  [{'ok' if check.passed else 'FAIL'}]  {check.name}")
        for line in check.lines:
            print(f"          {line}")
        print()

    passing = sum(c.passed for c in checks)
    print(f"M1 {'passing' if passing == len(checks) else 'FAILED'}, "
          f"{passing}/{len(checks)}")
    return 0 if passing == len(checks) else 1


if __name__ == "__main__":
    raise SystemExit(main())
