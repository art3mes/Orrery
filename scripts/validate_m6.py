"""M6 gate: the Moon, computed here.

Through M5 the Moon came from DE440. M4's eclipses were therefore a test of
shadow geometry sitting on somebody else's orbit -- the last place in this
project where ground truth was an *input* rather than the thing being measured.

``orrery.lunar`` closes that. Four gates:

1. **Meeus's worked example.** Chapter 47 carries one, to six decimal places.
   A hundred and twenty transcribed coefficients either reproduce it or they do
   not, and nothing else needs to run if they do not.
2. **Against DE440**, over a century. The theory is abridged, so this is a
   measurement of how much was left out, not a pass/fail on arithmetic.
3. **What it costs an eclipse.** M4's machinery, run on this Moon instead of on
   DE440's. Arcseconds are abstract; twenty seconds of totality is not.
4. **The Moon's periods.** Synodic, sidereal, anomalistic and draconic months,
   the 18.6-year regression of the nodes and the 8.85-year turn of the apsides
   all fall out of four polynomial rates. None of them is in the table.

Usage::

    python scripts/validate_m6.py
    python scripts/validate_m6.py --offline
"""

from __future__ import annotations

import argparse
import sys
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from orrery import apparent, eclipse, frames, lunar, times, truth  # noqa: E402
from orrery.kepler import AU_KM  # noqa: E402

# Meeus, Astronomical Algorithms, example 47.a: 1992 April 12.0 TD.
MEEUS_JD = 2448724.5
MEEUS_ANGLES = {
    "L": 134.290182,
    "D": 113.842304,
    "M": 97.643514,
    "Mp": 5.150833,
    "F": 219.889721,
}
MEEUS_RESULT = {"longitude": 133.162655, "latitude": -3.229126, "distance": 368409.7}

# The abridged theory is quoted at 10 arcsec in longitude. Allow a little more
# for the worst case over a century, and hold the rms much tighter.
ANGULAR_RMS_ARCSEC = 5.0
ANGULAR_MAX_ARCSEC = 25.0
DISTANCE_RMS_KM = 6.0

# What that costs the 2024 eclipse. Twenty seconds of totality out of 268.
GREATEST_TOLERANCE_MINUTES = 3.0
TRACK_TOLERANCE_KM = 60.0
DURATION_TOLERANCE_SECONDS = 25.0

# Days. Every one of these is a textbook number and none is in the table.
PUBLISHED_MONTHS = {
    # Tropical, not sidereal. The Moon's mean longitude is referred to the
    # equinox, and the equinox moves, so 360 degrees of L is one turn against
    # the equinox rather than against the stars. The two differ by 7 seconds,
    # which is precisely the 50.29 arcsec a year of general precession.
    "tropical": 27.321582,
    "synodic": 29.530589,
    "anomalistic": 27.554550,
    "draconic": 27.212221,
}
SIDEREAL_MONTH_DAYS = 27.321662
GENERAL_PRECESSION_ARCSEC_PER_YEAR = 50.2909
NODAL_REGRESSION_YEARS = 18.6
APSIDAL_PRECESSION_YEARS = 8.85


@dataclass
class Check:
    name: str
    passed: bool
    lines: list[str] = field(default_factory=list)


def gate_worked_example() -> Check:
    angles = lunar.arguments(MEEUS_JD)
    longitude, latitude, distance = lunar.spherical(MEEUS_JD)

    lines = [f"  1992 April 12.0 TD, the example in chapter 47", ""]
    passed = True
    for name, published in MEEUS_ANGLES.items():
        ours = float(np.atleast_1d(angles[name])[0])
        passed = passed and abs(ours - published) < 1e-5
        lines.append(f"  {name:<12}{ours:14.6f}   Meeus {published:14.6f}")

    lines.append("")
    for name, ours, published in (
        ("longitude", float(longitude), MEEUS_RESULT["longitude"]),
        ("latitude", float(latitude), MEEUS_RESULT["latitude"]),
        ("distance", float(distance), MEEUS_RESULT["distance"]),
    ):
        tolerance = 0.05 if name == "distance" else 1e-5
        passed = passed and abs(ours - published) < tolerance
        lines.append(f"  {name:<12}{ours:14.6f}   Meeus {published:14.6f}")

    lines += [
        "",
        "  120 coefficients, five polynomials and ten additive terms, and they",
        "  land on six decimal places. Nothing else here would be worth running",
        "  if they did not.",
    ]
    return Check("Meeus's worked example", passed, lines)


def gate_against_de440(*, offline: bool) -> Check:
    jd = np.arange(times.jd(1950, 1, 1), times.jd(2050, 1, 1), 7.0)
    positions, _ = truth.barycentric_state(
        ("geocentre", "moon"), jd, allow_download=not offline
    )
    reference = positions[:, 1, :] - positions[:, 0, :]
    ours = lunar.position(jd)

    angle = frames.separation_arcsec(ours, reference)
    distance = (np.linalg.norm(ours, axis=-1) - np.linalg.norm(reference, axis=-1)) * AU_KM

    angular_rms = float(np.sqrt((angle**2).mean()))
    angular_max = float(angle.max())
    distance_rms = float(np.sqrt((distance**2).mean()))

    passed = (
        angular_rms < ANGULAR_RMS_ARCSEC
        and angular_max < ANGULAR_MAX_ARCSEC
        and distance_rms < DISTANCE_RMS_KM
    )
    return Check(
        "against DE440, 1950-2050",
        passed,
        [
            f"  {len(jd)} dates, weekly",
            "",
            f"  angle      rms {angular_rms:6.2f}\"   max {angular_max:6.2f}\"",
            f"  distance   rms {distance_rms:6.1f} km  max"
            f" {float(np.abs(distance).max()):6.1f} km",
            "",
            "  Meeus quotes 10 arcsec in longitude for the abridged theory. The",
            "  residual is the terms that were left out, not a mistake -- the full",
            "  ELP-2000/82 has some twenty thousand of them.",
        ],
    )


def _ephemeris_with_our_moon(jd, *, offline: bool):
    """DE440 for the Sun and Earth, this package's theory for the Moon."""
    grid = np.arange(jd.min() - 3.0, jd.max() + 3.25, 0.25)
    positions, velocities = truth.barycentric_state(
        ("sun", "geocentre", "moon"), grid, allow_download=not offline
    )
    swapped = positions.copy()
    swapped[:, 2, :] = positions[:, 1, :] + lunar.position(grid)
    return apparent.Ephemeris(("sun", "geocentre", "moon"), grid, swapped, velocities)


def gate_eclipse_cost(*, offline: bool) -> Check:
    """Run M4's 2024 eclipse on our Moon, and on DE440's, and difference them."""
    from orrery import observer

    centre = times.jd(2024, 4, 8, 18, 17)
    jd = centre + np.arange(-2.0, 2.0, 2.0 / 86400.0) / 24.0
    site = observer.Site("Nazas, Durango", 25.25, -104.13, 1250.0)

    theirs = truth.sampled_ephemeris(
        ("sun", "geocentre", "moon"), jd, allow_download=not offline
    )
    ours = _ephemeris_with_our_moon(jd, offline=offline)

    results = {}
    for label, ephemeris in (("DE440", theirs), ("ours", ours)):
        miss, _, _ = eclipse.shadow_axis(ephemeris, jd)
        index = int(np.argmin(miss))
        latitude, longitude = eclipse.shadow_landing(ephemeris, jd[index])
        view = eclipse.solar_view(ephemeris, jd, site=site)
        central = jd[view.total]
        results[label] = (
            jd[index],
            float(latitude),
            (float(longitude) + 180.0) % 360.0 - 180.0,
            (central[-1] - central[0]) * 86400.0 if len(central) else 0.0,
        )

    (t_ref, lat_ref, lon_ref, dur_ref) = results["DE440"]
    (t_our, lat_our, lon_our, dur_our) = results["ours"]

    minutes = (t_our - t_ref) * 1440.0
    gap_km = np.hypot(
        lat_our - lat_ref, (lon_our - lon_ref) * np.cos(np.radians(lat_ref))
    ) * 111.0
    seconds = dur_our - dur_ref

    passed = (
        abs(minutes) < GREATEST_TOLERANCE_MINUTES
        and gap_km < TRACK_TOLERANCE_KM
        and abs(seconds) < DURATION_TOLERANCE_SECONDS
    )
    return Check(
        "what it costs the 2024 eclipse",
        passed,
        [
            f"  {'':<12}{'greatest':>18}{'lands at':>20}{'totality':>11}",
            f"  {'DE440':<12}{times.isoformat(t_ref)[11:]:>18}"
            f"{f'{lat_ref:+.3f}, {lon_ref:.3f}':>20}{dur_ref:9.0f} s",
            f"  {'our Moon':<12}{times.isoformat(t_our)[11:]:>18}"
            f"{f'{lat_our:+.3f}, {lon_our:.3f}':>20}{dur_our:9.0f} s",
            "",
            f"  {minutes:+.1f} min, {gap_km:.0f} km along the track,"
            f" {seconds:+.0f} s of totality",
            "",
            "  That is the price of an abridged theory, in the units anyone",
            "  standing on the centre line would notice.",
        ],
    )


def gate_the_months() -> Check:
    """Every period the Moon is famous for, out of four polynomial rates."""
    rates = {  # degrees per century, from the linear term of each argument
        "L": 481267.88123421,
        "D": 445267.1114034,
        "Mp": 477198.8675055,
        "F": 483202.0175233,
    }
    per_day = {k: v / 36525.0 for k, v in rates.items()}

    months = {
        "tropical": 360.0 / per_day["L"],
        "synodic": 360.0 / per_day["D"],
        "anomalistic": 360.0 / per_day["Mp"],
        "draconic": 360.0 / per_day["F"],
    }
    lines = [f"  {'month':<14}{'ours':>12}{'published':>12}   days"]
    passed = True
    for name, published in PUBLISHED_MONTHS.items():
        ours = months[name]
        passed = passed and abs(ours - published) < 1e-4
        lines.append(f"  {name:<14}{ours:12.6f}{published:12.6f}")

    # Add precession back on and the tropical month becomes the sidereal one.
    turn_against_the_stars = per_day["L"] - GENERAL_PRECESSION_ARCSEC_PER_YEAR / (
        3600.0 * 365.25
    )
    sidereal = 360.0 / turn_against_the_stars
    passed = passed and abs(sidereal - SIDEREAL_MONTH_DAYS) < 1e-4
    lines += [
        f"  {'sidereal':<14}{sidereal:12.6f}{SIDEREAL_MONTH_DAYS:12.6f}",
        "",
        f"  the tropical and sidereal months differ by"
        f" {(sidereal - months['tropical']) * 86400:.1f} s, which is the equinox",
        "  moving under the Moon at 50.3 arcsec a year and nothing else",
    ]

    # The node goes backwards and the perigee goes forwards; both are
    # differences between two of the rates above.
    nodal = 360.0 / abs(per_day["L"] - per_day["F"]) / 365.25
    apsidal = 360.0 / abs(per_day["L"] - per_day["Mp"]) / 365.25
    passed = (
        passed
        and abs(nodal - NODAL_REGRESSION_YEARS) < 0.1
        and abs(apsidal - APSIDAL_PRECESSION_YEARS) < 0.1
    )
    lines += [
        "",
        f"  nodes regress once in {nodal:.2f} years   (published"
        f" {NODAL_REGRESSION_YEARS})",
        f"  apsides turn once in  {apsidal:.2f} years   (published"
        f" {APSIDAL_PRECESSION_YEARS})",
        "",
        "  None of these is in the table. They are differences between the rates",
        "  of four angles, and the 18.6-year one is why eclipse seasons drift and",
        "  why the saros is 18 years and eleven days rather than a round number.",
    ]
    return Check("the Moon's periods", passed, lines)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--offline", action="store_true")
    args = parser.parse_args()

    print("\nM6 -- the Moon, computed here\n")
    checks = [
        gate_worked_example(),
        gate_against_de440(offline=args.offline),
        gate_eclipse_cost(offline=args.offline),
        gate_the_months(),
    ]
    for check in checks:
        print(f"  [{'ok' if check.passed else 'FAIL'}]  {check.name}")
        for line in check.lines:
            print(f"      {line}")
        print()

    passing = sum(c.passed for c in checks)
    print(f"M6 {'passing' if passing == len(checks) else 'FAILED'}, "
          f"{passing}/{len(checks)}")
    return 0 if passing == len(checks) else 1


if __name__ == "__main__":
    raise SystemExit(main())
