"""M4 gate: eclipses.

An eclipse is where everything built so far has to be right at once. Light-time,
aberration, the observer's own position on a rotating ellipsoid, the Earth's
orientation, and delta T -- and unlike every earlier milestone, the answers are
checked against *observations*, not against JPL. Eclipse circumstances are
published to the second and were watched by millions of people.

Four gates:

1. **The cones.** The umbra's length follows from similar triangles, and whether
   its tip reaches the Earth decides total against annular before any timing is
   done. Two eclipses, one of each, predicted from geometry alone.
2. **Two total solar eclipses**, 2017 and 2024: the moment of greatest eclipse,
   where on the ground the axis lands, and how long totality runs there.
3. **A total lunar eclipse**, March 2025: all four contact times.
4. **Saros.** The same eclipse comes back 6585.32 days later, a third of a turn
   west. Structure rather than a single number.

Usage::

    python scripts/validate_m4.py
    python scripts/validate_m4.py --offline
"""

from __future__ import annotations

import argparse
import sys
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from orrery import eclipse, events, observer, times, truth  # noqa: E402

SAROS_DAYS = 6585.3213

# Published circumstances. Greatest eclipse in UT, position on the ground, and
# the longest totality anywhere on the track.
SOLAR = {
    "2017-08-21": {
        "greatest": (2017, 8, 21, 18, 26),
        "latitude": 36.97,
        "longitude": -87.65,
        "duration_s": 160.2,
        "site": observer.Site("Hopkinsville KY", 36.86, -87.49, 170.0),
    },
    "2024-04-08": {
        "greatest": (2024, 4, 8, 18, 17),
        "latitude": 25.30,
        "longitude": -104.13,
        "duration_s": 268.0,
        "site": observer.Site("Nazas, Durango", 25.25, -104.13, 1250.0),
    },
}

# 2025 March 14, total lunar. Contact times in UT.
LUNAR_CONTACTS = {
    "P1 penumbra begins": (3, 57),
    "U1 partial begins": (5, 9),
    "U2 totality begins": (6, 26),
    "greatest": (6, 59),
    "U3 totality ends": (7, 31),
    "U4 partial ends": (8, 47),
    "P4 penumbra ends": (10, 0),
}

# Greatest eclipse is the instant the shadow axis passes closest to the Earth's
# centre, and that minimum is very flat: for 2024 the miss distance changes by
# under 3 km across three minutes. So the *time* is badly conditioned even
# though the geometry is not -- the same shape of problem as the great
# conjunction in M1, where a 2.5 arcmin position error moved a flat minimum by
# ten hours. The position and the duration are what this milestone can claim
# tightly; the instant gets a tolerance that reflects the conditioning.
GREATEST_TOLERANCE_MINUTES = 3.0
POSITION_TOLERANCE_DEGREES = 0.4
DURATION_TOLERANCE_SECONDS = 20.0
LUNAR_TOLERANCE_MINUTES = 5.0


@dataclass
class Check:
    name: str
    passed: bool
    lines: list[str] = field(default_factory=list)


def ut(jd) -> str:
    return times.isoformat(observer.ut1_from_tdb(np.atleast_1d(jd))[0])


def window(centre_jd: float, hours: float, step_seconds: float) -> np.ndarray:
    half = hours / 48.0
    return centre_jd + np.arange(-half, half, step_seconds / 86400.0)


def gate_cones(*, offline: bool) -> Check:
    lines = ["the tip of the Moon's shadow either reaches us or it does not:"]
    passed = True

    for date, expect in ((( 2024, 4, 8, 18), "total"), ((2023, 10, 14, 18), "annular")):
        jd = np.array([times.jd(*date)])
        eph = truth.sampled_ephemeris(
            ("sun", "geocentre", "moon"), jd, allow_download=not offline
        )
        earth = eph.at("geocentre")(jd)
        moon_km = float(np.linalg.norm(eph.at("moon")(jd) - earth)) * 149597870.7
        sun_km = float(np.linalg.norm(eph.at("sun")(jd) - eph.at("moon")(jd))) * 149597870.7
        length = float(eclipse.umbra_length_km(sun_km))

        reaches = length > moon_km
        kind = "total" if reaches else "annular"
        passed = passed and kind == expect
        verdict = (
            "reaches"
            if reaches
            else f"falls {moon_km - length:,.0f} km short"
        )
        lines.append(
            f"  {date[0]}-{date[1]:02d}-{date[2]:02d}   cone {length:,.0f} km,"
            f" Moon {moon_km:,.0f} km  ->  {verdict}, {kind}"
        )

    lines += [
        "",
        "The cone is 374 000 km long and the Moon averages 384 400 km away, so the",
        "tip usually misses. Totality exists in the few percent by which the Moon's",
        "distance varies -- and so does the Sun's, which is why the cone length is",
        "not a constant either.",
    ]
    return Check("the shadow cones", passed, lines)


def gate_solar(*, offline: bool) -> Check:
    lines: list[str] = []
    passed = True

    for label, published in SOLAR.items():
        centre = times.jd(*published["greatest"])
        jd = window(centre, 4.0, 2.0)
        eph = truth.sampled_ephemeris(
            ("sun", "geocentre", "moon"), jd, allow_download=not offline
        )

        miss, _, _ = eclipse.shadow_axis(eph, jd)
        index = int(np.argmin(miss))
        greatest = jd[index]
        latitude, longitude = eclipse.shadow_landing(eph, greatest)
        longitude = (float(longitude) + 180.0) % 360.0 - 180.0

        view = eclipse.solar_view(eph, jd, site=published["site"])
        central = jd[view.total]
        duration = (central[-1] - central[0]) * 86400.0 if len(central) else 0.0

        offset = (greatest - centre) * 1440.0
        gap = np.hypot(float(latitude) - published["latitude"],
                       (longitude - published["longitude"]) * np.cos(np.radians(float(latitude))))
        duration_error = duration - published["duration_s"]

        ok = (
            abs(offset) < GREATEST_TOLERANCE_MINUTES
            and gap < POSITION_TOLERANCE_DEGREES
            and abs(duration_error) < DURATION_TOLERANCE_SECONDS
        )
        passed = passed and ok

        flat = float(
            np.interp(centre, jd, miss) - miss[index]
        )
        lines += [
            f"{label}  ({view.kind()})",
            f"  greatest    {ut(greatest)} UT      {offset:+.1f} min vs published"
            f"   (a flat minimum: only {flat:.1f} km deeper than at the published"
            " instant)",
            f"  lands at    {float(latitude):+.2f}, {longitude:+.2f}"
            f"      {gap * 111:.0f} km from published",
            f"  totality at {published['site'].name}: {duration:.0f} s"
            f"   ({duration_error:+.0f} s vs the track maximum)",
            "",
        ]

    lines.append(
        "Published values are the observed circumstances, not a JPL prediction."
    )
    return Check("two total solar eclipses", passed, lines)


def gate_lunar(*, offline: bool) -> Check:
    centre = times.jd(2025, 3, 14, 7)
    jd = window(centre, 10.0, 2.0)
    eph = truth.sampled_ephemeris(
        ("sun", "geocentre", "moon"), jd, allow_download=not offline
    )
    view = eclipse.lunar_view(eph, jd)
    radius = eclipse.MOON_RADIUS_KM

    def crossings(level):
        return events.find_crossings(jd, view.miss_km - level, 0.0)

    found = {}
    penumbral = crossings(view.penumbra_km + radius)
    partial = crossings(view.umbra_km + radius)
    total = crossings(view.umbra_km - radius)
    if len(penumbral) == 2:
        found["P1 penumbra begins"], found["P4 penumbra ends"] = penumbral
    if len(partial) == 2:
        found["U1 partial begins"], found["U4 partial ends"] = partial
    if len(total) == 2:
        found["U2 totality begins"], found["U3 totality ends"] = total
    found["greatest"] = jd[int(np.argmin(view.miss_km))]

    lines = [f"2025-03-14, {view.kind()}", ""]
    worst = 0.0
    for label, (hour, minute) in LUNAR_CONTACTS.items():
        if label not in found:
            lines.append(f"  {label:<22}NOT FOUND")
            worst = np.inf
            continue
        wanted = times.jd(2025, 3, 14, hour, minute)
        offset = (observer.ut1_from_tdb(np.array([found[label]]))[0] - wanted) * 1440.0
        worst = max(worst, abs(offset))
        lines.append(
            f"  {label:<22}{ut(found[label])[11:]} UT"
            f"   published {hour:02d}:{minute:02d}   {offset:+.1f} min"
        )

    lines += [
        "",
        "The residual is the atmosphere. The Earth's shadow is fuzzy and bigger",
        "than geometry says; this uses Danjon's 2% enlargement, and sources differ",
        f"by a minute or two over which rule to use. Worst contact {worst:.1f} min.",
    ]
    return Check("a total lunar eclipse", worst < LUNAR_TOLERANCE_MINUTES, lines)


def gate_saros(*, offline: bool) -> Check:
    """One saros on: the same eclipse, a third of a turn west."""
    first = times.jd(*SOLAR["2024-04-08"]["greatest"])
    later = first + SAROS_DAYS

    places = []
    for centre in (first, later):
        jd = window(centre, 5.0, 10.0)
        eph = truth.sampled_ephemeris(
            ("sun", "geocentre", "moon"), jd, allow_download=not offline
        )
        miss, _, _ = eclipse.shadow_axis(eph, jd)
        index = int(np.argmin(miss))
        latitude, longitude = eclipse.shadow_landing(eph, jd[index])
        places.append(
            (jd[index], float(miss[index]), float(latitude),
             (float(longitude) + 180.0) % 360.0 - 180.0)
        )

    (when_a, miss_a, lat_a, lon_a), (when_b, miss_b, lat_b, lon_b) = places
    west = (lon_a - lon_b + 180.0) % 360.0 - 180.0

    lines = [
        f"  {ut(when_a)[:10]}   axis misses centre by {miss_a:6.0f} km"
        f"   lands {lat_a:+.1f}, {lon_a:+.1f}",
        f"  {ut(when_b)[:10]}   axis misses centre by {miss_b:6.0f} km"
        f"   lands {lat_b:+.1f}, {lon_b:+.1f}",
        "",
        f"  one saros later, and the track has moved {west:.0f} degrees west",
        "",
        "A saros is 6585.32 days -- 18 years and a third of a day. The third of a",
        "day is the point: the Earth has turned 120 degrees further, so the same",
        "eclipse lands a third of the way round the world.",
    ]
    both_central = miss_a < 6378.0 and miss_b < 6378.0
    return Check("the saros", both_central and 100.0 < abs(west) < 140.0, lines)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--offline", action="store_true")
    args = parser.parse_args()

    table = observer._tabulated_delta_t()
    print("\nM4 -- eclipses\n")
    print(
        f"  delta T from {'measured values' if table else 'the POLYNOMIAL FIT'}"
        f"{'' if table else ' -- run observer.build_delta_t_table() for real timing'}\n"
    )

    checks = []
    for gate in (gate_cones, gate_solar, gate_lunar, gate_saros):
        check = gate(offline=args.offline)
        checks.append(check)
        print(f"  [{'ok' if check.passed else 'FAIL'}]  {check.name}")
        for line in check.lines:
            print(f"        {line}")
        print()

    passing = sum(c.passed for c in checks)
    print(f"M4 {'passing' if passing == len(checks) else 'FAILED'}, "
          f"{passing}/{len(checks)}")
    return 0 if passing == len(checks) else 1


if __name__ == "__main__":
    raise SystemExit(main())
