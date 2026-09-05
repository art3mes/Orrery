"""One command, one date.

    orrery                      today
    orrery 2027-08-02           that date
    orrery 2027-08-02 --at cairo
    orrery 1969-07-20 --no-viewer

Prints where everything was, then opens the 3-D view on the same date. The
viewer's slider runs the whole 1850-2050 range, so the date is a starting
point rather than a commitment.

Everything below is computed here -- ``model.ephemeris`` packs this package's
own orbits into the same object DE440 comes out of -- so it needs no download
and no network. How far that is from JPL is measured in the README.
"""

from __future__ import annotations

import argparse
import sys

import numpy as np

from . import elements, eclipse, events, frames, lunar, model, observer, times
from .kepler import AU_KM

PLANETS = ("mercury", "venus", "mars", "jupiter", "saturn", "uranus", "neptune", "pluto")

# Named by the Moon's elongation in ecliptic longitude, which runs 0 to 360
# through the month. The lit fraction alone cannot name a phase: it is 33% for
# a waxing crescent and 33% for a waning one, and those are a fortnight apart.
PHASES = (
    (2.0, "new"), (88.0, "waxing crescent"), (92.0, "first quarter"),
    (178.0, "waxing gibbous"), (182.0, "full"), (268.0, "waning gibbous"),
    (272.0, "last quarter"), (358.0, "waning crescent"), (360.1, "new"),
)


def parse_date(text: str) -> float:
    """``YYYY-MM-DD`` or ``YYYY-MM-DD HH:MM``, read as TDB."""
    date, _, clock = text.partition(" ")
    parts = [int(p) for p in date.split("-")]
    if len(parts) != 3:
        raise argparse.ArgumentTypeError("date must be YYYY-MM-DD")
    hour = minute = 0
    if clock:
        bits = clock.split(":")
        hour, minute = int(bits[0]), int(bits[1]) if len(bits) > 1 else 0
    return times.jd(*parts, hour, minute)


def _hms(hours: float) -> str:
    h = int(hours)
    m = (hours - h) * 60.0
    return f"{h:02d}h {m:04.1f}m"


def _dms(degrees: float) -> str:
    sign = "-" if degrees < 0 else "+"
    d = abs(degrees)
    return f"{sign}{int(d):02d}d {(d - int(d)) * 60:04.1f}m"


def _phase_name(elongation_deg: float) -> str:
    for edge, name in PHASES:
        if elongation_deg < edge:
            return name
    return "new"


def _moon_elongation(jd_tdb: float) -> float:
    """Moon minus Sun in ecliptic longitude: 0 at new, 180 at full, degrees."""
    when = np.atleast_1d(float(jd_tdb))
    place, _ = model.states(("geocentre", "moon"), when)
    sun = frames.equatorial_to_ecliptic(-place[:, 0])
    moon = frames.equatorial_to_ecliptic(place[:, 1] - place[:, 0])

    def longitude(v):
        return np.degrees(np.arctan2(v[:, 1], v[:, 0]))

    return float((longitude(moon) - longitude(sun)) % 360.0)


def report(jd_tdb: float, site: observer.Site | None = None, out=sys.stdout) -> None:
    """Where everything is, from the Earth, on one date."""
    bodies = ("sun", "geocentre", "moon") + PLANETS
    eph = model.ephemeris(jd_tdb, bodies=bodies)
    when = np.atleast_1d(float(jd_tdb))

    where = "geocentric" if site is None else site.name
    print(f"\n{times.isoformat(jd_tdb)} TDB   --   as seen from {where}\n", file=out)
    print("  body        RA            Dec         from Earth   from Sun"
          "   elongation", file=out)
    print("  " + "-" * 74, file=out)

    sun = eph.look("sun", when, site=site)
    sun_direction = sun.apparent / np.linalg.norm(sun.apparent)

    for body in ("sun",) + PLANETS + ("moon",):
        sight = eph.look(body, when, site=site)
        ra, dec = frames.radec(sight.apparent)
        distance = float(sight.distance[0])

        if body == "sun":
            heliocentric = "--"
            elongation = "--"
        else:
            place, _ = model.states((body,), when)
            heliocentric = f"{np.linalg.norm(place[0, 0]):9.3f}"
            angle = frames.separation_arcsec(sight.apparent, sun_direction) / 3600.0
            elongation = f"{float(angle):7.1f}d"

        label = "Moon" if body == "moon" else body.capitalize()
        print(f"  {label:<10}  {_hms(float(ra[0]))}   {_dms(float(dec[0]))}"
              f"   {distance:9.4f}   {heliocentric:>9}   {elongation:>10}", file=out)

    # The Moon, in the terms people actually ask about it.
    moon = eph.look("moon", when, site=site)
    separation = float(frames.separation_arcsec(moon.apparent, sun.apparent)) / 3600.0
    age = _moon_elongation(jd_tdb)
    lit = 0.5 * (1.0 - np.cos(np.radians(age)))
    _, _, moon_km = lunar.spherical(jd_tdb)

    print(f"\n  The Moon is {_phase_name(age)}, {lit:.0%} lit, "
          f"{float(moon_km):,.0f} km away,", file=out)
    print(f"  and {separation:.1f} degrees from the Sun in the sky.", file=out)

    # An eclipse needs those two nearly in line, which is a single comparison.
    # Only then is it worth sweeping the day, and the sweep is what turns "the
    # Moon is near the Sun" into a time and a percentage.
    if separation < 12.0:
        _eclipse_today(jd_tdb, site, out)
    elif separation > 168.0:
        print("\n  Full moon near a node -- a lunar eclipse is close to this "
              "date. scripts/validate_m4.py times them properly.", file=out)
    print(file=out)


def _eclipse_today(jd_tdb: float, site: observer.Site | None, out) -> None:
    """Sweep the day either side and say whether the Moon covers the Sun here.

    288 samples at five minutes. The Moon takes hours to cross the Sun, so this
    resolves an eclipse comfortably, and ``find_extrema`` refines the peak to
    finer than the grid it was found on.
    """
    midnight = np.floor(jd_tdb - 0.5) + 0.5
    scan = midnight + np.arange(-0.5, 1.5, 5.0 / 1440.0)
    eph = model.ephemeris(scan, bodies=("sun", "geocentre", "moon"))
    view = eclipse.solar_view(eph, scan, site=site)

    peaks = events.find_extrema(scan, view.obscuration, kind="max", threshold=0.0)
    if not peaks:
        return

    at, covered = max(peaks, key=lambda pair: pair[1])
    where = "" if site is None else f" from {site.name}"
    print(f"\n  Solar eclipse{where}: {covered:.1%} of the Sun covered at "
          f"{times.isoformat(at)}.", file=out)

    miss, _, _ = eclipse.shadow_axis(eph, scan)
    index = int(np.argmin(miss))
    if float(miss[index]) < observer.EARTH_RADIUS_KM:
        latitude, longitude = eclipse.shadow_landing(eph, scan[index])
        east = (float(longitude) + 180.0) % 360.0 - 180.0
        print(f"  The centre of the shadow lands at {float(latitude):+.1f}, "
              f"{east:+.1f} at {times.isoformat(scan[index])}.", file=out)


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(
        prog="orrery",
        description="Where the planets actually were, on any date from 1850 to 2050.",
        epilog="Positions come from this package's own orbits; nothing is downloaded.",
    )
    parser.add_argument(
        "date", nargs="?", default=None,
        help="YYYY-MM-DD, optionally 'YYYY-MM-DD HH:MM'. Default: today.",
    )
    parser.add_argument(
        "--at", default=None, metavar="PLACE",
        help="a named site (" + ", ".join(observer.SITES) + "), or lat,lon",
    )
    parser.add_argument(
        "--no-viewer", action="store_true", help="print the report and stop"
    )
    parser.add_argument(
        "--view", default="inner", choices=("inner", "planets", "all"),
        help="opening framing of the 3-D view",
    )
    parser.add_argument(
        "--focus", default=None, metavar="BODY",
        help="open zoomed in on one body at its true shape: sun, "
             + ", ".join(elements.BODIES) + ". Saturn and Uranus bring rings.",
    )
    args = parser.parse_args(argv)

    if args.date is None:
        import datetime

        now = datetime.datetime.now(datetime.timezone.utc)
        jd_tdb = times.jd(now.year, now.month, now.day, now.hour, now.minute)
    else:
        jd_tdb = parse_date(args.date)

    low, high = elements.VALID_JD
    if not low <= jd_tdb <= high:
        parser.error(
            f"{times.isoformat(jd_tdb)} is outside 1800-2050, where the element "
            "table is defined"
        )

    site = None
    if args.at:
        if "," in args.at:
            latitude, _, longitude = args.at.partition(",")
            site = observer.Site(args.at, float(latitude), float(longitude), 0.0)
        elif args.at.lower() in observer.SITES:
            site = observer.SITES[args.at.lower()]
        else:
            parser.error(
                f"unknown place {args.at!r}; use lat,lon or one of: "
                + ", ".join(observer.SITES)
            )

    report(jd_tdb, site)

    if args.no_viewer:
        return 0

    try:
        from .view import Orrery
    except ImportError:
        print("  (the 3-D view needs polyscope: pip install -e \".[viz]\")")
        return 0

    focus = None
    if args.focus:
        focus = "sun" if args.focus.lower() == "sun" else elements.canonical(args.focus)

    Orrery(jd_tdb, view=args.view).run(focus=focus)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
