"""The path of a total eclipse, drawn on the ground.

Everything in this project meets here. The track is where the axis of the Moon's
shadow -- placed by DE440, corrected for light-time, intersected with a WGS84
ellipsoid turned by the IAU rotation elements and clocked by measured delta T --
crosses the Earth's surface.

The default is 2 August 2027: six minutes and twenty-three seconds of totality
over Luxor, the longest anywhere on land until 2114.

Writes ``docs/images/eclipse-<date>.png``.

    python scripts/demo_m4.py
    python scripts/demo_m4.py --date 2024-04-08 --hours 5
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from orrery import eclipse, globe, observer, times, truth  # noqa: E402

IMAGES = Path(__file__).resolve().parents[1] / "docs" / "images"

WATCHERS = {
    "2027-08-02": [
        observer.Site("Luxor, Egypt", 25.69, 32.64, 76.0),
        observer.Site("Tangier, Morocco", 35.76, -5.83, 20.0),
        observer.Site("Jeddah, Saudi Arabia", 21.49, 39.19, 12.0),
        observer.Site("Cadiz, Spain", 36.53, -6.29, 11.0),
        observer.Site("Rome, Italy", 41.90, 12.50, 21.0),
    ],
    "2024-04-08": [
        observer.Site("Nazas, Durango", 25.25, -104.13, 1250.0),
        observer.Site("Dallas, Texas", 32.78, -96.80, 131.0),
        observer.Site("Cleveland, Ohio", 41.50, -81.69, 199.0),
        observer.Site("New York City", 40.71, -74.01, 10.0),
    ],
}


def track(ephemeris, jd):
    """The central line: where the shadow axis lands, and when it misses."""
    latitude, longitude = eclipse.shadow_landing(ephemeris, jd)
    longitude = (longitude + 180.0) % 360.0 - 180.0
    return latitude, longitude


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--date", default="2027-08-02")
    parser.add_argument("--hours", type=float, default=6.0)
    parser.add_argument(
        "--centre-hour", type=float, default=None,
        help="UT hour to centre the window on; found automatically if omitted",
    )
    args = parser.parse_args()

    year, month, day = (int(p) for p in args.date.split("-"))
    midnight = times.jd(year, month, day)

    if args.centre_hour is None:
        # Sweep the whole day and let the geometry say when the eclipse is,
        # rather than making the caller already know. Ten-minute steps are
        # plenty to find a shadow that takes hours to cross.
        coarse = midnight + np.arange(0.0, 1.0, 10.0 / 1440.0)
        scan = truth.sampled_ephemeris(("sun", "geocentre", "moon"), coarse)
        sweep, _, _ = eclipse.shadow_axis(scan, coarse)
        centre = coarse[int(np.argmin(sweep))]
        found = times.isoformat(observer.ut1_from_tdb(np.array([centre]))[0])
        print(f"\n(found the shadow's closest approach near {found[11:]} UT)")
    else:
        centre = midnight + args.centre_hour / 24.0

    jd = centre + np.arange(-args.hours / 2, args.hours / 2, 60.0 / 86400.0)
    ephemeris = truth.sampled_ephemeris(("sun", "geocentre", "moon"), jd)
    miss, _, _ = eclipse.shadow_axis(ephemeris, jd)
    latitude, longitude = track(ephemeris, jd)

    central = ~np.isnan(latitude)
    if not np.any(central):
        print(f"\nNo central eclipse on {args.date}: the axis misses the Earth by "
              f"{miss.min():,.0f} km at closest.")
        return 1

    best = int(np.argmin(miss))
    print(f"\nEclipse of {args.date}\n")
    print(f"  greatest eclipse   {times.isoformat(observer.ut1_from_tdb(jd[best:best+1])[0])} UT")
    print(f"  axis misses centre by {miss[best]:,.0f} km   (Earth radius 6 378)")
    print(f"  lands at           {latitude[best]:+.2f}, {longitude[best]:+.2f}")
    print(
        f"  the track runs from {latitude[central][0]:+.1f},{longitude[central][0]:+.1f}"
        f" to {latitude[central][-1]:+.1f},{longitude[central][-1]:+.1f}"
    )
    minutes = central.sum()
    print(f"  the shadow is on the Earth for about {minutes:.0f} minutes")

    print(f"\n  {'place':<24}{'kind':>10}{'magnitude':>11}{'obscured':>10}{'central':>10}")
    for site in WATCHERS.get(args.date, []):
        fine = centre + np.arange(-args.hours / 2, args.hours / 2, 2.0 / 86400.0)
        view = eclipse.solar_view(ephemeris, fine, site=site)
        kind = view.kind()
        run = fine[view.total] if kind == "total" else fine[view.annular]
        length = (run[-1] - run[0]) * 86400.0 if len(run) else 0.0
        stamp = f"{int(length // 60)}m {length % 60:02.0f}s" if length else "--"
        print(
            f"  {site.name:<24}{kind:>10}{view.magnitude.max():11.3f}"
            f"{view.obscuration.max() * 100:9.1f}%{stamp:>10}"
        )

    plot(args.date, latitude[central], longitude[central],
         latitude[best], longitude[best])
    return 0


def plot(label, latitude, longitude, best_lat, best_lon) -> None:
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        print("\n(matplotlib not installed; skipping the figure)")
        return

    figure, axis = plt.subplots(figsize=(12, 6))

    earth = globe.load_texture("embary", download=False)
    if earth is not None:
        axis.imshow(earth, extent=(-180, 180, -90, 90), origin="upper")
    else:
        axis.set_facecolor("#101830")

    # Split the track where it crosses the antimeridian, so it does not draw a
    # line straight back across the map.
    breaks = np.flatnonzero(np.abs(np.diff(longitude)) > 180.0) + 1
    for piece_lon, piece_lat in zip(
        np.split(longitude, breaks), np.split(latitude, breaks)
    ):
        axis.plot(piece_lon, piece_lat, color="crimson", lw=2.5)

    axis.plot(best_lon, best_lat, "*", ms=18, color="yellow",
              markeredgecolor="black", label="greatest eclipse")
    axis.set_xlim(-180, 180)
    axis.set_ylim(-90, 90)
    axis.set_xlabel("longitude")
    axis.set_ylabel("latitude")
    axis.set_title(f"Path of the Moon's shadow, {label}")
    axis.legend(loc="lower left")
    figure.tight_layout()
    # Named for the date, so running this for a second eclipse does not quietly
    # overwrite the first.
    IMAGES.mkdir(parents=True, exist_ok=True)
    out = IMAGES / f"eclipse-{label}.png"
    figure.savefig(out, dpi=140)
    print(f"\nwrote docs/images/{out.name}")


if __name__ == "__main__":
    raise SystemExit(main())
