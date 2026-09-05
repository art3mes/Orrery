"""What M3 lets you say that M0 to M2 could not.

Three things:

**How old the view is.** Every planet is seen where it was, not where it is.

**How big the corrections are.** Aberration is twenty arcseconds; the bending of
light past the Sun is a hundredth of that; the frame drifts faster than either.
Ranking them is the whole reason to compute them separately.

**Retrograde.** Mars stops, reverses for eleven weeks, and stops again. Nothing
in this package knows that happens. It falls out of the Earth overtaking it on
the inside, and it is the observation the geocentric model could never explain
without wheels upon wheels.

Writes ``m3_retrograde.png``.

    python scripts/demo_m3.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from orrery import apparent, events, frames, precession, times, truth  # noqa: E402

OUT = Path(__file__).resolve().parents[1] / "docs" / "images" / "mars-retrograde.png"
BODIES = ("sun", "geocentre", "mercury", "venus", "mars", "jupiter", "saturn", "neptune")


def geometry(bodies, jd, pad=3.0):
    jd = np.atleast_1d(np.asarray(jd, dtype=float))
    grid = np.arange(jd.min() - pad, jd.max() + pad + 0.25, 0.25)
    positions, velocities = truth.barycentric_state(bodies, grid)
    at = {b: apparent.interpolator(grid, positions[:, i, :]) for i, b in enumerate(bodies)}
    speed = {b: apparent.interpolator(grid, velocities[:, i, :]) for i, b in enumerate(bodies)}
    return at, speed


def look(at, speed, body, jd, **kwargs):
    return apparent.observe(
        at[body], jd, at["geocentre"](jd), speed["geocentre"](jd),
        sun_at=at["sun"], **kwargs
    )


def show_light_time(jd: float) -> None:
    at, speed = geometry(BODIES, np.array([jd]))
    print(f"\nHow old the view is, on {times.isoformat(jd)[:10]}\n")
    print(f"  {'body':<9}{'distance':>10}{'you are seeing it':>22}")
    for body in BODIES[2:]:
        sight = look(at, speed, body, np.array([jd]))
        minutes = float(sight.light_minutes[0])
        if minutes < 60:
            when = f"{minutes:.1f} minutes ago"
        else:
            when = f"{minutes / 60:.1f} hours ago"
        print(f"  {body:<9}{float(sight.distance[0]):9.3f} au{when:>22}")


def show_corrections(jd: float) -> None:
    """Each correction, applied on its own, in arcseconds."""
    at, speed = geometry(BODIES, np.array([jd]))
    when = np.array([jd])
    print(f"\nSize of each correction for Mars, on {times.isoformat(jd)[:10]}\n")

    geometric = at["mars"](when) - at["geocentre"](when)
    astrometric = look(at, speed, "mars", when, deflection=False, aberration=False)
    bent = look(at, speed, "mars", when, deflection=True, aberration=False)
    tilted = look(at, speed, "mars", when, deflection=True, aberration=True)
    of_date = precession.to_equinox_of_date(tilted.apparent, when)

    steps = [
        ("light-time", geometric, astrometric.astrometric),
        ("gravitational bending", astrometric.astrometric, bent.apparent),
        ("aberration", bent.apparent, tilted.apparent),
        ("precession + nutation to date", tilted.apparent, of_date),
    ]
    for name, before, after in steps:
        print(f"  {name:<32}{float(frames.separation_arcsec(before, after)[0]):9.3f}\"")
    print(
        f"  {'--- total':<32}"
        f"{float(frames.separation_arcsec(geometric, of_date)[0]):9.3f}\""
    )


def retrograde(start=(2024, 9, 1), end=(2025, 6, 1)):
    jd = np.arange(times.jd(*start), times.jd(*end), 0.5)
    at, speed = geometry(("sun", "geocentre", "mars"), jd)
    of_date = precession.to_equinox_of_date(look(at, speed, "mars", jd).apparent, jd)
    ra_hours, dec_degrees = frames.radec(of_date)

    ra = np.unwrap(np.radians(ra_hours * 15.0))
    turning = [
        when
        for kind in ("max", "min")
        for when, _ in events.find_extrema(jd, ra, kind=kind)
    ]
    turning.sort()

    print("\nMars, apparent right ascension of date\n")
    for when in turning:
        print(f"  stationary   {times.isoformat(when)[:10]}")
    if len(turning) >= 2:
        print(f"  retrograde for {turning[1] - turning[0]:.0f} days")
    return jd, ra_hours, dec_degrees, turning


def plot(jd, ra_hours, dec_degrees, turning) -> None:
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        print("\n(matplotlib not installed; skipping the figure)")
        return

    figure, axis = plt.subplots(figsize=(7.5, 6.0))
    axis.plot(ra_hours, dec_degrees, lw=1.6)

    for month in range(1, 13):
        for year in (2024, 2025):
            stamp = times.jd(year, month, 1)
            if jd[0] <= stamp <= jd[-1]:
                i = int(np.argmin(np.abs(jd - stamp)))
                axis.plot(ra_hours[i], dec_degrees[i], "o", ms=4, color="k")
                axis.annotate(
                    f"{year}-{month:02d}",
                    (ra_hours[i], dec_degrees[i]),
                    textcoords="offset points", xytext=(6, -3), fontsize=8,
                )
    for when in turning:
        i = int(np.argmin(np.abs(jd - when)))
        axis.plot(ra_hours[i], dec_degrees[i], "*", ms=14, color="crimson")

    axis.invert_xaxis()  # right ascension increases to the east, i.e. leftward
    axis.set_xlabel("apparent right ascension (hours, of date)")
    axis.set_ylabel("apparent declination (degrees)")
    axis.set_title("Mars, Sep 2024 to Jun 2025: the retrograde loop")
    axis.grid(alpha=0.3)
    figure.tight_layout()
    OUT.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(OUT, dpi=140)
    print(f"\nwrote {OUT.name}")


if __name__ == "__main__":
    today = times.jd(2026, 9, 3)
    show_light_time(today)
    show_corrections(today)
    plot(*retrograde())
    print()
