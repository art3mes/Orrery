"""What M0 can already answer, with no graphics at all.

Four questions, in increasing order of how much they ask of the model:

1. Where is everything right now?
2. When is Earth closest to the Sun?  -- a date, from geometry alone.
3. When is Mars at opposition?        -- Earth and Mars together.
4. How close did Jupiter and Saturn get in December 2020?

The fourth is the interesting one. It looks like a triumph and is mostly luck,
which is exactly the kind of thing validate_m0.py exists to say out loud.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from orrery import elements, events, frames, kepler, times  # noqa: E402


def show_today() -> None:
    now = times.jd(2026, 9, 3)
    print(f"\nPositions on {times.isoformat(now)[:10]} (TDB)\n")
    print(f"{'body':<9}{'from Sun':>10}{'from Earth':>12}{'RA':>10}{'Dec':>9}")
    print(f"{'':<9}{'au':>10}{'au':>12}{'h':>10}{'deg':>9}")
    print("-" * 50)

    earth = kepler.position("embary", now)
    for body in elements.ORDER:
        pos = kepler.position(body, now)
        if body == "embary":
            ra_text, dec_text, from_earth = "  --", "  --", "  --"
        else:
            ra, dec = frames.radec(frames.ecliptic_to_equatorial(pos - earth))
            ra_text, dec_text = f"{ra:9.2f}", f"{dec:8.2f}"
            from_earth = f"{frames.norm(pos - earth):11.3f}"
        print(
            f"{body:<9}{frames.norm(pos):10.3f}{from_earth:>12}"
            f"{ra_text:>10}{dec_text:>9}"
        )


def show_perihelion(year: int) -> None:
    days = times.jd(year, 1, 1) + np.arange(0, 366, 0.01)
    r = frames.norm(kepler.position("embary", days))
    closest, farthest = days[np.argmin(r)], days[np.argmax(r)]
    print(f"\nEarth-Moon barycentre, {year}")
    print(f"  perihelion  {times.isoformat(closest)}   {r.min():.6f} au")
    print(f"  aphelion    {times.isoformat(farthest)}   {r.max():.6f} au")
    print(f"  the orbit is {100 * (r.max() / r.min() - 1):.1f}% wider at aphelion")


def elongation_deg(body: str, jd) -> np.ndarray:
    """Angle between the body and the Sun, as seen from Earth."""
    earth = kepler.position("embary", jd)
    toward_body = kepler.position(body, jd) - earth
    toward_sun = -earth
    return frames.separation_arcsec(toward_body, toward_sun) / 3600.0


def show_oppositions(body: str, start: int, end: int) -> None:
    days = np.arange(times.jd(start, 1, 1), times.jd(end, 1, 1), 0.05)
    angle = elongation_deg(body, days)
    peaks = np.flatnonzero(
        (angle[1:-1] > angle[:-2]) & (angle[1:-1] > angle[2:]) & (angle[1:-1] > 150)
    )
    print(f"\n{body.capitalize()} at opposition, {start}-{end}")
    for index in peaks:
        when = days[index + 1]
        distance = frames.norm(
            kepler.position(body, when) - kepler.position("embary", when)
        )
        print(
            f"  {times.isoformat(when)[:10]}   {distance:.3f} au"
            f"   elongation {angle[index + 1]:.2f} deg"
        )


def show_great_conjunction() -> None:
    """Jupiter and Saturn, December 2020. The closest pairing since 1623.

    DE440 puts closest approach at 2020-12-21 18:14 TDB, 6.104 arcmin apart.
    """
    days = times.jd(2020, 12, 1) + np.arange(0, 40, 0.005)
    earth = kepler.position("embary", days)
    separation = frames.separation_arcsec(
        kepler.position("jupiter", days) - earth,
        kepler.position("saturn", days) - earth,
    )
    # Through the same finder M1's gate uses, refinement included. The grid is
    # 7.2 minutes wide and a raw argmin is quantised to it; taking the vertex of
    # the fitted parabola instead is what lets this agree with the gate to the
    # minute rather than to the sample.
    when, closest_arcsec = min(
        events.find_extrema(days, separation, kind="min"), key=lambda pair: pair[1]
    )
    closest = closest_arcsec / 60.0

    hours_late = (when - times.jd(2020, 12, 21, 18, 14)) * 24.0

    print("\nJupiter-Saturn great conjunction, December 2020")
    print(
        f"  closest      {times.isoformat(when)}"
        f"   ({hours_late:+.1f} h vs DE440)"
    )
    print(f"  separation   {closest:.3f} arcmin   (DE440: 6.104 arcmin)")
    print(
        "\n  Do not read that 0.1 arcmin agreement as accuracy. On that date this\n"
        "  model puts Jupiter 0.4 arcmin and Saturn 2.5 arcmin from where they\n"
        "  really were -- both far larger than the 0.1 arcmin the separation is\n"
        "  out by. The two displacements happen to point much the same way, so\n"
        "  most of the error cancels in the difference between them.\n"
        "\n  The date is the weaker number here, not the stronger one: at closest\n"
        "  approach the separation curve is nearly flat, so Saturn's 2.5 arcmin\n"
        "  moves the minimum by ten hours. M1 gates conjunctions to a day for\n"
        "  that reason, and claims nothing about the separation."
    )


if __name__ == "__main__":
    show_today()
    show_perihelion(2026)
    show_oppositions("mars", 2020, 2032)
    show_great_conjunction()
    print()
