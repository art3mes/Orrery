"""Two things M2 can show that M0 and M1 could not.

**Why symplectic.** A Runge-Kutta step is more accurate than a leapfrog step of
the same size. Run both for a few centuries and Runge-Kutta is far worse,
because its energy error all points the same way and accumulates, while a
symplectic method conserves a slightly wrong energy exactly and its error just
oscillates. One curve is flat, the other is a ramp.

**The missing 43 arcseconds.** Mercury's perihelion turns faster than Newtonian
gravity can account for. Integrate the eight planets pulling on each other, and
the perihelion still falls behind DE440 by a fixed amount per century. Add the
one relativistic term and it catches up.

Writes ``m2_energy_and_precession.png``.

    python scripts/demo_m2.py
    python scripts/demo_m2.py --fast
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from orrery import frames, kepler, nbody, times, truth  # noqa: E402

MU_MERCURY = kepler.GM_SUN * (1.0 + 1.0 / nbody.SUN_OVER_BODY["mercury"])
OUT = Path(__file__).resolve().parents[1] / "docs" / "images" / "energy-and-precession.png"


def seed(jd0: float):
    gm = nbody.gm_vector(nbody.DEFAULT_BODIES)
    pos, vel = truth.barycentric_state(nbody.DEFAULT_BODIES, np.array([jd0]))
    return nbody.to_barycentric(pos[0], vel[0], gm)


def perihelion_longitude(r: np.ndarray, v: np.ndarray) -> np.ndarray:
    """Longitude of perihelion in arcsec, unwrapped, ecliptic of J2000."""
    r = frames.equatorial_to_ecliptic(r)
    v = frames.equatorial_to_ecliptic(v)
    degrees = kepler.elements_from_state(r, v, MU_MERCURY)["long_peri"]
    return np.degrees(np.unwrap(np.radians(degrees))) * 3600.0


def energy_comparison(years: int):
    jd0 = times.jd(1950, 1, 1)
    pos0, vel0 = seed(jd0)
    out = {}
    for method in ("yoshida4", "rk4"):
        run = nbody.integrate(
            pos0, vel0, jd0=jd0, dt=2.0, days=years * 365.25,
            method=method, sample_every=100,
        )
        out[method] = (
            (run.jd - run.jd[0]) / 365.25,
            np.abs(run.energy / run.energy[0] - 1.0),
        )

    print(f"\nEnergy over {years} years, both at a 2-day step\n")
    print(f"  {'years':>7}{'yoshida4':>14}{'rk4':>14}")
    span = out["yoshida4"][0]
    for fraction in (0.0, 0.25, 0.5, 0.75, 1.0):
        i = min(int(fraction * (len(span) - 1)), len(span) - 1)
        print(
            f"  {span[i]:7.0f}{out['yoshida4'][1][i]:14.2e}"
            f"{out['rk4'][1][i]:14.2e}"
        )
    print(
        "\n  The symplectic run wanders and comes back. Runge-Kutta only ever\n"
        "  goes one way, and after a few thousand years the planets have\n"
        "  visibly moved off their orbits."
    )
    return out


def precession_story(years: int):
    jd0 = times.jd(1970, 1, 1)
    pos0, vel0 = seed(jd0)
    days, dt, every = years * 365.25, 0.2, 100

    def run(**kw):
        return nbody.integrate(
            pos0, vel0, jd0=jd0, dt=dt, days=days, method="yoshida4",
            sample_every=every, **kw
        )

    newton, einstein = run(), run(relativity=True)

    mercury = nbody.DEFAULT_BODIES.index("mercury")
    control = nbody.integrate(
        np.array([pos0[0], pos0[mercury]]), np.array([vel0[0], vel0[mercury]]),
        bodies=("sun", "mercury"), jd0=jd0, dt=dt, days=days,
        method="yoshida4", sample_every=every,
    )

    jd = newton.jd
    reference, reference_v = truth.barycentric_state(("sun", "mercury"), jd)
    de440 = perihelion_longitude(
        reference[:, 1] - reference[:, 0], reference_v[:, 1] - reference_v[:, 0]
    )

    # Subtract the control run pointwise, not just its slope: it is the same
    # integrator on the same orbit with the same step, so it carries the same
    # manufactured precession, and a two-body orbit has none of its own.
    artifact = perihelion_longitude(*control.heliocentric("mercury"))
    artifact -= artifact[0]

    curves = {}
    for name, trajectory in (("Newton", newton), ("Newton + GR", einstein)):
        longitude = perihelion_longitude(*trajectory.heliocentric("mercury"))
        corrected = longitude - longitude[0] - artifact
        curves[name] = corrected - (de440 - de440[0])

    print(f"\nMercury's perihelion, {years} years from 1970, against DE440\n")
    print(f"  {'years':>7}{'Newton':>12}{'Newton + GR':>14}   arcsec")
    span = (jd - jd[0]) / 365.25
    for fraction in (0.25, 0.5, 0.75, 1.0):
        i = min(int(fraction * (len(span) - 1)), len(span) - 1)
        print(
            f"  {span[i]:7.0f}{curves['Newton'][i]:12.2f}"
            f"{curves['Newton + GR'][i]:14.2f}"
        )
    slope = np.polyfit(span, curves["Newton"], 1)[0] * 100
    print(
        f"\n  Newtonian gravity falls behind at {abs(slope):.1f} arcsec per century.\n"
        "  Adding one term -- GM/(c^2 r^3)[(4GM/r - v.v) r + 4(r.v) v] -- closes it."
    )
    return span, curves


def plot(energy, span, curves) -> None:
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        print("\n(matplotlib not installed; skipping the figure)")
        return

    figure, (left, right) = plt.subplots(1, 2, figsize=(11, 4.2))

    for method, style in (("rk4", "-"), ("yoshida4", "-")):
        years, error = energy[method]
        left.semilogy(years, np.maximum(error, 1e-16), style, label=method)
    left.set_xlabel("years")
    left.set_ylabel("|relative energy error|")
    left.set_title("energy: symplectic stays put, Runge-Kutta ramps")
    left.legend()
    left.grid(alpha=0.3)

    for name in curves:
        right.plot(span, curves[name], label=name)
    right.axhline(0, color="k", lw=0.8)
    right.set_xlabel("years from 1970")
    right.set_ylabel("perihelion longitude minus DE440 (arcsec)")
    right.set_title("Mercury's perihelion: what Newton leaves out")
    right.legend()
    right.grid(alpha=0.3)

    figure.tight_layout()
    OUT.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(OUT, dpi=140)
    print(f"\nwrote {OUT.name}")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--fast", action="store_true", help="halve both spans")
    args = parser.parse_args()

    energy = energy_comparison(100 if args.fast else 200)
    span, curves = precession_story(30 if args.fast else 60)
    plot(energy, span, curves)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
