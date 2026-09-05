"""M2 gate: does real gravity behave?

M0 and M1 had every planet on its own fixed ellipse. M2 throws those away and
integrates Newton's law directly, with every body pulling on every other. Three
questions follow, and they are not the same question:

1. **Does the integrator hold the system together?** A thousand years of energy,
   momentum and angular momentum, and a Runge-Kutta run alongside to show what
   choosing a symplectic method actually bought.
2. **Do the planets stay where DE440 says?** Fifty years, started from DE440's
   own state vector, compared back against it.
3. **Does Mercury's perihelion turn at the right rate?** The headline. Newtonian
   gravity gets most of the way and is then short by a fixed amount, and that
   amount is what made Einstein famous.

Gate 3 needs care, because the integrator produces a spurious apsidal precession
of its own -- see the control run below, and the note in the README. Energy
conservation does not imply correct orbits, and this milestone is where that
stops being an abstract remark.

Runtime is a couple of minutes; most of it is gate 3. Use --quick to halve it.

Usage::

    python scripts/validate_m2.py
    python scripts/validate_m2.py --offline
    python scripts/validate_m2.py --quick
"""

from __future__ import annotations

import argparse
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from orrery import frames, kepler, nbody, times, truth  # noqa: E402

# --- expectations -----------------------------------------------------------

# 6 pi GM / (c^2 a (1 - e^2)) per orbit, in arcsec per century. Not fitted to
# anything: it drops out of general relativity in two lines.
GR_PREDICTED_ARCSEC_PER_CENTURY = 42.98

# How far the measured GR contribution may sit from that.
GR_TOLERANCE = 1.0

# How far the full model (Newtonian + GR) may sit from DE440's own Mercury.
PRECESSION_TOLERANCE = 2.0

# A symplectic method must beat Runge-Kutta on energy by at least this factor
# over the same span, or there was no reason to prefer it.
SYMPLECTIC_ADVANTAGE = 100.0

MU_MERCURY = kepler.GM_SUN * (1.0 + 1.0 / nbody.SUN_OVER_BODY["mercury"])


@dataclass
class Check:
    name: str
    passed: bool
    lines: list[str] = field(default_factory=list)


def seed(jd0: float, *, offline: bool):
    """DE440 barycentric state for every body, recentred on the barycentre."""
    fetch_kw = {"allow_download": not offline}
    pos, vel = truth.barycentric_state(
        nbody.DEFAULT_BODIES, np.array([jd0]), **fetch_kw
    )
    gm = nbody.gm_vector(nbody.DEFAULT_BODIES)
    return nbody.to_barycentric(pos[0], vel[0], gm)


def perihelion_rate(r: np.ndarray, v: np.ndarray, jd: np.ndarray) -> float:
    """Rate of change of Mercury's longitude of perihelion, arcsec per century.

    Measured in the **ecliptic** of J2000, which is where the published figures
    are referred and where the JPL element table lives. Reading the same states
    against the equator instead moves the answer by 11 arcsec per century --
    a real effect of the reference plane, not an error, and a trap worth naming
    because both numbers look entirely plausible.
    """
    r = frames.equatorial_to_ecliptic(r)
    v = frames.equatorial_to_ecliptic(v)
    longitude = np.unwrap(np.radians(kepler.elements_from_state(r, v, MU_MERCURY)["long_peri"]))
    slope_per_day = np.polyfit(jd, longitude, 1)[0]
    return float(np.degrees(slope_per_day) * 3600.0 * 36525.0)


# --- gate 1: conservation ---------------------------------------------------


def gate_conservation(*, offline: bool, quick: bool) -> Check:
    jd0 = times.jd(1950, 1, 1)
    pos0, vel0 = seed(jd0, offline=offline)
    gm = nbody.gm_vector(nbody.DEFAULT_BODIES)
    years = 500 if quick else 1000

    run = nbody.integrate(
        pos0, vel0, jd0=jd0, dt=2.0, days=years * 365.25,
        method="yoshida4", sample_every=200,
    )
    e = run.energy
    scale = abs(e[0])
    swing = float(e.max() - e.min()) / scale

    # The point of a symplectic method is not that the error is small but that
    # it does not accumulate. So fit a line and compare its total rise against
    # the oscillation it rides on.
    centuries = (run.jd - run.jd[0]) / 36525.0
    trend = float(np.polyfit(centuries, e, 1)[0]) / scale
    total_trend = abs(trend) * centuries[-1]

    # Runge-Kutta over the same span and step, for contrast.
    rk = nbody.integrate(
        pos0, vel0, jd0=jd0, dt=2.0, days=years * 365.25,
        method="rk4", sample_every=200,
    )
    rk_centuries = (rk.jd - rk.jd[0]) / 36525.0
    rk_trend = abs(float(np.polyfit(rk_centuries, rk.energy, 1)[0]) / scale)
    advantage = rk_trend / max(abs(trend), 1e-30)

    momentum = np.linalg.norm(nbody.momentum(run.vel[-1], gm))
    angular = nbody.angular_momentum(run.pos[-1], run.vel[-1], gm)
    angular0 = nbody.angular_momentum(run.pos[0], run.vel[0], gm)
    angular_drift = float(
        np.linalg.norm(angular - angular0) / np.linalg.norm(angular0)
    )

    passed = (
        total_trend < 0.2 * swing
        and swing < 1e-4
        and advantage > SYMPLECTIC_ADVANTAGE
        and angular_drift < 1e-12
        and momentum < 1e-18
    )
    return Check(
        f"conservation over {years} years",
        passed,
        [
            f"energy swings by {swing:.2e} of itself and stays there",
            f"its linear trend over the whole span is {total_trend:.2e},"
            f" {total_trend / swing:.0%} of the swing",
            f"Runge-Kutta at the same step drifts {advantage:.0f}x faster"
            f"   (needed {SYMPLECTIC_ADVANTAGE:.0f}x)",
            f"angular momentum {angular_drift:.1e},  linear momentum {momentum:.1e}",
        ],
    )


# --- gate 2: agreement with DE440 -------------------------------------------


def gate_drift(*, offline: bool, quick: bool) -> Check:
    jd0 = times.jd(1990, 1, 1)
    pos0, vel0 = seed(jd0, offline=offline)
    gm = nbody.gm_vector(nbody.DEFAULT_BODIES)
    years = 25 if quick else 50
    dt = 0.1

    run = nbody.integrate(
        pos0, vel0, jd0=jd0, dt=dt, days=years * 365.25,
        method="yoshida4", relativity=True, sample_every=2000,
    )

    reference, _ = truth.barycentric_state(
        nbody.DEFAULT_BODIES, np.array([run.jd[-1]]), allow_download=not offline
    )
    reference = reference[0] - (gm @ reference[0]) / gm.sum()
    delta = run.pos[-1] - reference

    # Split the error along the direction of travel and across it. The two mean
    # completely different things and lumping them together hides the result:
    # *across* is whether the orbit is the right shape, in the right plane, at
    # the right orientation -- the force model. *Along* is only whether the
    # planet is at the right point on it, and a fixed-step integrator gets that
    # slightly wrong forever, because its shadow Hamiltonian implies a mean
    # motion a hair off the true one and the phase error accumulates.
    lines = [
        f"{years} years from {times.isoformat(jd0)[:10]}, dt = {dt} d, GR on",
        f"  {'body':<9}{'across track':>14}{'along track':>14}",
    ]
    across_km, along_km = {}, {}
    for i, body in enumerate(run.bodies):
        if body == "sun":
            continue
        velocity = run.vel[-1, i] - run.vel[-1, 0]
        direction = velocity / np.linalg.norm(velocity)
        along = float(delta[i] @ direction)
        across_km[body] = float(np.linalg.norm(delta[i] - along * direction)) * kepler.AU_KM
        along_km[body] = abs(along) * kepler.AU_KM
        lines.append(
            f"  {body:<9}{across_km[body]:11,.0f} km{along_km[body]:11,.0f} km"
        )

    worst_across = max(across_km.values())
    others = max(v for b, v in along_km.items() if b != "mercury")
    lines += [
        "",
        f"across track, worst case {worst_across:,.0f} km -- the orbits themselves"
        f" are right, so the masses and the force model are too",
        "along track is a phase offset that grows without bound; Mercury shows it"
        " first",
        "because it laps the others. A Wisdom-Holman map, which integrates the"
        " Kepler",
        "part exactly, is the fix and is not built here.",
    ]

    passed = bool(
        worst_across < 1_000 and along_km["mercury"] < 200_000 and others < 20_000
    )
    return Check(f"{years}-year drift against DE440", passed, lines)


# --- gate 3: Mercury's perihelion -------------------------------------------


def gate_precession(*, offline: bool, quick: bool) -> Check:
    jd0 = times.jd(1970, 1, 1)
    pos0, vel0 = seed(jd0, offline=offline)
    years = 30 if quick else 60
    days = years * 365.25
    dt = 0.2
    sample_every = 100

    def run(**kwargs):
        return nbody.integrate(
            pos0, vel0, jd0=jd0, dt=dt, days=days,
            method="yoshida4", sample_every=sample_every, **kwargs
        )

    newtonian = run()
    einstein = run(relativity=True)

    # Control: the Sun and Mercury alone, same integrator, same step. A two-body
    # orbit does not precess at all, so whatever this run reports is entirely
    # manufactured by the integrator, and it is not small.
    mercury = nbody.DEFAULT_BODIES.index("mercury")
    control = nbody.integrate(
        np.array([pos0[0], pos0[mercury]]),
        np.array([vel0[0], vel0[mercury]]),
        bodies=("sun", "mercury"), jd0=jd0, dt=dt, days=days,
        method="yoshida4", sample_every=sample_every,
    )

    artifact = perihelion_rate(*control.heliocentric("mercury"), control.jd)
    newton_rate = perihelion_rate(*newtonian.heliocentric("mercury"), newtonian.jd) - artifact
    einstein_rate = perihelion_rate(*einstein.heliocentric("mercury"), einstein.jd) - artifact

    # DE440's own Mercury, through the same element extraction and the same fit.
    jd = np.arange(jd0, jd0 + days, 20.0)
    pos, vel = truth.barycentric_state(
        ("sun", "mercury"), jd, allow_download=not offline
    )
    de440_rate = perihelion_rate(pos[:, 1] - pos[:, 0], vel[:, 1] - vel[:, 0], jd)

    gr_contribution = einstein_rate - newton_rate
    shortfall = de440_rate - newton_rate
    residual = einstein_rate - de440_rate

    passed = (
        abs(gr_contribution - GR_PREDICTED_ARCSEC_PER_CENTURY) < GR_TOLERANCE
        and abs(residual) < PRECESSION_TOLERANCE
    )
    return Check(
        "Mercury's perihelion",
        passed,
        [
            f"integrator artifact  {artifact:8.2f}\"/cy  (two-body control;"
            f" the true answer is 0)",
            f"Newtonian gravity    {newton_rate:8.2f}\"/cy",
            f"with the GR term     {einstein_rate:8.2f}\"/cy",
            f"DE440                {de440_rate:8.2f}\"/cy",
            "",
            f"Newton falls short by{shortfall:8.2f}\"/cy",
            f"the GR term supplies {gr_contribution:8.2f}\"/cy"
            f"   (theory says {GR_PREDICTED_ARCSEC_PER_CENTURY})",
            f"leaving             {residual:+8.2f}\"/cy against DE440"
            f"   (allowed {PRECESSION_TOLERANCE})",
        ],
    )


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--offline", action="store_true", help="require cached truth")
    parser.add_argument("--quick", action="store_true", help="halve every span")
    args = parser.parse_args()

    print("\nM2 -- Newton's law, integrated\n")
    checks = []
    for gate in (gate_conservation, gate_drift, gate_precession):
        started = time.time()
        check = gate(offline=args.offline, quick=args.quick)
        checks.append(check)
        print(f"  [{'ok' if check.passed else 'FAIL'}]  {check.name}"
              f"   ({time.time() - started:.0f}s)")
        for line in check.lines:
            print(f"          {line}")
        print()

    passing = sum(c.passed for c in checks)
    print(f"M2 {'passing' if passing == len(checks) else 'FAILED'}, "
          f"{passing}/{len(checks)}")
    return 0 if passing == len(checks) else 1


if __name__ == "__main__":
    raise SystemExit(main())
