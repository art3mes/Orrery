"""M3 gate: where things appear, not where they are.

M0 to M2 computed geometry. Nobody has ever observed geometry. Light takes
minutes to arrive, bends past the Sun, and arrives tilted by the observer's own
motion; the coordinate frame it is quoted in drifts; and the observer is not at
the centre of the Earth. Those corrections reach 20 arcsec, which is larger than
Mercury's entire M0 sky error.

The gates are layered so each measures one thing:

1. **The physics chain** -- light-time, deflection, aberration -- fed exactly the
   same DE440 geometry as Skyfield, so any disagreement is a transformation and
   not an orbit. This is the layer that can be made exact.
2. **The equinox of date** -- precession and nutation. Deliberately separate,
   because the truncated nutation series here has a known floor of about half an
   arcsec, and lumping it in would hide gate 1's six orders of magnitude.
3. **Standing somewhere** -- the observer's own position and velocity, checked
   by parallax, then used to time the 2004 and 2012 Venus transits from real
   sites.
4. **Retrograde** -- the stationary points where Mars appears to reverse, which
   is the observation that broke the geocentric model.

Usage::

    python scripts/validate_m3.py
    python scripts/validate_m3.py --offline
"""

from __future__ import annotations

import argparse
import sys
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from orrery import (  # noqa: E402
    apparent,
    events,
    frames,
    observer,
    precession,
    times,
    truth,
)

# Gate 1 is a transformation check with identical inputs, so it should agree to
# the noise floor of the interpolation, not to "good enough".
CHAIN_TOLERANCE_ARCSEC = 0.001

# Gate 2 is limited by the four-term nutation series, not by anything fixable
# here. Meeus quotes 0.5 arcsec; this allows a little more and separately
# insists the error is the same for every body, which is what proves it is the
# frame and not the physics.
FRAME_TOLERANCE_ARCSEC = 1.0
FRAME_SPREAD_ARCSEC = 0.02

TOPOCENTRIC_TOLERANCE_ARCSEC = 0.05
CONTACT_TOLERANCE_SECONDS = 30.0
STATIONARY_TOLERANCE_DAYS = 0.5

# Sampling step for the DE440 grid the light-time solver interpolates on. At a
# quarter day the cubic is good to 0.03 km on Mercury, the worst case.
GRID_STEP = 0.25

SUN_RADIUS_AU = events.SUN_RADIUS_AU
VENUS_RADIUS_AU = 6051.8 / 149597870.7


@dataclass
class Check:
    name: str
    passed: bool
    lines: list[str] = field(default_factory=list)


class Geometry:
    """DE440 positions on a grid, wrapped so the light-time solver can use them."""

    def __init__(self, bodies, jd, *, offline: bool, pad: float = 3.0):
        jd = np.atleast_1d(np.asarray(jd, dtype=float))
        self.bodies = tuple(bodies)
        self.grid = np.arange(jd.min() - pad, jd.max() + pad + GRID_STEP, GRID_STEP)
        positions, velocities = truth.barycentric_state(
            self.bodies, self.grid, allow_download=not offline
        )
        self._at = {
            body: apparent.interpolator(self.grid, positions[:, i, :])
            for i, body in enumerate(self.bodies)
        }
        self._velocity = {
            body: apparent.interpolator(self.grid, velocities[:, i, :])
            for i, body in enumerate(self.bodies)
        }

    def at(self, body: str):
        return self._at[body]

    def state(self, body: str, jd):
        return self._at[body](jd), self._velocity[body](jd)

    def look(self, body: str, jd, *, site=None, **kwargs):
        """Apparent place of *body*, optionally from a place on the Earth."""
        jd = np.atleast_1d(np.asarray(jd, dtype=float))
        position, velocity = self.state("geocentre", jd)
        if site is not None:
            offset, spin = site.offset_from_geocentre(jd)
            position = position + offset
            velocity = velocity + spin
        return apparent.observe(
            self.at(body), jd, position, velocity, sun_at=self.at("sun"), **kwargs
        )


def error_arcsec(ours: np.ndarray, reference: np.ndarray) -> np.ndarray:
    return frames.separation_arcsec(ours, reference)


# --- gate 1: the physics chain ----------------------------------------------


def gate_chain(*, offline: bool) -> Check:
    jd = times.jd(2026, 3, 15) + np.arange(0.0, 400.0, 37.0)
    bodies = ("sun", "geocentre", "mercury", "venus", "mars", "jupiter", "saturn")
    geometry = Geometry(bodies, jd, offline=offline)

    lines = [f"{len(jd)} dates through 2026-2027, geocentric"]
    lines.append(f"  {'body':<9}{'light-time only':>18}{'+ bending + tilt':>19}")
    worst = 0.0
    for body in bodies[2:]:
        sight = geometry.look(body, jd)
        astrometric = error_arcsec(
            sight.astrometric,
            truth.apparent_reference(
                body, jd, stage="astrometric", allow_download=not offline
            ),
        ).max()
        full = error_arcsec(
            sight.apparent,
            truth.apparent_reference(
                body, jd, stage="apparent", allow_download=not offline
            ),
        ).max()
        worst = max(worst, astrometric, full)
        lines.append(f"  {body:<9}{astrometric:15.2e}\"{full:16.2e}\"")

    # How big were the corrections that just cancelled to nothing?
    sight = geometry.look("jupiter", jd)
    bare = geometry.look("jupiter", jd, deflection=False, aberration=False)
    tilt = error_arcsec(sight.apparent, bare.astrometric).max()
    lines += [
        "",
        f"for scale: those corrections move Jupiter by up to {tilt:.1f}\","
        f" and it is {sight.light_minutes.min():.0f}-{sight.light_minutes.max():.0f}"
        " light-minutes away",
    ]
    return Check("light-time, deflection, aberration", worst < CHAIN_TOLERANCE_ARCSEC, lines)


# --- gate 2: the equinox of date --------------------------------------------


def gate_frame(*, offline: bool) -> Check:
    jd = np.array(
        [times.jd(y, 6, 1) for y in (1900, 1950, 1975, 2004, 2026, 2049)]
    )
    bodies = ("sun", "geocentre", "venus", "mars", "jupiter")

    # One small window per date, rather than one grid spanning them all. The
    # light-time solver never looks back more than an hour, so a continuous
    # quarter-day grid across a century and a half is 217000 samples of which
    # about fifty are ever read -- and 48 MB of cached fixture.
    per_body = {body: [] for body in bodies[2:]}
    for when in jd:
        one = np.array([when])
        geometry = Geometry(bodies, one, offline=offline)
        for body in bodies[2:]:
            ours = precession.to_equinox_of_date(
                geometry.look(body, one).apparent, one
            )
            reference = truth.apparent_reference(
                body, one, epoch_of_date=True, allow_download=not offline
            )
            per_body[body].append(float(error_arcsec(ours, reference)[0]))
    per_body = {body: np.array(values) for body, values in per_body.items()}

    worst = max(e.max() for e in per_body.values())
    stacked = np.stack(list(per_body.values()))
    spread = float(np.max(stacked.max(axis=0) - stacked.min(axis=0)))

    lines = [f"  {'year':<8}" + "".join(f"{b[:7]:>9}" for b in per_body)]
    for i, when in enumerate(jd):
        lines.append(
            f"  {times.isoformat(when)[:4]:<8}"
            + "".join(f"{per_body[b][i]:8.3f}\"" for b in per_body)
        )
    lines += [
        "",
        f"every body is wrong by the same amount to within {spread:.4f}\", which is"
        " what a frame error looks like",
        "and a physics error does not. The floor is the four-term nutation series;"
        " IAU 2000A would remove it.",
    ]
    passed = worst < FRAME_TOLERANCE_ARCSEC and spread < FRAME_SPREAD_ARCSEC
    return Check("precession and nutation", passed, lines)


# --- gate 3: standing somewhere ---------------------------------------------


def transit_contacts(geometry: Geometry, jd: np.ndarray, site) -> list[float]:
    """The four contact times, as Julian dates."""
    venus = geometry.look("venus", jd, site=site)
    sun = geometry.look("sun", jd, site=site)
    separation = frames.separation_arcsec(venus.apparent, sun.apparent)

    solar = np.degrees(np.arcsin(SUN_RADIUS_AU / sun.distance)) * 3600.0
    venusian = np.degrees(np.arcsin(VENUS_RADIUS_AU / venus.distance)) * 3600.0

    external = events.find_crossings(jd, separation - (solar + venusian), 0.0)
    internal = events.find_crossings(jd, separation - (solar - venusian), 0.0)
    return sorted(external + internal)


def gate_topocentric(*, offline: bool) -> Check:
    sites = [observer.SITES[name] for name in ("greenwich", "mauna_kea", "paranal")]
    jd = times.jd(2026, 3, 15) + np.arange(0.0, 300.0, 29.0)
    bodies = ("sun", "geocentre", "venus", "mars")
    geometry = Geometry(bodies, jd, offline=offline)

    lines = ["parallax, against Skyfield's own topocentric places:"]
    worst = 0.0
    for site in sites:
        row = []
        for body in ("venus", "mars"):
            ours = geometry.look(body, jd, site=site).apparent
            reference = truth.apparent_reference(
                body, jd, site=site, allow_download=not offline
            )
            shift = error_arcsec(
                ours, truth.apparent_reference(body, jd, allow_download=not offline)
            ).max()
            error = error_arcsec(ours, reference).max()
            worst = max(worst, error)
            row.append(f"{body} {error:.4f}\" (parallax up to {shift:.1f}\")")
        lines.append(f"  {site.name:<30}" + ",  ".join(row))

    # The 2004 and 2012 transits, timed from three places at once.
    for year, month, day in ((2004, 6, 8), (2012, 6, 6)):
        window = times.jd(year, month, day) + np.arange(-0.7, 0.9, 1.0 / 86400.0 * 10)
        geometry = Geometry(("sun", "geocentre", "venus"), window, offline=offline)
        lines.append("")
        geocentric = transit_contacts(geometry, window, None)
        stamps = "  ".join(
            times.isoformat(observer.ut1_from_tdb(np.array([c]))[0])[11:]
            for c in geocentric
        )
        lines.append(f"Venus transit of {year}, contacts in UT:")
        lines.append(f"  {'(geocentric)':<30}{stamps}")
        table = {}
        for site in sites:
            contacts = transit_contacts(geometry, window, site)
            table[site.name] = contacts
            stamps = "  ".join(
                times.isoformat(observer.ut1_from_tdb(np.array([c]))[0])[11:]
                for c in contacts
            )
            lines.append(f"  {site.name:<30}{stamps}")
        counts = {len(v) for v in table.values()}
        if counts == {4}:
            first = np.array([v[0] for v in table.values()])
            spread_minutes = (first.max() - first.min()) * 24 * 60
            lines.append(
                f"  first contact differs by {spread_minutes:.1f} minutes between"
                " sites -- that is the parallax, and it is how the astronomical"
                " unit was first measured"
            )
        else:
            lines.append(f"  contact counts differ between sites: {counts}")
            worst = np.inf

    return Check(
        "the observer, and the Venus transits",
        worst < TOPOCENTRIC_TOLERANCE_ARCSEC,
        lines,
    )


# --- gate 4: retrograde ------------------------------------------------------


def right_ascension_hours(direction: np.ndarray) -> np.ndarray:
    return frames.radec(direction)[0]


def gate_retrograde(*, offline: bool) -> Check:
    jd = np.arange(times.jd(2024, 9, 1), times.jd(2025, 6, 1), 0.25)
    geometry = Geometry(("sun", "geocentre", "mars"), jd, offline=offline)

    ours = precession.to_equinox_of_date(geometry.look("mars", jd).apparent, jd)
    reference = truth.apparent_reference(
        "mars", jd, epoch_of_date=True, allow_download=not offline
    )

    def stationary(direction):
        ra = np.unwrap(np.radians(right_ascension_hours(direction) * 15.0))
        return [
            when
            for kind in ("max", "min")
            for when, _ in events.find_extrema(jd, ra, kind=kind)
        ]

    ours_found = sorted(stationary(ours))
    reference_found = sorted(stationary(reference))

    lines = ["Mars appears to stop, reverse, and stop again:"]
    if len(ours_found) != len(reference_found):
        lines.append(f"found {len(ours_found)}, Skyfield found {len(reference_found)}")
        return Check("Mars retrograde", False, lines)

    worst = 0.0
    for mine, theirs in zip(ours_found, reference_found):
        offset = (mine - theirs) * 24 * 60
        worst = max(worst, abs(mine - theirs))
        lines.append(
            f"  {times.isoformat(mine)[:10]}   {offset:+.1f} min vs Skyfield"
        )
    if len(ours_found) >= 2:
        lines.append(
            f"  retrograde lasted {ours_found[1] - ours_found[0]:.0f} days"
        )
    lines.append("")
    lines.append(
        "Nothing in the model knows Mars ever moves backwards. It falls out of"
        " the Earth overtaking it on the inside."
    )
    return Check("Mars retrograde", worst < STATIONARY_TOLERANCE_DAYS, lines)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--offline", action="store_true", help="require cached truth")
    args = parser.parse_args()

    print("\nM3 -- where things appear\n")
    checks = []
    for gate in (gate_chain, gate_frame, gate_topocentric, gate_retrograde):
        check = gate(offline=args.offline)
        checks.append(check)
        print(f"  [{'ok' if check.passed else 'FAIL'}]  {check.name}")
        for line in check.lines:
            print(f"          {line}")
        print()

    passing = sum(c.passed for c in checks)
    print(f"M3 {'passing' if passing == len(checks) else 'FAILED'}, "
          f"{passing}/{len(checks)}")
    return 0 if passing == len(checks) else 1


if __name__ == "__main__":
    raise SystemExit(main())
