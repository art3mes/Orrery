"""Where a planet *appears*, as opposed to where it is.

M0 through M2 computed geometry: the true instantaneous separation between two
bodies. Nobody has ever seen that. What reaches an observer has been delayed,
bent and tilted on the way, and the corrections are not small next to the
accuracy the earlier milestones fought for:

============================  ==========================================
light-time                    8.3 min to the Sun, up to 52 min to Jupiter
aberration                    up to 20.5 arcsec
gravitational deflection      1.75 arcsec at the solar limb, 0.004 at 90 deg
============================  ==========================================

Aberration alone is comparable to Mercury's entire M0 sky error. A model correct
to an arcsecond and uncorrected for these is wrong by twenty.

Everything here takes a **callable** for the target's position rather than a
body name. That is what lets ``validate_m3.py`` push DE440's own geometry
through this same pipeline: if the answer then disagrees with Skyfield, the
fault is in these transformations and not in the orbits, and the two can be
measured apart. Vectors are barycentric, in au, on ICRF axes.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .frames import norm
from .kepler import GM_SUN
from .nbody import C_AU_PER_DAY

# Twice the Sun's gravitational radius, in au: 2GM/c^2 = 2.95 km. The whole of
# gravitational light bending is this number divided by an impact parameter.
SUN_SCHWARZSCHILD_AU = 2.0 * GM_SUN / C_AU_PER_DAY**2


@dataclass
class Sight:
    """One look at one body."""

    jd: np.ndarray
    astrometric: np.ndarray  # observer -> target, light-time corrected
    apparent: np.ndarray  # ...and bent, and tilted
    distance: np.ndarray  # au, at the moment the light left
    light_time_days: np.ndarray
    emitted_jd: np.ndarray

    @property
    def light_minutes(self) -> np.ndarray:
        return self.light_time_days * 24.0 * 60.0


def light_time(
    target_at,
    observer: np.ndarray,
    jd: np.ndarray,
    *,
    tolerance_days: float = 1e-9,
    max_iter: int = 12,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Solve for when the light now arriving actually left.

    ``t_emit = t - |target(t_emit) - observer(t)| / c`` is implicit in
    ``t_emit``, so it is iterated. It converges fast because the correction to
    the correction is of order (planet's speed / c) -- four decimal places a
    pass -- and two iterations already reach a metre.

    The tolerance cannot be made arbitrarily small. A Julian date is about
    2.46e6, where a float64 steps by 4.7e-10 days, so a threshold below that
    can only be met by two iterates landing on the *same* bit pattern. Distant
    planets do reach such a fixed point; the Moon, seen from a point on a
    spinning Earth, instead flips between two adjacent representable values
    forever. This was set to 1e-12 and the Moon never converged. 1e-9 days is
    86 microseconds, in which light travels 26 km and the Moon moves 9 cm.

    Returns ``(direction, distance, emitted_jd)``, where *direction* runs from
    the observer at *jd* to the target at the earlier time.
    """
    jd = np.asarray(jd, dtype=float)
    emitted = np.array(jd, dtype=float, copy=True)

    for _ in range(max_iter):
        separation = np.asarray(target_at(emitted)) - observer
        distance = norm(separation)
        updated = jd - distance / C_AU_PER_DAY
        moved = np.max(np.abs(updated - emitted))
        emitted = updated
        if moved < tolerance_days:
            break
    else:  # pragma: no cover - would need a target moving at relativistic speed
        raise RuntimeError(
            f"light-time iteration did not converge: last step {moved:.3e} days"
        )

    separation = np.asarray(target_at(emitted)) - observer
    return separation, norm(separation), emitted


def deflect(
    direction: np.ndarray,
    target_from_sun: np.ndarray,
    observer_from_sun: np.ndarray,
    *,
    body_radius_au: float = 0.00465047,
) -> np.ndarray:
    """Bend a light ray in the Sun's gravity.

    The Explanatory Supplement's formula, with unit vectors p (observer to
    target), q (Sun to target) and e (Sun to observer)::

        p' = p + (2GM/c^2 |E|) [ (p.q) e - (e.p) q ] / (1 + q.e)

    1.75 arcsec grazing the Sun, falling off roughly as 1/elongation: 0.004
    arcsec at a right angle, which still exceeds the milliarcsecond agreement
    this pipeline is gated to.

    Rays that would pass *through* the Sun are left alone. The formula diverges
    there, the geometry is a body behind the Sun, and nothing is visible anyway.
    """
    p = direction / norm(direction)[..., None]
    e = observer_from_sun / norm(observer_from_sun)[..., None]

    # Looking at the Sun itself puts the target *at* the source, and in a frame
    # whose origin is the Sun -- ``model.ephemeris`` -- that vector is exactly
    # zero rather than merely small. Same argument as the denominator below:
    # divide safely rather than mask a nan afterwards. Such a ray is always
    # occulted, so the value landed on here is never the one returned.
    reach = norm(target_from_sun)
    q = target_from_sun / np.where(reach > 0.0, reach, 1.0)[..., None]

    distance_to_sun = norm(observer_from_sun)
    scale = SUN_SCHWARZSCHILD_AU / distance_to_sun

    pq = np.sum(p * q, axis=-1)
    ep = np.sum(e * p, axis=-1)
    qe = np.sum(q * e, axis=-1)

    # Rays that would pass through the Sun. The impact parameter is the
    # perpendicular distance from the Sun to the line of sight, and "behind"
    # distinguishes an occultation from a body between here and the Sun.
    occulted = (norm(np.cross(p, observer_from_sun)) < body_radius_au) & (
        np.sum(p * observer_from_sun, axis=-1) < 0
    )

    # 1 + q.e goes to zero for exactly those rays, so the denominator has to be
    # made safe before the division rather than after it: masking the result
    # afterwards still evaluates the division, and a nan warning in the middle
    # of an otherwise clean run is how real ones get ignored.
    denominator = np.where(occulted, 1.0, 1.0 + qe)
    correction = (
        scale[..., None] * (pq[..., None] * e - ep[..., None] * q) / denominator[..., None]
    )
    return p + np.where(occulted[..., None], 0.0, correction)


def aberrate(direction: np.ndarray, observer_velocity: np.ndarray) -> np.ndarray:
    """Tilt a direction by the observer's own motion.

    The special-relativistic form, not the first-order one. Earth's 30 km/s is
    1e-4 of c, so the difference between them is of order 1e-8 radians -- two
    milliarcseconds, which is above the noise this is gated at.

    Returns a unit vector.
    """
    p = direction / norm(direction)[..., None]
    beta = np.asarray(observer_velocity, dtype=float) / C_AU_PER_DAY

    speed_squared = np.sum(beta * beta, axis=-1)
    inverse_gamma = np.sqrt(1.0 - speed_squared)
    cosine = np.sum(p * beta, axis=-1)

    factor = 1.0 + cosine / (1.0 + inverse_gamma)
    tilted = (inverse_gamma[..., None] * p + factor[..., None] * beta) / (
        1.0 + cosine
    )[..., None]
    return tilted / norm(tilted)[..., None]


def observe(
    target_at,
    jd,
    observer_position: np.ndarray,
    observer_velocity: np.ndarray,
    *,
    sun_at=None,
    deflection: bool = True,
    aberration: bool = True,
) -> Sight:
    """The whole chain, in the order the physics happens.

    Light leaves the target, is bent as it passes the Sun, and is then tilted by
    the observer's motion at the moment of arrival. Reversing the last two makes
    a difference of a few milliarcseconds -- small, but not nothing, and there
    is no reason to get the order wrong.

    *target_at* and *sun_at* take an array of Julian dates and return barycentric
    positions. Omitting *sun_at* skips deflection.
    """
    jd = np.atleast_1d(np.asarray(jd, dtype=float))
    separation, distance, emitted = light_time(target_at, observer_position, jd)

    direction = separation / distance[..., None]
    if deflection and sun_at is not None:
        sun_now = np.asarray(sun_at(jd))
        direction = deflect(
            direction,
            np.asarray(target_at(emitted)) - sun_now,
            observer_position - sun_now,
        )
    if aberration:
        direction = aberrate(direction, observer_velocity)

    return Sight(
        jd=jd,
        astrometric=separation,
        apparent=direction * distance[..., None],
        distance=distance,
        light_time_days=jd - emitted,
        emitted_jd=emitted,
    )


class Ephemeris:
    """Sampled positions and velocities, ready for the light-time solver.

    Holds one interpolator per body and knows how to look from the Earth's
    centre or from a place on it. The positions come from wherever the caller
    got them -- this module has no opinion, which is what lets the same code run
    on DE440 and on this package's own orbits.
    """

    def __init__(self, bodies, jd_grid, positions, velocities):
        self.bodies = tuple(bodies)
        self.jd_grid = np.asarray(jd_grid, dtype=float)
        self._at = {
            body: interpolator(jd_grid, positions[:, i, :])
            for i, body in enumerate(self.bodies)
        }
        self._velocity = {
            body: interpolator(jd_grid, velocities[:, i, :])
            for i, body in enumerate(self.bodies)
        }

    def at(self, body: str):
        return self._at[body]

    def state(self, body: str, jd):
        return self._at[body](jd), self._velocity[body](jd)

    def observer(self, jd, *, site=None, centre: str = "geocentre"):
        """Position and velocity of the observer, ICRF barycentric."""
        position, velocity = self.state(centre, jd)
        if site is not None:
            offset, spin = site.offset_from_geocentre(jd)
            position = position + offset
            velocity = velocity + spin
        return position, velocity

    def look(self, body: str, jd, *, site=None, centre: str = "geocentre", **kwargs):
        """Apparent place of *body*, optionally from a place on the Earth."""
        jd = np.atleast_1d(np.asarray(jd, dtype=float))
        position, velocity = self.observer(jd, site=site, centre=centre)
        return observe(
            self.at(body), jd, position, velocity, sun_at=self.at("sun"), **kwargs
        )


def interpolator(jd_grid: np.ndarray, positions: np.ndarray):
    """Wrap sampled positions as a callable the light-time solver can use.

    Cubic through the four nearest samples. The solver asks for positions at
    times a few minutes before the grid points it was given, which no tabulated
    ephemeris will have exactly, and linear interpolation of a curved orbit
    would put Mercury several hundred kilometres out.
    """
    jd_grid = np.asarray(jd_grid, dtype=float)
    positions = np.asarray(positions, dtype=float)
    if jd_grid.ndim != 1 or positions.shape[0] != jd_grid.size:
        raise ValueError("expected one position per date")
    if jd_grid.size < 4:
        raise ValueError("cubic interpolation needs at least 4 samples")

    def at(when):
        when = np.atleast_1d(np.asarray(when, dtype=float))
        index = np.clip(
            np.searchsorted(jd_grid, when) - 2, 0, jd_grid.size - 4
        )
        offsets = index[:, None] + np.arange(4)
        nodes = jd_grid[offsets]
        values = positions[offsets]

        # Lagrange basis over the four surrounding nodes.
        weights = np.ones_like(nodes)
        for k in range(4):
            for m in range(4):
                if k != m:
                    weights[:, k] *= (when - nodes[:, m]) / (nodes[:, k] - nodes[:, m])
        return np.einsum("nk,nkj->nj", weights, values)

    return at
