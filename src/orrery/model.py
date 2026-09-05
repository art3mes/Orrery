"""This package's own ephemeris, assembled from its own orbits.

``truth.py`` wraps DE440 and needs a 32 MB kernel. This wraps *us* and needs
nothing: Keplerian elements for the planets (M0) and the abridged lunar theory
(M6), packed into the same :class:`~orrery.apparent.Ephemeris` that DE440 comes
out of. Same class, so the same eclipse and apparent-place code runs on either
-- which is exactly how M6 measured what our Moon costs against JPL's.

**The frame is heliocentric**, ICRF equatorial, au and au/day. The Sun sits at
the origin rather than at its true place a little off the barycentre. Nothing
downstream minds: light-time, deflection and aberration are all differences of
positions and velocities. The one price is aberration, which wants the
observer's *barycentric* velocity and gets its heliocentric one instead -- the
Sun's own 16 m/s wander about the barycentre, so up to **0.011 arcsec**. That
is a hundredth of the Moon's error and a thousandth of the eclipse timings this
feeds, and it is stated here rather than corrected because correcting it would
mean knowing where the barycentre is, which is what not needing DE440 means.

What this costs against DE440, measured rather than assumed, is in the README's
recipes: the 2024 eclipse comes out of these orbits and out of JPL's, and the
two answers are differenced.
"""

from __future__ import annotations

import numpy as np

from . import lunar
from .apparent import Ephemeris
from .elements import BODIES, canonical
from .frames import ecliptic_to_equatorial
from .kepler import state

# DE440's value. The Moon's share of the Earth-Moon pair is what separates the
# geocentre from the barycentre our elements actually describe: 4671 km, which
# is three quarters of an Earth radius and so not remotely ignorable.
EARTH_OVER_MOON = 81.3005682214972
MOON_SHARE = 1.0 / (1.0 + EARTH_OVER_MOON)

#: Everything this module can place. ``embary`` is what the elements give; the
#: geocentre and the Moon are split out of it by :data:`MOON_SHARE`.
DEFAULT_BODIES = ("sun", "geocentre", "moon") + BODIES

# For the Moon's velocity, by central difference. The theory has no derivative
# in closed form and nothing here needs one: velocities are used for aberration
# and for the observer's own motion, where a part in 1e5 is far below the noise.
_VELOCITY_STEP_DAYS = 0.05


def _key(body: str) -> str:
    name = body.strip().lower().replace("-", " ")
    if name in {"sun", "moon"}:
        return name
    if name in {"geocentre", "geocenter", "earth centre", "earth center"}:
        return "geocentre"
    return canonical(body)


def _moon_from_earth(jd: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """The Moon relative to the Earth's centre, and how fast, ICRF equatorial."""
    h = _VELOCITY_STEP_DAYS
    position = lunar.position(jd)
    velocity = (lunar.position(jd + h) - lunar.position(jd - h)) / (2.0 * h)
    return position, velocity


def states(bodies, jd) -> tuple[np.ndarray, np.ndarray]:
    """Positions and velocities of *bodies*, heliocentric ICRF equatorial.

    Returns ``(positions, velocities)``, each shaped ``(len(jd), len(bodies), 3)``
    -- the layout :class:`~orrery.apparent.Ephemeris` wants.

    ``"earth"`` means the Earth-Moon barycentre, because that is what the
    element table describes; ask for ``"geocentre"`` to get the planet itself.
    """
    jd = np.atleast_1d(np.asarray(jd, dtype=float))
    keys = [_key(body) for body in bodies]

    positions = np.zeros((jd.size, len(keys), 3))
    velocities = np.zeros_like(positions)

    needs_split = {"geocentre", "moon"} & set(keys)
    if needs_split:
        embary, embary_velocity = state("embary", jd)
        embary = ecliptic_to_equatorial(embary)
        embary_velocity = ecliptic_to_equatorial(embary_velocity)
        offset, drift = _moon_from_earth(jd)
        geocentre = embary - MOON_SHARE * offset
        geocentre_velocity = embary_velocity - MOON_SHARE * drift

    for i, key in enumerate(keys):
        if key == "sun":
            continue  # the origin, and at rest in it
        if key == "geocentre":
            positions[:, i, :] = geocentre
            velocities[:, i, :] = geocentre_velocity
        elif key == "moon":
            positions[:, i, :] = geocentre + offset
            velocities[:, i, :] = geocentre_velocity + drift
        else:
            pos, vel = state(key, jd)
            positions[:, i, :] = ecliptic_to_equatorial(pos)
            velocities[:, i, :] = ecliptic_to_equatorial(vel)

    return positions, velocities


def ephemeris(
    jd,
    *,
    bodies=DEFAULT_BODIES,
    pad: float = 3.0,
    step: float = 0.25,
) -> Ephemeris:
    """This package's orbits, on a grid, ready for the light-time solver.

    Mirrors :func:`orrery.truth.sampled_ephemeris` argument for argument, so the
    two are interchangeable at a call site and any answer can be computed both
    ways. *jd* may be a single date or the whole span you intend to ask about;
    the grid is padded either side because the solver asks for positions minutes
    *before* the dates it was handed.
    """
    jd = np.atleast_1d(np.asarray(jd, dtype=float))
    grid = np.arange(jd.min() - pad, jd.max() + pad + step, step)
    positions, velocities = states(bodies, grid)
    return Ephemeris([_key(body) for body in bodies], grid, positions, velocities)
