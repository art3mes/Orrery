"""Reference frames, and the angles between vectors.

Two frames are in play:

**Ecliptic J2000** -- the plane of Earth's orbit. The JPL element table is
expressed here, and so is everything :mod:`orrery.kepler` returns. It is the
natural frame for a solar system: all the planets lie nearly in the xy plane.

**Equatorial J2000 (ICRF)** -- the plane of Earth's equator. What telescopes,
star catalogues and the DE ephemerides use. Right ascension and declination live
here.

The two share an x axis (the equinox) and differ by a tilt: the obliquity.

    eps = 23 deg 26' 21.448" = 84381.448 arcsec

That is the classical J2000 value. The IAU 2006 value is 84381.406", 42 mas
smaller, and the ICRF-to-mean-equinox frame bias is another ~20 mas. Both are
far below the arcsecond-scale error of the element table, but they are the
reason two independent implementations can disagree at the tens-of-mas level
without either being wrong.
"""

from __future__ import annotations

import numpy as np

OBLIQUITY_ARCSEC = 84381.448
OBLIQUITY_RAD = np.radians(OBLIQUITY_ARCSEC / 3600.0)

_COS_EPS = np.cos(OBLIQUITY_RAD)
_SIN_EPS = np.sin(OBLIQUITY_RAD)


def ecliptic_to_equatorial(v: np.ndarray) -> np.ndarray:
    """Rotate ``(..., 3)`` vectors from the ecliptic frame to the equatorial one."""
    v = np.asarray(v, dtype=float)
    x, y, z = v[..., 0], v[..., 1], v[..., 2]
    return np.stack(
        [x, _COS_EPS * y - _SIN_EPS * z, _SIN_EPS * y + _COS_EPS * z], axis=-1
    )


def equatorial_to_ecliptic(v: np.ndarray) -> np.ndarray:
    """Rotate ``(..., 3)`` vectors from the equatorial frame to the ecliptic one."""
    v = np.asarray(v, dtype=float)
    x, y, z = v[..., 0], v[..., 1], v[..., 2]
    return np.stack(
        [x, _COS_EPS * y + _SIN_EPS * z, -_SIN_EPS * y + _COS_EPS * z], axis=-1
    )


def norm(v: np.ndarray) -> np.ndarray:
    """Length of each ``(..., 3)`` vector."""
    return np.linalg.norm(np.asarray(v, dtype=float), axis=-1)


def separation_arcsec(u: np.ndarray, v: np.ndarray) -> np.ndarray:
    """Angle between two ``(..., 3)`` vectors, in arcseconds.

    Uses ``atan2(|u x v|, u . v)`` rather than ``acos`` of the normalised dot
    product: the dot form loses all its precision for small angles, which is the
    only regime this function is ever asked about.
    """
    u = np.asarray(u, dtype=float)
    v = np.asarray(v, dtype=float)
    cross = np.linalg.norm(np.cross(u, v), axis=-1)
    dot = np.sum(u * v, axis=-1)
    return np.degrees(np.arctan2(cross, dot)) * 3600.0


def radec(v: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Right ascension (hours) and declination (degrees) of equatorial vectors.

    Input must already be in the equatorial frame; this does not rotate.
    """
    v = np.asarray(v, dtype=float)
    x, y, z = v[..., 0], v[..., 1], v[..., 2]
    ra_deg = np.degrees(np.arctan2(y, x)) % 360.0
    dec_deg = np.degrees(np.arcsin(z / np.linalg.norm(v, axis=-1)))
    return ra_deg / 15.0, dec_deg
