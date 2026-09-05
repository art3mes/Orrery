"""Which way up each planet is, and how fast it turns.

A textured planet that does not spin, or spins upright, reads as broken however
good the map is. Orientation is not decoration: it is the difference between a
picture of Mars and a picture of where Mars's features actually were.

The IAU/IAG Working Group on Cartographic Coordinates publishes, for every body,
three numbers as functions of time:

    alpha0, delta0   right ascension and declination of the north pole, ICRF
    W                angle from the pole's node to the prime meridian

From those the body-fixed frame follows: **Z** along the pole, **X** through the
prime meridian, and everything a map needs.

Two conventions worth stating, because both are easy to get backwards:

*North* means two different things, depending on the body. For the planets, the
Sun and the Moon the IAU fixes the north pole as the one on the north side of
the invariable plane, and encodes the sense of rotation in the sign of the
``W`` rate. For dwarf planets and small bodies it instead uses the right-hand
rule, so the published pole *is* the angular momentum vector.

That distinction is not pedantry. Venus's tabulated pole sits 2.6 degrees from
its orbit normal, which would make it the most upright planet in the solar
system; its obliquity is in fact 177.4 degrees, because it turns backwards and
its angular momentum points the other way. Reading the table without the
convention gives 2.6, and 2.6 is a perfectly plausible-looking number.

*Longitude* here is planetocentric and increases eastward. The IAU's own
convention for most bodies is west longitude, which runs the other way; nothing
in this package quotes cartographic longitude, so nothing converts.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .elements import canonical
from .times import DAYS_PER_CENTURY, J2000

# --- the table --------------------------------------------------------------
#
# Polynomials in T (Julian centuries past J2000) for the pole, and in d (days
# past J2000) for the prime meridian. Degrees throughout.
#
# Transcribed from the WGCCRE reports, which makes them the same kind of input
# as the JPL element table: not derivable from anything else here, and checked
# by validate_m5.py against quantities they imply -- the rotation periods and
# the obliquities, both of which are published independently and neither of
# which was used to write the table.
#
# The small periodic libration terms (Mercury, Mars, Neptune, the Moon) are
# omitted. They are tens of arcseconds; the maps are 2048 pixels wide, which is
# 10 arcminutes a pixel.


@dataclass(frozen=True)
class Spin:
    """Pole direction and prime meridian for one body."""

    right_ascension: tuple[float, ...]  # degrees, in powers of T
    declination: tuple[float, ...]  # degrees, in powers of T
    prime_meridian: tuple[float, float]  # W0 in degrees, rate in degrees/day
    right_handed: bool = False  # True where the pole IS the angular momentum


SPIN = {
    "sun": Spin((286.13,), (63.87,), (84.176, 14.1844000)),
    "mercury": Spin((281.0103, -0.0328), (61.4155, -0.0049), (329.5988, 6.1385108)),
    "venus": Spin((272.76,), (67.16,), (160.20, -1.4813688)),
    "embary": Spin((0.00, -0.641), (90.00, -0.557), (190.147, 360.9856235)),
    "mars": Spin((317.68143, -0.1061), (52.88650, -0.0609), (176.630, 350.89198226)),
    "jupiter": Spin(
        (268.056595, -0.006499), (64.495303, 0.002413), (284.95, 870.5360000)
    ),
    "saturn": Spin((40.589, -0.036), (83.537, -0.004), (38.90, 810.7939024)),
    "uranus": Spin((257.311,), (-15.175,), (203.81, -501.1600928)),
    "neptune": Spin((299.36,), (43.46,), (253.18, 536.3128492)),
    # Pluto is a dwarf planet, so the IAU gives it the right-hand-rule pole
    # rather than the north-side one, and its W rate is positive by construction.
    "pluto": Spin((132.993,), (-6.163,), (302.695, 56.3625225), right_handed=True),
    "moon": Spin((269.9949, 0.0031), (66.5392, 0.0130), (38.3213, 13.17635815)),
}


def _key(body: str) -> str:
    name = body.strip().lower()
    if name in ("sun", "moon"):
        return name
    if name == "earth":
        return "embary"  # the rotation is the Earth's; the orbit is the barycentre's
    return canonical(name)


def _polynomial(coefficients: tuple[float, ...], x: np.ndarray) -> np.ndarray:
    return sum(c * x**i for i, c in enumerate(coefficients))


def pole(body: str, jd_tdb) -> np.ndarray:
    """Unit vector along the body's north rotation pole, ICRF axes."""
    spin = SPIN[_key(body)]
    T = (np.asarray(jd_tdb, dtype=float) - J2000) / DAYS_PER_CENTURY
    alpha = np.radians(_polynomial(spin.right_ascension, T))
    delta = np.radians(_polynomial(spin.declination, T))
    return np.stack(
        [np.cos(delta) * np.cos(alpha), np.cos(delta) * np.sin(alpha), np.sin(delta)],
        axis=-1,
    )


def prime_meridian_degrees(body: str, jd_tdb) -> np.ndarray:
    """Angle W from the pole's ascending node to the prime meridian."""
    spin = SPIN[_key(body)]
    days = np.asarray(jd_tdb, dtype=float) - J2000
    return (spin.prime_meridian[0] + spin.prime_meridian[1] * days) % 360.0


def rotation_period_days(body: str) -> float:
    """Sidereal rotation period. Negative where the tabulated W rate is."""
    rate = SPIN[_key(body)].prime_meridian[1]
    return 360.0 / rate


def turns_backwards(body: str) -> bool:
    """Whether the tabulated W rate is negative.

    Not the same question as :func:`spins_retrograde`. This one is about the
    table's sign convention; that one is about the physics.
    """
    return SPIN[_key(body)].prime_meridian[1] < 0.0


def spin_axis(body: str, jd_tdb) -> np.ndarray:
    """Unit vector along the angular momentum, ICRF axes.

    The same as :func:`pole` except for a north-side-convention body whose W
    rate is negative, where the angular momentum points the other way.
    """
    spin = SPIN[_key(body)]
    axis = pole(body, jd_tdb)
    if spin.right_handed or spin.prime_meridian[1] > 0:
        return axis
    return -axis


def spins_retrograde(body: str, jd_tdb) -> bool:
    """Whether the body turns against its own orbital motion, i.e. obliquity > 90."""
    return bool(np.all(obliquity_degrees(body, jd_tdb) > 90.0))


def body_to_icrf(body: str, jd_tdb) -> np.ndarray:
    """Rotation taking body-fixed vectors to ICRF axes, shaped ``(..., 3, 3)``.

    Built from vectors rather than a product of elementary rotations, because
    the definition is itself a statement about vectors: Z is the pole, X sits at
    angle W round from the node of the body's equator on the ICRF equator, and Y
    completes the right-handed set.
    """
    jd = np.asarray(jd_tdb, dtype=float)
    spin = SPIN[_key(body)]
    T = (jd - J2000) / DAYS_PER_CENTURY
    alpha = np.radians(_polynomial(spin.right_ascension, T))

    z = pole(body, jd)
    # Ascending node of the body equator on the ICRF equator, at alpha0 + 90.
    node = np.stack(
        [-np.sin(alpha), np.cos(alpha), np.zeros_like(alpha)], axis=-1
    )
    w = np.radians(prime_meridian_degrees(body, jd))

    x = node * np.cos(w)[..., None] + np.cross(z, node) * np.sin(w)[..., None]
    y = np.cross(z, x)
    return np.stack([x, y, z], axis=-1)


def surface_point(body: str, jd_tdb, direction: np.ndarray) -> tuple[float, float]:
    """Planetocentric latitude and east longitude beneath an ICRF *direction*.

    *direction* points from the body's centre outward.
    """
    matrix = body_to_icrf(body, jd_tdb)
    direction = np.asarray(direction, dtype=float)
    direction = direction / np.linalg.norm(direction, axis=-1, keepdims=True)
    fixed = np.einsum("...ji,...j->...i", matrix, direction)
    latitude = np.degrees(np.arcsin(np.clip(fixed[..., 2], -1.0, 1.0)))
    longitude = np.degrees(np.arctan2(fixed[..., 1], fixed[..., 0])) % 360.0
    # A point a hair *west* of the prime meridian gives atan2 a tiny negative
    # angle, and -1e-15 % 360 rounds up to exactly 360.0 -- one ulp outside the
    # half-open range this claims to return. Fold it back to zero.
    longitude = np.where(longitude >= 360.0, 0.0, longitude)
    return latitude, longitude


def obliquity_degrees(body: str, jd_tdb) -> np.ndarray:
    """Angle between the rotation pole and the orbit's own normal.

    Over 90 degrees means the body turns the other way round, which is why
    Venus reads 177 and Uranus 98 rather than 3 and 82.

    The Sun has no orbit to be measured against, so its tilt is taken to the
    ecliptic pole instead. That is the published 7.25 degrees, and it is a
    different question wearing the same name.
    """
    from .frames import ecliptic_to_equatorial as _to_equatorial
    from .kepler import state

    key = _key(body)
    if key == "sun":
        ecliptic_pole = _to_equatorial(np.array([0.0, 0.0, 1.0]))
        cosine = np.sum(spin_axis(body, jd_tdb) * ecliptic_pole, axis=-1)
        return np.degrees(np.arccos(np.clip(cosine, -1.0, 1.0)))

    orbit_body = "embary" if key in ("embary", "moon") else key
    position, velocity = state(orbit_body, jd_tdb)
    normal = np.cross(position, velocity)
    normal = normal / np.linalg.norm(normal, axis=-1, keepdims=True)

    from .frames import ecliptic_to_equatorial

    normal = ecliptic_to_equatorial(normal)
    # Measured to the angular momentum, not to the tabulated pole. For Venus and
    # Uranus those are opposite directions and the two answers are supplementary.
    cosine = np.sum(spin_axis(body, jd_tdb) * normal, axis=-1)
    return np.degrees(np.arccos(np.clip(cosine, -1.0, 1.0)))


def sub_solar_point(jd_tdb, sun_direction: np.ndarray, body: str = "embary"):
    """Latitude and east longitude where the Sun stands overhead."""
    return surface_point(body, jd_tdb, sun_direction)
