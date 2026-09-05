"""orrery -- solar system positions you can check against NASA.

    >>> from orrery import position, jd
    >>> position("mars", jd(2026, 9, 3))
    array([...])

Positions are heliocentric, in au, in the mean ecliptic and equinox of J2000.
Dates are TDB Julian dates. Accuracy is whatever ``scripts/validate_m0.py``
last measured; the README carries the numbers.
"""

from .elements import BODIES, J2000, elements_at
from .frames import (
    ecliptic_to_equatorial,
    equatorial_to_ecliptic,
    norm,
    radec,
    separation_arcsec,
)
from .kepler import (
    AU_KM,
    GM_SUN,
    ellipse,
    period,
    position,
    solve_kepler,
    state,
)
from .times import calendar, isoformat, jd

__all__ = [
    "AU_KM",
    "BODIES",
    "GM_SUN",
    "J2000",
    "calendar",
    "ecliptic_to_equatorial",
    "elements_at",
    "ellipse",
    "equatorial_to_ecliptic",
    "isoformat",
    "jd",
    "norm",
    "period",
    "position",
    "radec",
    "separation_arcsec",
    "solve_kepler",
    "state",
]
