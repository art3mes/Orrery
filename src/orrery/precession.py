"""Equinox of date: precession, nutation, obliquity, sidereal time.

Everything else in this package works on ICRF axes, which do not move. Almanacs
and telescopes mostly do not: they quote right ascension and declination against
the equator and equinox *of date*, and that frame drifts at 5028 arcsec per
century. Since J2000 it has moved by more than a third of a degree, which
dwarfs every error M0 to M2 worried about.

Two rotations take you there:

**Precession**, the slow conical wander of Earth's axis, 26000 years to a turn.
Implemented as IAU 1976 (Lieske), good to about 0.1 arcsec over the couple of
centuries this package covers.

**Nutation**, a small nodding on top of it, driven mostly by the 18.6-year
regression of the Moon's node. The principal term is 17.2 arcsec.

The nutation series here is Meeus's short form: four terms, and Meeus quotes
0.5 arcsec in the longitude and 0.1 in the obliquity. That is much coarser than
the microarcsecond agreement :mod:`orrery.apparent` reaches, and it is the
accuracy floor of anything quoted in coordinates of date. The full IAU 2000A
series, 1365 terms, is the upgrade; ``validate_m3.py`` measures what the short
form actually costs rather than taking Meeus's word for it.
"""

from __future__ import annotations

import numpy as np

from .times import DAYS_PER_CENTURY, J2000

ARCSEC = np.pi / (180.0 * 3600.0)


def centuries(jd_tdb) -> np.ndarray:
    return (np.asarray(jd_tdb, dtype=float) - J2000) / DAYS_PER_CENTURY


# --- elementary rotations ---------------------------------------------------
#
# Coordinate (frame) rotations, not vector rotations: R3(a) expresses the same
# vector in axes turned by +a about the third axis. Writing the precession
# matrix as a product of these rather than transcribing its nine expanded
# entries is the difference between a sign error being obvious and being
# invisible.


def _r1(angle: np.ndarray) -> np.ndarray:
    c, s = np.cos(angle), np.sin(angle)
    zero, one = np.zeros_like(c), np.ones_like(c)
    return np.array([[one, zero, zero], [zero, c, s], [zero, -s, c]])


def _r2(angle: np.ndarray) -> np.ndarray:
    c, s = np.cos(angle), np.sin(angle)
    zero, one = np.zeros_like(c), np.ones_like(c)
    return np.array([[c, zero, -s], [zero, one, zero], [s, zero, c]])


def _r3(angle: np.ndarray) -> np.ndarray:
    c, s = np.cos(angle), np.sin(angle)
    zero, one = np.zeros_like(c), np.ones_like(c)
    return np.array([[c, s, zero], [-s, c, zero], [zero, zero, one]])


def _apply(matrix: np.ndarray, vector: np.ndarray) -> np.ndarray:
    """Apply a (3, 3, ...) stack of matrices to (..., 3) vectors."""
    return np.einsum("ij...,...j->...i", matrix, np.asarray(vector, dtype=float))


# --- obliquity and nutation -------------------------------------------------


def mean_obliquity(jd_tdb) -> np.ndarray:
    """Mean obliquity of the ecliptic, in radians. IAU 1980."""
    T = centuries(jd_tdb)
    seconds = 84381.448 - 46.8150 * T - 0.00059 * T**2 + 0.001813 * T**3
    return seconds * ARCSEC


def nutation(jd_tdb) -> tuple[np.ndarray, np.ndarray]:
    """Nutation in longitude and obliquity, in radians.

    Meeus's short series. ``Omega`` is the longitude of the Moon's ascending
    node, whose 18.6-year circulation supplies the dominant 17.2 arcsec term;
    the rest are semiannual and semimonthly.
    """
    T = centuries(jd_tdb)
    node = np.radians(125.04452 - 1934.136261 * T)
    sun = np.radians(280.4665 + 36000.7698 * T)
    moon = np.radians(218.3165 + 481267.8813 * T)

    d_psi = (
        -17.20 * np.sin(node)
        - 1.32 * np.sin(2 * sun)
        - 0.23 * np.sin(2 * moon)
        + 0.21 * np.sin(2 * node)
    )
    d_eps = (
        9.20 * np.cos(node)
        + 0.57 * np.cos(2 * sun)
        + 0.10 * np.cos(2 * moon)
        - 0.09 * np.cos(2 * node)
    )
    return d_psi * ARCSEC, d_eps * ARCSEC


def true_obliquity(jd_tdb) -> np.ndarray:
    return mean_obliquity(jd_tdb) + nutation(jd_tdb)[1]


# --- the two matrices -------------------------------------------------------


def precession_matrix(jd_tdb) -> np.ndarray:
    """J2000 mean equator and equinox to the mean equator and equinox of date."""
    T = centuries(jd_tdb)
    zeta = (2306.2181 * T + 0.30188 * T**2 + 0.017998 * T**3) * ARCSEC
    z = (2306.2181 * T + 1.09468 * T**2 + 0.018203 * T**3) * ARCSEC
    theta = (2004.3109 * T - 0.42665 * T**2 - 0.041833 * T**3) * ARCSEC
    return np.einsum("ij...,jk...,kl...->il...", _r3(-z), _r2(theta), _r3(-zeta))


def nutation_matrix(jd_tdb) -> np.ndarray:
    """Mean equator and equinox of date to the true equator and equinox of date."""
    d_psi, d_eps = nutation(jd_tdb)
    eps = mean_obliquity(jd_tdb)
    return np.einsum(
        "ij...,jk...,kl...->il...", _r1(-(eps + d_eps)), _r3(-d_psi), _r1(eps)
    )


def to_equinox_of_date(vector: np.ndarray, jd_tdb) -> np.ndarray:
    """Rotate ICRF vectors onto the true equator and equinox of date.

    The ICRF-to-J2000-mean-equinox frame bias, about 23 milliarcsec, is not
    applied. It is well below the nutation series' own error, and pretending
    otherwise would be false precision.
    """
    spun = _apply(precession_matrix(jd_tdb), vector)
    return _apply(nutation_matrix(jd_tdb), spun)


# --- sidereal time ----------------------------------------------------------


def greenwich_mean_sidereal_time(jd_ut1) -> np.ndarray:
    """GMST in degrees. IAU 1982.

    Note the argument is **UT1**, not TDB: sidereal time measures the Earth's
    actual rotation, and the whole point of UT1 is that it tracks it.
    """
    jd_ut1 = np.asarray(jd_ut1, dtype=float)
    days = jd_ut1 - J2000
    T = days / DAYS_PER_CENTURY
    degrees = (
        280.46061837
        + 360.98564736629 * days
        + 0.000387933 * T**2
        - T**3 / 38710000.0
    )
    return degrees % 360.0


def greenwich_apparent_sidereal_time(jd_ut1, jd_tdb) -> np.ndarray:
    """GAST in degrees: GMST plus the equation of the equinoxes.

    The correction is the nutation in longitude projected onto the equator, at
    most about 1.1 arcsec of angle -- 0.07 seconds of time. Small, and free.
    """
    d_psi, _ = nutation(jd_tdb)
    equation = np.degrees(d_psi * np.cos(true_obliquity(jd_tdb)))
    return (greenwich_mean_sidereal_time(jd_ut1) + equation) % 360.0
