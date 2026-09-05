"""The Moon's position, computed here rather than looked up.

Everywhere else in this package the Moon has come from DE440. That made M4's
eclipses trustworthy and left one honest gap: the shadow geometry was tested,
the orbit under it was not.

This is the abridged ELP-2000/82 as set out in Meeus, *Astronomical
Algorithms*, chapter 47: five fundamental angles, sixty periodic terms for
longitude and distance, sixty more for latitude, and a handful of additive
corrections for the perturbations of Venus and Jupiter. Meeus quotes 10
arcseconds in longitude and 4 in latitude against the full theory.

Ten arcseconds is not much and it is not nothing. The Moon moves half an
arcsecond a second, so ten arcseconds is twenty seconds of eclipse contact
time and about twelve kilometres of shadow track. ``validate_m6.py`` measures
what it actually costs by running M4's eclipses on this instead of on DE440.

Why the Moon is hard, in one number: the largest periodic term is 6.29 degrees.
The Sun's largest is 1.9. Nothing else in the solar system is pulled about like
this, which is why a Moon theory needs sixty terms where a planet needs six.

The tables below are transcribed, like the JPL element table, and rest on the
same arrangement: they are not derivable from anything else here, and the gate
compares them against an independent ephemeris.
"""

from __future__ import annotations

import numpy as np

from .times import DAYS_PER_CENTURY, J2000

# Mean distance the periodic distance terms are measured from, km.
MEAN_DISTANCE_KM = 385000.56

# --- the periodic terms -----------------------------------------------------
#
# Each row is (D, M, M', F) -- multiples of the four arguments -- followed by
# the coefficient. Longitude and distance share arguments, so they share a
# table; latitude has its own.
#
# Terms in M carry a factor E, the slow decline of the Earth's orbital
# eccentricity, once per power of M. Leaving it out is a 0.1% error on those
# terms and grows with time either side of J2000.

# (D, M, M', F), sigma_l (1e-6 deg), sigma_r (1e-3 km)
_LONGITUDE_DISTANCE = (
    (0, 0, 1, 0, 6288774, -20905355),
    (2, 0, -1, 0, 1274027, -3699111),
    (2, 0, 0, 0, 658314, -2955968),
    (0, 0, 2, 0, 213618, -569925),
    (0, 1, 0, 0, -185116, 48888),
    (0, 0, 0, 2, -114332, -3149),
    (2, 0, -2, 0, 58793, 246158),
    (2, -1, -1, 0, 57066, -152138),
    (2, 0, 1, 0, 53322, -170733),
    (2, -1, 0, 0, 45758, -204586),
    (0, 1, -1, 0, -40923, -129620),
    (1, 0, 0, 0, -34720, 108743),
    (0, 1, 1, 0, -30383, 104755),
    (2, 0, 0, -2, 15327, 10321),
    (0, 0, 1, 2, -12528, 0),
    (0, 0, 1, -2, 10980, 79661),
    (4, 0, -1, 0, 10675, -34782),
    (0, 0, 3, 0, 10034, -23210),
    (4, 0, -2, 0, 8548, -21636),
    (2, 1, -1, 0, -7888, 24208),
    (2, 1, 0, 0, -6766, 30824),
    (1, 0, -1, 0, -5163, -8379),
    (1, 1, 0, 0, 4987, -16675),
    (2, -1, 1, 0, 4036, -12831),
    (2, 0, 2, 0, 3994, -10445),
    (4, 0, 0, 0, 3861, -11650),
    (2, 0, -3, 0, 3665, 14403),
    (0, 1, -2, 0, -2689, -7003),
    (2, 0, -1, 2, -2602, 0),
    (2, -1, -2, 0, 2390, 10056),
    (1, 0, 1, 0, -2348, 6322),
    (2, -2, 0, 0, 2236, -9884),
    (0, 1, 2, 0, -2120, 5751),
    (0, 2, 0, 0, -2069, 0),
    (2, -2, -1, 0, 2048, -4950),
    (2, 0, 1, -2, -1773, 4130),
    (2, 0, 0, 2, -1595, 0),
    (4, -1, -1, 0, 1215, -3958),
    (0, 0, 2, 2, -1110, 0),
    (3, 0, -1, 0, -892, 3258),
    (2, 1, 1, 0, -810, 2616),
    (4, -1, -2, 0, 759, -1897),
    (0, 2, -1, 0, -713, -2117),
    (2, 2, -1, 0, -700, 2354),
    (2, 1, -2, 0, 691, 0),
    (2, -1, 0, -2, 596, 0),
    (4, 0, 1, 0, 549, -1423),
    (0, 0, 4, 0, 537, -1117),
    (4, -1, 0, 0, 520, -1571),
    (1, 0, -2, 0, -487, -1739),
    (2, 1, 0, -2, -399, 0),
    (0, 0, 2, -2, -381, -4421),
    (1, 1, 1, 0, 351, 0),
    (3, 0, -2, 0, -340, 0),
    (4, 0, -3, 0, 330, 0),
    (2, -1, 2, 0, 327, 0),
    (0, 2, 1, 0, -323, 1165),
    (1, 1, -1, 0, 299, 0),
    (2, 0, 3, 0, 294, 0),
    (2, 0, -1, -2, 0, 8752),
)

# (D, M, M', F), sigma_b (1e-6 deg)
_LATITUDE = (
    (0, 0, 0, 1, 5128122),
    (0, 0, 1, 1, 280602),
    (0, 0, 1, -1, 277693),
    (2, 0, 0, -1, 173237),
    (2, 0, -1, 1, 55413),
    (2, 0, -1, -1, 46271),
    (2, 0, 0, 1, 32573),
    (0, 0, 2, 1, 17198),
    (2, 0, 1, -1, 9266),
    (0, 0, 2, -1, 8822),
    (2, -1, 0, -1, 8216),
    (2, 0, -2, -1, 4324),
    (2, 0, 1, 1, 4200),
    (2, 1, 0, -1, -3359),
    (2, -1, -1, 1, 2463),
    (2, -1, 0, 1, 2211),
    (2, -1, -1, -1, 2065),
    (0, 1, -1, -1, -1870),
    (4, 0, -1, -1, 1828),
    (0, 1, 0, 1, -1794),
    (0, 0, 0, 3, -1749),
    (0, 1, -1, 1, -1565),
    (1, 0, 0, 1, -1491),
    (0, 1, 1, 1, -1475),
    (0, 1, 1, -1, -1410),
    (0, 1, 0, -1, -1344),
    (1, 0, 0, -1, -1335),
    (0, 0, 3, 1, 1107),
    (4, 0, 0, -1, 1021),
    (4, 0, -1, 1, 833),
    (0, 0, 1, -3, 777),
    (4, 0, -2, 1, 671),
    (2, 0, 0, -3, 607),
    (2, 0, 2, -1, 596),
    (2, -1, 1, -1, 491),
    (2, 0, -2, 1, -451),
    (0, 0, 3, -1, 439),
    (2, 0, 2, 1, 422),
    (2, 0, -3, -1, 421),
    (2, 1, -1, 1, -366),
    (2, 1, 0, 1, -351),
    (4, 0, 0, 1, 331),
    (2, -1, 1, 1, 315),
    (2, -2, 0, -1, 302),
    (0, 0, 1, 3, -283),
    (2, 1, 1, -1, -229),
    (1, 1, 0, -1, 223),
    (1, 1, 0, 1, 223),
    (0, 1, -2, -1, -220),
    (2, 1, -1, -1, -220),
    (1, 0, 1, 1, -185),
    (2, -1, -2, -1, 181),
    (0, 1, 2, 1, -177),
    (4, 0, -2, -1, 176),
    (4, -1, -1, -1, 166),
    (1, 0, 1, -1, -164),
    (4, 0, 1, -1, 132),
    (1, 0, -1, -1, -119),
    (4, -1, 0, -1, 115),
    (2, -2, 0, 1, 107),
)

_LD = np.array(_LONGITUDE_DISTANCE, dtype=float)
_LB = np.array(_LATITUDE, dtype=float)


def centuries(jd_tdb) -> np.ndarray:
    return (np.asarray(jd_tdb, dtype=float) - J2000) / DAYS_PER_CENTURY


def arguments(jd_tdb) -> dict[str, np.ndarray]:
    """The five fundamental angles, in degrees, plus the eccentricity factor.

    ``L`` is the Moon's mean longitude, ``D`` its elongation from the Sun,
    ``M`` and ``Mp`` the Sun's and the Moon's mean anomalies, and ``F`` the
    argument of latitude -- the angle from the ascending node, which is what
    decides whether an eclipse happens at all.

    ``A1`` to ``A3`` stand in for Venus, and for Jupiter, and for the flattening
    of the Earth.
    """
    T = centuries(jd_tdb)
    return {
        "L": (
            218.3164477 + 481267.88123421 * T - 0.0015786 * T**2
            + T**3 / 538841 - T**4 / 65194000
        ) % 360.0,
        "D": (
            297.8501921 + 445267.1114034 * T - 0.0018819 * T**2
            + T**3 / 545868 - T**4 / 113065000
        ) % 360.0,
        "M": (
            357.5291092 + 35999.0502909 * T - 0.0001536 * T**2 + T**3 / 24490000
        ) % 360.0,
        "Mp": (
            134.9633964 + 477198.8675055 * T + 0.0087414 * T**2
            + T**3 / 69699 - T**4 / 14712000
        ) % 360.0,
        "F": (
            93.2720950 + 483202.0175233 * T - 0.0036539 * T**2
            - T**3 / 3526000 + T**4 / 863310000
        ) % 360.0,
        "A1": (119.75 + 131.849 * T) % 360.0,
        "A2": (53.09 + 479264.290 * T) % 360.0,
        "A3": (313.45 + 481266.484 * T) % 360.0,
        # The Earth's orbit is slowly rounding out, and terms involving the
        # Sun's anomaly scale with it.
        "E": 1.0 - 0.002516 * T - 0.0000074 * T**2,
    }


def _sums(angles: dict[str, np.ndarray]):
    """The three periodic sums, in units of 1e-6 deg, 1e-6 deg and 1e-3 km."""
    D = np.radians(angles["D"])[..., None]
    M = np.radians(angles["M"])[..., None]
    Mp = np.radians(angles["Mp"])[..., None]
    F = np.radians(angles["F"])[..., None]
    E = np.asarray(angles["E"])[..., None]

    def eccentricity(power):
        """E once per power of M, because those terms scale with it."""
        return E ** np.abs(power)

    a_ld = _LD[:, 0] * D + _LD[:, 1] * M + _LD[:, 2] * Mp + _LD[:, 3] * F
    weight_ld = eccentricity(_LD[:, 1])
    sigma_l = np.sum(_LD[:, 4] * weight_ld * np.sin(a_ld), axis=-1)
    sigma_r = np.sum(_LD[:, 5] * weight_ld * np.cos(a_ld), axis=-1)

    a_b = _LB[:, 0] * D + _LB[:, 1] * M + _LB[:, 2] * Mp + _LB[:, 3] * F
    sigma_b = np.sum(_LB[:, 4] * eccentricity(_LB[:, 1]) * np.sin(a_b), axis=-1)

    # Venus, Jupiter and the Earth's flattening, as three extra angles.
    L = np.radians(angles["L"])
    A1 = np.radians(angles["A1"])
    A2 = np.radians(angles["A2"])
    A3 = np.radians(angles["A3"])
    Mp0 = np.radians(angles["Mp"])
    F0 = np.radians(angles["F"])

    sigma_l = sigma_l + 3958 * np.sin(A1) + 1962 * np.sin(L - F0) + 318 * np.sin(A2)
    sigma_b = (
        sigma_b
        - 2235 * np.sin(L)
        + 382 * np.sin(A3)
        + 175 * np.sin(A1 - F0)
        + 175 * np.sin(A1 + F0)
        + 127 * np.sin(L - Mp0)
        - 115 * np.sin(L + Mp0)
    )
    return sigma_l, sigma_b, sigma_r


def spherical(jd_tdb) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Geocentric ecliptic longitude, latitude (degrees) and distance (km).

    Referred to the mean equinox of date, which is what the theory is built on.
    """
    angles = arguments(jd_tdb)
    sigma_l, sigma_b, sigma_r = _sums(angles)

    longitude = (angles["L"] + sigma_l / 1e6) % 360.0
    latitude = sigma_b / 1e6
    distance = MEAN_DISTANCE_KM + sigma_r / 1e3
    return longitude, latitude, distance


def position(jd_tdb) -> np.ndarray:
    """Geocentric position of the Moon, ``(..., 3)`` in au on ICRF axes.

    The theory works in the mean ecliptic and equinox *of date*, which is not
    the frame anything else here uses, so the result is carried back: tilted by
    the obliquity of date onto the mean equator of date, then un-precessed to
    J2000. Skipping that second step leaves the Moon drifting by the whole of
    precession -- 5000 arcseconds a century, five hundred times the error of
    the theory itself.
    """
    from .kepler import AU_KM
    from .precession import _apply, mean_obliquity, precession_matrix

    longitude, latitude, distance = spherical(jd_tdb)
    lam, beta = np.radians(longitude), np.radians(latitude)
    radius = distance / AU_KM

    ecliptic = radius[..., None] * np.stack(
        [
            np.cos(beta) * np.cos(lam),
            np.cos(beta) * np.sin(lam),
            np.sin(beta),
        ],
        axis=-1,
    )

    # Ecliptic of date -> equator of date, about the x axis by the obliquity.
    eps = mean_obliquity(jd_tdb)
    cos, sin = np.cos(eps), np.sin(eps)
    x = ecliptic[..., 0]
    y = cos * ecliptic[..., 1] - sin * ecliptic[..., 2]
    z = sin * ecliptic[..., 1] + cos * ecliptic[..., 2]
    of_date = np.stack([x, y, z], axis=-1)

    # Equator of date -> J2000, the transpose of the precession matrix.
    matrix = np.moveaxis(precession_matrix(jd_tdb), (0, 1), (-2, -1))
    return np.einsum("...ji,...j->...i", matrix, of_date)
