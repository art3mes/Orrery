"""Standing somewhere: a place on the Earth, and where that puts you.

Every position so far has been geocentric, which nobody is. The Earth's radius
subtends about 8.8 arcsec at the Sun and 30 arcsec at Venus during a transit, so
two observers on opposite sides of the planet see Venus in noticeably different
places -- which is exactly how the astronomical unit was first measured, and why
transit contact times differ by minutes between sites.

Three things are needed to put an observer in space:

**A shape for the Earth.** WGS84, an ellipsoid flattened by 1/298.257. Treating
it as a sphere misplaces a mid-latitude observer by up to 21 km.

**A rotation angle.** Greenwich apparent sidereal time, which needs UT1 -- the
time scale that follows the Earth's actual, slightly irregular rotation --
rather than the uniform TDB everything else here uses.

**A velocity.** The ground moves at up to 465 m/s, which tilts starlight by 0.32
arcsec. Diurnal aberration is small but it is not below the level this package
works at, so the observer carries a velocity as well as a position.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from . import precession
from .kepler import AU_KM
from .times import jd as _jd
from .truth import DATA_DIR as _DATA_DIR, _use_system_trust_store

# WGS84.
EARTH_RADIUS_KM = 6378.137
EARTH_FLATTENING = 1.0 / 298.257223563
EARTH_RADIUS_AU = EARTH_RADIUS_KM / AU_KM

# Earth's rotation rate, radians per day. The sidereal day is about four
# minutes shorter than the solar one, which is where the extra 0.0027 comes from.
EARTH_ROTATION_RAD_PER_DAY = 7.292115e-5 * 86400.0


DELTA_T_TABLE = None  # filled lazily from data/delta_t.npz


def _tabulated_delta_t():
    """Load the cached table of measured TT-UT1, if there is one."""
    global DELTA_T_TABLE
    if DELTA_T_TABLE is None:
        path = _DATA_DIR / "delta_t.npz"
        if path.exists():
            with np.load(path) as data:
                DELTA_T_TABLE = (data["jd"], data["seconds"])
        else:
            DELTA_T_TABLE = ()
    return DELTA_T_TABLE or None


def build_delta_t_table(start: int = 1850, end: int = 2050, *, step_days: float = 30.0):
    """Cache measured delta T, sampled from the IERS data Skyfield carries.

    The Earth's rotation is not predictable, only observed, so delta T is a
    measurement and not a formula. The polynomial below is a *fit*, and after
    about 2005 it extrapolates: by 2018 it is 1.6 s high and climbing. For
    pointing that was 3 milliarcsec and beneath notice. For timing an eclipse it
    is 1.6 s directly, and the Moon's shadow crosses the ground at 700 m/s.

    Written once to ``data/delta_t.npz`` and read from there afterwards.
    """
    from skyfield.api import Loader

    _use_system_trust_store()
    loader = Loader(str(_DATA_DIR), verbose=False)
    ts = loader.timescale()

    jd = np.arange(_jd(start, 1, 1), _jd(end, 1, 1) + step_days, step_days)
    seconds = np.asarray(ts.tdb_jd(jd).delta_t, dtype=float)

    _DATA_DIR.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(_DATA_DIR / "delta_t.npz", jd=jd, seconds=seconds)

    global DELTA_T_TABLE
    DELTA_T_TABLE = (jd, seconds)
    return jd, seconds


def delta_t_seconds(jd_tdb) -> np.ndarray:
    """TT minus UT1, in seconds.

    Interpolated from measured values where a table has been cached, and from
    Espenak & Meeus's polynomials otherwise. :func:`build_delta_t_table` writes
    the table; ``validate_m4.py`` reports which one is in use, because an
    eclipse timed on the polynomial is a second or two out and an eclipse timed
    on the table is not.
    """
    table = _tabulated_delta_t()
    if table is not None:
        jd = np.asarray(jd_tdb, dtype=float)
        inside = (jd >= table[0][0]) & (jd <= table[0][-1])
        if np.all(inside):
            return np.interp(jd, table[0], table[1])

    return _polynomial_delta_t(jd_tdb)


def _polynomial_delta_t(jd_tdb) -> np.ndarray:
    """TT minus UT1, in seconds. Espenak & Meeus polynomials.

    The Earth is not a good clock: it wobbles, and tidal friction slows it. The
    gap has grown from zero in 1900 to about 69 s now, and it is not
    predictable, only fitted. These polynomials are a fit, and after about 2005
    they extrapolate -- the Earth has since sped up slightly, so they run a few
    seconds high. ``validate_m3.py`` measures the discrepancy against Skyfield's
    tabulated values rather than assuming it away.

    A few seconds of error here moves an observer by a couple of km, which is
    3 milliarcsec at 1 au. It matters for *timing* an event, not for pointing.
    """
    jd = np.asarray(jd_tdb, dtype=float)
    year = 2000.0 + (jd - 2451545.0) / 365.25

    def piece(t, *coefficients):
        return sum(c * t**i for i, c in enumerate(coefficients))

    out = np.empty_like(year)
    for low, high, origin, coefficients in (
        (-np.inf, 1920, 1900, (-2.79, 1.494119, -0.0598939, 0.0061966, -0.000197)),
        (1920, 1941, 1920, (21.20, 0.84493, -0.076100, 0.0020936)),
        (1941, 1961, 1950, (29.07, 0.407, -1 / 233, 1 / 2547)),
        (1961, 1986, 1975, (45.45, 1.067, -1 / 260, -1 / 718)),
        (1986, 2005, 2000,
         (63.86, 0.3345, -0.060374, 0.0017275, 0.000651814, 0.00002373599)),
        (2005, np.inf, 2000, (62.92, 0.32217, 0.005589)),
    ):
        mask = (year >= low) & (year < high)
        if np.any(mask):
            out[mask] = piece(year[mask] - origin, *coefficients)
    return out


def ut1_from_tdb(jd_tdb) -> np.ndarray:
    """UT1 Julian date. TDB and TT differ by under 2 ms, ignored here."""
    return np.asarray(jd_tdb, dtype=float) - delta_t_seconds(jd_tdb) / 86400.0


@dataclass(frozen=True)
class Site:
    """A place on the Earth's surface.

    Latitude and longitude in degrees, east positive; height in metres.
    """

    name: str
    latitude: float
    longitude: float
    height_m: float = 0.0

    def geocentric_components(self) -> tuple[float, float]:
        """``(rho cos phi', rho sin phi')`` in Earth radii.

        The classical pair: the ellipsoid squashes an observer toward the axis,
        so the geocentric latitude phi' differs from the geodetic latitude by up
        to 11 arcmin, and the distance from the centre by up to 21 km.
        """
        phi = np.radians(self.latitude)
        reduced = np.arctan((1.0 - EARTH_FLATTENING) * np.tan(phi))
        height = self.height_m / 1000.0 / EARTH_RADIUS_KM
        rho_sin = (1.0 - EARTH_FLATTENING) * np.sin(reduced) + height * np.sin(phi)
        rho_cos = np.cos(reduced) + height * np.cos(phi)
        return float(rho_cos), float(rho_sin)

    def offset_from_geocentre(self, jd_tdb) -> tuple[np.ndarray, np.ndarray]:
        """Position and velocity relative to the Earth's centre, on ICRF axes.

        In au and au/day. Built in the true equator and equinox of date, where
        sidereal time is defined, then rotated back onto ICRF axes so it can be
        added to a barycentric position.
        """
        jd_tdb = np.atleast_1d(np.asarray(jd_tdb, dtype=float))
        rho_cos, rho_sin = self.geocentric_components()

        gast = precession.greenwich_apparent_sidereal_time(
            ut1_from_tdb(jd_tdb), jd_tdb
        )
        theta = np.radians(gast + self.longitude)

        radius = EARTH_RADIUS_AU
        of_date = radius * np.stack(
            [
                rho_cos * np.cos(theta),
                rho_cos * np.sin(theta),
                np.full_like(theta, rho_sin),
            ],
            axis=-1,
        )
        # d/dt of the above: only theta moves, at the rotation rate.
        spin = EARTH_ROTATION_RAD_PER_DAY
        of_date_velocity = radius * spin * np.stack(
            [
                -rho_cos * np.sin(theta),
                rho_cos * np.cos(theta),
                np.zeros_like(theta),
            ],
            axis=-1,
        )

        return (
            _from_date_to_icrf(of_date, jd_tdb),
            _from_date_to_icrf(of_date_velocity, jd_tdb),
        )


def _from_date_to_icrf(vector: np.ndarray, jd_tdb: np.ndarray) -> np.ndarray:
    """Undo nutation then precession. The transpose of each, being rotations."""
    nutation = np.moveaxis(precession.nutation_matrix(jd_tdb), (0, 1), (-2, -1))
    spin = np.moveaxis(precession.precession_matrix(jd_tdb), (0, 1), (-2, -1))
    mean = np.einsum("...ji,...j->...i", nutation, vector)
    return np.einsum("...ji,...j->...i", spin, mean)


# A few places, so the demos and gates do not have to invent coordinates.
SITES = {
    "greenwich": Site("Royal Observatory, Greenwich", 51.4779, -0.0015, 47.0),
    "mauna_kea": Site("Mauna Kea", 19.8207, -155.4681, 4205.0),
    "paranal": Site("Paranal", -24.6275, -70.4044, 2635.0),
    "svalbard": Site("Longyearbyen", 78.2232, 15.6267, 30.0),
    "delhi": Site("New Delhi", 28.6139, 77.2090, 216.0),
}
