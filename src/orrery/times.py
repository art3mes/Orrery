"""Calendar dates to Julian dates, and back.

Every date in this package is a **TDB** Julian date. TDB is the time scale the
ephemerides are built on; it runs uniformly, with no leap seconds.

That choice matters more than it looks. UTC differs from TDB by ~69 s today,
and the Earth covers about 2000 km in 69 s -- a tenth of the error budget of the
element table itself, thrown away for nothing. So the calendar helpers here do
*not* convert time scales: the date you hand in is taken to be TDB. When
comparing against an external ephemeris, build both sides in TDB and the
question never arises.
"""

from __future__ import annotations

import numpy as np

# Julian date of the J2000.0 epoch (2000 Jan 1.5 TDB) -- the zero point every
# element and rate in this package is referred to.
J2000 = 2451545.0

# Days per Julian century, the unit the element rates are quoted in.
DAYS_PER_CENTURY = 36525.0


def jd(
    year: int,
    month: int = 1,
    day: int = 1,
    hour: int = 0,
    minute: int = 0,
    second: float = 0.0,
) -> float:
    """Julian date (TDB) of a proleptic Gregorian calendar date.

    Meeus, *Astronomical Algorithms*, chapter 7.
    """
    y, m = int(year), int(month)
    if m <= 2:
        y -= 1
        m += 12
    a = y // 100
    b = 2 - a + a // 4  # Gregorian calendar correction
    day_frac = day + (hour + minute / 60.0 + second / 3600.0) / 24.0
    return (
        int(365.25 * (y + 4716))
        + int(30.6001 * (m + 1))
        + day_frac
        + b
        - 1524.5
    )


def calendar(jd_tdb: float) -> tuple[int, int, float]:
    """Inverse of :func:`jd`: ``(year, month, day_with_fraction)``."""
    z = jd_tdb + 0.5
    f = z - int(z)
    z = int(z)
    if z < 2299161:
        a = z
    else:
        alpha = int((z - 1867216.25) / 36524.25)
        a = z + 1 + alpha - alpha // 4
    b = a + 1524
    c = int((b - 122.1) / 365.25)
    d = int(365.25 * c)
    e = int((b - d) / 30.6001)
    day = b - d - int(30.6001 * e) + f
    month = e - 1 if e < 14 else e - 13
    year = c - 4716 if month > 2 else c - 4715
    return year, month, day


def isoformat(jd_tdb: float) -> str:
    """A readable ``YYYY-MM-DD HH:MM`` label for a TDB Julian date."""
    year, month, day = calendar(jd_tdb)
    whole = int(day)
    hours = (day - whole) * 24.0
    hh = int(hours)
    mm = int(round((hours - hh) * 60.0))
    if mm == 60:  # rounding can carry
        mm, hh = 0, hh + 1
    return f"{year:04d}-{month:02d}-{whole:02d} {hh:02d}:{mm:02d}"


def linspace(jd_start: float, jd_end: float, count: int) -> np.ndarray:
    """*count* Julian dates spanning ``[jd_start, jd_end]`` inclusive."""
    return np.linspace(jd_start, jd_end, count)
