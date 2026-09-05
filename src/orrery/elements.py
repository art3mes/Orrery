"""JPL approximate Keplerian elements for the major planets.

Source: E. M. Standish, *Keplerian Elements for Approximate Positions of the
Major Planets*, JPL Solar System Dynamics.
https://ssd.jpl.nasa.gov/planets/approx_pos.html

Six numbers describe an ellipse in space and where the planet sits on it. Six
more describe how those numbers drift, per Julian century past J2000. That is
the entire dataset -- 108 floats for the whole solar system.

Elements are **heliocentric**, referred to the mean ecliptic and equinox of
J2000:

    a           semi-major axis                          au
    e           eccentricity                             --
    I           inclination to the ecliptic              deg
    L           mean longitude                           deg
    long.peri.  longitude of perihelion  (varpi)         deg
    long.node.  longitude of ascending node (Omega)      deg

Table 1 is valid **1800 AD - 2050 AD**. Outside that range the rates are being
extrapolated and the answer degrades quickly; :func:`check_epoch` warns.

The table is transcribed verbatim below so the package needs no network. One
mistyped digit here would corrupt every position the package produces while
still looking entirely plausible, so ``tests/test_elements.py`` diffs this text
against the JPL file whenever the network is available. Keep the layout
byte-identical to the source; that is what makes the diff readable.
"""

from __future__ import annotations

import warnings

import numpy as np

from .times import DAYS_PER_CENTURY, J2000, jd as _jd

# Validity window of table 1. Computed rather than written out: the endpoints
# are inclusive, and a hardcoded value half a day short makes every date on the
# final day warn for no reason.
VALID_JD = (_jd(1800, 1, 1), _jd(2050, 1, 1))

# --- JPL table 1, 1800 AD - 2050 AD -----------------------------------------
# For each body: a line of elements at J2000, then a line of rates per century.
# (The source file labels the eccentricity column "rad, rad/Cy"; eccentricity is
# dimensionless. That is an error in the header, not in the numbers.)

_TABLE_1 = """
               a              e               I                L            long.peri.      long.node.
           au, au/Cy     rad, rad/Cy     deg, deg/Cy      deg, deg/Cy      deg, deg/Cy     deg, deg/Cy
Mercury   0.38709927      0.20563593      7.00497902      252.25032350     77.45779628     48.33076593
          0.00000037      0.00001906     -0.00594749   149472.67411175      0.16047689     -0.12534081
Venus     0.72333566      0.00677672      3.39467605      181.97909950    131.60246718     76.67984255
          0.00000390     -0.00004107     -0.00078890    58517.81538729      0.00268329     -0.27769418
EM Bary   1.00000261      0.01671123     -0.00001531      100.46457166    102.93768193      0.0
          0.00000562     -0.00004392     -0.01294668    35999.37244981      0.32327364      0.0
Mars      1.52371034      0.09339410      1.84969142       -4.55343205    -23.94362959     49.55953891
          0.00001847      0.00007882     -0.00813131    19140.30268499      0.44441088     -0.29257343
Jupiter   5.20288700      0.04838624      1.30439695       34.39644051     14.72847983    100.47390909
         -0.00011607     -0.00013253     -0.00183714     3034.74612775      0.21252668      0.20469106
Saturn    9.53667594      0.05386179      2.48599187       49.95424423     92.59887831    113.66242448
         -0.00125060     -0.00050991      0.00193609     1222.49362201     -0.41897216     -0.28867794
Uranus   19.18916464      0.04725744      0.77263783      313.23810451    170.95427630     74.01692503
         -0.00196176     -0.00004397     -0.00242939      428.48202785      0.40805281      0.04240589
Neptune  30.06992276      0.00859048      1.77004347      -55.12002969     44.96476227    131.78422574
          0.00026291      0.00005105      0.00035372      218.45945325     -0.32241464     -0.00508664
Pluto    39.48211675      0.24882730     17.14001206      238.92903833    224.06891629    110.30393684
         -0.00031596      0.00005170      0.00004818      145.20780515     -0.04062942     -0.01183482
"""

# JPL's current page publishes table 1 for the eight planets only; the Pluto row
# above is from an earlier revision of the same table. It is kept because it
# works -- validate_m0.py measures it at 60 arcsec against DE440, in family with
# Neptune -- but it is the one row the transcription test cannot check against
# the live source, so it rests on the gate alone.
PLUTO_IS_UNVERIFIABLE_AGAINST_SOURCE = True

ELEMENT_NAMES = ("a", "e", "I", "L", "long_peri", "long_node")

# Canonical key -> name as it appears in the JPL table.
_KEYS = {
    "mercury": "Mercury",
    "venus": "Venus",
    "embary": "EM Bary",
    "mars": "Mars",
    "jupiter": "Jupiter",
    "saturn": "Saturn",
    "uranus": "Uranus",
    "neptune": "Neptune",
    "pluto": "Pluto",
}

# "earth" is accepted, but the table's third row is the Earth-Moon barycentre,
# which the Earth itself orbits at up to ~4700 km. Anything that cares about
# that distinction should say embary and mean it.
_ALIASES = {
    "earth": "embary",
    "em bary": "embary",
    "embary": "embary",
    "earth-moon": "embary",
    "earth moon barycenter": "embary",
}

BODIES = tuple(_KEYS)

# Bodies in order of distance from the Sun, for iteration and display.
ORDER = BODIES


def _parse(
    table: str, require: set[str] | None = None
) -> dict[str, tuple[np.ndarray, np.ndarray]]:
    """Parse JPL table text into ``{key: (elements, rates)}``.

    Scans for lines that begin with a known body name and takes the following
    line as that body's rates, so headers, horizontal rules and trailing notes
    need no special-casing. That matters because this same parser is pointed at
    JPL's live page by ``tests/test_elements.py``, and their layout is theirs to
    change.

    Body names may contain spaces ("EM Bary"), so a data line is split from the
    right: the last six fields are numbers, whatever precedes them is the name.
    """
    lines = [ln for ln in table.splitlines() if ln.strip()]
    label_to_key = {v: k for k, v in _KEYS.items()}

    out: dict[str, tuple[np.ndarray, np.ndarray]] = {}
    for index, line in enumerate(lines):
        parts = line.split()
        if len(parts) < 7:
            continue
        name = " ".join(parts[:-6])
        key = label_to_key.get(name)
        if key is None:
            continue
        if key in out:
            raise ValueError(f"{name!r} appears twice in the element table")
        if index + 1 >= len(lines):
            raise ValueError(f"{name!r} has no rate line")

        values = np.array([float(x) for x in parts[-6:]])
        rates = np.array([float(x) for x in lines[index + 1].split()])
        if rates.size != 6:
            raise ValueError(f"expected 6 rates for {name!r}, got {rates.size}")
        out[key] = (values, rates)

    missing = (set(_KEYS) if require is None else require) - set(out)
    if missing:
        raise ValueError(f"element table is missing {sorted(missing)}")
    return out


_ELEMENTS = _parse(_TABLE_1)


def canonical(body: str) -> str:
    """Normalise a body name. Raises on anything unrecognised."""
    key = body.strip().lower().replace("_", " ")
    key = _ALIASES.get(key, key.replace(" ", ""))
    if key not in _ELEMENTS:
        raise KeyError(f"unknown body {body!r}; known: {', '.join(BODIES)}")
    return key


def check_epoch(jd_tdb) -> None:
    """Warn if any date falls outside table 1's 1800-2050 validity window."""
    jd = np.asarray(jd_tdb, dtype=float)
    if np.any(jd < VALID_JD[0]) or np.any(jd > VALID_JD[1]):
        warnings.warn(
            "date outside the 1800-2050 range of JPL element table 1; the "
            "linear rates are being extrapolated and errors grow fast",
            RuntimeWarning,
            stacklevel=3,
        )


def elements_at(body: str, jd_tdb) -> dict[str, np.ndarray]:
    """Osculating elements of *body* at Julian date(s) *jd_tdb* (TDB).

    Returns a dict with ``a`` in au, ``e`` dimensionless, and ``I``, ``L``,
    ``long_peri``, ``long_node`` in **degrees**. Each value broadcasts to the
    shape of *jd_tdb*.
    """
    key = canonical(body)
    check_epoch(jd_tdb)
    values, rates = _ELEMENTS[key]

    T = (np.asarray(jd_tdb, dtype=float) - J2000) / DAYS_PER_CENTURY
    # values/rates are (6,); T is (...,). Broadcast to (..., 6), then unpack
    # along the last axis so each element comes back with the shape of jd_tdb.
    current = values + rates * T[..., None]
    return dict(zip(ELEMENT_NAMES, np.moveaxis(current, -1, 0)))


def table_text() -> str:
    """The embedded JPL table, verbatim. Used by the transcription test."""
    return _TABLE_1
