"""orrery -- solar system positions you can check against NASA.

    >>> from orrery import position, jd
    >>> position("mars", jd(2026, 9, 3))
    array([...])

Positions are heliocentric, in au, in the mean ecliptic and equinox of J2000.
Dates are TDB Julian dates. Accuracy is whatever ``scripts/validate_m*.py`` last
measured; the README carries the numbers, and every one of them was measured
against an independent implementation of the same question rather than recalled.

**The module layer.** The names lifted to the top of this package are the ones
you reach for first; everything else lives in a module, and the modules are the
real surface. In rough order of what depends on what:

==============  ==============================================================
``times``       Julian dates in, calendar dates out. Everything is TDB.
``elements``    JPL's Keplerian element table, 1800-2050.
``kepler``      Elements to a position: solve Kepler's equation, then rotate.
``frames``      Ecliptic and equatorial axes; angles between vectors.
``events``      Extrema and crossings of a sampled curve -- conjunctions,
                oppositions, greatest elongations, contact times.
``nbody``       The planets as bodies rather than ellipses. Leapfrog,
                Yoshida-4, RK4, and the 1PN relativity term.
``precession``  The equinox moves: IAU 1976 precession, 1980 nutation,
                sidereal time.
``observer``    A place on a rotating, flattened Earth. Delta T.
``apparent``    What you actually see: light-time, deflection, aberration.
``model``       This package's own orbits, packed as an ``Ephemeris``.
``truth``       DE440, packed the same way. The only module that imports
                Skyfield, and no other module imports it for numbers.
``rotation``    Which way a body faces: IAU pole and prime meridian.
``lunar``       The Moon, from an abridged ELP-2000/82.
``eclipse``     Shadow cones, where they land, and how much they cover.
``globe``       Spheres, oblateness, rings, and surface textures.
``scene``       Display geometry -- radii, colours, orbit loops, trails.
``view``        The polyscope viewer.
==============  ==============================================================

``model`` and ``truth`` both hand back an :class:`~orrery.apparent.Ephemeris`.
That is what lets the same eclipse code run on this package's orbits and on
JPL's, and the two answers be differenced -- the method of the whole project in
one sentence.

Nothing here imports Skyfield, polyscope, Pillow or the network at import time.
``import orrery`` needs numpy and nothing else.
"""

from . import (
    apparent,
    eclipse,
    elements,
    events,
    frames,
    globe,
    kepler,
    lunar,
    model,
    nbody,
    observer,
    precession,
    rotation,
    scene,
    times,
    truth,
    view,
)
from .apparent import Ephemeris, Sight, observe
from .eclipse import lunar_view, shadow_landing, solar_view
from .elements import BODIES, J2000, elements_at
from .events import elongation_deg, find_crossings, find_extrema
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
from .lunar import position as moon_position
from .model import ephemeris
from .nbody import Trajectory, integrate
from .observer import SITES, Site, delta_t_seconds
from .rotation import body_to_icrf, obliquity_degrees, sub_solar_point
from .times import calendar, isoformat, jd

__version__ = "1.0.0"

__all__ = [
    # the modules, which are the real surface
    "apparent",
    "eclipse",
    "elements",
    "events",
    "frames",
    "globe",
    "kepler",
    "lunar",
    "model",
    "nbody",
    "observer",
    "precession",
    "rotation",
    "scene",
    "times",
    "truth",
    "view",
    # constants
    "AU_KM",
    "BODIES",
    "GM_SUN",
    "J2000",
    "SITES",
    "__version__",
    # dates
    "calendar",
    "isoformat",
    "jd",
    # where things are
    "elements_at",
    "ellipse",
    "moon_position",
    "period",
    "position",
    "solve_kepler",
    "state",
    # axes and angles
    "ecliptic_to_equatorial",
    "equatorial_to_ecliptic",
    "norm",
    "radec",
    "separation_arcsec",
    # when things happen
    "elongation_deg",
    "find_crossings",
    "find_extrema",
    # how they move
    "Trajectory",
    "integrate",
    # what you would actually see
    "Ephemeris",
    "Sight",
    "Site",
    "delta_t_seconds",
    "ephemeris",
    "observe",
    # which way they face
    "body_to_icrf",
    "obliquity_degrees",
    "sub_solar_point",
    # shadows
    "lunar_view",
    "shadow_landing",
    "solar_view",
]
