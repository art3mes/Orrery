"""Geometry for the 3-D view: orbit rings, trails, and how big to draw things.

No rendering here, and no polyscope import. Everything the viewer draws is built
in this module as plain arrays, so it can be checked by ``validate_m1.py``
without opening a window.

**Scene units are astronomical units, exactly.** Positions are never rescaled,
warped, compressed or log-mapped -- what you see is where the planet is. The
only lie is the size of the spheres, because at true scale the Earth is 1/23000
of its orbital radius and would occupy less than a pixel from anywhere you could
see the whole orbit from. The exaggeration factor is a number on screen, and it
is the only number on screen that is not real.

The Sun gets its own, much smaller factor. At the planets' default
exaggeration the Sun's sphere would reach past Mercury's perihelion and the
picture would be actively misleading rather than merely stylised.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .elements import ORDER, VALID_JD, canonical
from .kepler import AU_KM, ellipse, position

# Equatorial radii, km. IAU 2015 nominal values where they exist.
RADIUS_KM = {
    "sun": 695700.0,
    "mercury": 2439.7,
    "venus": 6051.8,
    # WGS84 equatorial, matching observer.EARTH_RADIUS_KM. It was 6371, the
    # *mean* radius, which is a different number and made the Earth come out
    # 0.00224 flattened instead of 0.00335.
    "embary": 6378.137,  # the barycentre has no radius of its own
    "mars": 3396.2,
    "jupiter": 71492.0,
    "saturn": 60268.0,
    "uranus": 25559.0,
    "neptune": 24764.0,
    "pluto": 1188.3,
}

COLOR = {
    "sun": (1.00, 0.85, 0.30),
    "mercury": (0.66, 0.62, 0.58),
    "venus": (0.90, 0.78, 0.55),
    "embary": (0.35, 0.58, 0.85),
    "mars": (0.80, 0.40, 0.28),
    "jupiter": (0.82, 0.68, 0.52),
    "saturn": (0.86, 0.78, 0.58),
    "uranus": (0.60, 0.82, 0.86),
    "neptune": (0.35, 0.48, 0.82),
    "pluto": (0.72, 0.66, 0.60),
}

# Defaults chosen so the Sun stays well inside Mercury's perihelion (0.307 au)
# and the Earth is still a visible dot when the whole inner system is in frame.
DEFAULT_PLANET_EXAGGERATION = 1000.0
DEFAULT_SUN_EXAGGERATION = 30.0

# Fraction of its own orbital period each body's trail covers. Scaling the trail
# to the body rather than to a fixed number of days is what stops Mercury from
# smearing into a solid ring while Neptune shows a stub.
DEFAULT_TRAIL_FRACTION = 0.25


def display_radius_au(body: str, exaggeration: float) -> float:
    """Radius to draw *body* at, in au. Not its real radius."""
    key = "sun" if body == "sun" else canonical(body)
    if exaggeration <= 0:
        raise ValueError("exaggeration must be positive")
    return RADIUS_KM[key] / AU_KM * exaggeration


MERCURY_PERIHELION_AU = 0.30750  # a(1 - e) at J2000


def sun_fits_inside_mercury(exaggeration: float) -> bool:
    """Would the drawn Sun stay clear of Mercury's closest approach?"""
    return display_radius_au("sun", exaggeration) < MERCURY_PERIHELION_AU


def largest_honest_sun() -> float:
    """The most the Sun can be exaggerated before it eats Mercury's orbit.

    66, as it happens: the Sun's radius is 0.00465 au and Mercury comes within
    0.3075. Past this the picture starts asserting something false -- that
    Mercury's orbit is inside the Sun -- and no caption undoes that, because a
    reader believes what a picture shows before they read the label.
    """
    return MERCURY_PERIHELION_AU * AU_KM / RADIUS_KM["sun"]


def orbit_loop(body: str, jd_tdb: float, samples: int = 512) -> np.ndarray:
    """Closed ring of ``samples`` points tracing *body*'s orbit at *jd_tdb*."""
    return ellipse(body, jd_tdb, samples=samples)


def trail_span_days(body: str, fraction: float = DEFAULT_TRAIL_FRACTION) -> float:
    """How far back a trail should reach, as a fraction of the orbital period."""
    from .kepler import period

    return period(body) * fraction


def trail(
    body: str,
    jd_tdb: float,
    span_days: float,
    samples: int = 200,
) -> np.ndarray:
    """The path *body* actually travelled over the last *span_days*.

    Unlike :func:`orbit_loop` this is a trajectory, sampled at real dates, so it
    is clamped to the element table's 1800-2050 window. Without the clamp,
    Pluto's 62-year trail would reach back past 1800 whenever the scrubber is
    near the start of the range, and quietly fill the scene with extrapolated
    positions.
    """
    if samples < 2:
        raise ValueError("a trail needs at least 2 samples")

    end = float(np.clip(jd_tdb, *VALID_JD))
    start = float(np.clip(end - abs(span_days), *VALID_JD))
    return position(body, np.linspace(start, end, samples))


# The scene spans 0.31 au to 49 au, a factor of 160, so no single set of sizes
# works across it: sizes that make Mercury visible turn Pluto's orbit into a
# solid wall, and sizes that suit Pluto put the Earth below one pixel. Rather
# than compress the distances to hide that -- which is what a log radial scale
# would do, and the reason most pretty solar systems are useless for measuring
# anything -- the view offers framings, and states the exaggeration of each.


@dataclass(frozen=True)
class ViewPreset:
    """One framing: how far out, how oversized, and what is worth showing."""

    scale_au: float
    planet_exaggeration: float
    sun_exaggeration: float
    bodies: tuple[str, ...]


# The wide framings used to open at 300x and 500x, which put the drawn Sun at
# 1.4 and 2.3 au -- past Mercury, past Venus, and most of the way past the
# Earth. It looked better and it was not true, and the viewer announced as much
# in its own status line on startup. They open at the honest ceiling now; the
# slider still runs to 1000x for anyone who wants the Sun visible from Pluto,
# and the warning then fires as a consequence of asking rather than by default.
_HONEST_SUN = float(int(largest_honest_sun()))  # 66

VIEW_PRESETS = {
    "inner": ViewPreset(2.0, 1000.0, 30.0, ORDER[:4]),
    "planets": ViewPreset(32.0, 1500.0, _HONEST_SUN, ORDER[:8]),
    "all": ViewPreset(52.0, 3000.0, _HONEST_SUN, ORDER),
}


def line_radius_au(view_scale: float) -> float:
    """Thickness for orbit rings and trails, in au, for a given framing.

    Orbit lines are annotation, not object: they mark where a path goes and have
    no physical width, so scaling them with the view is right where scaling a
    planet would not be. A fixed width in au is either invisible at 50 au or a
    solid tube at 2 au.
    """
    return 0.004 * view_scale


# The camera opens at 2.555 view_scale from the origin -- sqrt(2.2^2 + 1.3^2),
# from the look_at that frames each preset -- so this is the constant above,
# expressed against the one length that keeps changing.
LINE_RADIUS_PER_DISTANCE = 0.004 / 2.555


def line_radius_from_camera(distance_au: float) -> float:
    """Thickness for orbit lines, from how far the camera is from what it sees.

    ``view_scale`` is the framing a preset *opened* at, and it never changes
    again; the camera moves the instant anyone scrolls. Sizing annotation off
    the preset means a tube that is 0.2 au thick in the 52 au view stays 0.2 au
    thick when you fly in to look at the Earth -- a fifth of the radius of the
    orbit it is supposed to be marking, drawn as a doughnut you can lose a
    planet inside. Annotation has to track the camera or it is not annotation.
    """
    return LINE_RADIUS_PER_DISTANCE * max(distance_au, 1e-6)


def all_bodies() -> tuple[str, ...]:
    return ORDER
