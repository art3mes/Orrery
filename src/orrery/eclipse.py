"""Shadows: where the Moon's falls on the Earth, and where the Earth's falls on
the Moon.

An eclipse is the one place in this project where every correction built so far
has to be right at once. Light-time, aberration, the observer's own position,
the Earth's rotation, delta T: get any of them wrong and the answer is out by
kilometres or minutes rather than by arcseconds, because the geometry is
absurdly marginal.

How marginal: the Moon's umbral cone is about **374 000 km** long and the Moon
is on average **384 400 km** away. The shadow's point usually misses the Earth
entirely, which is why most central eclipses are annular, and why the total ones
run along a strip a hundred kilometres wide instead of covering a hemisphere.
The Sun and Moon happen to be the same angular size to within a few percent,
and totality exists in the slack.

Sizes here are radii in kilometres, because that is how the cones are written
and converting to au and back only loses digits.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .frames import norm, separation_arcsec
from .kepler import AU_KM
from .nbody import C_AU_PER_DAY
from .observer import EARTH_FLATTENING, EARTH_RADIUS_KM
from .rotation import body_to_icrf, surface_point

SUN_RADIUS_KM = 695700.0
MOON_RADIUS_KM = 1737.4

# The Earth's atmosphere makes its shadow bigger than geometry says. Danjon's
# rule -- a 2% enlargement -- is the convention published contact times use, so
# omitting it would disagree with every almanac by about a minute.
ATMOSPHERE_ENLARGEMENT = 1.02


def angular_radius_arcsec(radius_km: float, distance_au) -> np.ndarray:
    """Apparent radius of a sphere, in arcseconds."""
    distance_km = np.asarray(distance_au, dtype=float) * AU_KM
    return np.degrees(np.arcsin(radius_km / distance_km)) * 3600.0


def umbra_length_km(sun_distance_km, body_radius_km: float = MOON_RADIUS_KM) -> np.ndarray:
    """How far behind a body its cone of total shadow reaches.

    Similar triangles: ``L = r D / (R - r)`` for a body of radius *r* lit by a
    sphere of radius *R* at distance *D*. For the Moon this is about 374 000 km
    against a mean distance of 384 400, so the tip usually falls short of the
    Earth -- and an eclipse that would have been total is annular instead.
    """
    distance = np.asarray(sun_distance_km, dtype=float)
    return body_radius_km * distance / (SUN_RADIUS_KM - body_radius_km)


def overlap_fraction(separation, radius_a, radius_b) -> np.ndarray:
    """Fraction of disc *a* hidden by disc *b*. Two circles, one lens.

    Eclipse *magnitude* is a ratio of diameters and is the number usually
    quoted; this is the ratio of areas, which is what actually dims the sky.
    """
    separation = np.asarray(separation, dtype=float)
    radius_a = np.asarray(radius_a, dtype=float)
    radius_b = np.asarray(radius_b, dtype=float)

    apart = separation >= radius_a + radius_b
    swallowed = separation <= np.abs(radius_a - radius_b)

    safe = np.clip(separation, 1e-12, None)
    cos_a = np.clip(
        (safe**2 + radius_a**2 - radius_b**2) / (2 * safe * radius_a), -1.0, 1.0
    )
    cos_b = np.clip(
        (safe**2 + radius_b**2 - radius_a**2) / (2 * safe * radius_b), -1.0, 1.0
    )
    alpha, beta = np.arccos(cos_a), np.arccos(cos_b)
    # Half the square root of the Heron-style product: the area of the kite
    # formed by the two centres and the two intersection points. It is
    # subtracted once, not twice -- doing it twice gives 0.115 where two equal
    # discs a radius apart overlap by 0.391, and the error only ever shows up
    # on *partial* eclipses, because total and annular ones take the
    # one-disc-inside-the-other branch above.
    kite = 0.5 * np.sqrt(
        np.clip(
            (-safe + radius_a + radius_b)
            * (safe + radius_a - radius_b)
            * (safe - radius_a + radius_b)
            * (safe + radius_a + radius_b),
            0.0,
            None,
        )
    )
    lens = radius_a**2 * alpha + radius_b**2 * beta - kite

    covered = lens / (np.pi * radius_a**2)
    covered = np.where(apart, 0.0, covered)
    return np.where(swallowed, np.minimum(1.0, (radius_b / radius_a) ** 2), covered)


@dataclass
class SolarView:
    """What the Sun looks like from one place at a series of instants."""

    jd: np.ndarray
    separation: np.ndarray  # arcsec between the two centres
    sun_radius: np.ndarray  # arcsec
    moon_radius: np.ndarray  # arcsec
    magnitude: np.ndarray  # fraction of the solar diameter covered
    obscuration: np.ndarray  # fraction of the solar area covered

    @property
    def eclipsed(self) -> np.ndarray:
        return self.separation < self.sun_radius + self.moon_radius

    @property
    def total(self) -> np.ndarray:
        return self.separation + self.sun_radius <= self.moon_radius

    @property
    def annular(self) -> np.ndarray:
        return self.separation + self.moon_radius <= self.sun_radius

    def kind(self) -> str:
        if np.any(self.total):
            return "total"
        if np.any(self.annular):
            return "annular"
        if np.any(self.eclipsed):
            return "partial"
        return "none"


def solar_view(ephemeris, jd, *, site=None) -> SolarView:
    """Sun and Moon as seen together, with light-time and parallax included."""
    jd = np.atleast_1d(np.asarray(jd, dtype=float))
    sun = ephemeris.look("sun", jd, site=site)
    moon = ephemeris.look("moon", jd, site=site)

    separation = separation_arcsec(sun.apparent, moon.apparent)
    sun_radius = angular_radius_arcsec(SUN_RADIUS_KM, sun.distance)
    moon_radius = angular_radius_arcsec(MOON_RADIUS_KM, moon.distance)

    magnitude = np.clip(
        (sun_radius + moon_radius - separation) / (2.0 * sun_radius), 0.0, None
    )
    return SolarView(
        jd=jd,
        separation=separation,
        sun_radius=sun_radius,
        moon_radius=moon_radius,
        magnitude=magnitude,
        obscuration=overlap_fraction(separation, sun_radius, moon_radius),
    )


def _sun_when_the_light_left(ephemeris, jd, target: str):
    """The Sun's position at the moment the light now lighting *target* left it.

    A shadow is cast by the light arriving, not by where its source is now. It
    is worth **7 km** of the Sun's position and no more, because what moves in
    8.3 minutes is the *Sun's barycentric* motion -- 12 m/s -- and not the
    Earth's 30 km/s. Retarding the observer is a 15 000 km correction;
    retarding the source is not, and the two are easy to confuse.

    Kept because it is correct and free, not because it changes an answer.
    """
    jd = np.atleast_1d(np.asarray(jd, dtype=float))
    body = ephemeris.at(target)(jd)
    emitted = jd
    for _ in range(3):
        distance = norm(body - ephemeris.at("sun")(emitted))
        emitted = jd - distance / C_AU_PER_DAY
    return ephemeris.at("sun")(emitted)


def shadow_axis(ephemeris, jd):
    """Where the Sun-Moon line runs, relative to the Earth's centre.

    Returns ``(miss_distance_km, direction, moon_from_earth_km)``. The miss
    distance is how far the axis passes from the Earth's centre; when it is less
    than an Earth radius, the axis lands somewhere and there is a central
    eclipse.
    """
    jd = np.atleast_1d(np.asarray(jd, dtype=float))
    earth = ephemeris.at("geocentre")(jd)
    moon = (ephemeris.at("moon")(jd) - earth) * AU_KM
    sun = (_sun_when_the_light_left(ephemeris, jd, "moon") - earth) * AU_KM

    along = moon - sun
    direction = along / norm(along)[..., None]
    projection = np.sum(moon * direction, axis=-1)
    miss = norm(moon - projection[..., None] * direction)
    return miss, direction, moon


def shadow_landing(ephemeris, jd):
    """Latitude and east longitude where the shadow axis meets the Earth.

    NaN where the axis misses. Solved on the ellipsoid, not a sphere: the
    flattening moves the landing point by up to a fifth of a degree, which is
    twenty kilometres of a track a hundred kilometres wide.

    The trick is to squash the problem rather than the answer -- scale the polar
    axis so the Earth becomes a unit sphere, intersect, then unscale.
    """
    jd = np.atleast_1d(np.asarray(jd, dtype=float))
    _, direction, moon = shadow_axis(ephemeris, jd)

    latitude = np.full(jd.shape, np.nan)
    longitude = np.full(jd.shape, np.nan)

    for i, when in enumerate(jd):
        frame = np.asarray(body_to_icrf("embary", when), dtype=float)
        # Into Earth-fixed axes, where z is the pole and the flattening is simple.
        start = frame.T @ moon[i]
        step = frame.T @ direction[i]

        squash = np.array([1.0, 1.0, 1.0 / (1.0 - EARTH_FLATTENING)])
        start_s = start * squash / EARTH_RADIUS_KM
        step_s = step * squash
        step_s = step_s / np.linalg.norm(step_s)

        b = start_s @ step_s
        c = start_s @ start_s - 1.0
        discriminant = b * b - c
        if discriminant < 0:
            continue  # the axis misses the Earth

        k = -b - np.sqrt(discriminant)  # the near side, facing the Moon
        hit = (start_s + k * step_s) / squash
        latitude[i], longitude[i] = surface_point("embary", when, frame @ hit)

    return latitude, longitude


@dataclass
class LunarView:
    """The Moon against the Earth's shadow."""

    jd: np.ndarray
    miss_km: np.ndarray  # Moon's centre from the shadow axis
    umbra_km: np.ndarray  # radius of the umbra at the Moon's distance
    penumbra_km: np.ndarray

    @property
    def penumbral(self) -> np.ndarray:
        return self.miss_km < self.penumbra_km + MOON_RADIUS_KM

    @property
    def partial(self) -> np.ndarray:
        return self.miss_km < self.umbra_km + MOON_RADIUS_KM

    @property
    def total(self) -> np.ndarray:
        return self.miss_km + MOON_RADIUS_KM < self.umbra_km

    def kind(self) -> str:
        if np.any(self.total):
            return "total"
        if np.any(self.partial):
            return "partial"
        if np.any(self.penumbral):
            return "penumbral"
        return "none"


def lunar_view(ephemeris, jd) -> LunarView:
    """The Moon's distance from the axis of the Earth's shadow, and the cone there.

    The umbra narrows with distance and the penumbra widens::

        umbra    = R_earth - d (R_sun - R_earth) / D
        penumbra = R_earth + d (R_sun + R_earth) / D

    both enlarged by 2% for the atmosphere, which is the convention the
    published contact times follow.
    """
    jd = np.atleast_1d(np.asarray(jd, dtype=float))
    earth = ephemeris.at("geocentre")(jd)
    moon = (ephemeris.at("moon")(jd) - earth) * AU_KM
    sun = (_sun_when_the_light_left(ephemeris, jd, "geocentre") - earth) * AU_KM

    sun_distance = norm(sun)
    away = -sun / sun_distance[..., None]  # the shadow points away from the Sun

    along = np.sum(moon * away, axis=-1)
    miss = norm(moon - along[..., None] * away)

    umbra = EARTH_RADIUS_KM - along * (SUN_RADIUS_KM - EARTH_RADIUS_KM) / sun_distance
    penumbra = EARTH_RADIUS_KM + along * (SUN_RADIUS_KM + EARTH_RADIUS_KM) / sun_distance
    return LunarView(
        jd=jd,
        miss_km=miss,
        umbra_km=umbra * ATMOSPHERE_ENLARGEMENT,
        penumbra_km=penumbra * ATMOSPHERE_ENLARGEMENT,
    )
