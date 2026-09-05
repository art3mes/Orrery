"""Spheres to hang maps on, and the maps.

A planet drawn as a flat-coloured dot carries one bit of information: which
planet. A textured, correctly oriented one carries where its features were
pointing, which is a real answer to a real question and is checkable -- see the
analemma gate in ``validate_m5.py``.

The mesh is built **once**, as a unit sphere in body-fixed coordinates. Every
frame then only sets a 4x4 transform: the rotation from
:mod:`orrery.rotation`, scaled by the display radius and translated to the
planet's position. Rewriting 40000 vertex positions a frame in Python would not
keep up; rewriting nine matrices does.

Maps come from Solar System Scope, CC BY 4.0. They are cached under
``data/textures/`` after the first fetch, and their absence is not an error --
a body with no map falls back to the flat colour it had before.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np

from .frames import equatorial_to_ecliptic
from .rotation import body_to_icrf
from .truth import DATA_DIR, _use_system_trust_store

TEXTURE_DIR = DATA_DIR / "textures"

TEXTURE_SOURCE = "https://www.solarsystemscope.com/textures/download/"
TEXTURE_LICENCE = "Solar System Scope, CC BY 4.0"

# Pluto has no map at this source, and falls back to flat colour.
# Polar radii, km. Every giant planet is visibly squashed by its own spin --
# Saturn by a tenth, which is obvious by eye and was drawn as a perfect sphere
# until it was not. Flattening is derived from these rather than tabulated
# separately, so the two can never drift apart.
POLAR_RADIUS_KM = {
    "sun": 695700.0,
    "mercury": 2439.7,
    "venus": 6051.8,
    "embary": 6356.752,
    "moon": 1736.0,
    "mars": 3376.2,
    "jupiter": 66854.0,
    "saturn": 54364.0,
    "uranus": 24973.0,
    "neptune": 24341.0,
    "pluto": 1188.3,
}

# Ring systems, as the named rings they are actually made of, in kilometres
# from the planet's centre.
#
# Only Saturn has a published map at the texture source, so the others are built
# from their real radii instead. That turns out to be the better representation
# anyway: Uranus's rings are hairlines a few kilometres wide, and no
# 2048-pixel strip was ever going to show that.


@dataclass(frozen=True)
class RingBand:
    """One named ring. Opacity is a rough optical depth, not a measurement."""

    name: str
    inner_km: float
    outer_km: float
    opacity: float


@dataclass(frozen=True)
class RingSystem:
    bands: tuple[RingBand, ...]
    colour: tuple[float, float, float]
    texture: str | None = None
    radial_bands: int = 24  # mesh resolution across the rings
    # Narrowest a ring may be *drawn*. Uranus's are three kilometres wide
    # against a 51 000 km radius, which is a sixtieth of a pixel: correct, and
    # invisible. Widening them is the same bargain M5 already makes for planet
    # radii, and like that one it is stated rather than hidden.
    minimum_width_km: float = 0.0

    @property
    def inner_km(self) -> float:
        return min(b.inner_km for b in self.bands)

    @property
    def outer_km(self) -> float:
        return max(b.outer_km for b in self.bands)


RING_SYSTEMS = {
    # Bright and broad, and the only one anybody draws.
    "saturn": RingSystem(
        bands=(
            RingBand("C", 74_658, 92_000, 0.3),
            RingBand("B", 92_000, 117_580, 0.9),
            RingBand("Cassini division", 117_580, 122_170, 0.15),
            RingBand("A", 122_170, 136_775, 0.7),
        ),
        colour=(0.78, 0.72, 0.62),
        texture="2k_saturn_ring_alpha.png",
        radial_bands=32,
    ),
    # Dust, not ice. The main ring is the only part that carries any brightness;
    # the gossamer rings are so faint they were found by a spacecraft looking
    # back at the Sun through them.
    "jupiter": RingSystem(
        bands=(
            RingBand("halo", 92_000, 122_500, 0.02),
            RingBand("main", 122_500, 129_000, 0.22),
            RingBand("Amalthea gossamer", 129_000, 182_000, 0.03),
            RingBand("Thebe gossamer", 182_000, 226_000, 0.015),
        ),
        colour=(0.62, 0.42, 0.30),
        radial_bands=64,
    ),
    # Ten narrow, very dark rings. Widths are real: most are a few kilometres
    # across, and epsilon, the widest and brightest, reaches about 96.
    "uranus": RingSystem(
        bands=(
            RingBand("6", 41_837, 41_840, 0.30),
            RingBand("5", 42_234, 42_237, 0.35),
            RingBand("4", 42_571, 42_574, 0.30),
            RingBand("alpha", 44_718, 44_729, 0.45),
            RingBand("beta", 45_661, 45_672, 0.45),
            RingBand("eta", 47_176, 47_178, 0.25),
            RingBand("gamma", 47_627, 47_631, 0.40),
            RingBand("delta", 48_300, 48_307, 0.35),
            RingBand("lambda", 50_024, 50_026, 0.10),
            RingBand("epsilon", 51_149, 51_245, 0.70),
        ),
        colour=(0.34, 0.34, 0.38),
        radial_bands=8,
        minimum_width_km=250.0,
    ),
}

TEXTURE_FILES = {
    "sun": "2k_sun.jpg",
    "mercury": "2k_mercury.jpg",
    "venus": "2k_venus_atmosphere.jpg",
    "embary": "2k_earth_daymap.jpg",
    "moon": "2k_moon.jpg",
    "mars": "2k_mars.jpg",
    "jupiter": "2k_jupiter.jpg",
    "saturn": "2k_saturn.jpg",
    "uranus": "2k_uranus.jpg",
    "neptune": "2k_neptune.jpg",
}


def uv_sphere(rows: int = 64, columns: int = 128):
    """A unit sphere with texture coordinates. Returns ``(vertices, faces, uv)``.

    The seam at longitude zero is duplicated rather than wrapped: the vertex at
    u = 0 and the one at u = 1 sit at the same place in space but must carry
    different texture coordinates, or the last column of the map gets stretched
    all the way back round the planet.

    ``v`` runs 0 at the north pole to 1 at the south, matching the upper-left
    origin of an image file, so an equirectangular map lands the right way up
    without a flip anywhere.

    Longitude starts at **-180**, not at zero. An equirectangular map puts the
    prime meridian down the *middle* of the image, so u = 0.5 is longitude 0 and
    u = 0 is the antimeridian. Starting the sphere at longitude zero instead
    rotates every map by half a turn -- which looks entirely convincing, because
    a planet with the wrong hemisphere facing you is still a planet. The Earth
    is the one body where it is obvious: at noon UT the Sun should be over
    Africa, and it was over the Pacific.
    """
    if rows < 2 or columns < 3:
        raise ValueError("a sphere needs at least 2 rows and 3 columns")

    i = np.arange(rows + 1)
    j = np.arange(columns + 1)  # the +1 is the duplicated seam
    theta = np.pi * i / rows  # 0 at the north pole
    phi = 2 * np.pi * j / columns - np.pi

    sin_theta, cos_theta = np.sin(theta), np.cos(theta)
    x = np.outer(sin_theta, np.cos(phi))
    y = np.outer(sin_theta, np.sin(phi))
    z = np.repeat(cos_theta[:, None], columns + 1, axis=1)
    vertices = np.stack([x, y, z], axis=-1).reshape(-1, 3)

    u = np.repeat((j / columns)[None, :], rows + 1, axis=0)
    v = np.repeat((i / rows)[:, None], columns + 1, axis=1)
    uv = np.stack([u, v], axis=-1).reshape(-1, 2)

    index = np.arange((rows + 1) * (columns + 1)).reshape(rows + 1, columns + 1)
    top_left = index[:-1, :-1].ravel()
    top_right = index[:-1, 1:].ravel()
    bottom_left = index[1:, :-1].ravel()
    bottom_right = index[1:, 1:].ravel()
    faces = np.concatenate(
        [
            np.stack([top_left, bottom_left, bottom_right], axis=-1),
            np.stack([top_left, bottom_right, top_right], axis=-1),
        ]
    )

    # Drop the degenerate triangles at the poles. Every vertex in the top row
    # sits at the same point, so half of each quad there has zero area -- and a
    # zero-area triangle has no normal, which shades black. It showed up as a
    # dark blot on the limb of whichever planet had its pole turned toward the
    # camera, and Uranus, tipped over on its side, has one permanently.
    edge_a = vertices[faces[:, 1]] - vertices[faces[:, 0]]
    edge_b = vertices[faces[:, 2]] - vertices[faces[:, 0]]
    area = np.linalg.norm(np.cross(edge_a, edge_b), axis=-1)
    return vertices, faces[area > 1e-12], uv


def flattening(body: str) -> float:
    """How squashed a body is: ``1 - polar / equatorial``.

    Zero for the rocky bodies, 0.065 for Jupiter and 0.098 for Saturn.
    """
    from .scene import RADIUS_KM

    key = "sun" if body == "sun" else body
    polar = POLAR_RADIUS_KM.get(key)
    equatorial = RADIUS_KM.get(key)
    if polar is None or equatorial is None:
        return 0.0
    return 1.0 - polar / equatorial


def has_rings(body: str) -> bool:
    return body in RING_SYSTEMS


def _drawn_span_km(system: "RingSystem") -> tuple[float, float]:
    """Inner and outer edge of the geometry actually emitted, in km."""
    low, high = system.inner_km, system.outer_km
    for band in system.bands:
        centre = 0.5 * (band.inner_km + band.outer_km)
        half = 0.5 * max(band.outer_km - band.inner_km, system.minimum_width_km)
        low = min(low, centre - half)
        high = max(high, centre + half)
    return low, high


def ring_span(body: str) -> tuple[float, float]:
    """Inner and outer edge as *drawn*, in equatorial radii of the planet.

    Widening a hairline ring pushes its edges out either side of the published
    radius, so this reports the geometry rather than the catalogue -- it is what
    frames the camera and sizes the bounding box, and both want the truth about
    what is on screen.
    """
    from .scene import RADIUS_KM

    system = RING_SYSTEMS[body]
    low, high = _drawn_span_km(system)
    return low / RADIUS_KM[body], high / RADIUS_KM[body]


def ring_mesh(body: str, segments: int = 192, bands: int | None = None):
    """One flat annulus per named ring, concatenated into a single mesh.

    Returns ``(vertices, faces, radial_fraction, opacity)``, in units of the
    body's equatorial radius, with the fraction measured across the whole system
    so a texture can still be looked up per vertex.

    Geometry is emitted **only where there is a ring**. Building one continuous
    disc and leaning on opacity to hide the gaps works for Saturn and fails for
    Uranus, whose rings are three kilometres wide in a nine-thousand kilometre
    span: at any sane mesh resolution the radial samples land between the rings
    and miss them entirely. Putting the vertices on the ring edges instead makes
    a three-kilometre ring exactly three kilometres wide, and costs less.

    The rings sit at z = 0 in body-fixed coordinates, so the same transform that
    orients and squashes the planet carries them -- and the squash, being along
    z, leaves a z = 0 plane alone.
    """
    from .scene import RADIUS_KM

    system = RING_SYSTEMS[body]
    planet = RADIUS_KM[body]
    # Measured across the geometry actually drawn, so the fraction stays in
    # [0, 1] once narrow rings have been widened past the catalogue edges.
    origin, edge = _drawn_span_km(system)
    span = edge - origin
    budget = bands if bands is not None else system.radial_bands

    angle = np.linspace(0.0, 2 * np.pi, segments + 1)[:-1]
    cos, sin = np.cos(angle), np.sin(angle)

    all_vertices, all_faces, all_radial, all_opacity = [], [], [], []
    offset = 0
    for band in system.bands:
        centre = 0.5 * (band.inner_km + band.outer_km)
        width = max(band.outer_km - band.inner_km, system.minimum_width_km)
        low, high = centre - 0.5 * width, centre + 0.5 * width
        rows = max(2, int(round(budget * width / span)) + 1)
        edges = np.linspace(low, high, rows)

        radius = edges / planet
        x = np.outer(radius, cos)
        y = np.outer(radius, sin)
        all_vertices.append(
            np.stack([x, y, np.zeros_like(x)], axis=-1).reshape(-1, 3)
        )
        all_radial.append(
            np.repeat(((edges - origin) / span)[:, None], segments, axis=1).ravel()
        )
        # Opacity comes from the band itself, not from a rasterised profile.
        # Sampling a profile at a band edge can land just outside it and return
        # zero, which draws a black hairline across the planet.
        all_opacity.append(np.full(rows * segments, band.opacity))

        index = offset + np.arange(rows * segments).reshape(rows, segments)
        right = np.roll(index, -1, axis=1)  # wraps, so each annulus closes
        inner_left, inner_right = index[:-1].ravel(), right[:-1].ravel()
        outer_left, outer_right = index[1:].ravel(), right[1:].ravel()
        all_faces.append(
            np.concatenate(
                [
                    np.stack([inner_left, outer_left, outer_right], axis=-1),
                    np.stack([inner_left, outer_right, inner_right], axis=-1),
                ]
            )
        )
        offset += rows * segments

    return (
        np.concatenate(all_vertices),
        np.concatenate(all_faces),
        np.concatenate(all_radial),
        np.concatenate(all_opacity),
    )


def _profile_from_bands(body: str, samples: int) -> np.ndarray:
    """Rasterise the named rings into an RGBA radial strip.

    Every edge is a real published radius, so a ring that is three kilometres
    wide comes out three kilometres wide. At Uranus that is one sample in two
    thousand, which is the honest answer and looks like what Voyager saw.
    """
    system = RING_SYSTEMS[body]
    inner, outer = system.inner_km, system.outer_km
    edges = np.linspace(inner, outer, samples)

    opacity = np.zeros(samples)
    for band in system.bands:
        inside = (edges >= band.inner_km) & (edges <= band.outer_km)
        opacity = np.where(inside, np.maximum(opacity, band.opacity), opacity)

    strip = np.zeros((samples, 4), dtype=np.float32)
    strip[:, :3] = system.colour
    strip[:, 3] = opacity
    return strip


def ring_profile(
    body: str, *, samples: int = 2048, download: bool = True
) -> np.ndarray | None:
    """The ring system as an ``(N, 4)`` RGBA strip, inner edge first.

    Saturn has a published map -- 2048 by 125, varying only along its length, so
    a radial profile wearing the shape of an image -- and it is collapsed to
    one. Everything else is rasterised from its own ring radii.
    """
    if body not in RING_SYSTEMS:
        return None
    name = RING_SYSTEMS[body].texture
    if name is None:
        return _profile_from_bands(body, samples)

    path = TEXTURE_DIR / name
    if not path.exists():
        if not download or fetch_texture(body, filename=name) is None:
            return _profile_from_bands(body, samples)
    try:
        from PIL import Image
    except ImportError:  # pragma: no cover - depends on environment
        return _profile_from_bands(body, samples)
    with Image.open(path) as image:
        pixels = np.asarray(image.convert("RGBA"), dtype=np.float32) / 255.0
    return pixels.mean(axis=0)


def ring_samples(
    profile: np.ndarray, radial_fraction: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    """Per-vertex ``(colour, opacity)`` sampled from a ring profile.

    Opacity is returned separately rather than folded into the colour. Folding
    it in works for Saturn, where the rings are opaque nearly everywhere, and
    fails completely for Uranus: its rings are hairlines a few kilometres wide
    in a nine-thousand kilometre span, so almost every vertex has zero opacity,
    and zero premultiplied colour is *black*, which against a dark sky reads as
    a solid disc rather than as empty space.
    """
    index = np.clip(
        (radial_fraction * (len(profile) - 1)).astype(int), 0, len(profile) - 1
    )
    sample = profile[index]
    return sample[:, :3], sample[:, 3]


def texture_path(body: str) -> Path | None:
    name = TEXTURE_FILES.get(body)
    return None if name is None else TEXTURE_DIR / name


def fetch_texture(
    body: str, *, filename: str | None = None, timeout: float = 60.0
) -> Path | None:
    """Download the map for *body* if it is not already cached."""
    path = TEXTURE_DIR / filename if filename else texture_path(body)
    if path is None:
        return None
    if path.exists():
        return path

    import urllib.error
    import urllib.request

    TEXTURE_DIR.mkdir(parents=True, exist_ok=True)
    _use_system_trust_store()
    request = urllib.request.Request(
        TEXTURE_SOURCE + (filename or TEXTURE_FILES[body]),
        headers={"User-Agent": "orrery/0.1"},
    )
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:
            data = response.read()
    except (urllib.error.URLError, TimeoutError, OSError):
        return None

    path.write_bytes(data)
    return path


def load_texture(body: str, *, download: bool = True) -> np.ndarray | None:
    """The map for *body* as an ``(H, W, 3)`` float array in [0, 1], or None."""
    path = texture_path(body)
    if path is None:
        return None
    if not path.exists():
        if not download or fetch_texture(body) is None:
            return None

    try:
        from PIL import Image
    except ImportError:  # pragma: no cover - depends on environment
        return None

    with Image.open(path) as image:
        pixels = np.asarray(image.convert("RGB"), dtype=np.float32) / 255.0
    return pixels


def sample(pixels: np.ndarray, latitude, longitude) -> np.ndarray:
    """Colour of an equirectangular map at a planetocentric position.

    Longitude in degrees east, latitude in degrees north. The inverse of the
    mapping :func:`uv_sphere` builds, so anything that agrees with this agrees
    with what gets drawn.
    """
    pixels = np.asarray(pixels)
    height, width = pixels.shape[:2]
    u = ((np.asarray(longitude, dtype=float) + 180.0) % 360.0) / 360.0
    v = (90.0 - np.asarray(latitude, dtype=float)) / 180.0
    column = np.clip((u * width).astype(int), 0, width - 1)
    row = np.clip((v * height).astype(int), 0, height - 1)
    return pixels[row, column]


def looks_like_water(colour: np.ndarray, margin: float = 0.08) -> np.ndarray:
    """Blue dominant *by a margin*. Enough to tell an ocean from a continent.

    The margin is not fussiness. Snow and ice come out near-white, with blue
    ahead of red by about one part in a hundred, so a bare "is blue largest"
    test calls Antarctica an ocean. Deep water leads by 0.3.
    """
    colour = np.asarray(colour, dtype=float)
    return (colour[..., 2] > colour[..., 0] + margin) & (
        colour[..., 2] > colour[..., 1] + margin
    )


def orientation(body: str, jd_tdb: float, radius: float, centre: np.ndarray) -> np.ndarray:
    """The 4x4 placing a unit body-fixed sphere in the scene.

    Rotation from the IAU elements, scaled by the drawn radius, translated to
    where the body is. The scale is uniform, so the rotation stays a rotation.

    The IAU elements are referred to ICRF *equatorial* axes and the scene is
    *ecliptic*, so the orientation is rotated by the obliquity on the way in.
    Skipping that tilts every planet by 23.4 degrees -- which looks like a
    plausible axial tilt rather than like a bug, and would have made Jupiter,
    whose real tilt is 3 degrees, lean further than the Earth.
    """
    body_to_equatorial = np.asarray(body_to_icrf(body, jd_tdb), dtype=float)
    # Rotate each column, i.e. each body axis, into the ecliptic frame.
    body_to_ecliptic = equatorial_to_ecliptic(body_to_equatorial.T).T

    # Squash along the body's own polar axis *before* rotating it into the
    # scene, so the flattening follows the tilt. The result is no longer a pure
    # rotation times a scalar, which is the point.
    squash = np.diag([radius, radius, radius * (1.0 - flattening(body))])

    matrix = np.eye(4)
    matrix[:3, :3] = body_to_ecliptic @ squash
    matrix[:3, 3] = np.asarray(centre, dtype=float)
    return matrix
