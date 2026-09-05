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

from pathlib import Path

import numpy as np

from .frames import equatorial_to_ecliptic
from .rotation import body_to_icrf
from .truth import DATA_DIR, _use_system_trust_store

TEXTURE_DIR = DATA_DIR / "textures"

TEXTURE_SOURCE = "https://www.solarsystemscope.com/textures/download/"
TEXTURE_LICENCE = "Solar System Scope, CC BY 4.0"

# Pluto has no map at this source, and falls back to flat colour.
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
    return vertices, faces, uv


def texture_path(body: str) -> Path | None:
    name = TEXTURE_FILES.get(body)
    return None if name is None else TEXTURE_DIR / name


def fetch_texture(body: str, *, timeout: float = 60.0) -> Path | None:
    """Download the map for *body* if it is not already cached."""
    path = texture_path(body)
    if path is None:
        return None
    if path.exists():
        return path

    import urllib.error
    import urllib.request

    TEXTURE_DIR.mkdir(parents=True, exist_ok=True)
    _use_system_trust_store()
    request = urllib.request.Request(
        TEXTURE_SOURCE + TEXTURE_FILES[body], headers={"User-Agent": "orrery/0.1"}
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

    matrix = np.eye(4)
    matrix[:3, :3] = body_to_ecliptic * radius
    matrix[:3, 3] = np.asarray(centre, dtype=float)
    return matrix
