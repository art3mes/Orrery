"""The sphere the maps go on, and the mapping from a coordinate to a pixel.

No renderer here either. Everything the viewer draws a planet with is built by
``globe.py`` as plain arrays, and checked as plain arrays.
"""

import numpy as np
import pytest

from orrery import frames, globe, rotation, times

JD = times.jd(2026, 9, 3)


# --- the mesh ---------------------------------------------------------------


def test_every_vertex_is_on_the_unit_sphere():
    vertices, _, _ = globe.uv_sphere(12, 24)
    assert np.allclose(np.linalg.norm(vertices, axis=-1), 1.0)


def test_texture_coordinates_cover_the_whole_map():
    _, _, uv = globe.uv_sphere(12, 24)
    assert uv[:, 0].min() == 0.0 and uv[:, 0].max() == 1.0
    assert uv[:, 1].min() == 0.0 and uv[:, 1].max() == 1.0


def test_the_seam_is_duplicated_not_wrapped():
    """Two vertices in the same place carrying different texture coordinates.

    Without that, the last column of the map stretches all the way back round
    the planet instead of meeting the first.
    """
    rows, columns = 8, 16
    vertices, _, uv = globe.uv_sphere(rows, columns)
    grid = vertices.reshape(rows + 1, columns + 1, 3)
    texture = uv.reshape(rows + 1, columns + 1, 2)

    assert np.allclose(grid[:, 0], grid[:, -1])  # same point in space
    assert np.allclose(texture[:, 0, 0], 0.0)  # different u
    assert np.allclose(texture[:, -1, 0], 1.0)


def test_faces_index_real_vertices():
    vertices, faces, _ = globe.uv_sphere(10, 20)
    assert faces.min() == 0
    assert faces.max() == len(vertices) - 1
    assert faces.shape[1] == 3


def test_the_prime_meridian_is_down_the_middle_of_the_map():
    """Equirectangular maps start at the antimeridian, not at Greenwich.

    Getting this backwards rotates every planet by half a turn, which is
    invisible on Jupiter and put the Sun over the Pacific at noon UT.
    """
    rows, columns = 4, 8
    vertices, _, uv = globe.uv_sphere(rows, columns)
    grid = vertices.reshape(rows + 1, columns + 1, 3)
    texture = uv.reshape(rows + 1, columns + 1, 2)

    equator = rows // 2
    middle = columns // 2
    assert texture[equator, middle, 0] == pytest.approx(0.5)
    # u = 0.5 sits on the +x axis, which is the prime meridian by definition.
    assert grid[equator, middle] == pytest.approx([1.0, 0.0, 0.0], abs=1e-12)
    # u = 0 sits on -x, the antimeridian.
    assert grid[equator, 0] == pytest.approx([-1.0, 0.0, 0.0], abs=1e-12)


def test_a_sphere_needs_enough_of_it():
    with pytest.raises(ValueError, match="at least 2 rows"):
        globe.uv_sphere(1, 8)
    with pytest.raises(ValueError, match="at least 2 rows"):
        globe.uv_sphere(8, 2)


# --- placing it -------------------------------------------------------------


def test_orientation_is_a_scaled_rotation_plus_a_shift():
    centre = np.array([1.5, -2.0, 0.25])
    radius = 0.03
    matrix = globe.orientation("mars", JD, radius, centre)

    assert matrix.shape == (4, 4)
    assert np.allclose(matrix[3], [0.0, 0.0, 0.0, 1.0])
    assert np.allclose(matrix[:3, 3], centre)

    spin = matrix[:3, :3] / radius
    assert np.allclose(spin @ spin.T, np.eye(3), atol=1e-13)
    assert np.linalg.det(spin) == pytest.approx(1.0)


def test_the_orientation_is_rotated_into_the_scene_frame():
    """The IAU pole is equatorial; the scene is ecliptic.

    Skipping that rotation tips every planet by the obliquity, which reads as a
    plausible axial tilt rather than as a bug -- and would leave Jupiter, whose
    real tilt is 3 degrees, leaning further than the Earth.
    """
    matrix = globe.orientation("embary", JD, 1.0, np.zeros(3))
    axis = matrix[:3, 2]
    ecliptic_pole = np.array([0.0, 0.0, 1.0])
    tilt = np.degrees(np.arccos(abs(axis @ ecliptic_pole)))
    assert tilt == pytest.approx(23.44, abs=0.05)

    # And the same axis, read back in the equatorial frame, is the IAU pole.
    assert np.allclose(
        frames.ecliptic_to_equatorial(axis), rotation.pole("embary", JD), atol=1e-12
    )


def test_jupiter_leans_less_than_the_earth():
    earth = globe.orientation("embary", JD, 1.0, np.zeros(3))[:3, 2]
    jupiter = globe.orientation("jupiter", JD, 1.0, np.zeros(3))[:3, 2]
    up = np.array([0.0, 0.0, 1.0])
    assert np.degrees(np.arccos(abs(jupiter @ up))) < np.degrees(
        np.arccos(abs(earth @ up))
    )


# --- sampling ---------------------------------------------------------------


def test_sampling_matches_the_mesh_convention():
    """Longitude 0 lands halfway across the image, latitude 90 at the top."""
    pixels = np.zeros((180, 360, 3), dtype=float)
    pixels[:, 180] = [1.0, 0.0, 0.0]  # the column at longitude 0
    pixels[0, :] = [0.0, 1.0, 0.0]  # the row at the north pole

    assert globe.sample(pixels, 0.0, 0.0) == pytest.approx([1.0, 0.0, 0.0])
    assert globe.sample(pixels, 90.0, 123.0) == pytest.approx([0.0, 1.0, 0.0])


def test_sampling_wraps_in_longitude():
    pixels = np.random.default_rng(0).random((32, 64, 3))
    assert np.allclose(globe.sample(pixels, 10.0, 20.0), globe.sample(pixels, 10.0, 380.0))


def test_water_needs_blue_by_a_margin():
    assert bool(globe.looks_like_water(np.array([0.12, 0.23, 0.46])))  # ocean
    assert not bool(globe.looks_like_water(np.array([0.87, 0.87, 0.88])))  # ice
    assert not bool(globe.looks_like_water(np.array([0.21, 0.26, 0.09])))  # forest


# --- the real maps, when they have been fetched -----------------------------

_earth = globe.load_texture("embary", download=False)
needs_map = pytest.mark.skipif(_earth is None, reason="Earth map not cached")


@needs_map
def test_the_earth_map_is_equirectangular():
    assert _earth.shape[1] == 2 * _earth.shape[0]
    assert _earth.min() >= 0.0 and _earth.max() <= 1.0


@needs_map
@pytest.mark.parametrize(
    "name,latitude,longitude,water",
    [
        ("Gulf of Guinea", 0.0, 0.0, True),
        ("Congo", -2.0, 23.0, False),
        ("mid Pacific", 0.0, -150.0, True),
        ("Australia", -25.0, 133.0, False),
        ("Antarctica", -80.0, 0.0, False),
    ],
)
def test_the_map_is_the_right_way_round(name, latitude, longitude, water):
    colour = globe.sample(_earth, latitude, longitude)
    assert bool(globe.looks_like_water(colour)) is water, name
