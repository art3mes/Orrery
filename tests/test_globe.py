"""The sphere the maps go on, and the mapping from a coordinate to a pixel.

No renderer here either. Everything the viewer draws a planet with is built by
``globe.py`` as plain arrays, and checked as plain arrays.
"""

import numpy as np
import pytest

from orrery import frames, globe, rotation, scene, times

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


def test_no_degenerate_triangles_at_the_poles():
    """Every vertex in the top row is the same point, so half those quads have
    zero area -- and a zero-area triangle has no normal and shades black. It
    showed as a dark blot on the limb of whichever planet had a pole turned
    toward the camera, which for Uranus is always.
    """
    for rows, columns in ((6, 12), (48, 96)):
        vertices, faces, _ = globe.uv_sphere(rows, columns)
        edge_a = vertices[faces[:, 1]] - vertices[faces[:, 0]]
        edge_b = vertices[faces[:, 2]] - vertices[faces[:, 0]]
        area = np.linalg.norm(np.cross(edge_a, edge_b), axis=-1)
        assert area.min() > 0.0
        # Exactly one triangle per quad is dropped, at each pole.
        assert len(faces) == 2 * rows * columns - 2 * columns


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


def test_orientation_is_a_rotation_a_squash_and_a_shift():
    """Not a pure rotation any more -- the polar axis is shortened.

    Venus is the body to check the pure-rotation case on, being the only large
    one with no measurable flattening.
    """
    centre = np.array([1.5, -2.0, 0.25])
    radius = 0.03
    matrix = globe.orientation("venus", JD, radius, centre)

    assert matrix.shape == (4, 4)
    assert np.allclose(matrix[3], [0.0, 0.0, 0.0, 1.0])
    assert np.allclose(matrix[:3, 3], centre)

    spin = matrix[:3, :3] / radius
    assert np.allclose(spin @ spin.T, np.eye(3), atol=1e-13)
    assert np.linalg.det(spin) == pytest.approx(1.0)


@pytest.mark.parametrize(
    "body,expected",
    [
        ("venus", 0.0),
        ("embary", 1 / 298.257223563),
        ("mars", 0.00589),
        ("jupiter", 0.06487),
        ("saturn", 0.09796),
        ("uranus", 0.02293),
        ("neptune", 0.01708),
    ],
)
def test_flattening_matches_published_values(body, expected):
    assert globe.flattening(body) == pytest.approx(expected, abs=2e-5)


def test_saturn_is_the_most_squashed():
    everything = {b: globe.flattening(b) for b in ("embary", "mars", "jupiter",
                                                  "saturn", "uranus", "neptune")}
    assert max(everything, key=everything.get) == "saturn"
    assert everything["saturn"] > 0.09  # a tenth, and obvious by eye


def test_the_squash_is_along_the_polar_axis_only():
    """The equatorial axes keep the full radius; the pole loses its flattening."""
    radius = 0.05
    matrix = globe.orientation("saturn", JD, radius, np.zeros(3))
    lengths = np.linalg.norm(matrix[:3, :3], axis=0)
    assert lengths[0] == pytest.approx(radius)
    assert lengths[1] == pytest.approx(radius)
    assert lengths[2] == pytest.approx(radius * (1 - globe.flattening("saturn")))


def test_the_squash_follows_the_tilt():
    """Squashing after the rotation instead of before would leave it upright."""
    matrix = globe.orientation("saturn", JD, 1.0, np.zeros(3))
    short_axis = matrix[:3, 2] / np.linalg.norm(matrix[:3, 2])
    from orrery import frames, rotation

    pole = frames.equatorial_to_ecliptic(rotation.pole("saturn", JD))
    assert abs(short_axis @ pole) == pytest.approx(1.0, abs=1e-12)


# --- rings ------------------------------------------------------------------


@pytest.mark.parametrize("body", ("saturn", "jupiter", "uranus"))
def test_the_ringed_planets_have_rings(body):
    assert globe.has_rings(body)


def test_unringed_planets_do_not():
    assert not globe.has_rings("embary")
    assert not globe.has_rings("neptune")  # it has some; they are not drawn


@pytest.mark.parametrize("body", ("saturn", "jupiter", "uranus"))
def test_the_ring_mesh_is_flat_and_clears_the_planet(body):
    vertices, faces, radial, opacity = globe.ring_mesh(body, segments=32)
    inner, outer = globe.ring_span(body)

    assert np.all(vertices[:, 2] == 0.0)  # flat, so the polar squash misses it
    radius = np.linalg.norm(vertices[:, :2], axis=1)
    assert radius.min() >= inner - 1e-9
    assert radius.max() <= outer + 0.02  # a widened narrow ring may spill a little
    assert inner > 1.0  # nothing intersects the globe
    assert radial.min() >= 0.0 and radial.max() <= 1.0
    assert faces.max() == len(vertices) - 1
    assert len(opacity) == len(vertices)


def test_each_ring_gets_its_own_opacity():
    """Read off the band, not sampled from a rasterised profile.

    Sampling a profile at a band edge can land just outside the band and come
    back zero, which drew a black hairline across the planet.
    """
    _, _, _, opacity = globe.ring_mesh("uranus", segments=8)
    system = globe.RING_SYSTEMS["uranus"]
    assert set(np.unique(opacity)) == {b.opacity for b in system.bands}


def test_geometry_exists_only_where_a_ring_does():
    """Uranus's rings cover 3 km of a 9300 km span.

    One continuous disc would be almost entirely empty, and at any sane mesh
    resolution the radial samples land between the rings and miss them.
    """
    vertices, _, _, _ = globe.ring_mesh("uranus", segments=8)
    system = globe.RING_SYSTEMS["uranus"]
    covered = sum(b.outer_km - b.inner_km for b in system.bands)
    span = system.outer_km - system.inner_km
    assert covered / span < 0.02  # the rings really are that sparse
    assert len(vertices) == 8 * 2 * len(system.bands)  # two rows per ring, no more


def test_narrow_rings_are_widened_to_stay_visible():
    """Stated, like the planet radii, rather than hidden.

    Three kilometres against a 51000 km radius is a sixtieth of a pixel.
    """
    system = globe.RING_SYSTEMS["uranus"]
    assert system.minimum_width_km > 0
    assert min(b.outer_km - b.inner_km for b in system.bands) < 10.0

    vertices, _, _, _ = globe.ring_mesh("uranus", segments=4)
    radius = np.linalg.norm(vertices[:, :2], axis=1) * scene.RADIUS_KM["uranus"]
    widths = np.diff(np.unique(np.round(radius, 3)))
    assert widths.max() > 100.0  # something got widened

    # Saturn's rings are broad already and are left alone.
    assert globe.RING_SYSTEMS["saturn"].minimum_width_km == 0.0


def test_ring_samples_keep_colour_and_opacity_apart():
    profile = np.array([[1.0, 0.5, 0.25, 0.0], [1.0, 0.5, 0.25, 0.8]])
    colour, opacity = globe.ring_samples(profile, np.array([0.0, 1.0]))
    assert np.allclose(colour[0], [1.0, 0.5, 0.25])  # not blackened
    assert opacity[0] == 0.0 and opacity[1] == pytest.approx(0.8)


def test_ring_radii_are_the_published_ones():
    """Spot checks against numbers anybody can look up."""
    saturn = {b.name: b for b in globe.RING_SYSTEMS["saturn"].bands}
    assert saturn["B"].outer_km == pytest.approx(117_580)
    assert saturn["Cassini division"].outer_km == pytest.approx(122_170)

    uranus = {b.name: b for b in globe.RING_SYSTEMS["uranus"].bands}
    assert uranus["epsilon"].inner_km == pytest.approx(51_149)
    assert uranus["6"].inner_km == pytest.approx(41_837)

    jupiter = {b.name: b for b in globe.RING_SYSTEMS["jupiter"].bands}
    assert jupiter["main"].outer_km == pytest.approx(129_000)


def test_the_orientation_is_rotated_into_the_scene_frame():
    """The IAU pole is equatorial; the scene is ecliptic.

    Skipping that rotation tips every planet by the obliquity, which reads as a
    plausible axial tilt rather than as a bug -- and would leave Jupiter, whose
    real tilt is 3 degrees, leaning further than the Earth.
    """
    matrix = globe.orientation("embary", JD, 1.0, np.zeros(3))
    # Normalised, because the polar column now carries the flattening as well
    # as the direction.
    axis = matrix[:3, 2] / np.linalg.norm(matrix[:3, 2])
    ecliptic_pole = np.array([0.0, 0.0, 1.0])
    tilt = np.degrees(np.arccos(abs(axis @ ecliptic_pole)))
    assert tilt == pytest.approx(23.44, abs=0.05)

    # And the same axis, read back in the equatorial frame, is the IAU pole.
    assert np.allclose(
        frames.ecliptic_to_equatorial(axis), rotation.pole("embary", JD), atol=1e-12
    )


def test_jupiter_leans_less_than_the_earth():
    def axis(body):
        column = globe.orientation(body, JD, 1.0, np.zeros(3))[:3, 2]
        return column / np.linalg.norm(column)

    earth, jupiter = axis("embary"), axis("jupiter")
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
