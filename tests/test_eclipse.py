"""Shadow geometry, on configurations built by hand.

``validate_m4.py`` checks real eclipses against published circumstances. These
check the geometry against arrangements whose answers can be written down, using
a synthetic ephemeris so nothing here needs DE440 or a network.
"""

import numpy as np
import pytest

from orrery import apparent, eclipse, times
from orrery.kepler import AU_KM

JD = times.jd(2026, 9, 3)


def fixed_ephemeris(**places):
    """An ephemeris where every body sits still at a given place, in km."""
    grid = JD + np.arange(-2.0, 3.0, 1.0)
    bodies = tuple(places)
    positions = np.stack(
        [np.repeat(np.asarray(p, dtype=float)[None, :] / AU_KM, len(grid), axis=0)
         for p in places.values()],
        axis=1,
    )
    return apparent.Ephemeris(bodies, grid, positions, np.zeros_like(positions))


# --- the cones --------------------------------------------------------------


def test_the_moons_umbra_is_374000_km_long():
    """Similar triangles, and the reason annular eclipses exist at all."""
    length = float(eclipse.umbra_length_km(1.496e8))
    assert length == pytest.approx(374_000, rel=0.01)
    # And the Moon is further away than that, on average.
    assert length < 384_400


def test_a_bigger_body_throws_a_longer_cone():
    small = eclipse.umbra_length_km(1.496e8, body_radius_km=1000.0)
    large = eclipse.umbra_length_km(1.496e8, body_radius_km=6378.0)
    assert large > small
    # The Earth's own umbra reaches far past the Moon, which is why lunar
    # eclipses are never annular.
    assert float(large) > 1_300_000


def test_the_sun_is_sixteen_arcminutes_across():
    assert float(eclipse.angular_radius_arcsec(eclipse.SUN_RADIUS_KM, 1.0)) == (
        pytest.approx(959.2, abs=1.0)
    )


def test_the_moon_and_sun_are_nearly_the_same_size():
    """The coincidence the whole phenomenon rests on."""
    sun = float(eclipse.angular_radius_arcsec(eclipse.SUN_RADIUS_KM, 1.0))
    moon = float(eclipse.angular_radius_arcsec(eclipse.MOON_RADIUS_KM, 384_400 / AU_KM))
    assert 0.9 < moon / sun < 1.1


# --- overlapping discs ------------------------------------------------------


def test_discs_that_do_not_touch_hide_nothing():
    assert float(eclipse.overlap_fraction(3.0, 1.0, 1.0)) == 0.0


def test_a_disc_swallowed_whole_is_fully_hidden():
    assert float(eclipse.overlap_fraction(0.0, 1.0, 2.0)) == pytest.approx(1.0)


def test_an_annular_eclipse_leaves_a_ring():
    """The Moon inside the Sun's disc: covered goes as the ratio of areas."""
    covered = float(eclipse.overlap_fraction(0.0, 1.0, 0.95))
    assert covered == pytest.approx(0.9025, rel=1e-9)


def test_half_covered_when_the_centres_touch_the_rims():
    """Equal discs a radius apart cover a known lens fraction."""
    covered = float(eclipse.overlap_fraction(1.0, 1.0, 1.0))
    expected = (2 * np.pi / 3 - np.sqrt(3) / 2) / np.pi
    assert covered == pytest.approx(expected, rel=1e-9)


def test_coverage_falls_as_the_discs_separate():
    fractions = [
        float(eclipse.overlap_fraction(s, 1.0, 1.0)) for s in (0.0, 0.5, 1.0, 1.5, 2.0)
    ]
    assert fractions[0] == pytest.approx(1.0)
    assert all(a > b for a, b in zip(fractions, fractions[1:]))
    assert fractions[-1] == 0.0


# --- the solar shadow -------------------------------------------------------


def test_a_moon_on_the_sun_earth_line_puts_the_axis_through_the_centre():
    ephemeris = fixed_ephemeris(
        sun=[1.496e8, 0.0, 0.0],
        geocentre=[0.0, 0.0, 0.0],
        moon=[384_400.0, 0.0, 0.0],
    )
    miss, direction, moon = eclipse.shadow_axis(ephemeris, JD)
    assert float(miss[0]) < 1.0  # km
    # The shadow travels away from the Sun.
    assert direction[0] @ np.array([1.0, 0.0, 0.0]) == pytest.approx(-1.0, abs=1e-9)


def test_moving_the_moon_sideways_moves_the_axis_by_about_as_much():
    offset = 3000.0
    ephemeris = fixed_ephemeris(
        sun=[1.496e8, 0.0, 0.0],
        geocentre=[0.0, 0.0, 0.0],
        moon=[384_400.0, offset, 0.0],
    )
    miss, _, _ = eclipse.shadow_axis(ephemeris, JD)
    # Slightly more than the offset, because the axis keeps diverging past the
    # Moon on its way to the Earth.
    assert offset < float(miss[0]) < offset * 1.02


def test_an_axis_that_misses_the_earth_lands_nowhere():
    ephemeris = fixed_ephemeris(
        sun=[1.496e8, 0.0, 0.0],
        geocentre=[0.0, 0.0, 0.0],
        moon=[384_400.0, 50_000.0, 0.0],
    )
    latitude, longitude = eclipse.shadow_landing(ephemeris, JD)
    assert np.isnan(latitude[0]) and np.isnan(longitude[0])


def test_a_central_shadow_lands_on_the_surface():
    ephemeris = fixed_ephemeris(
        sun=[1.496e8, 0.0, 0.0],
        geocentre=[0.0, 0.0, 0.0],
        moon=[384_400.0, 0.0, 0.0],
    )
    latitude, longitude = eclipse.shadow_landing(ephemeris, JD)
    assert not np.isnan(latitude[0])
    assert -90.0 <= float(latitude[0]) <= 90.0
    assert 0.0 <= float(longitude[0]) < 360.0


def test_the_landing_uses_the_ellipsoid_not_a_sphere():
    """Over the pole the surface is 21 km closer in, which moves the answer.

    A sphere would put the landing point at a measurably different latitude, so
    this checks the squash is actually applied rather than trusting it.
    """
    from orrery.observer import EARTH_FLATTENING

    assert EARTH_FLATTENING > 0
    ephemeris = fixed_ephemeris(
        sun=[1.496e8, 0.0, 0.0],
        geocentre=[0.0, 0.0, 0.0],
        moon=[384_400.0, 0.0, 4_000.0],
    )
    latitude, _ = eclipse.shadow_landing(ephemeris, JD)
    assert not np.isnan(latitude[0])
    assert abs(float(latitude[0])) > 20.0  # well off the equator, where it matters


# --- the lunar shadow -------------------------------------------------------


def test_a_moon_behind_the_earth_is_totally_eclipsed():
    ephemeris = fixed_ephemeris(
        sun=[1.496e8, 0.0, 0.0],
        geocentre=[0.0, 0.0, 0.0],
        moon=[-384_400.0, 0.0, 0.0],
    )
    view = eclipse.lunar_view(ephemeris, JD)
    assert float(view.miss_km[0]) < 1.0
    assert view.kind() == "total"
    # The umbra is still much wider than the Moon out there.
    assert float(view.umbra_km[0]) > eclipse.MOON_RADIUS_KM


def test_a_moon_well_off_the_axis_is_not_eclipsed_at_all():
    ephemeris = fixed_ephemeris(
        sun=[1.496e8, 0.0, 0.0],
        geocentre=[0.0, 0.0, 0.0],
        moon=[-384_400.0, 40_000.0, 0.0],
    )
    assert eclipse.lunar_view(ephemeris, JD).kind() == "none"


def test_the_penumbra_is_wider_than_the_umbra():
    ephemeris = fixed_ephemeris(
        sun=[1.496e8, 0.0, 0.0],
        geocentre=[0.0, 0.0, 0.0],
        moon=[-384_400.0, 0.0, 0.0],
    )
    view = eclipse.lunar_view(ephemeris, JD)
    assert float(view.penumbra_km[0]) > float(view.umbra_km[0])


def test_the_atmosphere_makes_the_shadow_bigger():
    """Danjon's 2%, which is why the contact times agree with almanacs."""
    assert eclipse.ATMOSPHERE_ENLARGEMENT > 1.0


# --- the tolerance that could not be met ------------------------------------


def test_the_light_time_tolerance_is_above_the_resolution_of_a_julian_date():
    """It was not, and the Moon never converged.

    A float64 near 2.46e6 steps by 4.7e-10, so a threshold of 1e-12 days can
    only be met by two iterates landing on the same bits. Distant planets do;
    the Moon seen from a spinning Earth flips between neighbours forever.
    """
    import inspect

    default = inspect.signature(apparent.light_time).parameters["tolerance_days"].default
    assert default > np.spacing(times.jd(2026, 1, 1))
