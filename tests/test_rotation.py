"""Which way each body points, and how fast it turns.

The rotation table is transcribed, like the JPL element table, so these check
it against things it implies rather than against itself.
"""

import numpy as np
import pytest

from orrery import frames, kepler, rotation, times

JD = times.jd(2026, 9, 3)


@pytest.mark.parametrize("body", sorted(rotation.SPIN))
def test_the_pole_is_a_unit_vector(body):
    assert np.linalg.norm(rotation.pole(body, JD)) == pytest.approx(1.0)


@pytest.mark.parametrize("body", sorted(rotation.SPIN))
def test_the_body_frame_is_a_rotation(body):
    matrix = rotation.body_to_icrf(body, JD)
    assert np.allclose(matrix @ matrix.T, np.eye(3), atol=1e-14)
    assert np.linalg.det(matrix) == pytest.approx(1.0)


@pytest.mark.parametrize("body", sorted(rotation.SPIN))
def test_the_third_axis_is_the_pole(body):
    assert np.allclose(rotation.body_to_icrf(body, JD)[:, 2], rotation.pole(body, JD))


def test_one_period_turns_the_prime_meridian_once_round():
    for body in ("embary", "mars", "jupiter", "venus"):
        period = rotation.rotation_period_days(body)
        before = rotation.prime_meridian_degrees(body, JD)
        after = rotation.prime_meridian_degrees(body, JD + period)
        turned = (after - before + 180.0) % 360.0 - 180.0  # signed, so 0 is 0
        assert turned == pytest.approx(0.0, abs=1e-6)


def test_the_earth_turns_once_a_sidereal_day():
    """23h 56m 4s, not 24 hours: the extra four minutes are the orbit."""
    assert rotation.rotation_period_days("embary") * 24 * 3600 == pytest.approx(
        86164.1, abs=0.5
    )


@pytest.mark.parametrize("body", ("venus", "uranus"))
def test_backwards_bodies_have_a_supplementary_naive_obliquity(body):
    """The trap this module exists to document.

    The IAU calls the pole on the north side of the invariable plane "north"
    whatever the sense of rotation, so measuring the tilt to the *tabulated*
    pole gives 180 minus the real obliquity. Both numbers look ordinary: Venus
    reads 2.6 degrees, which would make it the most upright planet there is.
    """
    orbit_normal = frames.ecliptic_to_equatorial(_orbit_normal(body))
    to_pole = np.degrees(
        np.arccos(np.clip(rotation.pole(body, JD) @ orbit_normal, -1.0, 1.0))
    )
    to_spin = float(rotation.obliquity_degrees(body, JD))

    assert to_pole + to_spin == pytest.approx(180.0, abs=1e-6)
    assert to_spin > 90.0  # the physical answer
    assert rotation.turns_backwards(body)
    assert np.allclose(rotation.spin_axis(body, JD), -rotation.pole(body, JD))


def _orbit_normal(body: str) -> np.ndarray:
    position, velocity = kepler.state(body, JD)
    normal = np.cross(position, velocity)
    return normal / np.linalg.norm(normal)


def test_pluto_uses_the_other_convention():
    """Dwarf planets get the right-hand-rule pole, so no flip is applied."""
    assert rotation.SPIN["pluto"].right_handed
    assert np.allclose(rotation.spin_axis("pluto", JD), rotation.pole("pluto", JD))
    assert not rotation.turns_backwards("pluto")
    assert rotation.spins_retrograde("pluto", JD)  # tilted past 90 all the same


def test_earth_and_mercury_are_the_extremes_of_tilt():
    tilts = {
        body: float(rotation.obliquity_degrees(body, JD))
        for body in ("mercury", "embary", "mars", "jupiter", "uranus")
    }
    assert tilts["mercury"] < 1.0
    assert tilts["uranus"] > 90.0
    assert tilts["embary"] == pytest.approx(23.44, abs=0.05)


def test_a_surface_point_round_trips():
    matrix = rotation.body_to_icrf("mars", JD)
    for latitude, longitude in ((0.0, 0.0), (45.0, 90.0), (-30.0, 200.0), (89.0, 315.0)):
        phi, lam = np.radians(latitude), np.radians(longitude)
        fixed = np.array(
            [np.cos(phi) * np.cos(lam), np.cos(phi) * np.sin(lam), np.sin(phi)]
        )
        back_lat, back_lon = rotation.surface_point("mars", JD, matrix @ fixed)
        assert float(back_lat) == pytest.approx(latitude, abs=1e-9)
        offset = (float(back_lon) - longitude + 180.0) % 360.0 - 180.0
        assert offset == pytest.approx(0.0, abs=1e-9)


def test_longitude_never_reaches_360():
    """The range is half-open, and floating point wants to break that."""
    matrix = rotation.body_to_icrf("mars", JD)
    just_west = matrix @ np.array([1.0, -1e-16, 0.0])
    _, longitude = rotation.surface_point("mars", JD, just_west)
    assert 0.0 <= float(longitude) < 360.0


def test_the_sub_solar_latitude_is_the_suns_declination():
    """At the solstice it equals the obliquity, which is what a solstice is."""
    jd = times.jd(2026, 6, 21, 12)
    toward_sun = frames.ecliptic_to_equatorial(-kepler.position("embary", jd))
    latitude, _ = rotation.surface_point("embary", jd, toward_sun)
    assert float(latitude) == pytest.approx(23.44, abs=0.05)


def test_the_sub_solar_point_crosses_the_equator_at_the_equinox():
    jd = times.jd(2026, 9, 23, 12)
    toward_sun = frames.ecliptic_to_equatorial(-kepler.position("embary", jd))
    latitude, _ = rotation.surface_point("embary", jd, toward_sun)
    assert abs(float(latitude)) < 0.4


def test_the_sub_solar_longitude_is_near_greenwich_at_noon():
    """Within the equation of time, which is at most about four degrees."""
    jd = times.jd(2026, 9, 3, 12)
    toward_sun = frames.ecliptic_to_equatorial(-kepler.position("embary", jd))
    _, longitude = rotation.surface_point("embary", jd, toward_sun)
    offset = (float(longitude) + 180.0) % 360.0 - 180.0
    assert abs(offset) < 4.5


def test_the_sun_is_tilted_to_the_ecliptic_not_to_an_orbit():
    """The Sun has no orbit here, so "obliquity" has to mean something else.

    Its equator is tilted 7.25 degrees to the ecliptic. Asking the general code
    path for it used to raise, because it went looking for the Sun's orbital
    elements, and the viewer's focus mode was the first thing to ask.
    """
    assert float(rotation.obliquity_degrees("sun", JD)) == pytest.approx(7.25, abs=0.05)


def test_the_sun_turns_once_in_25_days():
    assert rotation.rotation_period_days("sun") == pytest.approx(25.38, abs=0.05)


def test_unknown_bodies_are_rejected():
    with pytest.raises(KeyError):
        rotation.pole("planet nine", JD)
