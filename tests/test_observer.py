"""The observer, and the frame their coordinates are quoted in.

Nothing here needs a network or an ephemeris. ``validate_m3.py`` compares the
combination against Skyfield; these pin the pieces to values that can be
checked by hand.
"""

import numpy as np
import pytest

from orrery import frames, observer, precession, times

ARCSEC = np.pi / (180.0 * 3600.0)


# --- the shape of the Earth -------------------------------------------------


def test_the_equator_is_one_earth_radius_out():
    rho_cos, rho_sin = observer.Site("equator", 0.0, 0.0).geocentric_components()
    assert rho_cos == pytest.approx(1.0)
    assert rho_sin == pytest.approx(0.0)


def test_the_pole_is_flattened():
    """The polar radius is 21.4 km shorter, which is the whole of the flattening."""
    _, rho_sin = observer.Site("pole", 90.0, 0.0).geocentric_components()
    polar_km = rho_sin * observer.EARTH_RADIUS_KM
    assert polar_km == pytest.approx(6356.752, abs=0.001)
    assert observer.EARTH_RADIUS_KM - polar_km == pytest.approx(21.385, abs=0.01)


def test_a_spherical_earth_would_misplace_a_mid_latitude_observer():
    """By about 10 km at 45 degrees -- 14 milliarcsec at 1 au, and 0.5" at the Moon."""
    site = observer.Site("halfway", 45.0, 0.0)
    rho_cos, rho_sin = site.geocentric_components()
    radius = np.hypot(rho_cos, rho_sin) * observer.EARTH_RADIUS_KM
    assert observer.EARTH_RADIUS_KM - radius == pytest.approx(10.7, abs=0.5)


@pytest.mark.parametrize("name", sorted(observer.SITES))
def test_every_named_site_sits_on_the_surface(name):
    site = observer.SITES[name]
    jd = np.array([times.jd(2026, 9, 3)])
    position, _ = site.offset_from_geocentre(jd)
    radius_km = frames.norm(position)[0] * 149597870.7
    assert 6350.0 < radius_km < 6390.0


def test_the_ground_moves_at_465_metres_per_second_on_the_equator():
    """Diurnal aberration is 0.32 arcsec, which is not negligible here."""
    site = observer.Site("equator", 0.0, 0.0)
    _, velocity = site.offset_from_geocentre(np.array([times.jd(2026, 9, 3)]))
    au_per_day = frames.norm(velocity)[0]
    metres_per_second = au_per_day * 149597870.7 * 1000.0 / 86400.0
    assert metres_per_second == pytest.approx(465.0, abs=1.0)


def test_the_poles_barely_move():
    site = observer.Site("pole", 90.0, 0.0)
    _, velocity = site.offset_from_geocentre(np.array([times.jd(2026, 9, 3)]))
    assert frames.norm(velocity)[0] < 1e-12


def test_longitude_separates_two_sites_by_the_right_angle():
    jd = np.array([times.jd(2026, 9, 3)])
    here, _ = observer.Site("a", 0.0, 0.0).offset_from_geocentre(jd)
    there, _ = observer.Site("b", 0.0, 90.0).offset_from_geocentre(jd)
    assert frames.separation_arcsec(here, there)[0] == pytest.approx(90 * 3600.0, abs=1)


# --- delta T ----------------------------------------------------------------


def test_delta_t_is_about_64_seconds_in_2000():
    assert observer.delta_t_seconds(np.array([times.jd(2000, 1, 1)]))[0] == pytest.approx(
        63.9, abs=0.5
    )


def test_delta_t_was_near_zero_in_1900():
    assert abs(observer.delta_t_seconds(np.array([times.jd(1902, 1, 1)]))[0]) < 5.0


@pytest.mark.parametrize("year", (1920, 1941, 1961, 1986, 2005))
def test_delta_t_does_not_jump_at_a_piece_boundary(year):
    """Espenak and Meeus's polynomials are fitted separately per interval.

    They are not required to join up, and a visible step would put a second or
    two of Earth rotation into any event timed across the boundary.
    """
    before = observer.delta_t_seconds(np.array([times.jd(year, 1, 1) - 1.0]))[0]
    after = observer.delta_t_seconds(np.array([times.jd(year, 1, 1) + 1.0]))[0]
    assert abs(after - before) < 0.6


def test_ut1_runs_behind_tt():
    jd = np.array([times.jd(2026, 9, 3)])
    assert observer.ut1_from_tdb(jd)[0] < jd[0]


# --- precession, nutation, sidereal time ------------------------------------


def test_precession_is_the_identity_at_its_own_epoch():
    matrix = precession.precession_matrix(np.array([times.J2000]))
    assert np.allclose(matrix[:, :, 0], np.eye(3), atol=1e-15)


@pytest.mark.parametrize("year", (1850, 1975, 2026, 2050))
def test_both_matrices_are_rotations(year):
    jd = np.array([times.jd(year, 3, 1)])
    for matrix in (precession.precession_matrix(jd), precession.nutation_matrix(jd)):
        m = matrix[:, :, 0]
        assert np.allclose(m @ m.T, np.eye(3), atol=1e-14)
        assert np.linalg.det(m) == pytest.approx(1.0)


def test_the_equinox_moves_by_1_4_degrees_a_century():
    """General precession is 5028 arcsec per century; this measures it."""
    x_axis = np.array([1.0, 0.0, 0.0])  # the J2000 equinox
    later = precession._apply(
        precession.precession_matrix(np.array([times.J2000 + 36525.0])), x_axis
    )
    moved = frames.separation_arcsec(x_axis, later)[0]
    assert moved == pytest.approx(5028.0, rel=0.01)


def test_obliquity_at_j2000_is_the_defining_value():
    assert precession.mean_obliquity(times.J2000) / ARCSEC == pytest.approx(84381.448)


def test_obliquity_is_decreasing():
    """By 47 arcsec per century, which is why the tropics are creeping."""
    now = precession.mean_obliquity(times.J2000)
    century = precession.mean_obliquity(times.J2000 + 36525.0)
    assert (now - century) / ARCSEC == pytest.approx(46.8, abs=0.2)


def test_nutation_is_dominated_by_the_18_year_term():
    jd = times.J2000 + np.linspace(0.0, 6798.0, 400)  # one node cycle
    d_psi, d_eps = precession.nutation(jd)
    assert np.max(np.abs(d_psi)) / ARCSEC == pytest.approx(17.5, abs=1.5)
    assert np.max(np.abs(d_eps)) / ARCSEC == pytest.approx(9.5, abs=1.0)


def test_sidereal_time_at_j2000():
    """A known anchor: GMST was 18h 41m 50.5s at 2000 January 1.5 UT1."""
    gmst = precession.greenwich_mean_sidereal_time(np.array([times.J2000]))[0]
    assert gmst / 15.0 == pytest.approx(18.697374, abs=1e-5)


def test_sidereal_time_gains_four_minutes_a_day():
    jd = np.array([times.J2000, times.J2000 + 1.0])
    gmst = precession.greenwich_mean_sidereal_time(jd)
    gained = (gmst[1] - gmst[0]) % 360.0 / 15.0 * 3600.0
    assert gained == pytest.approx(236.6, abs=0.5)  # seconds of time


def test_the_equation_of_the_equinoxes_is_about_a_second_of_time():
    """Nutation in longitude projected on the equator: 17 arcsec of angle.

    Which is 1.1 *seconds of time*, since the sky turns 15 arcsec a second.
    Mixing those two up is easy and this test did it first time round.
    """
    jd = times.J2000 + np.linspace(0.0, 6798.0, 200)
    difference = (
        precession.greenwich_apparent_sidereal_time(jd, jd)
        - precession.greenwich_mean_sidereal_time(jd)
    )
    arcsec = ((difference + 180.0) % 360.0 - 180.0) * 3600.0
    assert np.max(np.abs(arcsec)) == pytest.approx(17.0, abs=2.0)
    assert np.max(np.abs(arcsec)) / 15.0 < 1.5  # seconds of time
