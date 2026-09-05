"""Frame rotations and calendar arithmetic.

Small, dull, and worth having: a swapped sign in the obliquity rotation or an
off-by-one in the Julian date puts every planet somewhere plausible.
"""

import numpy as np
import pytest

from orrery import frames, times


# --- frames -----------------------------------------------------------------


def test_rotation_round_trips():
    v = np.array([[1.0, 2.0, 3.0], [-0.4, 0.0, 0.9]])
    back = frames.equatorial_to_ecliptic(frames.ecliptic_to_equatorial(v))
    assert np.allclose(back, v, atol=1e-15)


def test_rotation_preserves_length():
    rng = np.random.default_rng(0)
    v = rng.normal(size=(50, 3))
    assert np.allclose(frames.norm(frames.ecliptic_to_equatorial(v)), frames.norm(v))


def test_equinox_direction_is_shared_by_both_frames():
    """The frames differ by a tilt about x, so +x is fixed."""
    x = np.array([1.0, 0.0, 0.0])
    assert np.allclose(frames.ecliptic_to_equatorial(x), x)


def test_ecliptic_pole_tilts_by_the_obliquity():
    z = np.array([0.0, 0.0, 1.0])
    tilted = frames.ecliptic_to_equatorial(z)
    assert frames.separation_arcsec(z, tilted) == pytest.approx(84381.448, abs=1e-6)


def test_separation_of_identical_directions_is_zero():
    v = np.array([0.3, -1.2, 0.5])
    assert frames.separation_arcsec(v, 2.0 * v) == pytest.approx(0.0, abs=1e-9)


def test_separation_of_perpendicular_directions_is_a_right_angle():
    x = np.array([1.0, 0.0, 0.0])
    y = np.array([0.0, 1.0, 0.0])
    assert frames.separation_arcsec(x, y) == pytest.approx(90 * 3600.0)


def test_separation_stays_accurate_at_small_angles():
    """acos of a dot product would lose half its digits here; atan2 does not."""
    tiny = np.radians(1e-6 / 3600.0)  # one microarcsecond
    u = np.array([1.0, 0.0, 0.0])
    v = np.array([np.cos(tiny), np.sin(tiny), 0.0])
    assert frames.separation_arcsec(u, v) == pytest.approx(1e-6, rel=1e-6)


def test_radec_of_the_axes():
    ra, dec = frames.radec(np.array([1.0, 0.0, 0.0]))
    assert (ra, dec) == pytest.approx((0.0, 0.0))
    ra, dec = frames.radec(np.array([0.0, 0.0, 1.0]))
    assert dec == pytest.approx(90.0)
    ra, _ = frames.radec(np.array([0.0, 1.0, 0.0]))
    assert ra == pytest.approx(6.0)  # 90 degrees = 6 hours


# --- times ------------------------------------------------------------------


def test_j2000_is_noon_on_2000_january_1():
    assert times.jd(2000, 1, 1, 12) == times.J2000


def test_known_julian_dates():
    # Meeus, worked examples.
    assert times.jd(1957, 10, 4, 19, 26, 24) == pytest.approx(2436116.31, abs=1e-6)
    assert times.jd(1999, 1, 1) == pytest.approx(2451179.5, abs=1e-9)


@pytest.mark.parametrize(
    "date",
    [(1850, 1, 1), (1900, 2, 28), (2000, 3, 1), (2024, 2, 29), (2050, 12, 31)],
)
def test_calendar_round_trips(date):
    value = times.jd(*date)
    year, month, day = times.calendar(value)
    assert (year, month, int(day)) == date


def test_isoformat_reads_back():
    assert times.isoformat(times.jd(2026, 9, 3, 18, 30)) == "2026-09-03 18:30"
