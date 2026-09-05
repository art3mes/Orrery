"""Light-time, bending and aberration, on problems with known answers.

``validate_m3.py`` checks the whole chain against Skyfield. These check the
pieces against algebra, so that a failure says which piece.
"""

import numpy as np
import pytest

from orrery import apparent, frames
from orrery.nbody import C_AU_PER_DAY

ARCSEC_PER_RADIAN = np.degrees(1.0) * 3600.0


# --- light-time -------------------------------------------------------------


def test_stationary_target_gives_exactly_distance_over_c():
    target = np.array([3.0, 4.0, 0.0])  # 5 au away
    observer = np.zeros(3)
    _, distance, emitted = apparent.light_time(
        lambda t: np.broadcast_to(target, (np.size(t), 3)), observer, np.array([0.0])
    )
    assert distance[0] == pytest.approx(5.0)
    assert emitted[0] == pytest.approx(-5.0 / C_AU_PER_DAY)


def test_a_moving_target_is_seen_where_it_was():
    """Constant velocity, so the answer can be written down.

    The light that arrives at t=0 left when the target was closer (or further)
    by v * light-time, and the implicit equation has a closed-form root.
    """
    start = np.array([5.0, 0.0, 0.0])
    velocity = np.array([0.0, 0.01, 0.0])  # au/day

    def at(t):
        t = np.atleast_1d(t)
        return start + velocity * t[:, None]

    _, distance, emitted = apparent.light_time(at, np.zeros(3), np.array([0.0]))

    # Self-consistency is the whole definition: the distance to where it was
    # must equal c times how long ago that was.
    assert distance[0] == pytest.approx(-emitted[0] * C_AU_PER_DAY, rel=1e-12)
    # And it must not be the naive answer.
    assert distance[0] != pytest.approx(5.0, abs=1e-9)


def test_light_time_to_the_sun_is_about_499_seconds():
    at = lambda t: np.zeros((np.size(t), 3))  # noqa: E731
    observer = np.array([1.0, 0.0, 0.0])
    _, _, emitted = apparent.light_time(at, observer, np.array([0.0]))
    assert -emitted[0] * 86400.0 == pytest.approx(499.0, abs=0.5)


# --- aberration -------------------------------------------------------------


def test_standing_still_changes_nothing():
    p = np.array([[0.6, 0.8, 0.0]])
    assert np.allclose(apparent.aberrate(p, np.zeros((1, 3))), p)


def test_aberration_reaches_twenty_arcsec_at_earths_speed():
    """The aberration constant, 20.5 arcsec, is just v/c for the Earth."""
    speed = 2 * np.pi / 365.25  # au/day, circular orbit at 1 au
    p = np.array([[1.0, 0.0, 0.0]])
    velocity = np.array([[0.0, speed, 0.0]])  # square on to the line of sight
    shifted = apparent.aberrate(p, velocity)
    assert frames.separation_arcsec(p, shifted)[0] == pytest.approx(20.5, abs=0.2)


def test_aberration_vanishes_looking_along_the_motion():
    p = np.array([[1.0, 0.0, 0.0]])
    velocity = np.array([[1e-4 * C_AU_PER_DAY, 0.0, 0.0]])
    assert frames.separation_arcsec(p, apparent.aberrate(p, velocity))[0] < 1e-9


def test_aberration_returns_unit_vectors():
    rng = np.random.default_rng(5)
    p = rng.normal(size=(20, 3)) * 3.0
    velocity = rng.normal(size=(20, 3)) * 1e-4 * C_AU_PER_DAY
    assert np.allclose(frames.norm(apparent.aberrate(p, velocity)), 1.0)


# --- gravitational deflection -----------------------------------------------


def test_the_suns_schwarzschild_radius_is_three_kilometres():
    assert apparent.SUN_SCHWARZSCHILD_AU * 149597870.7 == pytest.approx(2.95, abs=0.02)


def _grazing_shift(impact_parameter_in_radii: float) -> tuple[float, float]:
    """Deflection of a ray passing the Sun at a given impact parameter.

    The observer sits 1 au out, so a ray leaving at angle theta from the Sun's
    direction passes the Sun at b = theta au. Placing the target at a large
    offset *in au* instead, which is the obvious thing to write, gives an impact
    parameter a thousand times too small and a deflection to match.
    """
    solar_radius = 0.00465047
    observer_from_sun = np.array([[1.0, 0.0, 0.0]])
    angle = solar_radius * impact_parameter_in_radii
    target_from_sun = observer_from_sun + 1000.0 * np.array(
        [[-np.cos(angle), np.sin(angle), 0.0]]
    )
    direction = target_from_sun - observer_from_sun
    bent = apparent.deflect(direction, target_from_sun, observer_from_sun)

    predicted = (
        np.degrees(2 * apparent.SUN_SCHWARZSCHILD_AU / (angle * 1.0)) * 3600.0
    )
    return float(frames.separation_arcsec(direction, bent)[0]), float(predicted)


def test_a_ray_grazing_the_sun_bends_by_1_75_arcsec():
    """Eddington's number, and the reason anyone looked at a 1919 eclipse."""
    measured, predicted = _grazing_shift(1.001)
    assert measured == pytest.approx(1.75, abs=0.01)
    # A shade under 4GM/c^2b, because the observer is 1 au away rather than
    # infinitely far, so the ray is not bent over its whole path.
    assert measured < predicted
    assert measured == pytest.approx(predicted, rel=0.002)


def test_deflection_goes_as_one_over_the_impact_parameter():
    close, _ = _grazing_shift(2.0)
    far, _ = _grazing_shift(20.0)
    assert close / far == pytest.approx(10.0, rel=0.01)


def test_a_ray_exactly_at_the_limb_is_treated_as_occulted():
    """The guard is deliberately inclusive: at b = R the Sun is in the way."""
    measured, _ = _grazing_shift(1.0)
    assert measured == pytest.approx(0.0, abs=1e-9)


def test_deflection_falls_away_from_the_sun():
    observer_from_sun = np.array([[1.0, 0.0, 0.0]])
    target_from_sun = np.array([[0.0, 1000.0, 0.0]])  # a right angle
    direction = target_from_sun - observer_from_sun
    bent = apparent.deflect(direction, target_from_sun, observer_from_sun)
    shift = frames.separation_arcsec(direction, bent)[0]
    assert 0.001 < shift < 0.01  # about 0.004 arcsec, small but not nothing


def test_a_ray_through_the_sun_is_left_alone():
    """The formula diverges there, and nothing is visible anyway."""
    observer_from_sun = np.array([[1.0, 0.0, 0.0]])
    target_from_sun = np.array([[-5.0, 0.0, 0.0]])  # directly behind the Sun
    direction = target_from_sun - observer_from_sun
    bent = apparent.deflect(direction, target_from_sun, observer_from_sun)
    assert np.allclose(bent, direction / frames.norm(direction)[:, None])


# --- interpolation ----------------------------------------------------------


def test_interpolation_is_exact_on_a_cubic():
    """Four-point Lagrange reproduces any polynomial of degree three."""
    grid = np.arange(0.0, 10.0, 0.5)
    positions = np.stack(
        [grid**3 - 2 * grid, 4 * grid**2 + 1, np.full_like(grid, 7.0)], axis=-1
    )
    at = apparent.interpolator(grid, positions)

    probe = np.array([1.234, 5.678, 8.9])
    want = np.stack(
        [probe**3 - 2 * probe, 4 * probe**2 + 1, np.full_like(probe, 7.0)], axis=-1
    )
    assert np.allclose(at(probe), want, rtol=1e-12, atol=1e-12)


def test_interpolation_needs_enough_samples():
    with pytest.raises(ValueError, match="at least 4"):
        apparent.interpolator(np.arange(3.0), np.zeros((3, 3)))
    with pytest.raises(ValueError, match="one position per date"):
        apparent.interpolator(np.arange(5.0), np.zeros((4, 3)))


def test_interpolation_clamps_at_the_ends():
    grid = np.arange(0.0, 5.0)
    positions = np.stack([grid, grid, grid], axis=-1)
    at = apparent.interpolator(grid, positions)
    assert np.allclose(at(np.array([0.0]))[0], [0.0, 0.0, 0.0], atol=1e-12)
    assert np.allclose(at(np.array([4.0]))[0], [4.0, 4.0, 4.0], atol=1e-12)


# --- the whole chain --------------------------------------------------------


def test_a_still_observer_watching_a_still_target_sees_it_where_it_is():
    target = np.array([0.0, 2.0, 0.0])
    at = lambda t: np.broadcast_to(target, (np.size(t), 3))  # noqa: E731
    sight = apparent.observe(
        at, np.array([0.0]), np.zeros(3), np.zeros(3), sun_at=None
    )
    assert np.allclose(sight.apparent, sight.astrometric)
    assert sight.distance[0] == pytest.approx(2.0)
    assert sight.light_minutes[0] == pytest.approx(2.0 / C_AU_PER_DAY * 1440.0)


def test_turning_the_corrections_off_leaves_pure_geometry():
    rng = np.random.default_rng(2)
    target = rng.normal(size=3) * 3.0
    at = lambda t: np.broadcast_to(target, (np.size(t), 3))  # noqa: E731
    observer_position = rng.normal(size=3)
    observer_velocity = rng.normal(size=3) * 1e-4

    bare = apparent.observe(
        at, np.array([0.0]), observer_position, observer_velocity,
        deflection=False, aberration=False,
    )
    assert np.allclose(bare.apparent, bare.astrometric)
