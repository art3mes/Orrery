"""Offline checks on the state vectors.

None of these need a reference ephemeris. They test the things that must be
true of *any* correct two-body solution -- conserved quantities, closure, the
plane the orbit lies in -- plus a few facts about the real solar system that a
unit or sign error would break loudly.

The point is that these run with no network and no downloads, so a mistake gets
caught in the second it is made, and ``validate_m0.py`` is left to answer the
one question they cannot: is it the *right* ellipse?
"""

import numpy as np
import pytest

from orrery import elements, frames, kepler, times

# Sample the whole validity window of the element table.
JD = times.linspace(times.jd(1850, 1, 1), times.jd(2050, 1, 1), 200)

# Perihelion and aphelion distances in au, from a(1 -/+ e), with a little slack.
DISTANCE_RANGE = {
    "mercury": (0.30, 0.47),
    "venus": (0.71, 0.73),
    "embary": (0.98, 1.02),
    "mars": (1.37, 1.67),
    "jupiter": (4.94, 5.47),
    "saturn": (8.99, 10.08),
    "uranus": (18.21, 20.14),
    "neptune": (29.79, 30.35),
    "pluto": (29.60, 49.36),
}


@pytest.mark.parametrize("body", elements.ORDER)
def test_energy_matches_the_semi_major_axis(body):
    """v^2/2 - GM/r = -GM/2a, the vis-viva relation.

    True by construction, which is exactly why it is worth asserting: it catches
    a mismatch between the position branch and the velocity branch of
    ``state()``, the two of which are written separately.
    """
    pos, vel = kepler.state(body, JD)
    a = elements.elements_at(body, JD)["a"]
    energy = 0.5 * np.sum(vel**2, axis=-1) - kepler.GM_SUN / frames.norm(pos)
    expected = -kepler.GM_SUN / (2 * a)
    assert np.max(np.abs(energy / expected - 1)) < 1e-10


@pytest.mark.parametrize("body", elements.ORDER)
def test_angular_momentum_matches_the_ellipse(body):
    pos, vel = kepler.state(body, JD)
    el = elements.elements_at(body, JD)
    h = frames.norm(np.cross(pos, vel))
    expected = np.sqrt(kepler.GM_SUN * el["a"] * (1 - el["e"] ** 2))
    assert np.max(np.abs(h / expected - 1)) < 1e-10


@pytest.mark.parametrize("body", elements.ORDER)
def test_distance_stays_between_perihelion_and_aphelion(body):
    r = frames.norm(kepler.position(body, JD))
    low, high = DISTANCE_RANGE[body]
    assert low <= r.min() and r.max() <= high


@pytest.mark.parametrize("body", elements.ORDER)
def test_velocity_agrees_with_a_numerical_derivative(body):
    """The analytic velocity should differentiate the position.

    They do not agree exactly, and the residual is informative rather than a
    bug: ``state()`` computes speed from the semi-major axis, while the position
    advances at the table's mean-longitude rate, and those two columns were
    fitted independently. For Pluto the implied periods differ by about 0.07%.
    """
    dt = 0.01  # days
    t = JD[::20]
    ahead = kepler.position(body, t + dt)
    behind = kepler.position(body, t - dt)
    numeric = (ahead - behind) / (2 * dt)
    _, analytic = kepler.state(body, t)
    relative = frames.norm(numeric - analytic) / frames.norm(analytic)
    assert relative.max() < 5e-3


def test_orbit_closes_after_one_period():
    period = kepler.period("embary")
    start = times.jd(2000, 1, 1)
    drift = frames.norm(
        kepler.position("embary", start) - kepler.position("embary", start + period)
    )
    assert drift < 1e-3  # au


def test_earth_stays_in_the_ecliptic_plane():
    """The ecliptic *is* Earth's orbital plane, so z should be ~0.

    Not exactly zero: the frame is the ecliptic of J2000 and the plane itself
    precesses, which the table records as a nonzero inclination rate. Over two
    centuries that lifts the Earth a few 1e-4 au out of plane. A sign error in
    the inclination rotation would put it thousands of times further.
    """
    z = kepler.position("embary", JD)[..., 2]
    assert np.max(np.abs(z)) < 5e-4


def test_earth_reaches_perihelion_in_early_january():
    """A real, checkable fact, and a trap for angle errors.

    Earth is closest to the Sun in early January -- perihelion fell on 5 January
    in 2020. Nothing in the code knows this; it falls out of the longitude of
    perihelion being right.
    """
    days = times.jd(2020, 1, 1) + np.arange(0, 366, 0.05)
    closest = days[np.argmin(frames.norm(kepler.position("embary", days)))]
    year, month, day = times.calendar(closest)
    assert (year, month) == (2020, 1)
    assert 2 <= day <= 7


def test_planets_are_ordered_by_distance():
    order = [frames.norm(kepler.position(b, times.J2000)) for b in elements.ORDER]
    # Pluto is closer than Neptune for part of its orbit, but not at J2000.
    assert order == sorted(order)


def test_scalar_and_array_dates_agree():
    single = kepler.position("mars", times.J2000)
    batch = kepler.position("mars", np.array([times.J2000, times.J2000 + 1]))
    assert single.shape == (3,)
    assert batch.shape == (2, 3)
    assert np.allclose(single, batch[0])


def test_extrapolating_past_2050_warns():
    with pytest.warns(RuntimeWarning, match="outside"):
        kepler.position("mars", times.jd(2200, 1, 1))
