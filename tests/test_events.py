"""Event finding, checked on functions whose answers are known exactly.

The finders are the only code the M1 gate runs on both this model and DE440, so
a bug here would move both sides together and the gate would report agreement
while both were wrong. Hence testing them against algebra rather than against
either ephemeris.
"""

import numpy as np
import pytest

from orrery import events, frames


def test_parabola_vertex_is_found_exactly():
    x = np.linspace(-3, 5, 81)
    y = 2.0 * (x - 1.37) ** 2 + 0.5
    (found, value), = events.find_extrema(x, y, kind="min")
    assert found == pytest.approx(1.37, abs=1e-9)
    assert value == pytest.approx(0.5, abs=1e-9)


def test_maximum_is_found_too():
    x = np.linspace(0, 10, 101)
    y = -((x - 6.42) ** 2)
    (found, _), = events.find_extrema(x, y, kind="max")
    assert found == pytest.approx(6.42, abs=1e-9)


def test_threshold_rejects_shallow_extrema():
    x = np.linspace(0, 30, 301)
    y = np.sin(x) + 0.001 * np.sin(13 * x)  # big peaks plus ripple
    big = events.find_extrema(x, y, kind="max", threshold=0.5)
    assert len(big) == 5  # five sine maxima in [0, 30]


def test_refinement_never_lands_further_than_half_a_step():
    """The clamp that the 2020 great conjunction forced into existence.

    A near-flat minimum makes the fitted curvature badly conditioned, and an
    unclamped vertex can fly past the neighbouring samples -- on a 0.02 day grid
    it put Jupiter and Saturn's closest approach 24 minutes further from DE440
    than simply taking the smallest sample would have. For a smooth function on
    a uniform grid the true extremum is always within half a step of the argmin,
    so clamping there is free when the fit is good and saves it when it is not.
    """
    x = np.linspace(0.0, 1.0, 11)
    step = x[1] - x[0]

    # Quartic minimum: locally very flat, poorly matched by a parabola.
    y = (x - 0.5) ** 4 + 1.0
    y += 1e-15 * np.arange(len(x))  # a whisper of asymmetric rounding

    for found, _ in events.find_extrema(x, y, kind="min"):
        nearest = x[int(np.argmin(np.abs(x - found)))]
        assert abs(found - nearest) <= 0.5 * step + 1e-12


def test_flat_regions_do_not_produce_extrema():
    x = np.linspace(0, 1, 51)
    y = np.zeros_like(x)
    assert events.find_extrema(x, y, kind="min") == []


def test_crossings_of_a_straight_line():
    x = np.linspace(0, 10, 11)
    assert events.find_crossings(x, 2 * x - 7, 0.0) == pytest.approx([3.5])


def test_crossings_are_found_in_both_directions():
    """Ingress and egress: a transit needs the downward crossing and the upward."""
    x = np.linspace(0.1, 2 * np.pi - 0.1, 361)
    found = events.find_crossings(x, np.sin(x), 0.0)
    assert found == pytest.approx([np.pi], abs=1e-3)

    dipping = np.abs(x - np.pi) - 0.5  # negative in a band around pi
    assert len(events.find_crossings(x, dipping, 0.0)) == 2


def test_solar_disc_is_about_sixteen_arcminutes():
    at_1au = events.solar_radius_arcsec(np.array([1.0, 0.0, 0.0]))
    assert at_1au == pytest.approx(959.2, abs=1.0)

    # Closer Sun, bigger disc: ~3% between perihelion and aphelion.
    near = events.solar_radius_arcsec(np.array([0.9833, 0.0, 0.0]))
    far = events.solar_radius_arcsec(np.array([1.0167, 0.0, 0.0]))
    assert near / far == pytest.approx(1.034, abs=0.002)


def test_opposition_is_180_degrees():
    earth = np.array([1.0, 0.0, 0.0])
    beyond = np.array([2.0, 0.0, 0.0])  # directly away from the Sun
    assert events.elongation_deg(beyond, earth) == pytest.approx(180.0)


def test_conjunction_is_zero_degrees():
    earth = np.array([1.0, 0.0, 0.0])
    behind = np.array([-3.0, 0.0, 0.0])  # on the far side of the Sun
    assert events.elongation_deg(behind, earth) == pytest.approx(0.0, abs=1e-9)


def test_angles_do_not_care_which_frame_they_are_measured_in():
    """Why the M1 gate can compare ecliptic positions against equatorial ones."""
    rng = np.random.default_rng(7)
    earth, a, b = rng.normal(size=(3, 3))

    plain = events.separation_from_earth(a, b, earth)
    rotated = events.separation_from_earth(
        *(frames.ecliptic_to_equatorial(v) for v in (a, b, earth))
    )
    assert rotated == pytest.approx(plain, rel=1e-12)
