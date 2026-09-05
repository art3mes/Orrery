"""Scene geometry: the orbit rings, the trails, and the one honest lie.

No polyscope here. Everything the viewer draws is built as plain arrays by
``scene.py`` precisely so it can be checked like this, with no window and no
graphics driver.
"""

import warnings

import numpy as np
import pytest

from orrery import elements, frames, kepler, scene, times, view

JD = times.jd(2026, 9, 3)


# --- the ellipse ------------------------------------------------------------


@pytest.mark.parametrize("body", elements.ORDER)
def test_ring_spans_perihelion_to_aphelion(body):
    el = elements.elements_at(body, JD)
    a, e = float(el["a"]), float(el["e"])
    r = frames.norm(kepler.ellipse(body, JD, samples=4096))
    assert r.min() == pytest.approx(a * (1 - e), rel=1e-6)
    assert r.max() == pytest.approx(a * (1 + e), rel=1e-6)


@pytest.mark.parametrize("body", elements.ORDER)
def test_the_sun_is_at_a_focus_not_the_centre(body):
    """Every point on the ring satisfies the focus-directrix property.

    r = a(1 - e cos E) for an ellipse with the Sun at a focus. A model that drew
    the orbit as a circle of radius a, or centred the ellipse on the Sun, would
    still look like an orbit and would fail here.
    """
    el = elements.elements_at(body, JD)
    a, e = float(el["a"]), float(el["e"])
    samples = 512
    E = np.linspace(0.0, 2 * np.pi, samples, endpoint=False)
    r = frames.norm(kepler.ellipse(body, JD, samples=samples))
    assert np.allclose(r, a * (1 - e * np.cos(E)), rtol=1e-12)


def test_ring_is_closed_but_not_duplicated():
    loop = kepler.ellipse("mars", JD, samples=64)
    assert loop.shape == (64, 3)
    assert not np.allclose(loop[0], loop[-1])  # the loop closes by wrapping
    steps = frames.norm(np.diff(np.vstack([loop, loop[:1]]), axis=0))
    assert steps.min() > 0  # no zero-length edge


def test_ring_needs_one_instant():
    with pytest.raises(ValueError, match="scalar"):
        kepler.ellipse("mars", np.array([JD, JD + 1]))


def test_outer_planets_are_drawable_without_extrapolating():
    """The reason the ring sweeps E rather than time.

    Neptune takes 165 years to go round. Sampling its actual path over a period
    would run straight out of the element table; sweeping the eccentric anomaly
    stays at one instant.
    """
    with warnings.catch_warnings():
        warnings.simplefilter("error", RuntimeWarning)
        for body in ("neptune", "pluto"):
            kepler.ellipse(body, times.jd(2049, 1, 1), samples=128)


# --- trails -----------------------------------------------------------------


def test_trail_ends_at_the_planet():
    for body in elements.ORDER:
        span = scene.trail_span_days(body)
        path = scene.trail(body, JD, span)
        assert frames.norm(path[-1] - kepler.position(body, JD)) < 1e-9


def test_trail_is_clamped_to_the_element_table():
    """Pluto's trail is 62 years long; near 1850 it would reach past 1800."""
    with warnings.catch_warnings():
        warnings.simplefilter("error", RuntimeWarning)
        path = scene.trail("pluto", elements.VALID_JD[0], scene.trail_span_days("pluto"))
    assert path.shape == (200, 3)


def test_trail_length_scales_with_the_orbit():
    """A fixed number of days would smear Mercury and leave Neptune a stub."""
    mercury = scene.trail_span_days("mercury")
    neptune = scene.trail_span_days("neptune")
    assert neptune / mercury == pytest.approx(kepler.period("neptune") / kepler.period("mercury"))


def test_trail_needs_two_points():
    with pytest.raises(ValueError, match="at least 2"):
        scene.trail("mars", JD, 100.0, samples=1)


# --- sizes ------------------------------------------------------------------


def test_unexaggerated_radius_is_the_real_radius():
    assert scene.display_radius_au("embary", 1.0) * kepler.AU_KM == pytest.approx(
        6378.137
    )


def test_the_two_modules_agree_on_the_size_of_the_earth():
    """They did not. scene had the mean radius, observer the equatorial one.

    Six kilometres apart, invisible in every picture, and enough to give the
    Earth two thirds of its real flattening.
    """
    from orrery import observer

    assert scene.RADIUS_KM["embary"] == pytest.approx(observer.EARTH_RADIUS_KM)


def test_exaggeration_is_linear_and_so_preserves_size_ratios():
    ratio = scene.display_radius_au("jupiter", 1.0) / scene.display_radius_au("embary", 1.0)
    for factor in (1.0, 300.0, 5000.0):
        scaled = scene.display_radius_au("jupiter", factor) / scene.display_radius_au(
            "embary", factor
        )
        assert scaled == pytest.approx(ratio)


def test_exaggeration_must_be_positive():
    with pytest.raises(ValueError, match="positive"):
        scene.display_radius_au("mars", 0.0)


def test_default_sun_stays_inside_mercurys_orbit():
    assert scene.sun_fits_inside_mercury(scene.VIEW_PRESETS["inner"].sun_exaggeration)
    assert not scene.sun_fits_inside_mercury(500.0)


def test_view_presets_name_real_bodies():
    for name, preset in scene.VIEW_PRESETS.items():
        assert preset.bodies, name
        for body in preset.bodies:
            assert elements.canonical(body) == body
        assert preset.scale_au > 0 and preset.planet_exaggeration > 0


def test_inner_preset_frames_mars_aphelion():
    preset = scene.VIEW_PRESETS["inner"]
    mars_aphelion = 1.666
    assert preset.scale_au > mars_aphelion


def test_line_width_follows_the_framing():
    assert scene.line_radius_au(52.0) > scene.line_radius_au(2.0)


# --- the scrubber -----------------------------------------------------------


def test_day_number_and_julian_date_round_trip():
    for jd in (times.jd(1850, 1, 1), times.jd(1987, 4, 5), times.jd(2050, 1, 1)):
        assert view.day_to_jd(view.jd_to_day(jd)) == pytest.approx(jd, abs=1e-9)


def test_the_scrubber_counts_days_for_a_reason():
    """Single precision cannot resolve a Julian date, and this is the fix.

    A Julian date is ~2.45e6, where a float32 steps by 0.25 days -- a scrubber
    bound to it would snap in six-hour jumps. Days since 1850 top out near
    73000, where the step is under 12 minutes.
    """
    jd = times.jd(2026, 9, 3)

    def float32_step(value):
        return float(np.nextafter(np.float32(value), np.float32(np.inf)) - np.float32(value))

    assert float32_step(jd) == pytest.approx(0.25)
    assert float32_step(view.jd_to_day(jd)) < 12 / (24 * 60)


# --- the drawn Sun ----------------------------------------------------------


def test_no_preset_opens_on_a_picture_that_is_not_true():
    """Every framing has to satisfy the check the viewer itself applies.

    Two of the three used to fail it. The viewer opened at 300x and 500x, put
    the drawn Sun at 1.4 and 2.3 au, swallowed Mercury's orbit and Venus's, and
    printed a line in its own status panel saying so. A caption does not undo a
    picture -- a reader believes what they can see first.
    """
    for name, preset in scene.VIEW_PRESETS.items():
        assert scene.sun_fits_inside_mercury(preset.sun_exaggeration), name


def test_the_ceiling_is_where_the_sun_reaches_mercury():
    """66, and it follows from two radii rather than from taste."""
    ceiling = scene.largest_honest_sun()
    assert ceiling == pytest.approx(66.1, abs=0.1)
    assert scene.sun_fits_inside_mercury(ceiling * 0.999)
    assert not scene.sun_fits_inside_mercury(ceiling * 1.001)


def test_the_slider_can_still_be_pushed_past_it():
    """Being honest by default is not the same as forbidding the exaggeration.

    Somebody framing all 52 au wants a Sun they can see, and the viewer will
    draw one -- and say what it costs, which is the whole difference.
    """
    assert not scene.sun_fits_inside_mercury(500.0)
    assert scene.display_radius_au("sun", 500.0) > scene.MERCURY_PERIHELION_AU
