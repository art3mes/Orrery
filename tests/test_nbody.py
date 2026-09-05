"""N-body forces, integrators, and the elements read back out of a state vector.

Checked against two-body problems, where the answer is known in closed form, and
against the conservation laws, which hold whatever the configuration. Nothing
here needs an ephemeris; ``validate_m2.py`` does the comparison against DE440.
"""

import numpy as np
import pytest

from orrery import frames, kepler, nbody

GM_SUN = kepler.GM_SUN


def two_body(a=1.0, e=0.0, ratio=1e6):
    """Sun plus a light companion at perihelion, in the xy plane."""
    gm = np.array([GM_SUN, GM_SUN / ratio])
    r = a * (1 - e)
    speed = np.sqrt(GM_SUN * (1 + e) / (a * (1 - e)))
    pos = np.array([[0.0, 0.0, 0.0], [r, 0.0, 0.0]])
    vel = np.array([[0.0, 0.0, 0.0], [0.0, speed, 0.0]])
    return (*nbody.to_barycentric(pos, vel, gm), gm)


# --- forces -----------------------------------------------------------------


def test_acceleration_is_inverse_square():
    pos = np.array([[0.0, 0.0, 0.0], [2.0, 0.0, 0.0]])
    gm = np.array([GM_SUN, 0.0])
    acc = nbody.accelerations(pos, gm)
    assert acc[1] == pytest.approx([-GM_SUN / 4.0, 0.0, 0.0])
    assert acc[0] == pytest.approx([0.0, 0.0, 0.0])  # massless companion pulls nothing


def test_no_body_pulls_on_itself():
    """The diagonal is removed by setting its separation to infinity."""
    pos = np.array([[1.0, 2.0, 3.0]])
    assert np.all(nbody.accelerations(pos, np.array([GM_SUN])) == 0.0)


def test_third_law_holds_body_by_body():
    rng = np.random.default_rng(3)
    pos = rng.normal(scale=5.0, size=(6, 3))
    gm = rng.uniform(1e-8, 1e-4, size=6)
    total = gm @ nbody.accelerations(pos, gm)
    assert np.max(np.abs(total)) < 1e-18


def test_relativistic_term_is_a_small_correction():
    """Order (v/c)^2 -- a few parts in 1e8 for Mercury, and never zero."""
    pos, vel, gm = two_body(a=0.387, e=0.206, ratio=6023600.0)
    newtonian = nbody.accelerations(pos, gm)
    correction = nbody.relativistic_correction(pos, vel, gm)
    ratio = np.linalg.norm(correction[1]) / np.linalg.norm(newtonian[1])
    assert 1e-9 < ratio < 1e-6
    assert np.all(correction[0] == 0.0)  # the Sun does not orbit itself


# --- conserved quantities ---------------------------------------------------


def test_barycentric_shift_zeroes_the_momentum():
    rng = np.random.default_rng(11)
    pos = rng.normal(size=(5, 3))
    vel = rng.normal(size=(5, 3))
    gm = rng.uniform(1e-6, 1e-3, size=5)
    _, shifted = nbody.to_barycentric(pos, vel, gm)
    residual = np.max(np.abs(nbody.momentum(shifted, gm)))
    assert residual / np.sum(gm * np.linalg.norm(vel, axis=-1)) < 1e-14


def test_circular_orbit_has_the_vis_viva_energy():
    pos, vel, gm = two_body(a=1.0, e=0.0, ratio=1e12)
    # A test particle on a circular orbit: E = -GM/2a, times its own GM here.
    assert nbody.energy(pos, vel, gm) == pytest.approx(-gm[1] * GM_SUN / 2.0, rel=1e-6)


# --- integrators ------------------------------------------------------------


@pytest.mark.parametrize("method", sorted(nbody.SYMPLECTIC))
def test_symplectic_energy_stays_in_a_band(method):
    pos, vel, gm = two_body(a=1.0, e=0.2)
    run = nbody.integrate(
        pos, vel, bodies=("sun", "mercury"), jd0=0.0, dt=1.0,
        days=200 * 365.25, method=method, sample_every=50,
    )
    relative = np.abs(run.energy / run.energy[0] - 1.0)
    # The band does not widen: the second half is no worse than the first.
    half = len(relative) // 2
    assert relative[half:].max() < 3.0 * relative[:half].max() + 1e-14


def test_runge_kutta_leaks_energy_where_leapfrog_does_not():
    """The reason nothing outside demo_m2 uses rk4."""
    pos, vel, gm = two_body(a=1.0, e=0.2)
    trend = {}
    for method in ("yoshida4", "rk4"):
        run = nbody.integrate(
            pos, vel, bodies=("sun", "mercury"), jd0=0.0, dt=2.0,
            days=300 * 365.25, method=method, sample_every=50,
        )
        e = run.energy / run.energy[0] - 1.0
        trend[method] = abs(np.polyfit(run.jd, e, 1)[0])
    assert trend["rk4"] > 100 * trend["yoshida4"]


def test_the_last_step_is_always_recorded():
    """Even when the span is not a whole number of sampling intervals.

    It was not, once. The reversibility test below asked for 14610 steps
    sampled every 10000 and got its "final" state from step 10000, which looked
    like a large integration error and was nothing of the kind.
    """
    pos, vel, _ = two_body()
    run = nbody.integrate(
        pos, vel, bodies=("sun", "mercury"), jd0=0.0, dt=1.0, days=1461.0,
        method="leapfrog", sample_every=1000,
    )
    assert run.jd[-1] == pytest.approx(1461.0)


def test_leapfrog_runs_backwards_to_where_it_started():
    """Time reversibility. A dissipative method could not do this."""
    pos, vel, _ = two_body(a=1.0, e=0.3)
    days = 40 * 365.25
    out = nbody.integrate(
        pos, vel, bodies=("sun", "mercury"), jd0=0.0, dt=1.0, days=days,
        method="leapfrog", sample_every=10_000,
    )
    back = nbody.integrate(
        out.pos[-1], out.vel[-1], bodies=("sun", "mercury"), jd0=out.jd[-1],
        dt=-1.0, days=-days, method="leapfrog", sample_every=10_000,
    )
    # Not exactly zero, but only because of round-off: 29220 steps of
    # floating-point arithmetic leave about a metre. The method itself has no
    # preferred direction in time.
    assert np.max(np.abs(back.pos[-1] - pos)) < 1e-10   # au; 1e-10 au is 15 m
    assert np.max(np.abs(back.vel[-1] - vel)) < 1e-12


def test_integrate_rejects_nonsense():
    pos, vel, _ = two_body()
    common = dict(bodies=("sun", "mercury"), jd0=0.0, days=10.0)
    with pytest.raises(ValueError, match="unknown method"):
        nbody.integrate(pos, vel, dt=1.0, method="euler", **common)
    with pytest.raises(ValueError, match="non-zero"):
        nbody.integrate(pos, vel, dt=0.0, **common)
    with pytest.raises(ValueError, match="same direction"):
        nbody.integrate(pos, vel, dt=-1.0, **common)
    with pytest.raises(ValueError, match="shaped"):
        nbody.integrate(pos[:1], vel[:1], dt=1.0, **common)


def test_masses_are_ordered_as_expected():
    """Jupiter is the heaviest planet and Pluto the lightest, by a long way."""
    planets = {b: nbody.GM[b] for b in nbody.SUN_OVER_BODY}
    assert max(planets, key=planets.get) == "jupiter"
    assert min(planets, key=planets.get) == "pluto"
    assert nbody.GM["sun"] / nbody.GM["jupiter"] == pytest.approx(1047.3486)


def test_light_takes_about_499_seconds_to_cross_an_au():
    assert 1.0 / nbody.C_AU_PER_DAY * 86400.0 == pytest.approx(499.0, abs=0.5)


# --- elements from a state vector -------------------------------------------


@pytest.mark.parametrize(
    "body", ("mercury", "venus", "embary", "mars", "jupiter", "pluto")
)
def test_elements_survive_a_round_trip(body):
    """elements -> state -> elements must come back unchanged."""
    from orrery import elements, times

    jd = times.jd(2026, 9, 3)
    r, v = kepler.state(body, jd)
    mu = kepler.GM_SUN
    back = kepler.elements_from_state(r, v, mu)
    want = elements.elements_at(body, jd)

    assert back["a"] == pytest.approx(float(want["a"]), rel=1e-10)
    assert back["e"] == pytest.approx(float(want["e"]), rel=1e-9)
    assert back["long_peri"] == pytest.approx(float(want["long_peri"]) % 360.0, abs=1e-8)

    # I and the node are only compared where the inclination is safely positive;
    # see the Earth-Moon barycentre test below for why.
    if float(want["I"]) > 0.1:
        assert back["I"] == pytest.approx(float(want["I"]), abs=1e-9)
        assert back["long_node"] == pytest.approx(
            float(want["long_node"]) % 360.0, abs=1e-8
        )


def test_the_earth_moon_barycentre_has_a_negative_tabulated_inclination():
    """And an extracted orbit is not obliged to agree with it.

    JPL fits the Earth-Moon barycentre's inclination as a straight line through
    zero, so after 2000 it goes negative -- which is not a thing an inclination
    can be. Read back from a state vector it comes out positive, with the node
    turned by 180 degrees. Same plane, different convention, and the longitude
    of perihelion is identical either way because the two flips cancel inside
    it. Worth a test, because "the inclination has the wrong sign" is exactly
    what a broken rotation would also look like.
    """
    from orrery import elements as element_table, times as time_module

    jd = time_module.jd(2026, 9, 3)
    r, v = kepler.state("embary", jd)
    got = kepler.elements_from_state(r, v, GM_SUN)
    want = element_table.elements_at("embary", jd)

    assert float(want["I"]) < 0
    assert got["I"] == pytest.approx(-float(want["I"]), abs=1e-9)
    assert got["long_node"] == pytest.approx(
        (float(want["long_node"]) + 180.0) % 360.0, abs=1e-6
    )
    assert got["long_peri"] == pytest.approx(float(want["long_peri"]) % 360.0, abs=1e-8)


def test_circular_orbit_has_zero_eccentricity():
    r = np.array([1.0, 0.0, 0.0])
    v = np.array([0.0, np.sqrt(GM_SUN), 0.0])
    el = kepler.elements_from_state(r, v, GM_SUN)
    assert el["e"] == pytest.approx(0.0, abs=1e-15)
    assert el["a"] == pytest.approx(1.0)
    assert el["I"] == pytest.approx(0.0, abs=1e-12)


def test_elements_broadcast_over_time():
    r, v = kepler.state("mars", np.linspace(2451545.0, 2451545.0 + 300, 7))
    el = kepler.elements_from_state(r, v, GM_SUN)
    assert el["a"].shape == (7,)
    assert np.ptp(el["a"]) < 1e-4  # a barely moves over a year


# --- the headline, in miniature ---------------------------------------------


def test_the_relativistic_term_advances_the_perihelion_by_43_arcsec():
    """Two bodies, so the only precession is the one under test.

    General relativity predicts 6 pi GM / (c^2 a (1 - e^2)) per orbit, which for
    Mercury is 42.98 arcsec per century. The integrator manufactures a spurious
    precession far larger than that, so the measurement is made as a
    *difference* between two runs identical but for the GR flag -- the artifact
    depends on the step and the orbit, not on the flag, and cancels.
    """
    a, e = 0.38709927, 0.20563593
    pos, vel, _ = two_body(a=a, e=e, ratio=6023600.0)
    mu = GM_SUN

    rates = {}
    for relativity in (False, True):
        run = nbody.integrate(
            pos, vel, bodies=("sun", "mercury"), jd0=0.0, dt=0.5,
            days=8 * 365.25, method="yoshida4", relativity=relativity,
            sample_every=20,
        )
        r, v = run.heliocentric("mercury")
        longitude = np.unwrap(
            np.radians(kepler.elements_from_state(r, v, mu)["long_peri"])
        )
        rates[relativity] = np.degrees(np.polyfit(run.jd, longitude, 1)[0]) * 3600 * 36525

    predicted = (
        np.degrees(6 * np.pi * mu / (nbody.C_AU_PER_DAY**2 * a * (1 - e * e)))
        * 3600.0
        / (2 * np.pi * np.sqrt(a**3 / mu))
        * 36525.0
    )
    assert predicted == pytest.approx(42.98, abs=0.05)
    assert rates[True] - rates[False] == pytest.approx(predicted, rel=0.02)


def test_the_frame_changes_the_answer():
    """Perihelion longitude is referred to a plane, and the plane matters.

    Measuring Mercury's against the equator instead of the ecliptic shifts it by
    degrees. Both numbers look perfectly reasonable, which is exactly why M2
    states the plane every time it quotes a rate.
    """
    r, v = kepler.state("mercury", 2451545.0)
    equatorial = kepler.elements_from_state(
        frames.ecliptic_to_equatorial(r), frames.ecliptic_to_equatorial(v), GM_SUN
    )
    ecliptic = kepler.elements_from_state(r, v, GM_SUN)
    assert abs(equatorial["long_peri"] - ecliptic["long_peri"]) > 1.0
