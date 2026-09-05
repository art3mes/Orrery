"""This package's own orbits, packed as an ephemeris.

``model.py`` exists so that an answer needing light-time -- an eclipse, an
apparent place -- can be computed without DE440 at all. These check the packing
rather than the astronomy: the Moon and the Earth have to come out of one set of
elements and stay consistent with each other, and the frame has to be the one
everything downstream assumes.

The astronomy is measured in the README's recipes, where the 2024 eclipse is run
through these orbits and through JPL's and the two are differenced.
"""

import numpy as np
import pytest

from orrery import frames, kepler, lunar, model, times
from orrery.kepler import AU_KM

JD = times.jd(2026, 9, 3)


def test_the_sun_is_the_origin_and_stays_there():
    """The frame is heliocentric, which is what lets it need nothing external.

    Not the barycentre: finding that would mean knowing every mass and every
    position, which is the thing being avoided.
    """
    positions, velocities = model.states(("sun",), JD + np.arange(5.0))
    assert np.all(positions == 0.0)
    assert np.all(velocities == 0.0)


def test_the_moon_hangs_off_the_geocentre_exactly():
    """Whatever else is going on, ``moon - geocentre`` is the lunar theory.

    Both are built from the same barycentre, so this is the one relation that
    must hold to machine precision rather than to some tolerance.
    """
    jd = JD + np.arange(10.0)
    positions, _ = model.states(("geocentre", "moon"), jd)
    offset = positions[:, 1, :] - positions[:, 0, :]
    assert np.abs(offset - lunar.position(jd)).max() < 1e-15


def test_the_barycentre_sits_between_them_where_the_masses_say():
    """The elements describe the Earth-Moon barycentre, not the Earth.

    It is 4671 km from the Earth's centre -- three quarters of an Earth radius,
    and inside the planet, which is why it is so easy to forget it is not the
    planet. Forgetting it would move an eclipse track by that much.
    """
    jd = JD + np.arange(10.0)
    positions, _ = model.states(("geocentre", "moon", "embary"), jd)
    geocentre, moon, embary = positions[:, 0], positions[:, 1], positions[:, 2]

    weighted = (1.0 - model.MOON_SHARE) * geocentre + model.MOON_SHARE * moon
    assert np.abs(weighted - embary).max() * AU_KM < 1e-6

    wobble = np.linalg.norm(embary - geocentre, axis=-1) * AU_KM
    assert 4500.0 < wobble.mean() < 4800.0


def test_the_frame_is_equatorial_not_ecliptic():
    """``kepler`` works in the ecliptic; everything downstream of ``apparent``
    is on ICRF axes. Handing over the wrong one tilts the answer by 23.4
    degrees, which is not subtle but is invisible in a scalar distance."""
    positions, _ = model.states(("mars",), JD)
    ecliptic = kepler.position("mars", JD)
    assert np.abs(positions[0, 0] - frames.ecliptic_to_equatorial(ecliptic)).max() < 1e-15
    assert float(frames.separation_arcsec(positions[0, 0], ecliptic)) > 3600.0


def _velocity_error(bodies):
    """Fraction by which the reported velocity misses a differenced position."""
    h = 1e-3
    _, velocities = model.states(bodies, JD)
    ahead, _ = model.states(bodies, JD + h)
    behind, _ = model.states(bodies, JD - h)

    numerical = (ahead - behind) / (2.0 * h)
    error = np.linalg.norm(velocities - numerical, axis=-1)[0]
    return error / np.linalg.norm(velocities, axis=-1)[0]


def test_the_earth_velocity_is_the_one_aberration_gets_told_about():
    """A part in 1e5 on the geocentre, including the Moon's tug on it.

    This is the velocity that matters: ``Ephemeris.observer`` hands it to the
    aberration term, where 30 km/s buys 20 arcsec of tilt. A part in 1e5 of it
    is 0.0002 arcsec, comfortably under the 0.011 arcsec this frame already
    concedes by putting the Sun at the origin.
    """
    assert np.all(_velocity_error(("geocentre", "moon")) < 1e-5)


def test_the_planet_velocities_are_keplers_two_body_ones():
    """A part in 1e4, and that is not this module's doing.

    ``kepler.state`` returns the velocity of the ellipse *through* the point,
    ignoring the century drift of the elements themselves. Its docstring says
    so. Nothing here reads a planet's velocity for anything but aberration on a
    body nobody is standing on, so it is inherited rather than improved.
    """
    error = _velocity_error(("jupiter", "mars"))
    assert np.all(error < 1e-3)
    assert error.max() > 1e-5  # and it really is the drift, not a copy of above


def test_names_are_resolved_the_way_the_rest_of_the_package_resolves_them():
    assert model._key("Earth") == "embary"  # the table's body, not the planet
    assert model._key("geocenter") == "geocentre"
    assert model._key("MOON") == "moon"


# --- the shape the ephemeris comes out in -----------------------------------


def test_the_ephemeris_interpolates_back_to_the_orbits_it_was_built_from():
    """Between grid points, not just on them -- the light-time solver only ever
    asks between them."""
    jd = JD + np.array([0.3, 1.7, 2.45])
    eph = model.ephemeris(jd, bodies=("sun", "geocentre", "moon"))

    direct, _ = model.states(("geocentre", "moon"), jd)
    for i, body in enumerate(("geocentre", "moon")):
        drift = np.linalg.norm(eph.at(body)(jd) - direct[:, i, :], axis=-1) * AU_KM
        assert drift.max() < 0.5  # km, against a quarter-day cubic


def test_the_grid_reaches_past_the_dates_asked_for():
    """The solver asks for positions minutes before the dates it was handed; a
    grid that stopped at them would extrapolate at both ends."""
    eph = model.ephemeris(JD, pad=3.0)
    assert eph.jd_grid.min() <= JD - 3.0
    assert eph.jd_grid.max() >= JD + 3.0


def test_it_can_be_looked_through_like_the_de440_one():
    """The point of the module: same class, same call, no download."""
    jd = JD + np.arange(0.0, 0.5, 0.1)
    eph = model.ephemeris(jd, bodies=("sun", "geocentre", "moon", "mars"))

    sight = eph.look("mars", jd)
    assert sight.distance.shape == jd.shape
    assert np.all(sight.light_time_days > 0.0)
    assert np.all(np.isfinite(sight.apparent))


def test_looking_at_the_sun_from_the_sun_frame_does_not_divide_by_zero():
    """The Sun is exactly at the origin here, so ``target - sun`` is exactly
    zero and the deflection formula's unit vector is 0/0. Such a ray is
    occulted and its correction is discarded, but the division still happens,
    and a nan warning in a clean run is how real ones get ignored."""
    jd = JD + np.arange(0.0, 0.5, 0.1)
    eph = model.ephemeris(jd, bodies=("sun", "geocentre"))

    with np.errstate(all="raise"):
        sight = eph.look("sun", jd)
    assert np.all(np.isfinite(sight.apparent))
    assert np.allclose(sight.distance, 1.0, atol=0.02)  # an au, near enough


# --- and what it costs, measured the way everything else here is -------------


def test_the_2027_eclipse_comes_out_the_same_as_de440s():
    """The README's recipe, gated.

    The claim in the prose is that swapping ``ephemeris`` for
    ``truth.sampled_ephemeris`` moves the answer by a few hundredths of a
    percent and no minutes at all. This is that claim, on a one-day window so
    the fixture stays at four kilobytes, and it is the only measurement of
    ``model.py`` against anything outside itself.

    Everything is right at once here or not at all: the Moon's own theory, the
    Earth-Moon split, light-time, the observer on the ellipsoid, delta T.
    """
    from orrery import Site, ephemeris, events, eclipse, observer, truth

    site = Site("Cairo", 30.044, 31.236, 23.0)
    scan = times.jd(2027, 8, 2) + np.arange(0.0, 1.0, 1.0 / 1440.0)
    bodies = ("sun", "geocentre", "moon")

    answers = {}
    for label, eph in (
        ("ours", ephemeris(scan, bodies=bodies)),
        ("de440", truth.sampled_ephemeris(bodies, scan, allow_download=False)),
    ):
        view = eclipse.solar_view(eph, scan, site=site)
        (answers[label],) = events.find_extrema(
            scan, view.obscuration, kind="max", threshold=0.0
        )

    (our_time, our_depth) = answers["ours"]
    (their_time, their_depth) = answers["de440"]

    assert abs(our_time - their_time) * 1440.0 < 0.5      # minutes
    assert abs(our_depth - their_depth) < 0.001           # fraction of the disc
    assert 0.94 < our_depth < 0.95                        # and both are the real eclipse
