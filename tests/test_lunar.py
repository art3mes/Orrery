"""The lunar theory, against the one worked example and against itself.

``validate_m6.py`` measures it against DE440. These check the transcription and
the structure, offline, so a broken coefficient says so in a second rather than
after a hundred-year comparison.
"""

import numpy as np
import pytest

from orrery import frames, lunar, times

# Meeus, chapter 47, example 47.a: 1992 April 12.0 TD.
MEEUS_JD = 2448724.5


def test_the_fundamental_arguments_match_the_worked_example():
    angles = lunar.arguments(MEEUS_JD)
    for name, published in (
        ("L", 134.290182),
        ("D", 113.842304),
        ("M", 97.643514),
        ("Mp", 5.150833),
        ("F", 219.889721),
    ):
        assert float(np.atleast_1d(angles[name])[0]) == pytest.approx(
            published, abs=1e-6
        ), name


def test_the_position_matches_the_worked_example():
    """120 coefficients either land on six decimals or they do not."""
    longitude, latitude, distance = lunar.spherical(MEEUS_JD)
    assert float(longitude) == pytest.approx(133.162655, abs=1e-6)
    assert float(latitude) == pytest.approx(-3.229126, abs=1e-6)
    assert float(distance) == pytest.approx(368409.7, abs=0.05)


def test_the_eccentricity_factor_is_applied():
    """Terms in the Sun's anomaly scale with the Earth's eccentricity.

    E is 1.000194 in 1992 and drifts by a part in 400 a century. Dropping it
    would leave the worked example right at J2000 and slowly wrong either side.
    """
    assert float(lunar.arguments(MEEUS_JD)["E"]) == pytest.approx(1.000194, abs=1e-6)
    assert float(lunar.arguments(times.J2000)["E"]) == pytest.approx(1.0, abs=1e-9)
    assert float(lunar.arguments(times.J2000 + 36525.0)["E"]) < 1.0


# --- structure --------------------------------------------------------------


def test_the_largest_term_is_six_degrees():
    """The Moon's equation of the centre, and why it needs sixty terms.

    The Sun's largest periodic term is 1.9 degrees. Nothing else in the solar
    system is pulled about like this.
    """
    largest = max(row[4] for row in lunar._LONGITUDE_DISTANCE)
    assert largest / 1e6 == pytest.approx(6.289, abs=0.01)


def test_the_tables_are_the_right_size():
    assert len(lunar._LONGITUDE_DISTANCE) == 60
    assert len(lunar._LATITUDE) == 60
    assert all(len(row) == 6 for row in lunar._LONGITUDE_DISTANCE)
    assert all(len(row) == 5 for row in lunar._LATITUDE)


def test_latitude_terms_all_involve_the_argument_of_latitude():
    """F is what takes the Moon off the ecliptic, so every term must carry it.

    A latitude term with F = 0 would be a constant offset from the ecliptic,
    which is not a thing an orbit through the origin can have.
    """
    assert all(row[3] != 0 for row in lunar._LATITUDE)


def test_longitude_terms_never_involve_odd_multiples_of_F():
    """And the mirror of it: longitude and distance are symmetric about the
    ecliptic, so they can only contain F in even multiples."""
    assert all(row[3] % 2 == 0 for row in lunar._LONGITUDE_DISTANCE)


def test_the_distance_never_leaves_its_real_range():
    jd = times.J2000 + np.arange(0, 36525, 0.5)
    _, _, distance = lunar.spherical(jd)
    assert distance.min() == pytest.approx(356500, abs=1500)
    assert distance.max() == pytest.approx(406700, abs=1500)


def test_the_latitude_stays_within_the_inclination():
    """The Moon's orbit is tilted 5.15 degrees, and the ecliptic latitude
    swings a little further because the node moves."""
    jd = times.J2000 + np.arange(0, 36525, 0.5)
    _, latitude, _ = lunar.spherical(jd)
    assert 5.0 < np.abs(latitude).max() < 5.4


# --- the periods, out of the rates ------------------------------------------


@pytest.mark.parametrize(
    "argument,days",
    [("L", 27.321582), ("D", 29.530589), ("Mp", 27.554550), ("F", 27.212221)],
)
def test_each_argument_turns_at_its_own_month(argument, days):
    """Tropical, synodic, anomalistic and draconic, from four rates.

    ``L`` gives the *tropical* month, not the sidereal one: the mean longitude
    is measured from the equinox, and the equinox moves. Seven seconds apart.
    """
    start = lunar.arguments(times.J2000)[argument]
    later = lunar.arguments(times.J2000 + days)[argument]
    assert float((later - start + 180.0) % 360.0 - 180.0) == pytest.approx(0.0, abs=1e-3)


def test_the_nodes_go_backwards_and_the_apsides_forwards():
    """18.6 years and 8.85 years, and they run opposite ways."""
    jd = times.J2000 + np.arange(0.0, 8000.0, 1.0)
    angles = lunar.arguments(jd)

    node = np.unwrap(np.radians(angles["L"] - angles["F"]))
    perigee = np.unwrap(np.radians(angles["L"] - angles["Mp"]))

    node_rate = np.polyfit(jd, node, 1)[0]  # radians per day
    perigee_rate = np.polyfit(jd, perigee, 1)[0]

    assert node_rate < 0 < perigee_rate
    assert 2 * np.pi / abs(node_rate) / 365.25 == pytest.approx(18.6, abs=0.1)
    assert 2 * np.pi / perigee_rate / 365.25 == pytest.approx(8.85, abs=0.1)


# --- the frame the answer comes back in -------------------------------------


def test_the_position_is_the_spherical_answer():
    from orrery.kepler import AU_KM

    jd = times.jd(2026, 9, 3)
    _, _, distance = lunar.spherical(jd)
    assert np.linalg.norm(lunar.position(jd)) * AU_KM == pytest.approx(
        float(distance), rel=1e-12
    )


def test_the_answer_is_un_precessed_to_j2000():
    """The theory works in the equinox of date; everything else here is J2000.

    Skipping the rotation back would drift the Moon by the whole of precession,
    5000 arcsec a century -- five hundred times the error of the theory itself.
    """
    from orrery.precession import _apply, precession_matrix

    jd = times.jd(2049, 1, 1)
    ours = lunar.position(jd)
    of_date = _apply(precession_matrix(jd), ours)

    drift = frames.separation_arcsec(ours, of_date)
    assert 2000.0 < float(drift) < 3000.0  # half a century of precession


def test_it_broadcasts_over_dates():
    jd = times.jd(2026, 9, 3) + np.arange(5.0)
    assert lunar.position(jd).shape == (5, 3)
    assert lunar.position(jd[0]).shape == (3,)
