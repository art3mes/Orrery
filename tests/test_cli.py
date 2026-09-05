"""The one command.

``orrery <date>`` is the whole application for anyone who has just cloned this:
a date in, everything about that date out, then the 3-D view. It has to be fast
enough to feel like a lookup rather than a computation, and it has to be right
about the things a person can check by looking up -- the phase of the Moon, and
whether there was an eclipse.
"""

import io

import numpy as np
import pytest

from orrery import __main__ as cli
from orrery import observer, times


def run(*argv) -> str:
    out = io.StringIO()
    jd = cli.parse_date(argv[0])
    site = observer.SITES[argv[1]] if len(argv) > 1 else None
    cli.report(jd, site, out=out)
    return out.getvalue()


# --- dates ------------------------------------------------------------------


def test_it_reads_a_date_and_an_optional_clock():
    assert cli.parse_date("2027-08-02") == times.jd(2027, 8, 2)
    assert cli.parse_date("2027-08-02 10:07") == times.jd(2027, 8, 2, 10, 7)
    assert cli.parse_date("2027-08-02 10") == times.jd(2027, 8, 2, 10)


def test_a_date_outside_the_table_is_refused_rather_than_extrapolated():
    with pytest.raises(SystemExit):
        cli.main(["1500-01-01", "--no-viewer"])


# --- the phase of the Moon --------------------------------------------------


@pytest.mark.parametrize(
    "date,phase",
    [
        ("2026-09-11", "new"),             # 03:29 by the crossing finder
        ("2026-09-19", "first quarter"),
        ("2026-09-26", "waxing gibbous"),  # full at 16:50, so still waxing at 00:00
        ("2026-10-03", "waning gibbous"),
    ],
)
def test_the_moon_gets_its_phase_named_right(date, phase):
    assert phase in run(date)


def test_the_lit_fraction_cannot_name_a_phase_on_its_own():
    """Which is why the name comes from the elongation instead.

    Six days before new and six days after are both crescents of about the same
    brightness, a fortnight apart. A namer keyed on the lit fraction calls them
    the same thing, and calls a 33% waxing crescent 'gibbous' if its table is
    off. This is that bug, kept as a test.
    """
    before, after = run("2026-09-05"), run("2026-09-17")
    assert "waning crescent" in before
    assert "waxing crescent" in after


# --- what it says about a date with something on it -------------------------


def test_it_notices_the_eclipse_and_says_how_deep_it_is():
    text = run("2027-08-02", "delhi")
    assert "Solar eclipse" in text
    assert "shadow lands at" in text


def test_an_ordinary_date_says_nothing_about_eclipses():
    assert "eclipse" not in run("2026-09-19")


def test_every_body_appears_once_with_a_finite_distance():
    text = run("2026-09-06")
    for name in ("Sun", "Mercury", "Venus", "Mars", "Jupiter", "Saturn",
                 "Uranus", "Neptune", "Pluto", "Moon"):
        assert sum(line.strip().startswith(name) for line in text.splitlines()) == 1
    assert "nan" not in text.lower()


def test_standing_somewhere_changes_the_answer():
    """Parallax on the Moon is about a degree, which no rounding hides."""
    geocentric = run("2026-09-06")
    from_a_place = run("2026-09-06", "paranal")
    assert geocentric != from_a_place
    assert "Paranal" in from_a_place


# --- the flags that reach the viewer ----------------------------------------


def test_focus_accepts_the_names_the_rest_of_the_package_accepts():
    """`--focus earth` has to mean the same body as `position("earth", ...)`.

    Which is the Earth-Moon barycentre, because that is what the element table
    describes. Two naming schemes in one program is one too many.
    """
    from orrery import elements

    assert elements.canonical("earth") == "embary"
    assert elements.canonical("SATURN") == "saturn"
    with pytest.raises(KeyError):
        elements.canonical("vulcan")


def test_the_viewer_takes_a_focus_argument():
    """Wired through, so `orrery --focus saturn` opens on Saturn rather than
    opening on the system and waiting for a click."""
    import inspect

    from orrery.view import Orrery

    assert "focus" in inspect.signature(Orrery.run).parameters
