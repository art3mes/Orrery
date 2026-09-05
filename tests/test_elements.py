"""The element table: parsing, and whether it was transcribed correctly.

A single wrong digit in ``elements.py`` would move a planet without breaking
anything else -- every conserved quantity would still check out, every orbit
would still close, and the position would still be an ellipse. Only a
comparison against the source catches it, so the network test at the bottom is
doing real work even though it is skipped by default.
"""

import html
import re

import numpy as np
import pytest

from orrery import elements

# JPL retired the standalone p_elem_t1.txt; the table now lives in a <pre>
# block on the page itself.
JPL_URL = "https://ssd.jpl.nasa.gov/planets/approx_pos.html"


def test_every_body_parsed():
    assert set(elements.BODIES) == set(elements._ELEMENTS)
    assert len(elements.BODIES) == 9


@pytest.mark.parametrize("body", elements.ORDER)
def test_elements_have_six_values_and_six_rates(body):
    values, rates = elements._ELEMENTS[body]
    assert values.shape == rates.shape == (6,)
    assert np.all(np.isfinite(values)) and np.all(np.isfinite(rates))


def test_at_the_epoch_the_elements_are_the_table_values():
    values, _ = elements._ELEMENTS["mars"]
    at_epoch = elements.elements_at("mars", elements.J2000)
    assert np.allclose([at_epoch[k] for k in elements.ELEMENT_NAMES], values)


def test_eccentricities_are_elliptical():
    for body in elements.ORDER:
        e = elements._ELEMENTS[body][0][1]
        assert 0.0 <= e < 0.3, body


def test_earth_is_the_earth_moon_barycentre():
    assert elements.canonical("earth") == "embary"
    assert elements.canonical("EM Bary") == "embary"


def test_unknown_body_is_an_error():
    with pytest.raises(KeyError, match="unknown body"):
        elements.canonical("planet nine")


def test_dates_outside_the_table_range_warn():
    with pytest.warns(RuntimeWarning, match="1800-2050"):
        elements.elements_at("venus", elements.J2000 + 200 * 36525.0)


def test_rates_are_per_century_not_per_year():
    """Earth's mean longitude advances ~36000 degrees per century, not ~360.

    Mixing the two is the single easiest way to build a solar system that runs
    a hundred times too slow and still looks completely convincing.
    """
    _, rates = elements._ELEMENTS["embary"]
    assert 35900 < rates[3] < 36100


def _strip_rules(text: str) -> str:
    """Drop blank lines and horizontal rules, keeping the data lines."""
    keep = [
        line
        for line in text.splitlines()
        if line.strip() and set(line.strip()) != {"-"}
    ]
    return "\n".join(keep)


@pytest.mark.network
def test_table_matches_the_jpl_source():
    """Diff the embedded table, digit for digit, against JPL's published page."""
    import urllib.error
    import urllib.request

    from orrery.truth import _use_system_trust_store

    _use_system_trust_store()
    try:
        with urllib.request.urlopen(JPL_URL, timeout=30) as response:
            page = response.read().decode("utf-8", "replace")
    except (urllib.error.URLError, TimeoutError, OSError) as exc:
        pytest.skip(f"JPL page unreachable: {exc}")

    blocks = [
        html.unescape(re.sub(r"<[^>]+>", "", block))
        for block in re.findall(r"<pre[^>]*>(.*?)</pre>", page, re.S)
    ]
    # The page carries three tables; the first is the 1800-2050 one this
    # package uses. The others are the 3000 BC - 3000 AD variant and its extra
    # correction terms, which need a different model and are not embedded here.
    table_1 = next((b for b in blocks if "Mercury" in b and "long.peri." in b), None)
    if table_1 is None:
        pytest.skip("could not locate the element table on the JPL page")

    # JPL now publishes table 1 for the eight planets only.
    expected = set(elements.BODIES) - {"pluto"}
    published = elements._parse(_strip_rules(table_1), require=expected)

    assert set(published) == expected, (
        "the set of bodies JPL publishes in table 1 has changed; if Pluto is "
        "back, drop it from `expected` so it gets checked too"
    )

    for body in sorted(expected):
        ours_values, ours_rates = elements._ELEMENTS[body]
        their_values, their_rates = published[body]
        assert np.array_equal(ours_values, their_values), f"{body} elements differ"
        assert np.array_equal(ours_rates, their_rates), f"{body} rates differ"
