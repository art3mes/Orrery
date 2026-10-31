"""What the renderer actually draws, read back from its pixels.

Everything else in this suite checks arrays. That is not enough on its own:
``uv_sphere`` and ``globe.sample`` agreed perfectly while polyscope drew every
map upside down. This opens a real window for a moment, so it needs polyscope
and a display, and it skips without them, which includes CI.
"""

import pytest

from orrery import view


def test_every_corner_of_the_map_lands_where_it_belongs():
    quadrants = view.rendered_map_quadrants()
    if quadrants is None:
        pytest.skip("nothing can draw here: needs polyscope and a display")
    assert len(quadrants) == 4
    assert quadrants == {corner: corner for corner in quadrants}
