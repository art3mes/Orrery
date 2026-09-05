"""The README's recipes, run as written.

Documentation rots quietly: a rename lands, the prose keeps the old call, and
nobody finds out until a stranger pastes it into a terminal. These blocks are
the first thing anyone will try, so they are executed verbatim and their printed
output is compared character for character with what the README claims.

If this fails after a deliberate change, run the block and paste the new output
into the README. That is the intended workflow -- the numbers in the prose are
measurements, and measurements move.
"""

import os
import re
import subprocess
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
SECTION = "## Using it as a library"
NEXT = "## How accurate is it"


def recipes():
    text = (ROOT / "README.md").read_text(encoding="utf-8")
    section = text[text.index(SECTION) : text.index(NEXT)]
    pairs = re.findall(r"```python\n(.*?)```\n\n```\n(.*?)```", section, re.S)
    assert len(pairs) == 5, f"expected five recipes, found {len(pairs)}"
    return pairs


@pytest.mark.parametrize("code,expected", recipes(), ids=lambda v: None)
def test_the_readme_recipe_prints_what_the_readme_says(code, expected):
    """Verbatim, from a subprocess, with only src on the path.

    A subprocess because a recipe importing something the suite already imported
    would pass here and fail for a reader -- and because the recipes are the one
    place the package is used the way an outsider uses it.
    """
    result = subprocess.run(
        [sys.executable, "-c", code],
        capture_output=True,
        text=True,
        cwd=str(ROOT),
        env={**os.environ, "PYTHONPATH": str(ROOT / "src")},
    )
    assert result.returncode == 0, result.stderr
    assert result.stdout == expected


# --- the figures ------------------------------------------------------------

MARKDOWN = ("README.md", "docs/milestones.md", "docs/notes.md", "data/README.md")


def _figures_referenced():
    used = set()
    for name in MARKDOWN:
        text = (ROOT / name).read_text(encoding="utf-8")
        for link in re.findall(r"!\[[^\]]*\]\(([^)]+)\)", text):
            if not link.startswith("http"):
                used.add((ROOT / name).parent.joinpath(link).resolve())
    return used


def test_every_figure_referenced_exists():
    for path in sorted(_figures_referenced()):
        assert path.exists(), path


def test_no_figure_sits_in_the_repository_unused():
    """Eleven of sixteen were orphaned once, several of them stale renders from
    before the viewer could draw rings. An image nobody links to is either a
    deletion someone forgot or a section someone forgot -- both worth knowing.
    """
    on_disk = {p.resolve() for p in (ROOT / "docs" / "images").glob("*.png")}
    orphans = sorted(p.name for p in on_disk - _figures_referenced())
    assert not orphans, f"unreferenced: {orphans}"


def test_no_markdown_link_points_at_a_missing_file():
    for name in MARKDOWN:
        path = ROOT / name
        text = path.read_text(encoding="utf-8")
        for link in re.findall(r"\]\(([^)#][^)]*)\)", text):
            if link.startswith("http"):
                continue
            assert (path.parent / link).exists(), f"{name} -> {link}"
