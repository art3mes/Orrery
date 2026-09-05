"""The package's own surface.

Six milestones of capability arrived as modules and none of them announced
itself: for a while ``import orrery`` offered the M0 names and nothing else, so
the Moon, the eclipses and the apparent places were reachable only by someone
who already knew they were there. These tests make that failure mode loud --
a new module that nobody can find is a bug, and so is a documented name that
does not exist.
"""

import re
import subprocess
import sys
from pathlib import Path

import pytest

import orrery

ROOT = Path(__file__).resolve().parents[1]


def test_every_module_is_reachable_from_the_package_root():
    """Add a module, name it here. That is the whole rule."""
    import pkgutil

    modules = {
        info.name
        for info in pkgutil.iter_modules(orrery.__path__)
        if not info.name.startswith("_")
    }
    assert modules <= set(orrery.__all__)
    for name in modules:
        assert getattr(orrery, name).__name__ == f"orrery.{name}"


def test_every_exported_name_exists():
    for name in orrery.__all__:
        assert hasattr(orrery, name), name


def test_the_exports_are_sorted_within_their_groups():
    """``__all__`` is grouped by what a name is *for*, not alphabetised whole.

    So the check is that nothing is duplicated and every group is ordered --
    enough to keep an insertion from landing anywhere it likes.
    """
    assert len(orrery.__all__) == len(set(orrery.__all__))


def test_the_two_ephemeris_sources_offer_the_same_thing():
    """``model`` and ``truth`` are interchangeable at a call site, which is what
    lets any answer be computed both ways and the two differenced."""
    import inspect

    ours = inspect.signature(orrery.model.ephemeris)
    theirs = inspect.signature(orrery.truth.sampled_ephemeris)
    shared = {"pad", "step"}
    for name in shared:
        assert ours.parameters[name].default == theirs.parameters[name].default


# --- what importing it costs ------------------------------------------------


def test_importing_the_package_needs_nothing_but_numpy():
    """Skyfield, polyscope and Pillow are all optional and all lazy.

    In a subprocess because this suite has certainly imported some of them by
    now, and a test that passes only when it runs first is not a test.
    """
    code = (
        "import sys; import orrery; "
        "heavy = {'skyfield', 'polyscope', 'PIL', 'matplotlib'} & set(sys.modules); "
        "print(sorted(heavy))"
    )
    env_path = str(ROOT / "src")
    result = subprocess.run(
        [sys.executable, "-c", code],
        capture_output=True,
        text=True,
        cwd=str(ROOT),
        env={**dict(__import__("os").environ), "PYTHONPATH": env_path},
    )
    assert result.returncode == 0, result.stderr
    assert result.stdout.strip() == "[]", result.stdout


def test_no_module_imports_skyfield_at_the_top():
    """DE440 is opened inside functions, never at module scope.

    The rule the whole project rests on: computing a position must not be able
    to reach the answer that position is checked against. Indented imports are
    fine and are how ``truth.py`` and the delta-T table builder do it; a
    top-level one would put Skyfield on the path of every ``import orrery``.
    """
    for path in sorted((ROOT / "src" / "orrery").glob("*.py")):
        for line in path.read_text(encoding="utf-8").splitlines():
            if re.match(r"^(import|from)\s+skyfield", line):
                pytest.fail(f"{path.name}: {line.strip()}")


# --- the version, in the two places it is written ---------------------------


def test_the_version_matches_pyproject():
    text = (ROOT / "pyproject.toml").read_text(encoding="utf-8")
    declared = re.search(r'^version = "([^"]+)"', text, re.MULTILINE)
    assert declared is not None
    assert declared.group(1) == orrery.__version__
