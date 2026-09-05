"""The Kepler solver, checked by putting its answer back into the equation.

No reference values needed: ``E - e sin(E) - M`` is the definition, so the
residual is a complete test of correctness. What is left to check is that it
converges everywhere it claims to, and fails loudly where it does not.
"""

import numpy as np
import pytest

from orrery.kepler import solve_kepler


def wrap(x):
    return (x + np.pi) % (2 * np.pi) - np.pi


@pytest.mark.parametrize("e", [0.0, 0.0068, 0.0934, 0.2056, 0.2488, 0.5, 0.9])
def test_residual_is_machine_zero(e):
    M = np.linspace(-4 * np.pi, 4 * np.pi, 2001)
    E = solve_kepler(M, e)
    residual = E - e * np.sin(E) - wrap(M)
    assert np.max(np.abs(residual)) < 1e-12


def test_circular_orbit_is_the_identity():
    """With e = 0 the equation collapses to E = M, up to the [-pi, pi) wrap."""
    M = np.linspace(-3 * np.pi, 3 * np.pi, 101)
    assert np.allclose(solve_kepler(M, 0.0), wrap(M), atol=1e-14)


def test_broadcasts_bodies_against_times():
    M = np.linspace(0, 6, 7)[:, None]  # 7 times
    e = np.array([0.01, 0.1, 0.25])  # 3 bodies
    E = solve_kepler(M, e)
    assert E.shape == (7, 3)
    assert np.max(np.abs(E - e * np.sin(E) - wrap(M))) < 1e-12


def test_solution_is_monotonic_in_M():
    """E increases with M. A solver landing on the wrong branch would not."""
    M = np.linspace(-np.pi + 1e-9, np.pi - 1e-9, 500)
    E = solve_kepler(M, 0.2)
    assert np.all(np.diff(E) > 0)


@pytest.mark.parametrize("e", [-0.1, 1.0, 1.5])
def test_rejects_non_elliptical(e):
    with pytest.raises(ValueError):
        solve_kepler(0.5, e)


def test_reports_failure_rather_than_returning_junk():
    with pytest.raises(RuntimeError, match="converge"):
        solve_kepler(np.array([1.0]), np.array([0.999]), tol=1e-16, max_iter=2)
