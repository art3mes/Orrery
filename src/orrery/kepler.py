"""Kepler's equation, and elements -> position/velocity.

A planet moves on an ellipse with the Sun at one focus, and it moves fast near
the Sun and slow far away. So "what fraction of the orbital period has elapsed"
is *not* "what fraction of the way round the ellipse the planet has gone", and
converting between the two is the only genuinely awkward step in the whole
package::

    M = E - e sin(E)

*M* (the mean anomaly) is proportional to elapsed time and is trivial to
compute. *E* (the eccentric anomaly) is what fixes the planet's actual position.
No rearrangement gives E in closed form, so it is solved numerically -- Newton's
method, a handful of iterations, :func:`solve_kepler` below.

Everything after that is rotating a 2-D ellipse into 3-D by three angles.
"""

from __future__ import annotations

import numpy as np

from .elements import elements_at

# Gaussian gravitational constant, au^(3/2) / day. GM_SUN is its square.
GAUSS_K = 0.01720209895
GM_SUN = GAUSS_K**2  # au^3 / day^2

# IAU value of the astronomical unit.
AU_KM = 149597870.7


def solve_kepler(
    M: np.ndarray,
    e: np.ndarray,
    *,
    tol: float = 1e-13,
    max_iter: int = 60,
) -> np.ndarray:
    """Solve ``M = E - e sin(E)`` for the eccentric anomaly *E*.

    Parameters
    ----------
    M, e
        Mean anomaly in **radians** and eccentricity. Broadcast against each
        other.
    tol
        Convergence threshold on the residual ``E - e sin(E) - M``, in radians.
        1e-13 rad is ~2e-8 arcsec: far below anything else in the error budget,
        and still reached in a handful of iterations.

    Returns
    -------
    E, in radians, in the same branch as the wrapped *M* (i.e. in [-pi, pi]).

    Raises
    ------
    RuntimeError
        If any element fails to converge. Silence here would mean a planet
        placed at a plausible but wrong point on its orbit, so this is loud.
    """
    M = np.asarray(M, dtype=float)
    e = np.asarray(e, dtype=float)
    if np.any(e < 0) or np.any(e >= 1):
        raise ValueError("solve_kepler handles elliptical orbits only (0 <= e < 1)")

    # Wrap to [-pi, pi). Kepler's equation is periodic in M, and starting near
    # zero keeps Newton well behaved.
    M = (M + np.pi) % (2 * np.pi) - np.pi

    E = M + e * np.sin(M)  # first-order starting guess
    E = np.broadcast_arrays(E, e)[0].astype(float, copy=True)

    for _ in range(max_iter):
        residual = E - e * np.sin(E) - M
        if np.max(np.abs(residual)) < tol:
            return E
        # Clip the Newton step. The unclipped step is fine for planetary
        # eccentricities but can overshoot badly for e near 1, and the clip
        # costs nothing when it never binds.
        step = residual / (1.0 - e * np.cos(E))
        E = E - np.clip(step, -1.0, 1.0)

    residual = np.max(np.abs(E - e * np.sin(E) - M))
    raise RuntimeError(
        f"Kepler solver failed to converge: residual {residual:.3e} rad "
        f"after {max_iter} iterations (max e = {np.max(e):.3f})"
    )


def _perifocal_to_ecliptic(
    x: np.ndarray,
    y: np.ndarray,
    omega: np.ndarray,
    inc: np.ndarray,
    node: np.ndarray,
) -> np.ndarray:
    """Rotate in-plane coordinates to the J2000 ecliptic frame.

    ``omega`` (argument of perihelion), ``inc`` (inclination) and ``node``
    (longitude of ascending node) are in radians. Composition is
    ``Rz(node) Rx(inc) Rz(omega)``, applied to a vector with zero z.
    """
    cw, sw = np.cos(omega), np.sin(omega)
    ci, si = np.cos(inc), np.sin(inc)
    cn, sn = np.cos(node), np.sin(node)

    X = (cw * cn - sw * sn * ci) * x + (-sw * cn - cw * sn * ci) * y
    Y = (cw * sn + sw * cn * ci) * x + (-sw * sn + cw * cn * ci) * y
    Z = (sw * si) * x + (cw * si) * y
    return np.stack([X, Y, Z], axis=-1)


def state(body: str, jd_tdb) -> tuple[np.ndarray, np.ndarray]:
    """Heliocentric state of *body* at Julian date(s) *jd_tdb* (TDB).

    Returns ``(position, velocity)``, both shaped ``(..., 3)`` with the leading
    axes of *jd_tdb*, in the **mean ecliptic and equinox of J2000**. Position is
    in au, velocity in au/day.

    The velocity is that of the two-body ellipse through the point -- it uses
    GM of the Sun alone and ignores the century drift of the elements. That is
    the right thing for seeding an N-body run (M2) and roughly a part in 1e4 off
    from a numerically differentiated position; do not read more into it.
    """
    el = elements_at(body, jd_tdb)
    a = el["a"]
    e = el["e"]
    inc = np.radians(el["I"])
    node = np.radians(el["long_node"])
    peri = np.radians(el["long_peri"])

    omega = peri - node  # argument of perihelion
    M = np.radians(el["L"]) - peri
    E = solve_kepler(M, e)

    cosE, sinE = np.cos(E), np.sin(E)
    root = np.sqrt(1.0 - e * e)

    # In-plane coordinates, perihelion on the +x axis.
    x = a * (cosE - e)
    y = a * root * sinE

    # dE/dt from differentiating Kepler's equation: n = (1 - e cos E) dE/dt.
    n = np.sqrt(GM_SUN / a**3)  # mean motion, rad/day
    Edot = n / (1.0 - e * cosE)
    vx = -a * sinE * Edot
    vy = a * root * cosE * Edot

    pos = _perifocal_to_ecliptic(x, y, omega, inc, node)
    vel = _perifocal_to_ecliptic(vx, vy, omega, inc, node)
    return pos, vel


def position(body: str, jd_tdb) -> np.ndarray:
    """Heliocentric J2000-ecliptic position in au. See :func:`state`."""
    return state(body, jd_tdb)[0]


def ellipse(body: str, jd_tdb: float, samples: int = 512) -> np.ndarray:
    """The orbit as a closed ring of ``samples`` points, shaped ``(samples, 3)``.

    This is the *osculating* ellipse: the elements are evaluated once, at
    *jd_tdb*, and then the eccentric anomaly is swept right round. It is a
    geometric object, not a trajectory.

    That distinction is what makes the outer planets drawable at all. Sampling
    Neptune's actual path over one period would run 165 years past the date
    asked for, straight out of the element table's 1800-2050 window and into
    extrapolated nonsense. Sweeping E instead stays at one instant, so the ring
    is exactly as valid as the position is.

    The ring drifts slowly with the elements, so it is the orbit *now*, not a
    fixed track the planet returns to. Over the table's two centuries that drift
    is small enough to be invisible at solar-system zoom, and real.
    """
    if np.ndim(jd_tdb) != 0:
        raise ValueError("ellipse() draws one orbit at one instant; pass a scalar date")
    if samples < 3:
        raise ValueError("an ellipse needs at least 3 samples")

    el = elements_at(body, jd_tdb)
    a, e = float(el["a"]), float(el["e"])
    inc = np.radians(float(el["I"]))
    node = np.radians(float(el["long_node"]))
    omega = np.radians(float(el["long_peri"])) - node

    # endpoint=False: the loop closes by wrapping, so a duplicated final point
    # would only add a zero-length edge.
    E = np.linspace(0.0, 2 * np.pi, samples, endpoint=False)
    x = a * (np.cos(E) - e)
    y = a * np.sqrt(1.0 - e * e) * np.sin(E)
    return _perifocal_to_ecliptic(x, y, omega, inc, node)


def elements_from_state(
    r: np.ndarray, v: np.ndarray, mu: float
) -> dict[str, np.ndarray]:
    """Osculating elements of a state vector. The inverse of :func:`state`.

    *r* and *v* are position and velocity **relative to the central body**,
    shaped ``(..., 3)``, and *mu* is ``G(M + m)``. Angles come back in degrees,
    referred to whatever frame the vectors were given in.

    ``long_peri`` is the longitude of perihelion, ``Omega + omega``. Those two
    angles are measured in different planes, which makes it a bent angle rather
    than a real one -- but it is the quantity the JPL element table tabulates,
    the quantity Mercury's 43 arcsec per century refers to, and it stays
    well-defined as the inclination goes to zero, where ``omega`` alone does not.
    """
    r = np.asarray(r, dtype=float)
    v = np.asarray(v, dtype=float)

    h = np.cross(r, v)
    r_len = np.linalg.norm(r, axis=-1)
    v2 = np.sum(v * v, axis=-1)
    rv = np.sum(r * v, axis=-1)

    # The Laplace-Runge-Lenz vector, pointing from the focus to perihelion. Its
    # slow rotation is exactly what M2 measures.
    e_vec = ((v2 - mu / r_len)[..., None] * r - rv[..., None] * v) / mu
    e = np.linalg.norm(e_vec, axis=-1)

    inc = np.arccos(np.clip(h[..., 2] / np.linalg.norm(h, axis=-1), -1.0, 1.0))

    # Node vector n = zhat x h = (-h_y, h_x, 0).
    node = np.arctan2(h[..., 0], -h[..., 1]) % (2 * np.pi)

    # Argument of perihelion: the angle from the node to perihelion, in the
    # orbital plane. Computed with atan2 of components resolved along and across
    # the node direction, which stays accurate at small e where the arccos form
    # loses digits.
    cos_node, sin_node = np.cos(node), np.sin(node)
    along = e_vec[..., 0] * cos_node + e_vec[..., 1] * sin_node
    cos_inc = np.cos(inc)
    across = (
        (e_vec[..., 1] * cos_node - e_vec[..., 0] * sin_node) * cos_inc
        + e_vec[..., 2] * np.sin(inc)
    )
    argp = np.arctan2(across, along) % (2 * np.pi)

    return {
        "a": 1.0 / (2.0 / r_len - v2 / mu),
        "e": e,
        "I": np.degrees(inc),
        "long_node": np.degrees(node),
        "arg_peri": np.degrees(argp),
        "long_peri": np.degrees(node + argp) % 360.0,
    }


def period(body: str, jd_tdb=None) -> float:
    """Orbital period in days, from the semi-major axis alone.

    Uses GM of the Sun only, so this is the period of a massless test particle;
    for the Earth-Moon barycentre that is ~0.002 d longer than the true sidereal
    year.
    """
    from .elements import J2000

    a = np.asarray(elements_at(body, J2000 if jd_tdb is None else jd_tdb)["a"])
    return float(2 * np.pi * np.sqrt(a**3 / GM_SUN))
