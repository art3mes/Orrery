"""Real gravity: every body pulling on every other, integrated forward.

M0 and M1 put each planet on its own fixed ellipse, drifting slowly because JPL
fitted the drift. Nothing in them knows that Jupiter tugs on Saturn. This module
throws the ellipses away and integrates Newton's law directly.

Two choices carry the whole module.

**Barycentric, with the Sun as a body.** The obvious thing is to keep the Sun at
the origin and integrate the planets around it. That frame accelerates -- the
Sun is pulled about by Jupiter -- and a symplectic integrator in a
non-inertial frame is no longer symplectic, which quietly gives back the energy
behaviour that was the whole reason for choosing one. So the Sun is body zero,
the origin is the solar system barycentre, and the frame is inertial.

**Symplectic, not merely accurate.** A Runge-Kutta step is more accurate than a
leapfrog step of the same size, and after a few centuries it is far worse,
because its error in energy accumulates in one direction and the planets spiral.
Symplectic integrators conserve a slightly wrong energy *exactly*, so the error
oscillates and never grows. ``demo_m2.py`` shows the two curves side by side;
the difference is the difference between a model that survives a thousand years
and one that does not.

Positions and velocities are in au and au/day, in whatever frame they were
handed in. Masses are carried as GM in au^3/day^2, because that is what the
equations actually use and because the mass of the Sun in kilograms is known to
far fewer digits than GM of the Sun is.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .kepler import GM_SUN

# Speed of light, au/day. Exact, since the metre is defined from it and the au
# is now a defined number of metres.
C_AU_PER_DAY = 299792.458 * 86400.0 / 149597870.7

# Sun-to-body mass ratios, from the DE440 header. Planet values are for the
# whole system -- planet plus its moons -- which is what a barycentre orbits and
# what the JPL element table tracks.
#
# These are the one set of numbers in the package with no independent check
# inside it: they are not derivable from anything else here. What validates them
# is gate 2, the hundred-year drift against DE440, which a mass wrong by a part
# in a thousand would visibly spoil.
SUN_OVER_BODY = {
    "mercury": 6023600.0,
    "venus": 408523.71,
    "embary": 328900.5614,
    "mars": 3098708.0,
    "jupiter": 1047.3486,
    "saturn": 3497.898,
    "uranus": 22902.98,
    "neptune": 19412.24,
    "pluto": 135200000.0,
}

GM = {"sun": GM_SUN} | {
    body: GM_SUN / ratio for body, ratio in SUN_OVER_BODY.items()
}

# The Sun first, then outward. Order matters only in that it fixes the row order
# of every array in this module.
DEFAULT_BODIES = ("sun",) + tuple(SUN_OVER_BODY)


def gm_vector(bodies) -> np.ndarray:
    return np.array([GM[b] for b in bodies], dtype=float)


# --- forces -----------------------------------------------------------------


def accelerations(pos: np.ndarray, gm: np.ndarray) -> np.ndarray:
    """Newtonian acceleration on each body, shaped like *pos*.

    ``a_i = sum_j GM_j (r_j - r_i) / |r_j - r_i|^3``, with the self-term removed
    by setting the diagonal separation to infinity rather than by masking, which
    keeps the whole thing one array expression.
    """
    delta = pos[None, :, :] - pos[:, None, :]
    r2 = np.einsum("ijk,ijk->ij", delta, delta)
    np.fill_diagonal(r2, np.inf)
    return np.einsum("j,ij,ijk->ik", gm, r2**-1.5, delta)


def relativistic_correction(
    pos: np.ndarray, vel: np.ndarray, gm: np.ndarray, sun: int = 0
) -> np.ndarray:
    """The 1PN Schwarzschild term for motion in the Sun's field.

    General relativity, to the order that matters here::

        a = GM/(c^2 r^3) [ (4GM/r - v.v) r + 4 (r.v) v ]

    with *r* and *v* measured from the Sun. It is a correction of order
    ``(v/c)^2`` -- about 3e-8 for Mercury -- and it is entirely responsible for
    the 43 arcsec per century of Mercury's perihelion advance that Newtonian
    gravity cannot produce. Off by default: M2's job is to show how far Newton
    gets, and then how much is left over.

    This omits the planet-planet relativistic terms and the Sun's motion, both
    far below the Mercury signal.
    """
    mu = gm[sun]
    r = pos - pos[sun]
    v = vel - vel[sun]
    r_len = np.linalg.norm(r, axis=-1)
    r_len[sun] = np.inf  # the Sun does not orbit itself

    v2 = np.einsum("ij,ij->i", v, v)
    rv = np.einsum("ij,ij->i", r, v)
    scale = mu / (C_AU_PER_DAY**2 * r_len**3)
    return scale[:, None] * ((4 * mu / r_len - v2)[:, None] * r + 4 * rv[:, None] * v)


def _acceleration_fn(gm: np.ndarray, relativity: bool):
    if not relativity:
        return lambda pos, vel: accelerations(pos, gm)
    return lambda pos, vel: accelerations(pos, gm) + relativistic_correction(
        pos, vel, gm
    )


# --- integrators ------------------------------------------------------------

# Yoshida's fourth-order symplectic composition: three leapfrog steps, the
# middle one backwards, with these weights. Costs three force evaluations
# instead of one and buys two orders of accuracy, which at a one-day step is
# the difference between resolving Mercury's orbit and merely sampling it.
_CBRT2 = 2.0 ** (1.0 / 3.0)
_W1 = 1.0 / (2.0 - _CBRT2)
_W0 = -_CBRT2 * _W1
YOSHIDA_WEIGHTS = (_W1, _W0, _W1)


def _leapfrog(pos, vel, acc_fn, dt, acc):
    """One kick-drift-kick step. Returns the new state and its acceleration."""
    vel = vel + 0.5 * dt * acc
    pos = pos + dt * vel
    acc = acc_fn(pos, vel)
    vel = vel + 0.5 * dt * acc
    return pos, vel, acc


def _yoshida4(pos, vel, acc_fn, dt, acc):
    for weight in YOSHIDA_WEIGHTS:
        pos, vel, acc = _leapfrog(pos, vel, acc_fn, weight * dt, acc)
    return pos, vel, acc


def _rk4(pos, vel, acc_fn, dt, acc):
    """Classical Runge-Kutta. Present only so demo_m2 can show it losing energy."""

    def derivative(p, v):
        return v, acc_fn(p, v)

    k1p, k1v = derivative(pos, vel)
    k2p, k2v = derivative(pos + 0.5 * dt * k1p, vel + 0.5 * dt * k1v)
    k3p, k3v = derivative(pos + 0.5 * dt * k2p, vel + 0.5 * dt * k2v)
    k4p, k4v = derivative(pos + dt * k3p, vel + dt * k3v)

    pos = pos + dt / 6.0 * (k1p + 2 * k2p + 2 * k3p + k4p)
    vel = vel + dt / 6.0 * (k1v + 2 * k2v + 2 * k3v + k4v)
    return pos, vel, acc_fn(pos, vel)


INTEGRATORS = {"leapfrog": _leapfrog, "yoshida4": _yoshida4, "rk4": _rk4}
SYMPLECTIC = frozenset({"leapfrog", "yoshida4"})


# --- conserved quantities ---------------------------------------------------


def energy(pos: np.ndarray, vel: np.ndarray, gm: np.ndarray) -> float:
    """Total energy, in units where a body's mass is its GM.

    That is the true energy times G, so it is not in joules and is not meant to
    be. All that is asked of it is that it stay constant, and a constant times
    a constant is still constant.
    """
    kinetic = 0.5 * np.sum(gm * np.einsum("ij,ij->i", vel, vel))

    delta = pos[None, :, :] - pos[:, None, :]
    r = np.sqrt(np.einsum("ijk,ijk->ij", delta, delta))
    np.fill_diagonal(r, np.inf)
    potential = -0.5 * np.sum(np.outer(gm, gm) / r)  # 0.5 undoes double counting
    return float(kinetic + potential)


def momentum(vel: np.ndarray, gm: np.ndarray) -> np.ndarray:
    return gm @ vel


def angular_momentum(pos: np.ndarray, vel: np.ndarray, gm: np.ndarray) -> np.ndarray:
    return np.einsum("i,ij->j", gm, np.cross(pos, vel))


def to_barycentric(
    pos: np.ndarray, vel: np.ndarray, gm: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    """Shift into the frame where the whole system's momentum is zero.

    Any leftover drift of the barycentre is harmless for the orbits but shows up
    as a growing position error against an ephemeris that has none, so it is
    removed once at the start rather than explained away later.
    """
    total = np.sum(gm)
    return pos - (gm @ pos) / total, vel - (gm @ vel) / total


# --- driver -----------------------------------------------------------------


@dataclass
class Trajectory:
    """Sampled output of a run. ``pos``/``vel`` are (samples, bodies, 3)."""

    bodies: tuple[str, ...]
    jd: np.ndarray
    pos: np.ndarray
    vel: np.ndarray
    energy: np.ndarray
    method: str
    dt: float
    relativity: bool

    def index(self, body: str) -> int:
        return self.bodies.index(body)

    def of(self, body: str) -> tuple[np.ndarray, np.ndarray]:
        i = self.index(body)
        return self.pos[:, i, :], self.vel[:, i, :]

    def heliocentric(self, body: str) -> tuple[np.ndarray, np.ndarray]:
        sun = self.index("sun")
        i = self.index(body)
        return (
            self.pos[:, i, :] - self.pos[:, sun, :],
            self.vel[:, i, :] - self.vel[:, sun, :],
        )


def integrate(
    pos0: np.ndarray,
    vel0: np.ndarray,
    *,
    bodies=DEFAULT_BODIES,
    jd0: float,
    dt: float,
    days: float,
    method: str = "yoshida4",
    relativity: bool = False,
    sample_every: int = 1,
) -> Trajectory:
    """Integrate *days* forward from ``jd0``, sampling every *sample_every* steps.

    *dt* may be negative to run backwards; *days* is then also negative. The
    step count is derived from the two, so the final sample lands on or just
    past the requested span.
    """
    if method not in INTEGRATORS:
        raise ValueError(f"unknown method {method!r}; try {', '.join(INTEGRATORS)}")
    if dt == 0:
        raise ValueError("dt must be non-zero")
    if np.sign(dt) != np.sign(days):
        raise ValueError("dt and days must run in the same direction")
    if sample_every < 1:
        raise ValueError("sample_every must be at least 1")

    gm = gm_vector(bodies)
    step = INTEGRATORS[method]
    acc_fn = _acceleration_fn(gm, relativity)

    pos = np.array(pos0, dtype=float, copy=True)
    vel = np.array(vel0, dtype=float, copy=True)
    if pos.shape != (len(bodies), 3) or vel.shape != pos.shape:
        raise ValueError(f"expected state shaped ({len(bodies)}, 3)")

    steps = int(abs(days / dt))
    # The last step is always recorded, whether or not it lands on a sampling
    # boundary. A caller who asks for a span expects its far end back, and
    # silently returning a state from partway through is the kind of thing that
    # looks like a physics result.
    samples = steps // sample_every + 1 + (steps % sample_every != 0)

    out_jd = np.empty(samples)
    out_pos = np.empty((samples, len(bodies), 3))
    out_vel = np.empty((samples, len(bodies), 3))
    out_energy = np.empty(samples)

    acc = acc_fn(pos, vel)
    written = 0
    for n in range(steps + 1):
        if (n % sample_every == 0 or n == steps) and written < samples:
            out_jd[written] = jd0 + n * dt
            out_pos[written] = pos
            out_vel[written] = vel
            out_energy[written] = energy(pos, vel, gm)
            written += 1
        if n < steps:
            pos, vel, acc = step(pos, vel, acc_fn, dt, acc)

    return Trajectory(
        bodies=tuple(bodies),
        jd=out_jd[:written],
        pos=out_pos[:written],
        vel=out_vel[:written],
        energy=out_energy[:written],
        method=method,
        dt=dt,
        relativity=relativity,
    )
