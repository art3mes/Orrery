"""Finding events: conjunctions, oppositions, transits.

Everything here works on **arrays of positions**, never on body names. That is
deliberate. The M1 gate has to ask the same question twice -- once of this
package's orbits and once of JPL's DE440 -- and if the two used different
finders, a disagreement would not say which half was wrong. Feeding both through
identical code means any difference that shows up is a difference in the
positions, which is the thing being measured.

Positions are heliocentric, in au, in any consistent frame; angles come out in
arcseconds or degrees as named. Nothing here corrects for light travel time, so
these are geometric events, not observed ones. That correction lands in M3.
"""

from __future__ import annotations

import numpy as np

from .frames import norm, separation_arcsec
from .kepler import AU_KM

# IAU nominal solar radius, in au.
SUN_RADIUS_KM = 695700.0
SUN_RADIUS_AU = SUN_RADIUS_KM / AU_KM


def separation_from_earth(
    target: np.ndarray, other: np.ndarray, earth: np.ndarray
) -> np.ndarray:
    """Angle between two bodies as seen from Earth, in arcseconds."""
    return separation_arcsec(target - earth, other - earth)


def elongation_deg(target: np.ndarray, earth: np.ndarray) -> np.ndarray:
    """Angle between a body and the Sun as seen from Earth, in degrees.

    0 is toward the Sun, 180 is opposition.
    """
    return separation_arcsec(target - earth, -earth) / 3600.0


def solar_radius_arcsec(earth: np.ndarray) -> np.ndarray:
    """Apparent radius of the Sun's disc from Earth, in arcseconds.

    Varies by about 3% over the year: roughly 975" at perihelion in January and
    945" at aphelion in July. A transit check that used a fixed value would be
    wrong by more than the width of Venus.
    """
    return np.degrees(np.arcsin(SUN_RADIUS_AU / norm(earth))) * 3600.0


def _refine(x: np.ndarray, y: np.ndarray, index: int) -> tuple[float, float]:
    """Fit a parabola through three samples and return its vertex.

    A sampled minimum is only ever as good as the grid step, and near a
    conjunction the curve is very flat, so the nearest sample can sit hours from
    the true extremum. Three points and a parabola cost nothing and remove that
    quantisation almost entirely.
    """
    if index <= 0 or index >= len(x) - 1:
        return float(x[index]), float(y[index])

    x0, x1, x2 = x[index - 1 : index + 2]
    y0, y1, y2 = y[index - 1 : index + 2]
    denominator = (x0 - x1) * (x0 - x2) * (x1 - x2)
    if denominator == 0:
        return float(x1), float(y1)

    a = (x2 * (y1 - y0) + x1 * (y0 - y2) + x0 * (y2 - y1)) / denominator
    b = (
        x2 * x2 * (y0 - y1) + x1 * x1 * (y2 - y0) + x0 * x0 * (y1 - y2)
    ) / denominator
    if a == 0:
        return float(x1), float(y1)

    # Clamp to half a step either side of the smallest sample.
    #
    # For a smooth function on a uniform grid the extremum cannot be further
    # than that: if it were, a neighbouring sample would be nearer to it and
    # therefore smaller, and this would not be the argmin. So the clamp costs
    # nothing when the fit is sound.
    #
    # It is not decoration. At a very flat minimum -- a conjunction is exactly
    # that -- the three samples differ by barely more than rounding, the fitted
    # curvature is badly conditioned, and the vertex can fly well outside the
    # bracket. On a 0.02 d grid the 2020 great conjunction refined to 24 minutes
    # *worse* than its own raw argmin. Clamped, refinement can never do worse
    # than not refining at all.
    half = 0.5 * min(x1 - x0, x2 - x1)
    vertex = float(np.clip(-b / (2 * a), x1 - half, x1 + half))
    c = y1 - a * x1 * x1 - b * x1
    return vertex, float(a * vertex * vertex + b * vertex + c)


def find_extrema(
    x: np.ndarray,
    y: np.ndarray,
    *,
    kind: str = "min",
    threshold: float | None = None,
) -> list[tuple[float, float]]:
    """Locate interior local extrema of *y* against *x*, refined to sub-step.

    *threshold* keeps only minima below it, or maxima above it -- the way to
    say "oppositions, not every wiggle in the elongation curve".
    """
    if kind not in ("min", "max"):
        raise ValueError("kind must be 'min' or 'max'")

    middle = y[1:-1]
    if kind == "min":
        interior = (middle < y[:-2]) & (middle < y[2:])
        if threshold is not None:
            interior &= middle < threshold
    else:
        interior = (middle > y[:-2]) & (middle > y[2:])
        if threshold is not None:
            interior &= middle > threshold

    return [_refine(x, y, int(i) + 1) for i in np.flatnonzero(interior)]


def find_crossings(x: np.ndarray, y: np.ndarray, level: float) -> list[float]:
    """Where *y* crosses *level*, by linear interpolation between samples."""
    above = y > level
    changes = np.flatnonzero(above[:-1] != above[1:])
    out = []
    for i in changes:
        y0, y1 = y[i], y[i + 1]
        if y1 == y0:
            out.append(float(x[i]))
        else:
            out.append(float(x[i] + (level - y0) * (x[i + 1] - x[i]) / (y1 - y0)))
    return out
