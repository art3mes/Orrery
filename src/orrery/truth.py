"""Ground truth: JPL DE ephemeris positions, via Skyfield.

This module is the *independent implementation* the M0 gate is measured
against. Nothing in :mod:`orrery.kepler` may import it, and it is an optional
dependency, so a wrong answer here can never quietly become the answer.

Results are cached to ``data/fixtures/*.npz`` keyed by the exact query, so the
gate needs the network once and then runs offline forever.

Positions returned are **geometric, heliocentric, ICRF equatorial**, in au:

* *geometric* -- Skyfield's ``.at(t)`` with no ``observe()``, so no light-time
  correction and no aberration. Those are real effects, but they belong to an
  observer; comparing them against a bare orbit model would be comparing two
  different questions. Light-time arrives in M3.
* *heliocentric* -- relative to the Sun's centre, not the solar system
  barycentre. The element table is heliocentric, and the difference is about one
  solar radius, roughly 1e-3 au: 300 times larger than the errors being measured
  here. Getting this wrong would sink the gate on its own.
"""

from __future__ import annotations

import hashlib
from pathlib import Path

import numpy as np

from .elements import canonical

DATA_DIR = Path(__file__).resolve().parents[2] / "data"
FIXTURE_DIR = DATA_DIR / "fixtures"

DEFAULT_EPHEMERIS = "de440s.bsp"

# de440s covers roughly 1849-12 to 2150-01, so the usable overlap with the
# element table (which stops at 2050) starts in 1850, not 1800. The exact
# endpoints are read from the kernel rather than written down here -- a
# hardcoded span that is half a day out rejects perfectly good dates.

# Names that are not planets in the element table's sense. "geocentre" is the
# Earth itself, which orbits the Earth-Moon barycentre at up to 4700 km -- 6.5
# arcsec seen from Mars, and the difference between a geocentric place and a
# barycentric one. M3 needs the distinction; M0 to M2 did not.
_NOT_IN_THE_TABLE = {"sun": "sun", "geocentre": "earth", "moon": "moon"}

_SKYFIELD_NAMES = {
    "sun": "sun",
    "geocentre": "earth",
    "moon": "moon",
    "mercury": "mercury barycenter",
    "venus": "venus barycenter",
    "embary": "earth barycenter",
    "mars": "mars barycenter",
    "jupiter": "jupiter barycenter",
    "saturn": "saturn barycenter",
    "uranus": "uranus barycenter",
    "neptune": "neptune barycenter",
    "pluto": "pluto barycenter",
}


def _use_system_trust_store() -> None:
    """Verify TLS against the OS certificate store, if ``truststore`` is present.

    Machines behind a TLS-inspecting proxy or antivirus present a locally
    installed root CA that Python's bundled certificate list does not contain,
    so the ephemeris download fails with CERTIFICATE_VERIFY_FAILED even though
    the browser and pip are perfectly happy. Deferring to the OS store fixes
    that without weakening verification. Absent the package, nothing changes.
    """
    try:
        import truststore
    except ImportError:
        return
    truststore.inject_into_ssl()


def _key(body: str) -> str:
    """Canonical name, with the Sun, the Earth's centre and the Moon let through."""
    name = body.strip().lower()
    return name if name in _NOT_IN_THE_TABLE else canonical(name)


def _cache_path(
    bodies: tuple[str, ...], jd: np.ndarray, ephemeris: str, kind: str = "helio"
) -> Path:
    h = hashlib.sha1()
    h.update(kind.encode())
    h.update(ephemeris.encode())
    h.update(",".join(bodies).encode())
    h.update(np.ascontiguousarray(jd, dtype=np.float64).tobytes())
    return FIXTURE_DIR / f"truth_{kind}_{h.hexdigest()[:16]}.npz"


def heliocentric_equatorial(
    bodies,
    jd_tdb,
    *,
    ephemeris: str = DEFAULT_EPHEMERIS,
    allow_download: bool = True,
    verbose: bool = False,
) -> dict[str, np.ndarray]:
    """Geometric heliocentric ICRF positions, in au, shaped ``(len(jd), 3)``.

    Cached on disk. Set ``allow_download=False`` to require a cache hit and
    fail loudly rather than reach for the network.
    """
    keys = tuple(_key(b) for b in bodies)
    jd = np.atleast_1d(np.asarray(jd_tdb, dtype=float))

    path = _cache_path(keys, jd, ephemeris)
    if path.exists():
        with np.load(path) as data:
            return {k: data[k] for k in keys}

    if not allow_download:
        raise FileNotFoundError(
            f"no cached ground truth at {path.name}; rerun with network access "
            "to populate data/fixtures/"
        )

    try:
        from skyfield.api import Loader
    except ImportError as exc:  # pragma: no cover - depends on environment
        raise ImportError(
            "ground truth needs Skyfield: pip install -e \".[truth]\""
        ) from exc

    FIXTURE_DIR.mkdir(parents=True, exist_ok=True)
    _use_system_trust_store()
    loader = Loader(str(DATA_DIR), verbose=verbose)
    eph = loader(ephemeris)
    ts = loader.timescale()
    t = ts.tdb_jd(jd)

    lo, hi = ephemeris_span(eph)
    if jd.min() < lo or jd.max() > hi:
        raise ValueError(
            f"requested dates (JD {jd.min():.1f} to {jd.max():.1f}) fall outside "
            f"the span of {ephemeris} (JD {lo:.1f} to {hi:.1f})"
        )

    sun = eph["sun"]
    out = {}
    for key in keys:
        target = eph[_SKYFIELD_NAMES[key]]
        # .at() is geometric: the true instantaneous separation, no light-time.
        out[key] = np.asarray((target - sun).at(t).position.au).T.copy()

    np.savez_compressed(path, **out)
    return out


def barycentric_state(
    bodies,
    jd_tdb,
    *,
    ephemeris: str = DEFAULT_EPHEMERIS,
    allow_download: bool = True,
    verbose: bool = False,
) -> tuple[np.ndarray, np.ndarray]:
    """Positions and velocities about the solar system barycentre, ICRF.

    Returns two arrays shaped ``(len(jd), len(bodies), 3)`` in au and au/day.

    This is what seeds an N-body run. Using DE440 for the initial conditions and
    then comparing against DE440 is not circular: the integrator is what is
    being tested, and starting it anywhere else would confound its error with
    the error of the starting point. ``validate_m2.py`` also runs it from the
    Keplerian elements, which shows exactly how much that costs.

    The barycentre is the natural origin here because it does not accelerate,
    and a symplectic integrator in an accelerating frame is not symplectic.
    """
    keys = tuple(_key(b) for b in bodies)
    jd = np.atleast_1d(np.asarray(jd_tdb, dtype=float))

    path = _cache_path(keys, jd, ephemeris, kind="bary")
    if path.exists():
        with np.load(path) as data:
            return data["pos"], data["vel"]

    if not allow_download:
        raise FileNotFoundError(
            f"no cached barycentric state at {path.name}; rerun with network "
            "access to populate data/fixtures/"
        )

    try:
        from skyfield.api import Loader
    except ImportError as exc:  # pragma: no cover - depends on environment
        raise ImportError(
            "ground truth needs Skyfield: pip install -e \".[truth]\""
        ) from exc

    FIXTURE_DIR.mkdir(parents=True, exist_ok=True)
    _use_system_trust_store()
    loader = Loader(str(DATA_DIR), verbose=verbose)
    eph = loader(ephemeris)
    ts = loader.timescale()
    t = ts.tdb_jd(jd)

    lo, hi = ephemeris_span(eph)
    if jd.min() < lo or jd.max() > hi:
        raise ValueError(
            f"requested dates (JD {jd.min():.1f} to {jd.max():.1f}) fall outside "
            f"the span of {ephemeris} (JD {lo:.1f} to {hi:.1f})"
        )

    pos = np.empty((len(jd), len(keys), 3))
    vel = np.empty_like(pos)
    for i, key in enumerate(keys):
        at = eph[_SKYFIELD_NAMES[key]].at(t)
        pos[:, i, :] = np.asarray(at.position.au).T
        vel[:, i, :] = np.asarray(at.velocity.au_per_d).T

    np.savez_compressed(path, pos=pos, vel=vel)
    return pos, vel


def sampled_ephemeris(
    bodies,
    jd,
    *,
    pad: float = 3.0,
    step: float = 0.25,
    ephemeris: str = DEFAULT_EPHEMERIS,
    allow_download: bool = True,
):
    """DE440 on a grid, wrapped so the light-time solver can interpolate it.

    The solver asks for positions minutes before the dates it was handed, which
    no tabulated ephemeris has exactly. A quarter-day grid and a cubic is good
    to 0.03 km on Mercury, the worst case, and far below anything measured here.
    """
    from .apparent import Ephemeris

    jd = np.atleast_1d(np.asarray(jd, dtype=float))
    grid = np.arange(jd.min() - pad, jd.max() + pad + step, step)
    positions, velocities = barycentric_state(
        bodies, grid, ephemeris=ephemeris, allow_download=allow_download
    )
    return Ephemeris(bodies, grid, positions, velocities)


def apparent_reference(
    body: str,
    jd_tdb,
    *,
    site=None,
    epoch_of_date: bool = False,
    stage: str = "apparent",
    ephemeris: str = DEFAULT_EPHEMERIS,
    allow_download: bool = True,
) -> np.ndarray:
    """Skyfield's own apparent place, as unit vectors shaped ``(len(jd), 3)``.

    The reference for M3. Skyfield does light-time, gravitational deflection,
    aberration, precession and nutation with the full IAU 2000A machinery, so
    comparing against it isolates this package's transformations from its
    orbits: both sides are handed the same DE440 geometry, and any difference
    is in the corrections.

    *stage* is "astrometric" (light-time only) or "apparent" (also deflected and
    aberrated). *site* is an :class:`orrery.observer.Site` for a topocentric
    place, or None for geocentric.
    """
    jd = np.atleast_1d(np.asarray(jd_tdb, dtype=float))

    tag = f"{body}|{stage}|{'date' if epoch_of_date else 'icrf'}"
    if site is not None:
        tag += f"|{site.latitude:.6f},{site.longitude:.6f},{site.height_m:.1f}"
    path = _cache_path((tag,), jd, ephemeris, kind="apparent")
    if path.exists():
        with np.load(path) as data:
            return data["direction"]

    if not allow_download:
        raise FileNotFoundError(
            f"no cached apparent place at {path.name}; rerun with network access"
        )

    try:
        from skyfield.api import Loader, wgs84
    except ImportError as exc:  # pragma: no cover - depends on environment
        raise ImportError("the M3 reference needs Skyfield") from exc

    FIXTURE_DIR.mkdir(parents=True, exist_ok=True)
    _use_system_trust_store()
    loader = Loader(str(DATA_DIR), verbose=False)
    eph = loader(ephemeris)
    ts = loader.timescale()
    t = ts.tdb_jd(jd)

    origin = eph["earth"]
    if site is not None:
        origin = origin + wgs84.latlon(
            site.latitude, site.longitude, elevation_m=site.height_m
        )

    seen = origin.at(t).observe(eph[_SKYFIELD_NAMES[_key(body)]])
    if stage == "apparent":
        seen = seen.apparent()
    elif stage != "astrometric":
        raise ValueError("stage must be 'astrometric' or 'apparent'")

    if epoch_of_date:
        ra, dec, _ = seen.radec(epoch="date")
        a, d = np.radians(ra.hours * 15.0), np.radians(dec.degrees)
        direction = np.stack(
            [np.cos(d) * np.cos(a), np.cos(d) * np.sin(a), np.sin(d)], axis=-1
        )
    else:
        vector = np.asarray(seen.position.au).T
        direction = vector / np.linalg.norm(vector, axis=-1)[:, None]

    np.savez_compressed(path, direction=direction)
    return direction


def ephemeris_span(eph) -> tuple[float, float]:
    """Julian date range covered by *every* segment of a loaded kernel.

    The intersection, not the union: a date is only usable if all the bodies
    being asked about are covered at it.
    """
    spans = [seg.spk_segment for seg in eph.segments]
    return max(s.start_jd for s in spans), min(s.end_jd for s in spans)


def cached_only(bodies, jd_tdb, **kwargs) -> dict[str, np.ndarray]:
    """:func:`heliocentric_equatorial` with the network refused."""
    return heliocentric_equatorial(bodies, jd_tdb, allow_download=False, **kwargs)


def have_skyfield() -> bool:
    try:
        import skyfield  # noqa: F401
    except ImportError:
        return False
    return True
