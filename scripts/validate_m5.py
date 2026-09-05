"""M5 gate: are the planets the right way up, and turning at the right rate?

A textured planet is the easiest thing in this project to get wrong without
noticing. Every failure mode still looks like a planet: half a turn out, tilted
by the wrong 23 degrees, spinning backwards, or with the map upside down. None
of that shows up as an error, only as a picture that is quietly of somewhere
else.

So none of these gates look at the rendering. They check the numbers the
rendering is built from, against quantities published independently of the
table they came from.

1. **Rotation periods**, derived from the tabulated W rates, against the
   sidereal periods in every reference.
2. **Obliquities**, derived from the tabulated poles and the orbits from M0,
   against published axial tilts -- including the two bodies where the IAU's
   pole convention makes the naive answer supplementary.
3. **The analemma.** The sub-solar point on the Earth over a year has to trace
   the Sun's declination and the equation of time, and the solstices and
   equinoxes have to land on the right days. Nothing in the rotation table knows
   any of that.
4. **The map is on the right way round.** Sample the Earth's texture where the
   mesh says a coordinate is, and check ocean is ocean.

Usage::

    python scripts/validate_m5.py
    python scripts/validate_m5.py --offline    # skip anything needing a download
"""

from __future__ import annotations

import argparse
import sys
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from orrery import frames, globe, kepler, observer, rotation, times  # noqa: E402

# Sidereal rotation periods, days. Negative where the body turns backwards.
PUBLISHED_PERIOD = {
    "mercury": 58.6462,
    "venus": -243.0180,
    "embary": 0.997270,
    "mars": 1.025957,
    "jupiter": 0.413540,
    "saturn": 0.444010,
    "uranus": -0.718330,
    "neptune": 0.671250,
    "pluto": 6.387200,
}

# Axial tilt, degrees, measured to the orbit. Over 90 means it turns backwards.
PUBLISHED_OBLIQUITY = {
    "mercury": 0.03,
    "venus": 177.36,
    "embary": 23.44,
    "mars": 25.19,
    "jupiter": 3.13,
    "saturn": 26.73,
    "uranus": 97.77,
    "neptune": 28.32,
    "pluto": 119.60,
}

# Neptune's pole carries periodic terms of amplitude 0.7 degrees that this
# package omits, so its obliquity is allowed to sit further out than the rest.
OBLIQUITY_TOLERANCE = 0.05
NEPTUNE_TOLERANCE = 0.6

# Known geography, to check the map is not rotated or flipped.
COASTLINES = [
    ("Gulf of Guinea", 0.0, 0.0, True),
    ("Congo basin", -2.0, 23.0, False),
    ("Egypt", 26.0, 30.0, False),
    ("mid Pacific", 0.0, -150.0, True),
    ("Australia", -25.0, 133.0, False),
    ("Kansas", 38.0, -98.0, False),
    ("South Atlantic", -35.0, -15.0, True),
    ("Bay of Bengal", 15.0, 88.0, True),
    ("Siberia", 65.0, 100.0, False),
    ("Antarctica", -80.0, 0.0, False),
]


@dataclass
class Check:
    name: str
    passed: bool
    lines: list[str] = field(default_factory=list)


def gate_periods() -> Check:
    lines = [f"  {'body':<9}{'ours':>13}{'published':>13}{'sense':>13}"]
    worst = 0.0
    for body, published in PUBLISHED_PERIOD.items():
        ours = rotation.rotation_period_days(body)
        worst = max(worst, abs(ours - published) / abs(published))
        lines.append(
            f"  {body:<9}{ours:13.5f}{published:13.5f}"
            f"{'backwards' if rotation.turns_backwards(body) else 'forwards':>13}"
        )
    lines += [
        "",
        f"worst disagreement {worst:.2e} of the period -- the W rates are the"
        " periods, restated",
    ]
    return Check("rotation periods", worst < 1e-4, lines)


def gate_obliquities() -> Check:
    jd = times.jd(2026, 9, 3)
    lines = [f"  {'body':<9}{'ours':>10}{'published':>12}{'tabulated pole':>17}"]
    passed = True
    for body, published in PUBLISHED_OBLIQUITY.items():
        ours = float(rotation.obliquity_degrees(body, jd))
        naive = float(
            np.degrees(
                np.arccos(
                    np.clip(
                        np.sum(
                            rotation.pole(body, jd)
                            * frames.ecliptic_to_equatorial(
                                _orbit_normal(body, jd)
                            )
                        ),
                        -1.0,
                        1.0,
                    )
                )
            )
        )
        allowed = NEPTUNE_TOLERANCE if body == "neptune" else OBLIQUITY_TOLERANCE
        passed = passed and abs(ours - published) < allowed
        lines.append(
            f"  {body:<9}{ours:9.2f}d{published:11.2f}d{naive:16.2f}d"
        )
    lines += [
        "",
        "the last column measures to the tabulated pole rather than to the angular",
        "momentum. For Venus and Uranus the two are supplementary -- 2.6 against",
        "177.4, and 82.2 against 97.8 -- because the IAU calls the northern one",
        "north whichever way the body turns. Both readings look reasonable.",
        "Neptune is allowed 0.6d: its pole has periodic terms this package omits.",
    ]
    return Check("obliquities, and the pole convention", passed, lines)


def _orbit_normal(body: str, jd) -> np.ndarray:
    orbiting = "embary" if body in ("embary", "moon") else body
    position, velocity = kepler.state(orbiting, jd)
    normal = np.cross(position, velocity)
    return normal / np.linalg.norm(normal, axis=-1, keepdims=True)


def gate_analemma() -> Check:
    """The sub-solar point over a year, which the rotation table cannot know."""
    jd_ut = times.jd(2026, 1, 1, 12) + np.arange(0, 366, 1.0)
    jd_tdb = jd_ut + observer.delta_t_seconds(jd_ut) / 86400.0

    toward_sun = frames.ecliptic_to_equatorial(-kepler.position("embary", jd_tdb))
    latitude, longitude = rotation.surface_point("embary", jd_tdb, toward_sun)
    longitude = (longitude + 180.0) % 360.0 - 180.0

    tilt = float(max(latitude.max(), -latitude.min()))
    spread_minutes = float(longitude.max() - longitude.min()) * 4.0

    events = {
        "June solstice": int(np.argmax(latitude)),
        "December solstice": int(np.argmin(latitude)),
        "March equinox": int(np.argmin(np.abs(latitude[:200]))),
        "September equinox": 200 + int(np.argmin(np.abs(latitude[200:]))),
    }
    wanted = {
        "June solstice": "2026-06-21",
        "December solstice": "2026-12-21",
        "March equinox": "2026-03-20",
        "September equinox": "2026-09-23",
    }

    lines = [
        f"  sub-solar latitude reaches +-{tilt:.2f}d   (the obliquity, {23.44})",
        f"  longitude at noon UT spans {spread_minutes:.1f} minutes of time"
        f"   (equation of time, 30.5)",
        "",
    ]
    dates_ok = True
    for label, index in events.items():
        stamp = times.isoformat(jd_ut[index])[:10]
        agrees = stamp == wanted[label]
        dates_ok = dates_ok and agrees
        lines.append(
            f"  {label:<20}{stamp}   {'ok' if agrees else 'want ' + wanted[label]}"
        )
    lines += [
        "",
        "None of this is in the rotation table. It is the Earth's tilt and the",
        "eccentricity of its orbit, arriving through W.",
    ]

    passed = (
        abs(tilt - 23.44) < 0.05
        and abs(spread_minutes - 30.5) < 1.5
        and dates_ok
    )
    return Check("the analemma", passed, lines)


def gate_map(*, offline: bool) -> Check:
    vertices, faces, uv = globe.uv_sphere(16, 32)
    radius = np.linalg.norm(vertices, axis=-1)

    lines = [
        f"  mesh: {len(vertices)} vertices, {len(faces)} faces,"
        f" radius {radius.min():.6f}..{radius.max():.6f}",
        f"  u spans {uv[:, 0].min():.1f}..{uv[:, 0].max():.1f},"
        f" v spans {uv[:, 1].min():.1f}..{uv[:, 1].max():.1f}",
    ]
    mesh_ok = (
        np.allclose(radius, 1.0)
        and uv.min() == 0.0
        and uv.max() == 1.0
        and faces.max() == len(vertices) - 1
    )

    pixels = globe.load_texture("embary", download=not offline)
    if pixels is None:
        lines.append("  (no Earth map cached; skipping the geography check)")
        return Check("the mesh, and which way round the map is", mesh_ok, lines)

    height, width = pixels.shape[:2]
    lines.append(f"  map: {width}x{height}, aspect {width / height:.2f}")

    wrong = []
    for name, latitude, longitude, want_water in COASTLINES:
        is_water = bool(globe.looks_like_water(globe.sample(pixels, latitude, longitude)))
        if is_water != want_water:
            wrong.append(name)
    lines.append(
        f"  {len(COASTLINES) - len(wrong)}/{len(COASTLINES)} known places came out"
        f" land or sea as they should"
    )
    if wrong:
        lines.append(f"  wrong: {', '.join(wrong)}")
    lines += [
        "",
        "Half a turn out and the Gulf of Guinea becomes the mid Pacific -- both",
        "ocean, both plausible. It takes the whole list to pin the map down.",
        f"Maps: {globe.TEXTURE_LICENCE}.",
    ]
    return Check(
        "the mesh, and which way round the map is",
        mesh_ok and not wrong and abs(width / height - 2.0) < 1e-9,
        lines,
    )


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--offline", action="store_true")
    args = parser.parse_args()

    print("\nM5 -- orientation, rotation, and the maps\n")
    checks = [
        gate_periods(),
        gate_obliquities(),
        gate_analemma(),
        gate_map(offline=args.offline),
    ]
    for check in checks:
        print(f"  [{'ok' if check.passed else 'FAIL'}]  {check.name}")
        for line in check.lines:
            print(f"        {line}")
        print()

    passing = sum(c.passed for c in checks)
    print(f"M5 {'passing' if passing == len(checks) else 'FAILED'}, "
          f"{passing}/{len(checks)}")
    return 0 if passing == len(checks) else 1


if __name__ == "__main__":
    raise SystemExit(main())
