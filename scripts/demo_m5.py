"""One body, close up, oriented for a date.

The orrery view draws the Earth about five pixels across, which is honest and
useless for looking at a map. This draws a single body at unit radius with its
real orientation: the IAU pole, the real axial tilt, and the prime meridian
where it actually was at that instant.

It doubles as a check anyone can make by eye. Run it for the Earth at noon UT
and the sub-solar longitude is near zero, so **Africa and Europe face the Sun**.
If the chain from the rotation elements through the body-fixed frame to the
texture coordinates were wrong anywhere, the wrong ocean would be lit.

    python scripts/demo_m5.py --body earth
    python scripts/demo_m5.py --body mars --date 2026-09-03 --hour 12
    python scripts/demo_m5.py --body earth --from-sun --screenshot earth.png
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from orrery import frames, globe, kepler, rotation, scene, times  # noqa: E402


def sun_direction(body: str, jd: float) -> np.ndarray:
    """Unit vector from the body toward the Sun, in the ecliptic frame."""
    orbiting = "embary" if body in ("embary", "moon") else body
    return -kepler.position(orbiting, jd) / np.linalg.norm(
        kepler.position(orbiting, jd)
    )


def report(body: str, jd: float) -> None:
    toward_sun = frames.ecliptic_to_equatorial(sun_direction(body, jd))
    latitude, longitude = rotation.surface_point(body, jd, toward_sun)
    period = rotation.rotation_period_days(body)
    tilt = float(rotation.obliquity_degrees(body, jd))

    print(f"\n{body} on {times.isoformat(jd)} TDB\n")
    print(f"  axial tilt          {tilt:8.2f} deg")
    print(
        f"  rotation period     {abs(period) * 24:8.3f} h"
        f"   {'retrograde' if rotation.turns_backwards(body) else 'prograde'}"
    )
    print(f"  sub-solar point     {float(latitude):+7.2f} lat"
          f"  {float(longitude):7.2f} east lon")
    print(f"  prime meridian W    {float(rotation.prime_meridian_degrees(body, jd)):8.2f} deg")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--body", default="earth")
    parser.add_argument("--date", default="2026-09-03")
    parser.add_argument("--hour", type=float, default=12.0, help="UT hour")
    parser.add_argument(
        "--from-sun", action="store_true", help="look from the Sun, so the lit face shows"
    )
    parser.add_argument("--frames", type=int, default=None)
    parser.add_argument("--screenshot", default=None)
    parser.add_argument("--backend", default="")
    args = parser.parse_args()

    year, month, day = (int(p) for p in args.date.split("-"))
    jd = times.jd(year, month, day) + args.hour / 24.0
    body = rotation._key(args.body)

    report(body, jd)

    import polyscope as ps

    ps.init(args.backend) if args.backend else ps.init()
    ps.set_up_dir("z_up")
    ps.set_ground_plane_mode("none")
    ps.set_background_color((0.02, 0.02, 0.05))
    ps.set_bounding_box((-1.6, -1.6, -1.6), (1.6, 1.6, 1.6))

    vertices, faces, uv = globe.uv_sphere(96, 192)
    mesh = ps.register_surface_mesh(body.capitalize(), vertices, faces, smooth_shade=True)
    mesh.set_color(scene.COLOR.get(body, (0.7, 0.7, 0.7)))
    mesh.set_material("flat" if body == "sun" else "clay")

    pixels = globe.load_texture(body)
    if pixels is not None:
        mesh.add_parameterization_quantity("uv", uv, enabled=False)
        mesh.add_color_quantity(
            "map", pixels, defined_on="texture", param_name="uv", enabled=True
        )
    else:
        print(f"  (no map for {body}; showing flat colour)")

    # Orientation only: unit radius, centred on the origin.
    mesh.set_transform(globe.orientation(body, jd, 1.0, np.zeros(3)))

    # Draw the rotation axis, so the tilt is visible rather than inferred.
    axis = frames.equatorial_to_ecliptic(rotation.spin_axis(body, jd))
    pole_line = ps.register_curve_network(
        "rotation axis", np.stack([-1.45 * axis, 1.45 * axis]), "line"
    )
    pole_line.set_color((1.0, 0.85, 0.3))
    pole_line.set_radius(0.006, relative=False)

    toward_sun = sun_direction(body, jd)
    if args.from_sun:
        ps.look_at(tuple(3.2 * toward_sun), (0.0, 0.0, 0.0))
    else:
        ps.look_at((3.0, -1.6, 1.2), (0.0, 0.0, 0.0))

    if args.screenshot:
        state = {"n": 0}

        def grab():
            state["n"] += 1
            if state["n"] == 2:
                ps.screenshot(args.screenshot, transparent_bg=False)

        ps.set_user_callback(grab)
        ps.show(forFrames=max(args.frames or 4, 3))
    elif args.frames is None:
        ps.show()
    else:
        ps.show(forFrames=args.frames)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
