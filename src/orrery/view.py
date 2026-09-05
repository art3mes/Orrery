"""The polyscope viewer.

Nothing in here computes anything. Positions come from :mod:`orrery.kepler`,
geometry from :mod:`orrery.scene`, and this module only registers structures and
pushes new coordinates into them each frame. That split is what lets
``validate_m1.py`` check the picture without a display attached: everything the
viewer can get wrong about *where things are* is checkable, and what is left
here is wiring.

Run it with ``scripts/demo_m1.py``.
"""

from __future__ import annotations

import numpy as np

from . import globe, scene
from .elements import ORDER, VALID_JD
from .kepler import position
from .times import isoformat, jd as julian_date

# The scrubber slides on days-since-1850, not on the Julian date itself.
#
# ImGui sliders are single precision. A Julian date is ~2.45e6, where float32
# steps in units of 0.25 days, so a slider bound straight to it would snap in
# visible six-hour jumps. Counting days from 1850 gives numbers up to 73048,
# where the step is 0.0078 days -- about 11 minutes. (Fractional years, the
# obvious alternative, are worse: ~2027 steps by 64 minutes.) The readable date
# is printed above the slider, so nothing is lost by the axis being a count.
EPOCH_JD = julian_date(1850, 1, 1)
DAY_MIN = 0.0
DAY_MAX = VALID_JD[1] - EPOCH_JD


def day_to_jd(day: float) -> float:
    return EPOCH_JD + day


def jd_to_day(jd: float) -> float:
    return jd - EPOCH_JD


class Orrery:
    """Registers the scene once, then rewrites its coordinates each frame."""

    def __init__(
        self,
        jd_start: float | None = None,
        *,
        view: str = "inner",
        orbit_samples: int = 512,
        trail_samples: int = 200,
        textures: bool = True,
        sphere_rows: int = 48,
    ) -> None:
        self.jd = float(
            np.clip(julian_date(2026, 9, 3) if jd_start is None else jd_start, *VALID_JD)
        )
        self.orbit_samples = orbit_samples
        self.trail_samples = trail_samples
        self.textures = textures
        self.sphere_rows = sphere_rows

        self.playing = False
        self.days_per_frame = 2.0
        if view not in scene.VIEW_PRESETS:
            raise ValueError(
                f"unknown view {view!r}; try {', '.join(scene.VIEW_PRESETS)}"
            )
        self.view_name = view
        preset = scene.VIEW_PRESETS[view]
        self.view_scale = preset.scale_au
        self.planet_exaggeration = preset.planet_exaggeration
        self.sun_exaggeration = preset.sun_exaggeration
        self.visible = set(preset.bodies)
        self.trail_fraction = scene.DEFAULT_TRAIL_FRACTION
        self.show_orbits = True
        self.show_trails = True

        # Focus mode: one body, at true size, at the origin. None means the
        # ordinary solar-system view.
        self.focus: str | None = None
        self.focus_from_sun = True

        self._planets: dict[str, object] = {}
        self._orbits: dict[str, object] = {}
        self._trails: dict[str, object] = {}
        self._line_radius = scene.line_radius_au(self.view_scale)
        self._rings: dict[str, object] = {}
        self._sun = None

        self._frame = 0
        self._screenshot_path: str | None = None
        self._screenshot_frame = -1

    # -- construction --------------------------------------------------------

    def build(self) -> None:
        import polyscope as ps

        ps.set_up_dir("z_up")
        ps.set_ground_plane_mode("none")
        ps.set_background_color((0.02, 0.02, 0.05))
        # Rings need real per-vertex alpha, not a premultiplied stand-in.
        ps.set_transparency_mode("pretty")
        # Pluto's aphelion is 49 au out. Without a fixed box, polyscope reframes
        # the whole scene whenever a trail grows, and the view jumps.
        ps.set_bounding_box((-52.0, -52.0, -52.0), (52.0, 52.0, 52.0))

        vertices, faces, uv = globe.uv_sphere(self.sphere_rows, self.sphere_rows * 2)

        def sphere(label: str, body: str):
            """A unit sphere carrying this body's map, or its flat colour."""
            mesh = ps.register_surface_mesh(label, vertices, faces, smooth_shade=True)
            mesh.set_color(scene.COLOR[body])
            mesh.set_material("flat" if body == "sun" else "clay")
            if self.textures:
                pixels = globe.load_texture(body)
                if pixels is not None:
                    mesh.add_parameterization_quantity("uv", uv, enabled=False)
                    mesh.add_color_quantity(
                        "map", pixels, defined_on="texture", param_name="uv",
                        enabled=True,
                    )
            return mesh

        self._sun = sphere("Sun", "sun")

        for body in ORDER:
            label = body.capitalize()
            self._planets[body] = sphere(label, body)

            if globe.has_rings(body):
                nodes, ring_faces, radial, opacity = globe.ring_mesh(body)
                rings = ps.register_surface_mesh(f"{label} rings", nodes, ring_faces)
                rings.set_material("flat")  # rings are lit from every side at once
                rings.set_back_face_policy("identical")
                system = globe.RING_SYSTEMS[body]
                profile = globe.ring_profile(body) if system.texture else None
                if profile is not None:
                    # Saturn has a real map, so take both colour and opacity
                    # from it rather than from the coarse band table.
                    colour, opacity = globe.ring_samples(profile, radial)
                    rings.add_color_quantity("rings", colour, enabled=True)
                else:
                    rings.set_color(system.colour)
                rings.add_scalar_quantity("opacity", opacity, enabled=False)
                rings.set_transparency_quantity("opacity")
                self._rings[body] = rings

            orbit = ps.register_curve_network(
                f"{label} orbit",
                scene.orbit_loop(body, self.jd, self.orbit_samples),
                "loop",
            )
            orbit.set_color(scene.COLOR[body])
            self._orbits[body] = orbit

            path = ps.register_curve_network(
                f"{label} trail", self._trail_nodes(body), "line"
            )
            path.set_color(scene.COLOR[body])
            self._trails[body] = path

        self.apply_view(self.view_name)

    def _trail_nodes(self, body: str) -> np.ndarray:
        return scene.trail(
            body,
            self.jd,
            scene.trail_span_days(body, self.trail_fraction),
            samples=self.trail_samples,
        )

    # -- per-frame updates ---------------------------------------------------

    def _place_globes(self) -> None:
        """Rotation, size and position, as one transform per body."""
        if self.focus is not None:
            # One body, true size, at the origin. Putting it at the origin
            # rather than at its real position is not cosmetic: a planet is
            # 4e-5 au across and sits 1 au out, so a camera close enough to see
            # the map would be resolving one part in 25000 of the scene, and
            # both the depth buffer and float32 give up long before that.
            mesh = self._sun if self.focus == "sun" else self._planets[self.focus]
            matrix = globe.orientation(self.focus, self.jd, 1.0, np.zeros(3))
            mesh.set_transform(matrix)
            if self.focus in self._rings:
                self._rings[self.focus].set_transform(matrix)
            return

        self._sun.set_transform(
            globe.orientation(
                "sun", self.jd,
                scene.display_radius_au("sun", self.sun_exaggeration),
                np.zeros(3),
            )
        )
        for body, mesh in self._planets.items():
            matrix = globe.orientation(
                body, self.jd,
                scene.display_radius_au(body, self.planet_exaggeration),
                position(body, self.jd),
            )
            mesh.set_transform(matrix)
            if body in self._rings:
                # The rings lie at z = 0 in body coordinates, so the same
                # transform carries them, tilt and all -- and the polar squash,
                # being along z, leaves that plane untouched.
                self._rings[body].set_transform(matrix)

    def _apply_radii(self) -> None:
        self._place_globes()
        self._set_line_radius(scene.line_radius_au(self.view_scale))

    def _set_line_radius(self, line: float) -> None:
        self._line_radius = line
        for orbit in self._orbits.values():
            orbit.set_radius(line, relative=False)
        for path in self._trails.values():
            path.set_radius(line * 1.8, relative=False)

    def track_camera(self) -> None:
        """Rescale the orbit lines to however far away the camera now is.

        Called every frame. Without it the lines keep the thickness the opening
        framing gave them, and flying in from 52 au to look at the Earth leaves
        a 0.2 au tube wrapped round a 1 au orbit.
        """
        import polyscope as ps

        eye = np.asarray(ps.get_view_camera_parameters().get_position(), dtype=float)
        distance = float(np.linalg.norm(eye - np.asarray(ps.get_view_center())))
        wanted = scene.line_radius_from_camera(distance)

        # Only when it has actually moved: eighteen curve networks a frame for
        # a change nobody can see is not worth the redraw.
        if abs(wanted - self._line_radius) > 0.02 * self._line_radius:
            self._set_line_radius(wanted)

    def apply_view(self, name: str) -> None:
        """Reframe the scene, and resize spheres and lines to suit."""
        import polyscope as ps

        preset = scene.VIEW_PRESETS[name]
        self.view_name = name
        self.view_scale = preset.scale_au
        self.planet_exaggeration = preset.planet_exaggeration
        self.sun_exaggeration = preset.sun_exaggeration
        self.visible = set(preset.bodies)
        self._apply_radii()
        self._apply_visibility()
        s = self.view_scale
        ps.look_at((0.0, -2.2 * s, 1.3 * s), (0.0, 0.0, 0.0))

    def _apply_visibility(self) -> None:
        if self.focus is not None:
            # Everything else is either off-screen or a subpixel speck, and the
            # orbit rings would be straight lines through the frame.
            self._sun.set_enabled(self.focus == "sun")
            for body in ORDER:
                self._planets[body].set_enabled(body == self.focus)
                self._orbits[body].set_enabled(False)
                self._trails[body].set_enabled(False)
                if body in self._rings:
                    self._rings[body].set_enabled(body == self.focus)
            return

        self._sun.set_enabled(True)
        for body in ORDER:
            shown = body in self.visible
            self._planets[body].set_enabled(shown)
            self._orbits[body].set_enabled(shown and self.show_orbits)
            self._trails[body].set_enabled(shown and self.show_trails)
            if body in self._rings:
                self._rings[body].set_enabled(shown)

    # -- focus ---------------------------------------------------------------

    def focus_on(self, body: str | None) -> None:
        """Fly to one body at its true size, or back out to the system."""
        import polyscope as ps

        if body is not None and body != "sun" and body not in self._planets:
            raise ValueError(f"cannot focus on {body!r}")
        self.focus = body

        if body is None:
            ps.set_bounding_box((-52.0, -52.0, -52.0), (52.0, 52.0, 52.0))
            self.apply_view(self.view_name)
            return

        # The sphere is a unit sphere in focus mode, so the scene is a few
        # radii across and the near and far planes have something sane to work
        # with.
        reach = (globe.ring_span(body)[1] * 1.25) if globe.has_rings(body) else 1.6
        ps.set_bounding_box((-reach,) * 3, (reach,) * 3)
        self._place_globes()
        self._apply_visibility()
        self.aim_camera()

    def aim_camera(self) -> None:
        """Point the camera at the focused body, from the Sun or from the side."""
        import polyscope as ps

        if self.focus is None:
            return
        if self.focus_from_sun and self.focus != "sun":
            toward_sun = -position(self.focus, self.jd)
            toward_sun = toward_sun / np.linalg.norm(toward_sun)
            reach = 3.2 * self._frame_radius()
            ps.look_at(tuple(reach * toward_sun), (0.0, 0.0, 0.0))
        else:
            reach = self._frame_radius()
            ps.look_at((3.0 * reach, -1.6 * reach, 1.2 * reach), (0.0, 0.0, 0.0))

    def _frame_radius(self) -> float:
        """How far out the camera has to sit to fit the body and its rings."""
        if self.focus and globe.has_rings(self.focus):
            return globe.ring_span(self.focus)[1] * 0.85
        return 1.0

    def focus_report(self) -> list[str]:
        """A line or two about what is being looked at."""
        from . import frames, rotation

        if self.focus is None:
            return []
        body = self.focus
        lines = [
            f"{body}: tilt {float(rotation.obliquity_degrees(body, self.jd)):.1f} deg,"
            f" day {abs(rotation.rotation_period_days(body)) * 24:.2f} h"
        ]
        if body != "sun":  # the Sun does not have the Sun overhead anywhere
            toward_sun = frames.ecliptic_to_equatorial(-position(body, self.jd))
            latitude, longitude = rotation.surface_point(body, self.jd, toward_sun)
            lines.append(
                f"sun overhead at {float(latitude):+.1f} lat,"
                f" {float(longitude):.1f} east lon"
            )
        return lines

    def refresh_positions(self) -> None:
        """Move everything to the current date. Node counts never change."""
        self._place_globes()
        if self.focus is not None:
            return  # the rings and trails are hidden; do not pay to move them
        for body in self._planets:
            self._orbits[body].update_node_positions(
                scene.orbit_loop(body, self.jd, self.orbit_samples)
            )
            self._trails[body].update_node_positions(self._trail_nodes(body))

    def advance(self) -> None:
        if not self.playing:
            return
        self.jd += self.days_per_frame
        if self.jd >= VALID_JD[1]:
            self.jd = VALID_JD[0]  # wrap rather than extrapolate
        self.refresh_positions()

    # -- interface -----------------------------------------------------------

    def gui(self) -> None:
        import polyscope as ps
        import polyscope.imgui as psim

        self._frame += 1
        self.track_camera()
        if self._screenshot_path and self._frame == self._screenshot_frame:
            ps.screenshot(self._screenshot_path, transparent_bg=False)

        moved = False

        psim.TextUnformatted(f"{isoformat(self.jd)}  TDB")
        psim.PushItemWidth(220)

        changed, day = psim.SliderFloat(
            "date", jd_to_day(self.jd), v_min=DAY_MIN, v_max=DAY_MAX
        )
        if changed:
            self.jd = float(np.clip(day_to_jd(day), *VALID_JD))
            moved = True

        for name in scene.VIEW_PRESETS:
            if psim.Button(name):
                self.focus_on(None)
                self.apply_view(name)
            psim.SameLine()
        psim.TextUnformatted("")

        # Focus: leave the solar system view and go and look at one body at its
        # real size. At system scale the Earth is fourteen pixels across, so the
        # maps are there but there is nothing to see.
        choices = ["-- system --", "sun", *ORDER]
        current = 0 if self.focus is None else choices.index(self.focus)
        changed_focus, picked = psim.Combo("focus", current, choices)
        if changed_focus:
            self.focus_on(None if picked == 0 else choices[picked])

        if self.focus is not None:
            if psim.Button("re-aim"):
                self.aim_camera()
            psim.SameLine()
            flipped, self.focus_from_sun = psim.Checkbox(
                "lit face", self.focus_from_sun
            )
            if flipped:
                self.aim_camera()
            for line in self.focus_report():
                psim.TextUnformatted(line)
            psim.Separator()

        if psim.Button("play" if not self.playing else "pause"):
            self.playing = not self.playing
        psim.SameLine()
        if psim.Button("now"):
            self.jd = julian_date(2026, 9, 3)
            moved = True

        _, self.days_per_frame = psim.SliderFloat(
            "days/frame", self.days_per_frame, v_min=0.1, v_max=200.0
        )

        psim.Separator()
        psim.TextUnformatted("distances are real; radii are not")

        changed_planet, planet_log = psim.SliderFloat(
            "planet size x10^", np.log10(self.planet_exaggeration), v_min=0.0, v_max=4.0
        )
        changed_sun, sun_log = psim.SliderFloat(
            "sun size x10^", np.log10(self.sun_exaggeration), v_min=0.0, v_max=3.0
        )
        if changed_planet or changed_sun:
            self.planet_exaggeration = float(10.0**planet_log)
            self.sun_exaggeration = float(10.0**sun_log)
            self._apply_radii()

        psim.TextUnformatted(
            f"planets x{self.planet_exaggeration:.0f}, sun x{self.sun_exaggeration:.0f}"
        )
        if not scene.sun_fits_inside_mercury(self.sun_exaggeration):
            psim.TextUnformatted("the drawn Sun now swallows Mercury's orbit")

        psim.Separator()
        changed_orbits, self.show_orbits = psim.Checkbox("orbits", self.show_orbits)
        psim.SameLine()
        changed_trails, self.show_trails = psim.Checkbox("trails", self.show_trails)
        if changed_orbits or changed_trails:
            self._apply_visibility()

        changed_trail, self.trail_fraction = psim.SliderFloat(
            "trail (orbits)", self.trail_fraction, v_min=0.01, v_max=1.0
        )

        psim.PopItemWidth()

        if moved or changed_trail:
            self.refresh_positions()
        self.advance()

    def run(
        self,
        *,
        frames: int | None = None,
        screenshot: str | None = None,
        backend: str = "",
        focus: str | None = None,
    ) -> None:
        import polyscope as ps

        ps.init(backend) if backend else ps.init()
        self.build()
        ps.set_user_callback(self.gui)

        # After build(), because focusing needs the meshes it registers.
        if focus is not None:
            self.focus_on(focus)

        # Screenshots are taken from inside the callback, not after show()
        # returns: by then polyscope has torn the window down and there is
        # nothing left to capture.
        self._screenshot_path = screenshot
        self._screenshot_frame = 2

        if frames is None:
            ps.show()
        else:
            ps.show(forFrames=max(frames, self._screenshot_frame + 1 if screenshot else 1))
