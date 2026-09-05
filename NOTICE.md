# Notices and attribution

This project is MIT licensed (see `LICENSE`). It bundles and downloads third
party data, which carries its own terms.

## Bundled in this repository

**Planet, moon and ring maps** — `data/textures/*.jpg`, `*.png`
Solar System Scope, <https://www.solarsystemscope.com/textures/>,
licensed **CC BY 4.0** (<https://creativecommons.org/licenses/by/4.0/>).

Attribution is a condition of that licence, so it is repeated here and applies
to the figures derived from those maps: every image in `docs/images/` that shows
a planet's surface, including the world map behind the eclipse tracks and the
close-up portraits.

**Cached ephemeris values** — `data/fixtures/*.npz`, `data/delta_t.npz`
Sampled from JPL DE440 and from IERS Earth-orientation data by way of Skyfield.
Committed so the validation gates reproduce offline. Both sources are US
government works and not subject to copyright.

## Downloaded on first run, not committed

**JPL DE440s planetary ephemeris** — `data/de440s.bsp`, about 32 MB
Jet Propulsion Laboratory, California Institute of Technology.
Fetched automatically by `orrery.truth` the first time a gate runs.

## Referenced, not redistributed

- **Keplerian elements** for the major planets: E. M. Standish, JPL Solar System
  Dynamics, <https://ssd.jpl.nasa.gov/planets/approx_pos.html>. The table is
  transcribed into `src/orrery/elements.py` and diffed against the live page by
  `tests/test_elements.py`.
- **Rotation elements**: IAU/IAG Working Group on Cartographic Coordinates and
  Rotational Elements, transcribed into `src/orrery/rotation.py`.
- **Delta T polynomials**: Espenak & Meeus, used as the fallback when no
  measured table is cached.
- **Skyfield** (MIT) is an optional dependency, used only as the independent
  reference the gates are measured against. Nothing in `src/orrery` imports it
  except `truth.py`, which is the module whose job that is.
