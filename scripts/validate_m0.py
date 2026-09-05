"""M0 gate: are the positions right?

Compares every position this package produces against JPL's DE440 ephemeris,
over a grid of dates spanning the element table's useful range, and reports the
error per planet in kilometres and in arcseconds of apparent sky position.

Why this runs before a single pixel is drawn: a wrong orbit still looks like an
ellipse. Planets still circle, still speed up near perihelion, still make a
pretty picture. There is no visual symptom to catch a bad angle conversion or a
mis-typed element, so the only honest check is an independent implementation of
the same question -- and the JPL ephemeris is not an approximation of the same
model, it is a numerical integration fitted to radar and spacecraft tracking.

Two numbers are reported per body:

* **position error** -- how far the modelled planet is from where it really was,
  in km. The physically meaningful one.
* **sky error** -- the angle between where this package says the planet appears
  from Earth and where it actually appeared. What an observer would notice, and
  the number that decides whether M1's conjunction checks can pass. It uses
  *this package's* Earth as well as its target, so Earth's own error is in it.

Usage::

    python scripts/validate_m0.py                    # check against the baseline
    python scripts/validate_m0.py --update-baseline  # record current numbers
    python scripts/validate_m0.py --offline          # refuse to hit the network
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from orrery import elements, frames, kepler, times, truth  # noqa: E402

BASELINE = Path(__file__).resolve().parents[1] / "data" / "baseline_m0.json"

# Gross-blunder ceiling: half a degree of sky error is the apparent diameter of
# the Moon. Nothing that passes M0 should be visibly wrong to the naked eye.
CEILING_SKY_ARCSEC = 1800.0

# A regression must be this much worse than the baseline before the gate fails.
# Loose enough to absorb a changed date grid, tight enough that a real mistake
# shows up.
REGRESSION_FACTOR = 1.25


def build_dates(start_year: int, end_year: int, count: int) -> np.ndarray:
    return times.linspace(times.jd(start_year, 1, 1), times.jd(end_year, 1, 1), count)


def measure(jd_grid: np.ndarray, *, offline: bool) -> dict[str, dict[str, float]]:
    bodies = list(elements.ORDER)

    fetch = truth.cached_only if offline else truth.heliocentric_equatorial
    reference = fetch(bodies, jd_grid)

    # Ours, rotated into the equatorial frame the reference is expressed in.
    # Rotation preserves both distances and angles, so this changes nothing
    # about the measured error -- it just removes a conversion from the
    # reference side, where a mistake would be indistinguishable from a win.
    ours = {
        b: frames.ecliptic_to_equatorial(kepler.position(b, jd_grid)) for b in bodies
    }

    results: dict[str, dict[str, float]] = {}
    for body in bodies:
        delta = ours[body] - reference[body]
        dist_km = frames.norm(delta) * kepler.AU_KM

        row = {
            "max_pos_km": float(dist_km.max()),
            "rms_pos_km": float(np.sqrt(np.mean(dist_km**2))),
            "max_helio_arcsec": float(
                frames.separation_arcsec(ours[body], reference[body]).max()
            ),
        }

        if body == "embary":
            # The observer cannot be displaced relative to itself.
            row["max_sky_arcsec"] = float("nan")
        else:
            seen_ours = ours[body] - ours["embary"]
            seen_ref = reference[body] - reference["embary"]
            row["max_sky_arcsec"] = float(
                frames.separation_arcsec(seen_ours, seen_ref).max()
            )
        results[body] = row

    return results


def report(results: dict[str, dict[str, float]], jd_grid: np.ndarray) -> None:
    span = f"{times.isoformat(jd_grid[0])[:10]} .. {times.isoformat(jd_grid[-1])[:10]}"
    print(f"\nM0 -- Keplerian elements vs JPL DE440")
    print(f"{len(jd_grid)} dates, {span}\n")
    print(f"{'body':<9}{'max pos':>12}{'rms pos':>12}{'helio':>11}{'sky':>11}")
    print(f"{'':<9}{'km':>12}{'km':>12}{'arcsec':>11}{'arcsec':>11}")
    print("-" * 55)
    for body, row in results.items():
        sky = row["max_sky_arcsec"]
        sky_text = "  --" if np.isnan(sky) else f"{sky:10.1f}"
        print(
            f"{body:<9}{row['max_pos_km']:12.0f}{row['rms_pos_km']:12.0f}"
            f"{row['max_helio_arcsec']:11.1f}{sky_text:>11}"
        )
    print()


def compare_to_baseline(results: dict[str, dict[str, float]]) -> list[str]:
    """Return a list of failure descriptions; empty means the gate passes."""
    failures: list[str] = []

    for body, row in results.items():
        sky = row["max_sky_arcsec"]
        if not np.isnan(sky) and sky > CEILING_SKY_ARCSEC:
            failures.append(
                f"{body}: sky error {sky:.0f}\" exceeds the "
                f"{CEILING_SKY_ARCSEC:.0f}\" blunder ceiling"
            )

    if not BASELINE.exists():
        failures.append(
            "no baseline recorded; inspect the table above, then rerun with "
            "--update-baseline to lock it in"
        )
        return failures

    saved = json.loads(BASELINE.read_text())["results"]
    for body, row in results.items():
        for metric in ("max_pos_km", "max_sky_arcsec"):
            was, now = saved.get(body, {}).get(metric), row[metric]
            if was is None or np.isnan(now) or (was is not None and np.isnan(was)):
                continue
            if now > was * REGRESSION_FACTOR:
                failures.append(
                    f"{body}: {metric} regressed, {was:.4g} -> {now:.4g}"
                )
    return failures


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--start", type=int, default=1850)
    parser.add_argument("--end", type=int, default=2050)
    # The error is oscillatory, not monotonic, so a coarse grid samples the
    # envelope by luck and the reported maximum moves around between runs.
    # Every ~36 days is dense enough for the maximum to be stable.
    parser.add_argument("--dates", type=int, default=2001)
    parser.add_argument("--offline", action="store_true", help="require cached truth")
    parser.add_argument("--update-baseline", action="store_true")
    args = parser.parse_args()

    jd_grid = build_dates(args.start, args.end, args.dates)
    results = measure(jd_grid, offline=args.offline)
    report(results, jd_grid)

    if args.update_baseline:
        BASELINE.write_text(
            json.dumps(
                {
                    "grid": {
                        "start": args.start,
                        "end": args.end,
                        "dates": args.dates,
                    },
                    "results": results,
                },
                indent=2,
            )
        )
        print(f"baseline written to {BASELINE.relative_to(BASELINE.parents[1])}")
        return 0

    failures = compare_to_baseline(results)
    if failures:
        print("M0 FAILED")
        for line in failures:
            print(f"  - {line}")
        return 1

    print("M0 passing")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
