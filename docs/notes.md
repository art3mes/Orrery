# Notes

[← back to the README](../README.md) · [the milestones](milestones.md)

Longer-form material that would crowd the README: how the model works, what
went wrong on the way, the decisions that are not obvious from the code, and the
full list of things this does not do.

---

## The idea

Six numbers describe an ellipse in space and where a planet sits on it. Six more
describe how those numbers drift, per century. JPL publishes both, for every
planet, in a table small enough to paste into a source file — 108 floats for
the whole solar system.

Turning them into a position needs one awkward step. A planet moves fast near
the Sun and slow far away, so *how much of the orbital period has elapsed* is
not *how far round the ellipse the planet has gone*. Converting between them is
Kepler's equation:

```
M = E - e sin(E)
```

`M` follows from elapsed time. `E` fixes the position. No rearrangement gives
`E` in closed form, so it is solved by Newton's method — about ten lines, four
iterations, machine precision. Everything after that is rotating a 2-D ellipse
into 3-D by three angles.

That is the whole model. The interesting question is not whether it produces
ellipses; it is *how wrong* the ellipses are, which is what M0 answers.


## What the gate caught

**The error had to be shown to oscillate.** The first run gave numbers a few
times larger than JPL's published accuracy figures, which is exactly the size a
subtle frame or units bug would produce. Plotting error against time settled it:
every body passes through near-zero at some epoch — Venus 0.3″ in 2024, Mars
1.0″ in 2000, Neptune 0.6″ in 1850. A constant offset cannot do that, so the
frame rotation and unit conversions are clean and the residual is model error.
Reporting a number without this check would have been reporting a number without
knowing what it meant.

**Refining a minimum made it worse.** The event finder fits a parabola through
the three samples around an extremum, which normally beats taking the smallest
sample. At a conjunction it did the opposite: on a 0.02-day grid it placed the
2020 closest approach 24 minutes *further* from DE440 than its own raw argmin.
A near-flat minimum makes the fitted curvature badly conditioned, and the vertex
flies outside the bracket. The fix is a theorem, not a fudge — for a smooth
function on a uniform grid the extremum is always within half a step of the
argmin — so clamping there costs nothing when the fit is good and saves it when
it is not. Caught only because two scripts computing the same DE440 quantity
disagreed by 28 minutes.

**And then the clamp hid a real bug for six milestones.** The same parabola,
fitted in raw Julian dates. A date is 2.46 × 10⁶ and a two-minute grid step is
1.4 × 10⁻³, so the *x*² terms in the fitted coefficients agree to sixteen digits
and cancel to nothing — the linear coefficient came out as pure rounding noise,
the vertex flew off, and the clamp caught it every single time and quietly
returned the raw argmin displaced by half a step. Nothing failed. Every gate
passed. It surfaced only when one of the README's recipes asked for the *value* at
an extremum rather than its time, and a 95%-covered Sun reported 0.0%. Fitting
in *x* − *x*₁ instead costs one subtraction; Mars's stationary points went from
5.5 minutes off Skyfield to 1.1, and the 2020 conjunction moved six minutes. The
lesson is not about floating point. It is that a safety net added for one
failure will happily absorb a different one and report success.

**Conserving energy is not the same as getting the orbit right.** The symplectic
integrator holds energy in a band of 5.7 × 10⁻⁷ for a thousand years, which
looks like a licence to trust it. Run the Sun and Mercury alone — a two-body
orbit, which does not precess at all — and it turns the apse line at 67″ per
century out of nothing, from a fixed step and finite-difference truncation
alone. That is an eighth of the Newtonian signal M2 sets out to measure. The
control run is therefore not a nicety: without it, "532″ per century" would have
been reported as 596. It converges as the fourth power of the step (2636″ at
dt = 0.5 d, 67.6″ at 0.2, 4.2″ at 0.1) and it cancels almost exactly between two
runs that differ only in whether the GR term is switched on, which is why the
43″ result is far sturdier than the 529″ one.

**The plane you measure the perihelion against changes the answer.** The trap
flagged before building M2 was equinox precession — 5028″/century of pure
coordinate motion. The one actually hit was quieter: longitude of perihelion is
Ω + ω, both referred to a reference plane, and reading the same state vectors
against the equator instead of the ecliptic moves Mercury's rate by 11″/century.
That is a quarter of the entire GR signal, and both numbers look perfectly
reasonable. Every rate M2 quotes names its plane.

**A tolerance finer than the numbers could represent.** The light-time solver
iterated until two successive emission times agreed to 1e-12 days. A Julian date
is about 2.46 × 10⁶, where a float64 steps by 4.7 × 10⁻¹⁰ — so that threshold
could only ever be met by two iterates landing on the *same bit pattern*.
Distant planets do reach such a fixed point and it worked for three milestones.
The Moon, seen from a point on a spinning Earth, flips between two neighbouring
representable values forever, and M4 was the first thing to ask. Now 1e-9 days:
86 microseconds, in which the Moon moves 9 cm.

**A triangle subtracted twice.** The area where two discs overlap is two
circular segments minus the kite between the intersection points; the kite came
off twice. Two equal discs a radius apart read 0.115 instead of 0.391. It could
only ever show on *partial* eclipses — total and annular ones take the
one-disc-inside-the-other branch — so every headline number was right and
Rome's 2027 obscuration was reported as 58.6% when it is 74.6%.

**A correction I talked myself into.** Retarding the Sun by the light-time to
the Moon looked like it should shift the shadow 40 km, and the comment said so.
It moves it **7 km**: what travels in those 8.3 minutes is the Sun's
*barycentric* motion, 12 m/s, not the Earth's 30 km/s. Retarding the observer is
the big correction; retarding the source is not. The code kept the term because
it is correct, and the comment was rewritten because it was not.

**The flat minimum came back.** Greatest eclipse is when the shadow axis passes
closest to the Earth's centre — and for 2024 that miss distance changes by under
3 km across three minutes. The time is badly conditioned even though the
geometry is not, exactly as a 2.5′ position error moved M1's great conjunction
by ten hours. So M4 claims the landing point to 17 km and the duration to 2 s,
and the instant only to three minutes.

**"North" means two different things.** The IAU fixes a planet's north pole as
the one on the north side of the invariable plane, whatever way the body turns,
and puts the sense of rotation in the sign of the W rate. So Venus's tabulated
pole sits 2.6 degrees from its orbit normal — which would make it the most
upright planet in the solar system, and it is a perfectly plausible number. Its
obliquity is 177.4: it turns backwards, and its angular momentum points the
other way. Uranus reads 82.2 instead of 97.8 the same way. Dwarf planets use the
right-hand rule instead, so Pluto needs no flip and Pluto alone. Caught by
checking obliquities against published values rather than eyeballing a globe.

**The map starts at the antimeridian.** An equirectangular planet map puts the
prime meridian down the *middle* of the image, so longitude 0 is u = 0.5. The
sphere was built starting at longitude 0, which rotated every planet by half a
turn. On Jupiter that is invisible. On the Earth it put the Sun over the Pacific
at noon UT, which is how it was found — and the fix is checked by sampling ten
known coastlines rather than by looking.

**A test that agreed for the wrong reason.** M3's deflection test placed a
target a thousand au behind the Sun, offset sideways by one solar radius *in au*
— which sounds like grazing the limb and is not. The impact parameter of that
ray is a thousand times smaller, so it passes straight through the Sun, where
the guard clause correctly suppresses the whole correction. The test measured
zero and reported the answer to be 1.75 arcsec off. The geometry, not the code,
was wrong: an observer 1 au out has to aim at the Sun's *angular* radius.

**A "final" state that was nothing of the kind.** `integrate()` recorded samples
every *n* steps and stopped there, so when the span was not a whole number of
sampling intervals the last recorded state was from partway through the run. The
time-reversibility test surfaced it, and only because the numbers were absurd
rather than merely bad. Its earlier version had been passing vacuously: with the
sampling interval larger than the whole run, it compared the initial state
against itself and got exactly zero.

**The great conjunction is a trap.** `demo_m0.py` finds Jupiter and Saturn 6.000
arcmin apart in December 2020 against DE440's 6.104 — apparently arcminute
accuracy from a model with arcminute errors. It is not. On that date the model
puts Jupiter 0.4′ and Saturn 2.5′ from their true positions; both displacements
point much the same way, and most of the error cancels in the difference. The
*date* is the weaker number: the separation curve is nearly flat at closest
approach, so Saturn's 2.5′ moves the minimum by ten hours. M1 gates conjunctions
to about a day and does not claim arcminute separations.


## Choices worth knowing about

**Everything is TDB.** No leap seconds, no UTC conversion anywhere. UTC differs
from TDB by ~69 s today and the Earth covers 2000 km in 69 s — a tenth of the
error budget, thrown away for nothing. Both sides of the gate are built in TDB,
so the question never arises.

**"Earth" is the Earth–Moon barycentre.** That is what the table's third row
is. The Earth itself orbits it by up to ~4700 km. `canonical("earth")` returns
`embary` and the name is used everywhere the distinction could matter.

**Ground truth is geometric, not apparent.** `truth.py` uses Skyfield's `.at()`,
not `.observe()`, so there is no light-time correction and no aberration on
either side. Those are real effects belonging to an observer, and folding them
into one side only would be comparing two different questions. Light-time
arrives in M3.

**Scene units are au, exactly.** Positions are never rescaled, compressed or
log-mapped — the usual trick for fitting Pluto and Mercury in one picture, and
the reason most pretty solar systems are useless for measuring anything. The
only lie is sphere size, it is linear so size *ratios* survive, and the factor
is displayed. The Sun gets its own much smaller factor; at the planets' setting
its sphere would reach past Mercury's perihelion, and the viewer says so when it
does.

**Orbit rings are geometry, not trajectory.** A ring is the osculating ellipse:
elements evaluated once, then the eccentric anomaly swept right round. Sampling
Neptune's actual path over a period would run 165 years past the date asked for,
out of the table and into extrapolation. Sweeping E stays at one instant, so the
ring is exactly as valid as the position is.

**The scrubber counts days, not Julian dates.** ImGui sliders are single
precision, and float32 steps by 0.25 days at 2.45 × 10⁶ — a slider bound to the
Julian date snaps in six-hour jumps. Days since 1850 reach 73048, where the step
is 11 minutes. Fractional years, the obvious alternative, are worse at 64.

**M2 integrates in the barycentric frame, with the Sun as a body.** Keeping the
Sun at the origin is the obvious thing and it is wrong: Jupiter pulls the Sun
about, so that frame accelerates, and a symplectic integrator in an accelerating
frame is not symplectic. The energy behaviour that was the entire reason for
choosing one quietly goes away.

**Masses are carried as GM, not as kilograms.** GM of the Sun is known to ten
significant figures; the mass of the Sun in kilograms is known to five, because
*G* is. Dividing by G to get a mass and multiplying by it again to get a force
throws away five digits for nothing.

**The mass ratios are the one unchecked input.** Every other number in the
package is derived from something else here or diffed against its source. The
Sun-to-planet mass ratios are not: they are typed in from the DE440 header. What
stands behind them is the 50-year drift gate, where a mass wrong by a part in a
thousand would show up immediately in the across-track column.

**The velocity is the two-body one.** `state()` computes speed from the
semi-major axis using GM of the Sun alone, ignoring the century drift of the
elements. It is the right thing for seeding an N-body run and agrees with a
numerical derivative of the position to a part in 1e4 or better. For Pluto the
gap reaches 0.07%, because the table's semi-major axis and its mean-longitude
rate were fitted independently and imply periods differing by that much.


## Caveats

- **1850–2050, and no further.** The element table is specified for 1800–2050,
  and DE440s starts in December 1849, so 1850 is where the two overlap. Dates
  outside the table's range raise a `RuntimeWarning`; the rates are linear
  extrapolations and degrade fast. JPL publishes a second table for
  3000 BC – 3000 AD with extra periodic terms for Jupiter outward; it is not
  implemented here.
- **Nothing is validated against an observation.** DE440 is the reference, and
  DE440 is itself a fit. That is the right reference for this milestone, but it
  means the gate measures agreement with JPL, not truth.
- **The integrator is not a Wisdom-Holman map.** It splits kinetic from
  potential energy, not Kepler motion from perturbation, so it manufactures its
  own apsidal precession (67″/century at a 0.2-day step) and its own mean-motion
  offset (Mercury's 112,000 km of along-track drift over fifty years). Both are
  measured rather than assumed — the two-body control run, and the across/along
  split — and both would largely disappear under a mixed-variable symplectic
  map. That is the obvious next thing to build and it is not built.
- **Gravity is Newtonian plus one term.** The relativistic correction is the
  1PN Schwarzschild term in the Sun's field only: no planet-planet relativistic
  terms, no solar oblateness, no asteroids, no lunar dynamics beyond treating
  the Earth-Moon barycentre as a point. For Mercury's perihelion those are all
  far below the 43″ signal. For Mars over centuries, the asteroids would not be.
- **M1's transit contact times are geometric.** M3's are not: they carry
  light-time and a real observer. The two differ by minutes, which is the point.
- **The nutation series is four terms.** That puts a floor of about half an
  arcsec under anything quoted in coordinates of date, while the underlying
  apparent place is good to microarcseconds. IAU 2000A would remove it. The
  ICRF-to-J2000 frame bias, 23 mas, is also not applied.
- **Delta T now comes from measured values**, cached to `data/delta_t.npz`, and
  falls back to Espenak & Meeus's polynomials if that file is absent. The
  polynomial was 6.2 s out by 2026 and 18.6 s by 2045; eclipse timing needed
  the real thing, which is why M4 depended on it.
- **The lunar theory is abridged.** 120 terms against the full ELP-2000/82's
  twenty thousand, which is 3″ rms and 15″ at worst. M4's eclipse gates still
  run on DE440's Moon; M6 measures the difference rather than swapping it in,
  so the eclipse numbers stay pinned to the best available orbit.
- **Only the centre line, not the path limits.** The track drawn is where the
  shadow *axis* lands. The northern and southern limits of totality, and the
  width of the band, are not computed.
- **The Earth's shadow is enlarged by Danjon's 2%** for the atmosphere. Sources
  differ over the rule, which is most of the 2-3 minute spread on the late lunar
  contacts.
- **No refraction, no rise and set times, no planetary magnitudes.** Refraction
  only matters near the horizon, and it depends on the weather.
- **The globes are lit from everywhere.** No night side, no clouds moving, no
  body casting a shadow on another. Oblateness and the ring systems of Jupiter,
  Saturn and Uranus are drawn and gated in M5; Neptune's rings are not built.
  Pluto has no map at the source used and falls back to flat colour.
- **The rotation table omits its periodic terms.** Mercury, Mars, Neptune and
  the Moon have libration terms of up to 0.7 degrees that are not implemented;
  Neptune's obliquity is 0.47 degrees off because of it. The maps are 2048
  pixels wide, which is 10 arcminutes a pixel, so it does not show.
- **`ephemeris()` puts the Sun at the origin**, not at the barycentre, which is
  exactly what lets it need nothing external. Everything downstream works on
  differences and does not mind, except aberration, which wants the observer's
  *barycentric* velocity and gets its heliocentric one: up to 0.011″. The gates
  all run on `truth.sampled_ephemeris`, which is barycentric.
- **The first gate run needs the network** — 32 MB of DE440s, and Skyfield
  caches it under `data/`. After that the fixture in `data/fixtures/` is enough,
  and `--offline` enforces it. On a machine behind a TLS-inspecting proxy or
  antivirus the download fails certificate verification; installing
  `truststore` (included in the `truth` extra) defers to the OS certificate
  store and fixes it without weakening verification.
- **The regression check is a ratchet, not a specification.** `data/baseline_m0.json`
  records what the model measured, and the gate fails if a change makes it 25%
  worse. It does not know what the numbers *should* be; only the blunder
  ceiling — half a degree of sky error, the apparent width of the Moon — is an
  absolute claim.

