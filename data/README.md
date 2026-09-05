# data

Split three ways, by where each thing comes from and whether it is committed.

## Committed

| | |
|---|---|
| `baseline_m0.json` | What `validate_m0.py` last measured. The gate fails if a change makes it 25% worse. |
| `fixtures/*.npz` | DE440 positions and Skyfield apparent places, cached at the exact dates each gate asks for. About 3 MB. |
| `delta_t.npz` | Measured TT − UT1, 1850–2050, sampled from IERS data. |
| `textures/*.jpg` | Planet maps. About 5 MB, CC BY 4.0 — see `../NOTICE.md`. |

These are committed on purpose. Every gate runs with `--offline` and reproduces
its published numbers from this directory alone, with no network and no
downloads. A result nobody else can reproduce is not much of a result.

## Downloaded on first use, not committed

| | |
|---|---|
| `de440s.bsp` | JPL's planetary ephemeris, 32 MB, covering 1849–2150. Fetched automatically the first time a gate needs ground truth. |

## Regenerating

```bash
python scripts/validate_m0.py            # repopulates the fixtures it needs
python -c "from orrery import observer; observer.build_delta_t_table()"
python -c "from orrery import globe; [globe.fetch_texture(b) for b in globe.TEXTURE_FILES]"
```
