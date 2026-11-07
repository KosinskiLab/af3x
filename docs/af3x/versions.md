# Versions

AF3x versions are git tags `af3x-vX.Y.Z`. The minor version increases with each sync to a newer AlphaFold 3 and with
notable features, the patch version with fixes.

| AF3x | AlphaFold 3 base | Python / JAX | Changes |
|---|---|---|---|
| `af3x-v0.3.0` | v3.0.5 | 3.12 / 0.10.2 | Sync to AlphaFold 3 v3.0.5 (stable MSA pairing order, faster structure code); documentation in `docs/af3x/` |
| `af3x-v0.2.0` | commit `87bd9e6` (after v3.0.1) | 3.12 / 0.9.1 | Sync to AlphaFold 3 of April 2026; `--ablate_crosslinkers`; version stamp in every output; `<job>_data.json` written before XL expansion (reusable as input); fixes for user-given bonded atom pairs, `RIGID<N>` and XL chain ids |
| `af3x-v0.1.0` | v3.0.1 | 3.11 / 0.4.34 | First stable public version |

Install a specific version with `git checkout af3x-v0.3.0` before installing.

## Reproducibility

- The same input and seed give different structures on different AF3x versions (same model parameters, different
  code and packages). Compare predictions seed by seed only within one version; cite the version you used.
- Every run (from v0.2.0) writes `<job>_af3x_version.json`: AF3x commit and branch, whether the code had local
  changes, the AlphaFold 3 base commit and the versions of key packages. The AF3x version id is also in the mmCIF
  `_software` entry.
- `pip show alphafold3` reports the AF3x version (e.g. `0.3.0`); the AlphaFold 3 version is in
  `alphafold3.version.__version__`.
