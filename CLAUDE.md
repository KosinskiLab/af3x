# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Overview

AF3x is a custom fork of AlphaFold 3 that enables explicit modeling of crosslinks by adding crosslinker molecules as ligands with covalent bonds to crosslinked residues (rather than using distance restraints).

## Environment

AF3x is already installed as an editable install. Activate the conda environment before working:

```bash
mamba activate af3x
```

HMMER is not installed system-wide — load it via the module system when the data pipeline is needed:

```bash
module load HMMER/3.4-gompi-2023a
```

Note: `--run_data_pipeline=false` can be used to skip the data pipeline (and HMMER) when pre-computed MSA features are already available.

## Installation Notes (for reference)

The package has a C++ extension built via CMake (fetches abseil-cpp, pybind11, libcifpp, dssp). `build_data` generates CCD pickle files in `src/alphafold3/constants/converters/`. To reinstall from scratch:

```bash
pip install -r dev-requirements.txt
pip install -e . --no-deps
build_data
```

## Running Tests

Tests use `absltest` (not pytest). The CI runs CPU-only data pipeline tests:

```bash
python run_alphafold_data_test.py
python run_alphafold_test.py
```

Test data lives in `src/alphafold3/test_data/crosslinks/` (cases: 4G3Y, 8WW0, 9G5K).

## Main Entry Point

```bash
python run_alphafold.py \
  --json_path=<input.json> \
  --output_dir=<output/> \
  --model_dir=~/models \
  --db_dir=<database_dir>
```

AF3x-specific flags:
- `--sample_crosslink_combinations=<int>`: Enumerate crosslink combinations
- `--remove_overlapping_crosslinks=true|false` (default: true)
- `--num_seeds=<int>`: Number of random seeds

## Architecture

### Data Flow

1. **Input**: JSON with sequences + crosslinks section (specifying crosslinker type and residue pairs)
2. **Data pipeline** (`src/alphafold3/data/`): MSA via HMMER/Jackhmmer, template search, feature prep
3. **Crosslink integration** (`src/alphafold3/crosslinks/`): Parse crosslink definitions → create crosslinker ligands with chemical structures → add covalent bonds to residues
4. **Model inference** (`src/alphafold3/model/`): Evoformer + diffusion head + confidence head
5. **Output**: Predicted structures (CIF/PDB), pLDDT/pAE confidence scores

### Key Modules

- `src/alphafold3/crosslinks/crosslink_definitions.py` — Crosslinker type definitions (SMILES, bond attachment points). This is the primary file to edit when adding new crosslinkers.
- `src/alphafold3/crosslinks/dynamic_crosslink.py` — Runtime crosslink handling, combination sampling, overlap removal
- `src/alphafold3/crosslinks/create_crosslinker.py` — Builds crosslinker ligand structures
- `src/alphafold3/common/folding_input.py` — Parses JSON input and integrates crosslink specs
- `src/alphafold3/model/model.py` — Core model; orchestrates feature batching, Evoformer, diffusion, scoring
- `src/alphafold3/constants/` — Atom types, residue names, chemical components

### Supported Crosslinkers

DSSO, DSS, DSG, CDI, BS3, BS2G, azide-A-DSBSO, DSBU, PHOX, BSPEG5, BSPEG9, SDA, LCSDA, SDAD, SDA25A (and experimental disulfide bonds).

### Adding a New Crosslinker

Define it in `crosslink_definitions.py` following the existing pattern (SMILES string, reactive group positions, crosslinker name). See recent commits for examples (CDI, SDA, LCSDA, SDAD, SDA25A).

## Input JSON Format

The crosslinks section specifies the crosslinker type and pairs of residues:
```json
{
  "crosslinks": [
    {
      "crosslinker": "DSSO",
      "residue1": {"chain": "A", "residue_number": 10},
      "residue2": {"chain": "B", "residue_number": 25}
    }
  ]
}
```

See `docs/input.md` for full documentation and `src/alphafold3/test_data/crosslinks/` for example inputs.

## CI

GitHub Actions (`.github/workflows/ci.yaml`) runs `run_alphafold_data_test.py` on push/PR to main using Python 3.11 on ubuntu-latest.
