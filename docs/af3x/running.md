# Running and output

## Running

Run AF3x like AlphaFold 3, with an input JSON that lists the crosslinks (XLs) ([Crosslink input](crosslink_input.md)).
AF3x adds the crosslinker ligands automatically:

```sh
python run_alphafold.py \
  --json_path=input.json \
  --output_dir=output \
  --model_dir=<MODEL_DIR> \
  --db_dir=<DB_DIR> \
  --num_seeds=20
```

All AlphaFold 3 flags work as usual (`python run_alphafold.py --helpfull`).

### AF3x flags

| Flag | Default | Effect |
|---|---|---|
| `--remove_overlapping_crosslinks` | `true` | Drops every XL that uses a residue already used by an earlier XL (input order, across all XL types). Dropped XLs are only logged. Disulfides are not checked. Set `false` to keep all |
| `--sample_crosslink_combinations=N` | `0` (off) | For each input with XLs, runs one job per combination of N of its XLs (all `C(n, N)` combinations). Job names get one suffix per XL, e.g. `9G5K_DSSO_A104-B43`. `N=1`: one prediction per XL. Fails if N is larger than the number of XLs |
| `--ablate_crosslinkers` | `false` | Control experiment: XL tokens are zeroed before the diffusion module, and the XL ligand chains are removed from the output structure. The trunk (Evoformer) still sees them |

Combinations are built first, then overlaps are removed. Both apply only when inference runs (not with
`--norun_inference`).

### Seeds

AlphaFold 3 gives different structures per seed. Preferably use 20 seeds (`--num_seeds=20` with one seed in the
JSON): 100 models with the default 5 diffusion samples per seed. More seeds increase the compute time
proportionally.

## Output

The output files are those of AlphaFold 3 ([AlphaFold 3 output](../output.md)), plus:

- XL ligand chains in the structures, with the first unused chain ids (`A`…`Z`, then `AA`, `BA`, …) after all your
  chains.
- `<job>_data.json` is written **before** the XLs are expanded: it keeps the `crosslinks` key and contains no XL
  ligand chains, so it can be used again as input.
- `<job>_af3x_version.json` (AF3x commit, AlphaFold 3 base, package versions). The mmCIF `_software` entry carries
  the same AF3x version id.

AF3x outputs are AlphaFold 3 outputs: the
[AlphaFold 3 output terms of use](https://github.com/KosinskiLab/af3x/blob/main/OUTPUT_TERMS_OF_USE.md) apply.

### Checking XL satisfaction

When not all XLs can be satisfied at once, AlphaFold 3 sometimes forms some of the bonds and stretches or breaks
the others. `eval_crosslinks.py` measures, in each model, the distance between the crosslinked atoms (or Cα atoms)
and reports which XLs are within a threshold:

```bash
python eval_crosslinks.py -j input.json -f output/*/*_model.cif -t 30 --summary_out_csv summary.csv
```

Per-XL thresholds: `-T DSS:30_DSG:20`. Other options: `--use_calpha`, `--save_distances_csv`,
`--find_shortest_homooligomeric`, `--extract_avg_plddt`. Tested on AlphaFold 3 mmCIF models only.
