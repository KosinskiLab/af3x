# Crosslink input

List the crosslinks (XLs) in an AlphaFold 3 input JSON; AF3x adds the crosslinker ligands when you run it
([Running and output](running.md)).

**Easiest:** the [AF3x input generator](https://kosinskilab.github.io/af3x-input-generator/) builds the input JSON in
the browser. It fetches sequences from UniProt and imports XLs from xiVIEW/xiNET CSV files. The rest of this page
describes the format it writes.

## JSON format

Add a top-level `crosslinks` list to an AlphaFold 3 input JSON
(see the [AlphaFold 3 input format](../input.md)):

```json
"crosslinks": [
  {
    "name": "DSSO",
    "residue_pairs": [
      [["A", 53], ["B", 66]],
      [["A", 54], ["B", 113]]
    ]
  },
  {
    "name": "SDA",
    "residue_pairs": [
      [["A", 120], ["B", 7]]
    ]
  }
]
```

| Field | Meaning |
|---|---|
| `name` | XL type, exactly as in the tables below (case-sensitive) |
| `residue_pairs` | List of pairs `[[chain_id, residue], [chain_id, residue]]` |
| `chain_id` | Id of a **protein** chain in `sequences`. For a homomer (`"id": ["A", "B"]`) use the id of the copy you mean |
| `residue` | 1-based position in that chain's `sequence` |

- The residue type is read from the one-letter code in `sequence` (modifications are ignored); only standard
  one-letter codes work at XL positions. Each residue must be a type the XL can react with (tables below), or the
  run stops with `Residue A:12 GLY does not match expected [...]`.
- XLs to RNA, DNA or ligand chains are not supported.
- Each residue pair becomes its own ligand chain, so 10 XLs add 10 ligand chains.
- A complete example: `src/alphafold3/test_data/crosslinks/9G5K/9G5K_input.json`.
- Do not use a `*_data.json` written by AF3x before September 2026 (commit `9f100ac`) as input: it already contains
  the XL ligands, which would be added a second time. Newer `*_data.json` files are fine.

## Supported crosslinkers

### NHS-ester crosslinkers (both ends react with K, S, T, Y or the N-terminus)

| Name | Bonded atoms (both ends) |
|---|---|
| `DSSO` | Lys NZ, Ser OG, Thr OG1, Tyr OH, N-terminal N |
| `DSS` | same |
| `DSG` | same |
| `BS3` | same |
| `BS2G` | same |
| `DSBU` | same |
| `PHOX` | same |
| `BSPEG5` | same |
| `BSPEG9` | same |
| `azide-A-DSBSO` | same |
| `CDI` | same; both residues bond to the same carbonyl carbon (zero-length) |

The N-terminus counts only for residue 1 of a chain, and only if residue 1 is not itself K, S, T or Y (those bond
through their side chain).

### Photo-crosslinkers (NHS ester + diazirine)

| Name | Also known as | End 1 (NHS ester) | End 2 (diazirine) |
|---|---|---|---|
| `SDA` | Sulfo-SDA | K, S, T, Y, N-terminus | any of the 20 standard amino acids |
| `LCSDA` | Sulfo-LC-SDA | same | same |
| `SDAD` | Sulfo-SDAD | same | same |
| `SDA25A` | | same | same |

**Order matters:** in each pair, the first residue is the one that reacted with the NHS ester, the second the one
that reacted with the diazirine. The diazirine end bonds to the side-chain atom listed in
[Dynamic crosslinkers](#dynamic-crosslinkers) (except Trp: NE1).

### Dynamic crosslinkers

Generic linkers of any length, built at run time. Both ends react with **any** standard amino acid.

| Name | Linker |
|---|---|
| `LINK<N>` | Flexible chain of N carbon atoms, e.g. `LINK5` |
| `RIGID<N>` | Rigid scaffold of N fused benzene rings, e.g. `RIGID3` |

N is an integer ≥ 1. Each end bonds to the last side-chain atom of the residue:

| Residue | Atom | Residue | Atom | Residue | Atom | Residue | Atom |
|---|---|---|---|---|---|---|---|
| Ala | CB | Gln | NE2 | Leu | CD2 | Ser | OG |
| Arg | NH2 | Glu | OE2 | Lys | NZ | Thr | OG1 |
| Asn | ND2 | Gly | CA | Met | CE | Trp | CH2 |
| Asp | OD2 | His | NE2 | Phe | CZ | Tyr | OH |
| Cys | SG | Ile | CD1 | Pro | CD | Val | CG2 |

## Disulfide bonds

```json
"disulfide_bonds": [
  {
    "residue_pairs": [
      [["A", 14], ["A", 20]],
      [["A", 8], ["A", 26]]
    ]
  }
]
```

Each listed residue is mutated to alanine (in the sequence and in the MSAs) and an `S-S` ligand is bonded to the
two alanine CB atoms. List cysteines only: the code mutates whatever residue you give. The output therefore shows
ALA at these positions plus an `S-S` ligand chain. Example:
`src/alphafold3/test_data/crosslinks/8WW0/8WW0_data_disulfide.json`.

## Adding a crosslinker

1. Add an entry to `src/alphafold3/crosslinks/crosslink_definitions.py` following the existing ones: the CCD
   definition of the ligand and, for each end, the reactive residue types with their bonded atoms.
2. `src/alphafold3/crosslinks/create_crosslinker.py` prints a CCD entry for a SMILES string (edit the SMILES in
   its `__main__` and run it from inside `src/alphafold3/crosslinks/`).
3. With an editable install (`pip install -e . --no-deps`) the new XL works right away. Otherwise reinstall, or
   rebuild the Docker image.
