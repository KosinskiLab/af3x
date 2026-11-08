# FAQ

## AF3x

<details class="question" markdown>
<summary>Which residue goes first for SDA, LCSDA, SDAD, SDA25A?</summary>

The one that reacted with the NHS ester (K, S, T, Y or the N-terminus); the second reacted with the diazirine.
See [Photo-crosslinkers](crosslink_input.md#photo-crosslinkers-nhs-ester-diazirine).

</details>

<details class="question" markdown>
<summary>The run stops with <code>Residue A:12 GLY does not match expected [...]</code>.</summary>

The residue at that position cannot react with this crosslinker. Check the residue number (1-based in the chain's
`sequence`) and the [supported residues](crosslink_input.md#supported-crosslinkers).

</details>

<details class="question" markdown>
<summary>Some of my crosslinks are missing from the model.</summary>

`--remove_overlapping_crosslinks` (default `true`) drops every XL that uses a residue already used by an earlier XL;
the log lists them. Use `--remove_overlapping_crosslinks=false` to keep all.

</details>

<details class="question" markdown>
<summary>How do I predict with one crosslink at a time?</summary>

`--sample_crosslink_combinations=1` runs a separate prediction for each XL in the input, named after it
(e.g. `9G5K_DSSO_A104-B43`). With `=2`, every pair of XLs, and so on.

</details>

<details class="question" markdown>
<summary>How do I crosslink to the N-terminus?</summary>

Use residue 1 of the chain. If residue 1 is K, S, T or Y, the bond goes to its side chain instead.

</details>

<details class="question" markdown>
<summary>Why are my cysteines alanines in the output?</summary>

Disulfide bonds are modelled as Cys → Ala plus an `S-S` ligand. See [Disulfide bonds](crosslink_input.md#disulfide-bonds).

</details>

<details class="question" markdown>
<summary>How many seeds should I use?</summary>

Preferably 20 (`--num_seeds=20` with one seed in the JSON): 100 models with the default 5 diffusion samples per seed.
More seeds increase the compute time proportionally.

</details>

<details class="question" markdown>
<summary>Can I use a <code>*_data.json</code> from an earlier run as input?</summary>

Yes, from AF3x v0.2.0 on. Files written before September 2026 already contain the XL ligands; do not reuse those.

</details>

<details class="question" markdown>
<summary>Did the model satisfy my crosslinks?</summary>

Check with `eval_crosslinks.py` ([Checking XL satisfaction](running.md#checking-xl-satisfaction)).

</details>

<details class="question" markdown>
<summary>Where do I ask a question or report a bug?</summary>

Open an issue at [KosinskiLab/af3x](https://github.com/KosinskiLab/af3x/issues), with the AF3x version
(`<job>_af3x_version.json`), the input JSON and the error message. Problems that also occur with plain AlphaFold 3
belong in the [AlphaFold 3 issues](https://github.com/google-deepmind/alphafold3/issues). To contribute code or
documentation, see [CONTRIBUTING.md](https://github.com/KosinskiLab/af3x/blob/main/CONTRIBUTING.md).

</details>

## AlphaFold 3 (answers in upstream GitHub issues)

- [SMILES strings with a `\` need to be escaped in the input JSON?](https://github.com/google-deepmind/alphafold3/issues/110)
- [Are SMILES required in the user-provided CCD format?](https://github.com/google-deepmind/alphafold3/issues/128)
- [The model doesn't always keep the chirality](https://github.com/google-deepmind/alphafold3/issues/107)
- [Post-translational modifications (PTMs), including unusual ones](https://github.com/google-deepmind/alphafold3/issues/54)
- [MMseqs2 vs jackhmmer: performance difference?](https://github.com/google-deepmind/alphafold3/issues/125#issuecomment-2500808661)
- [With precomputed MSAs, can I skip downloading the databases?](https://github.com/google-deepmind/alphafold3/issues/115)
- [Converting precomputed `.sto` alignments into the input JSON](https://github.com/google-deepmind/alphafold3/issues/112)
- [Adding new PDB templates](https://github.com/google-deepmind/alphafold3/issues/94)
- [Running without templates, or setting the max template date](https://github.com/google-deepmind/alphafold3/issues/88#issuecomment-2497980574)
- [More CPUs for HMMER than the default 8?](https://github.com/google-deepmind/alphafold3/issues/134)
- [What does `--jax_compilation_cache_dir` speed up?](https://github.com/google-deepmind/alphafold3/issues/118#issuecomment-2498839488)
- [Running a single test](https://github.com/google-deepmind/alphafold3/issues/129)
