![header](docs/header3x.jpg)

# AF3x - Custom modification of AlphaFold 3 enabling explicit modeling of crosslinks

**Warning: This is a custom fork and not the official version. In active development. Code quick and dirty, don't look. Bugs expected. Not always in sync with upstream. Use at your own risk. Fixes and PRs welcome.**

By Agnieszka Obarska-Kosinska (idea and crosslink definitions) and Jan Kosinski (implementation).

Most crosslink-based modeling programs implicitly model crosslinks by adding distance restraints between crosslinked residues. This program explicitly models crosslinks by adding crosslinker molecules as ligands and setting bonds to the crosslinked residues. This allows for more realistic modeling of crosslinks, and sometimes gives good results. The results are also more visually appealing. 

Read more in the prep-print: [Improving AlphaFold 3 structural modeling by incorporating explicit crosslinks](https://www.biorxiv.org/content/10.1101/2024.12.03.626671v2).

Usage:

1. Install AlphaFold 3 as per the instructions in the original README below.
1. Clone this repo
1. Install, for example like this:
    ```bash
    module load GCCcore/12.3.0
    module load Mamba

    mamba create -n af3x python=3.11 -y
    source activate af3x

    cd af3x
    pip install -r dev-requirements.txt

    # Install the package in editable mode without dependencies
    pip install --no-deps .

    build_data
    ```
1. Add crosslinks to your JSON following this format:
    ```json
        "crosslinks": [
            {
                "name": "DSSO",
                "residue_pairs": [
                    [["B", 53], ["C", 66]],
                    [["B", 54], ["C", 66]],
                    [["B", 54], ["C", 113]],
                    [["B", 53], ["C", 113]],
                    [["B", 54], ["C", 106]],
                    [["B", 67], ["C", 129]],
                    [["B", 208], ["C", 129]],
                    [["A", 1], ["C", 91]],
                    [["A", 1], ["C", 99]],
                    [["A", 1], ["C", 90]],
                    [["A", 49], ["C", 113]],
                    [["A", 145], ["C", 129]]
                ]
            },
            {
                "name": "azide-A-DSBSO",
                "residue_pairs": [
                    ...
                ]
            }
        ]
    ```
    for example:
    ```json
    {
        "name": "9G5K",
        "modelSeeds": [
            1
        ],
        "sequences": [
            {
                "protein": {
                    "id": "A",
                    "sequence": "MDCYRTSLSSSWIYPTVILCLFGFFSMMRPSEPFLIPYLSGPDKNLTSAEITNEIFPVWTYSYLVLLLPVFVLTDYVRYKPVIILQGISFIITWLLLLFGQGVKTMQVVEFFYGMVTAAEVAYYAYIYSVVSPEHYQRVSGYCRSVTLAAYTAGSVLAQLLVSLANMSYFYLNVISLASVSVAFLFSLFLPMPKKSMFFHAKPSREIKKSSSVNPVLEETHEGEAPGCEEQKPTSEILSTSGKLNKGQLNSLKPSNVTVDVFVQWFQDLKECYSSKRLFYWSLWWAFATAGFNQVLNYVQILWDYKAPSQDSSIYNGAVEAIATFGGAVAAFAVGYVKVNWDLLGELALVVFSVVNAGSLFLMHYTANIWACYAGYLIFKSSYMLLITIAVFQIAVNLNVERYALVFGINTFIALVIQTIMTVIVVDQRGLNLPVSIQFLVYGSYFAVIAGIFLMRSMYITYSTKSQKDVQSPAPSENPDVSHPEEESNIIMSTKL"
                }
            },
            {
                "protein": {
                    "id": "B",
                    "sequence": "QVQLVESGGGLVQAGDSLRLSCAASGRTFSNYYMAWFRQAPGKEREFVAAIRLSYGSTYYADSVRGRFTISKDNAKNTVNLRMNSLKSEDTAIYYCAAAEDRWALAVRTATTYNYWGQGTQVTVSSHHHHHHEPEA"
                }
            }
        ],
        "dialect": "alphafold3",
        "version": 1,
        "crosslinks": [
            {
                "name": "azide-A-DSBSO",
                "residue_pairs": [
                    [["A", 104], ["B", 43]]
                ]
            }
        ]
    }

    ```
    Supported crosslinkers:
    - DSSO
    - DSS
    - DSG
    - BS3
    - BS2G
    - azide-A-DSBSO
    - DSBU
    - PHOX
    - BSPEG5
    - BSPEG9
1. Run AlphaFold 3 as per the instructions in the original README below. Crosslinks will be added automatically.

    Key flags:
    ```bash
    --sample_crosslink_combinations=<integer> # Enumarate all possible combinations of crosslinks with length equal to sample_crosslinks value and run inference for each combination. Default is do not sample.
    --remove_overlapping_crosslinks=true|false # If set to true, remove any crosslinks that link to residues with already added crosslinks. Default is true.
    --num_seeds=<integer> # Number of seeds to sample. Default is use the seeds from your input JSON but we recommend to use at least 20 seeds (with the default 5 diffusion samples per seed, it gives 100 models).
    ```

# Even more experimental features

## Disulfide bonds

```json
  "disulfide_bonds": [
      {
          "residue_pairs": [
              [["A", 14], ["A", 20]],
              [["A", 8], ["A", 26]],
              [["A", 6], ["A", 28]],
              [["A", 2], ["A", 32]]
          ]
      }
  ]
``` 
Disulfide bonds will be added by mutating the cysteines to alanine and adding an S-S covalent ligand. Thanks Konstantin Gilep for the idea.

## Zero-length crosslinks (coming soon)

By this point you might know how it will be done.

## Observations:

- This sometimes works.
- Sometimes it doesn't.
- When not all crosslinks can be satisfied at the same time, AF3 will sometimes form some crosslinks but break bonds for others, kind off ignoring imcompatible crosslinks (this is good).

## How to add your own crosslinker type?

Find the fle `af3x/src/alphafold3/crosslinks/crosslink_definitions.py` and add your own crosslinker definition by following the provided examples. Reinstall the package with `pip install --no-deps .` and you are ready to go.

## Test data
The test data is available in `af3x/src/alphafold3/test_data/crosslinks` folder.

# OIRIGINAL ALPHAFOLD3 README

![header](docs/header.jpg)

This package provides an implementation of the inference pipeline of AlphaFold
3. See below for how to access the model parameters. You may only use AlphaFold
3 model parameters if received directly from Google. Use is subject to these
[terms of use](https://github.com/google-deepmind/alphafold3/blob/main/WEIGHTS_TERMS_OF_USE.md).

Any publication that discloses findings arising from using this source code, the
model parameters or outputs produced by those should [cite](#citing-this-work)
the
[Accurate structure prediction of biomolecular interactions with AlphaFold 3](https://doi.org/10.1038/s41586-024-07487-w)
paper.

Please also refer to the Supplementary Information for a detailed description of
the method.

AlphaFold 3 is also available at
[alphafoldserver.com](https://alphafoldserver.com) for non-commercial use,
though with a more limited set of ligands and covalent modifications.

If you have any questions, please contact the AlphaFold team at
[alphafold@google.com](mailto:alphafold@google.com).

## Obtaining Model Parameters

This repository contains all necessary code for AlphaFold 3 inference. To
request access to the AlphaFold 3 model parameters, please complete
[this form](https://forms.gle/svvpY4u2jsHEwWYS6). Access will be granted at
Google DeepMind’s sole discretion. We will aim to respond to requests within 2–3
business days. You may only use AlphaFold 3 model parameters if received
directly from Google. Use is subject to these
[terms of use](https://github.com/google-deepmind/alphafold3/blob/main/WEIGHTS_TERMS_OF_USE.md).

## Installation and Running Your First Prediction

See the [installation documentation](docs/installation.md).

Once you have installed AlphaFold 3, you can test your setup using e.g. the
following input JSON file named `fold_input.json`:

```json
{
  "name": "2PV7",
  "sequences": [
    {
      "protein": {
        "id": ["A", "B"],
        "sequence": "GMRESYANENQFGFKTINSDIHKIVIVGGYGKLGGLFARYLRASGYPISILDREDWAVAESILANADVVIVSVPINLTLETIERLKPYLTENMLLADLTSVKREPLAKMLEVHTGAVLGLHPMFGADIASMAKQVVVRCDGRFPERYEWLLEQIQIWGAKIYQTNATEHDHNMTYIQALRHFSTFANGLHLSKQPINLANLLALSSPIYRLELAMIGRLFAQDAELYADIIMDKSENLAVIETLKQTYDEALTFFENNDRQGFIDAFHKVRDWFGDYSEQFLKESRQLLQQANDLKQG"
      }
    }
  ],
  "modelSeeds": [1],
  "dialect": "alphafold3",
  "version": 1
}
```

You can then run AlphaFold 3 using the following command:

```
docker run -it \
    --volume $HOME/af_input:/root/af_input \
    --volume $HOME/af_output:/root/af_output \
    --volume <MODEL_PARAMETERS_DIR>:/root/models \
    --volume <DATABASES_DIR>:/root/public_databases \
    --gpus all \
    alphafold3 \
    python run_alphafold.py \
    --json_path=/root/af_input/fold_input.json \
    --model_dir=/root/models \
    --output_dir=/root/af_output
```

There are various flags that you can pass to the `run_alphafold.py` command, to
list them all run `python run_alphafold.py --help`. Two fundamental flags that
control which parts AlphaFold 3 will run are:

*   `--run_data_pipeline` (defaults to `true`): whether to run the data
    pipeline, i.e. genetic and template search. This part is CPU-only, time
    consuming and could be run on a machine without a GPU.
*   `--run_inference` (defaults to `true`): whether to run the inference. This
    part requires a GPU.

## AlphaFold 3 Input

See the [input documentation](docs/input.md).

## AlphaFold 3 Output

See the [output documentation](docs/output.md).

## Performance

See the [performance documentation](docs/performance.md).

## Known Issues

Known issues are documented in the
[known issues documentation](docs/known_issues.md).

Please
[create an issue](https://github.com/google-deepmind/alphafold3/issues/new/choose)
if it is not already listed in [Known Issues](docs/known_issues.md) or in the
[issues tracker](https://github.com/google-deepmind/alphafold3/issues).

## Citing This Work

Any publication that discloses findings arising from using this source code, the
model parameters or outputs produced by those should cite:

```bibtex
@article{Abramson2024,
  author  = {Abramson, Josh and Adler, Jonas and Dunger, Jack and Evans, Richard and Green, Tim and Pritzel, Alexander and Ronneberger, Olaf and Willmore, Lindsay and Ballard, Andrew J. and Bambrick, Joshua and Bodenstein, Sebastian W. and Evans, David A. and Hung, Chia-Chun and O’Neill, Michael and Reiman, David and Tunyasuvunakool, Kathryn and Wu, Zachary and Žemgulytė, Akvilė and Arvaniti, Eirini and Beattie, Charles and Bertolli, Ottavia and Bridgland, Alex and Cherepanov, Alexey and Congreve, Miles and Cowen-Rivers, Alexander I. and Cowie, Andrew and Figurnov, Michael and Fuchs, Fabian B. and Gladman, Hannah and Jain, Rishub and Khan, Yousuf A. and Low, Caroline M. R. and Perlin, Kuba and Potapenko, Anna and Savy, Pascal and Singh, Sukhdeep and Stecula, Adrian and Thillaisundaram, Ashok and Tong, Catherine and Yakneen, Sergei and Zhong, Ellen D. and Zielinski, Michal and Žídek, Augustin and Bapst, Victor and Kohli, Pushmeet and Jaderberg, Max and Hassabis, Demis and Jumper, John M.},
  journal = {Nature},
  title   = {Accurate structure prediction of biomolecular interactions with AlphaFold 3},
  year    = {2024},
  volume  = {630},
  number  = {8016},
  pages   = {493–-500},
  doi     = {10.1038/s41586-024-07487-w}
}
```

## Acknowledgements

AlphaFold 3's release was made possible by the invaluable contributions of the
following people:

Andrew Cowie, Bella Hansen, Charlie Beattie, Chris Jones, Grace Margand,
Jacob Kelly, James Spencer, Josh Abramson, Kathryn Tunyasuvunakool, Kuba Perlin,
Lindsay Willmore, Max Bileschi, Molly Beck, Oleg Kovalevskiy,
Sebastian Bodenstein, Sukhdeep Singh, Tim Green, Toby Sargeant, Uchechi Okereke,
Yotam Doron, and Augustin Žídek (engineering lead).

We also extend our gratitude to our collaborators at Google and Isomorphic Labs.

AlphaFold 3 uses the following separate libraries and packages:

*   [abseil-cpp](https://github.com/abseil/abseil-cpp) and
    [abseil-py](https://github.com/abseil/abseil-py)
*   [Chex](https://github.com/deepmind/chex)
*   [Docker](https://www.docker.com)
*   [DSSP](https://github.com/PDB-REDO/dssp)
*   [HMMER Suite](https://github.com/EddyRivasLab/hmmer)
*   [Haiku](https://github.com/deepmind/dm-haiku)
*   [JAX](https://github.com/jax-ml/jax/)
*   [jax-triton](https://github.com/jax-ml/jax-triton)
*   [jaxtyping](https://github.com/patrick-kidger/jaxtyping)
*   [libcifpp](https://github.com/pdb-redo/libcifpp)
*   [NumPy](https://github.com/numpy/numpy)
*   [pybind11](https://github.com/pybind/pybind11) and
    [pybind11_abseil](https://github.com/pybind/pybind11_abseil)
*   [RDKit](https://github.com/rdkit/rdkit)
*   [Tree](https://github.com/deepmind/tree)
*   [Triton](https://github.com/triton-lang/triton)
*   [tqdm](https://github.com/tqdm/tqdm)

We thank all their contributors and maintainers!

## Get in Touch

If you have any questions not covered in this overview, please contact the
AlphaFold team at alphafold@google.com.

We would love to hear your feedback and understand how AlphaFold 3 has been
useful in your research. Share your stories with us at
[alphafold@google.com](mailto:alphafold@google.com).

## Licence and Disclaimer

This is not an officially supported Google product.

Copyright 2024 DeepMind Technologies Limited.

### AlphaFold 3 Source Code and Model Parameters

The AlphaFold 3 source code is licensed under the Creative Commons
Attribution-Non-Commercial ShareAlike International License, Version 4.0
(CC-BY-NC-SA 4.0) (the "License"); you may not use this file except in
compliance with the License. You may obtain a copy of the License at
[https://github.com/google-deepmind/alphafold3/blob/main/LICENSE](https://github.com/google-deepmind/alphafold3/blob/main/LICENSE).

The AlphaFold 3 model parameters are made available under the
[AlphaFold 3 Model Parameters Terms of Use](https://github.com/google-deepmind/alphafold3/blob/main/WEIGHTS_TERMS_OF_USE.md)
(the "Terms"); you may not use these except in compliance with the Terms. You
may obtain a copy of the Terms at
[https://github.com/google-deepmind/alphafold3/blob/main/WEIGHTS_TERMS_OF_USE.md](https://github.com/google-deepmind/alphafold3/blob/main/WEIGHTS_TERMS_OF_USE.md).

Unless required by applicable law, AlphaFold 3 and its output are distributed on
an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express
or implied. You are solely responsible for determining the appropriateness of
using AlphaFold 3, or using or distributing its source code or output, and
assume any and all risks associated with such use or distribution and your
exercise of rights and obligations under the relevant terms. Output are
predictions with varying levels of confidence and should be interpreted
carefully. Use discretion before relying on, publishing, downloading or
otherwise using the AlphaFold 3 Assets.

AlphaFold 3 and its output are for theoretical modeling only. They are not
intended, validated, or approved for clinical use. You should not use the
AlphaFold 3 or its output for clinical purposes or rely on them for medical or
other professional advice. Any content regarding those topics is provided for
informational purposes only and is not a substitute for advice from a qualified
professional. See the relevant terms for the specific language governing
permissions and limitations under the terms.

### Third-party Software

Use of the third-party software, libraries or code referred to in the
[Acknowledgements](#acknowledgements) section above may be governed by separate
terms and conditions or license provisions. Your use of the third-party
software, libraries or code is subject to any such terms and you should check
that you can comply with any applicable restrictions or terms and conditions
before use.

### Mirrored and Reference Databases

The following databases have been: (1) mirrored by Google DeepMind; and (2) in
part, included with the inference code package for testing purposes, and are
available with reference to the following:

*   [BFD](https://bfd.mmseqs.com/) (modified), by Steinegger M. and Söding J.,
    modified by Google DeepMind, available under a
    [Creative Commons Attribution 4.0 International License](https://creativecommons.org/licenses/by/4.0/deed.en).
    See the Methods section of the
    [AlphaFold proteome paper](https://www.nature.com/articles/s41586-021-03828-1)
    for details.
*   [PDB](https://wwpdb.org) (unmodified), by H.M. Berman et al., available free
    of all copyright restrictions and made fully and freely available for both
    non-commercial and commercial use under
    [CC0 1.0 Universal (CC0 1.0) Public Domain Dedication](https://creativecommons.org/publicdomain/zero/1.0/).
*   [MGnify: v2022\_05](https://ftp.ebi.ac.uk/pub/databases/metagenomics/peptide_database/2022_05/README.txt)
    (unmodified), by Mitchell AL et al., available free of all copyright
    restrictions and made fully and freely available for both non-commercial and
    commercial use under
    [CC0 1.0 Universal (CC0 1.0) Public Domain Dedication](https://creativecommons.org/publicdomain/zero/1.0/).
*   [UniProt: 2021\_04](https://www.uniprot.org/) (unmodified), by The UniProt
    Consortium, available under a
    [Creative Commons Attribution 4.0 International License](https://creativecommons.org/licenses/by/4.0/deed.en).
*   [UniRef90: 2022\_05](https://www.uniprot.org/) (unmodified) by The UniProt
    Consortium, available under a
    [Creative Commons Attribution 4.0 International License](https://creativecommons.org/licenses/by/4.0/deed.en).
*   [NT: 2023\_02\_23](https://www.ncbi.nlm.nih.gov/nucleotide/) (modified) See
    the Supplementary Information of the
    [AlphaFold 3 paper](https://nature.com/articles/s41586-024-07487-w) for
    details.
*   [RFam: 14\_4](https://rfam.org/) (modified), by I. Kalvari et al., available
    free of all copyright restrictions and made fully and freely available for
    both non-commercial and commercial use under
    [CC0 1.0 Universal (CC0 1.0) Public Domain Dedication](https://creativecommons.org/publicdomain/zero/1.0/).
    See the Supplementary Information of the
    [AlphaFold 3 paper](https://nature.com/articles/s41586-024-07487-w) for
    details.
*   [RNACentral: 21\_0](https://rnacentral.org/) (modified), by The RNAcentral
    Consortium available free of all copyright restrictions and made fully and
    freely available for both non-commercial and commercial use under
    [CC0 1.0 Universal (CC0 1.0) Public Domain Dedication](https://creativecommons.org/publicdomain/zero/1.0/).
    See the Supplementary Information of the
    [AlphaFold 3 paper](https://nature.com/articles/s41586-024-07487-w) for
    details.
