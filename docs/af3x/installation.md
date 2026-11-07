# Installing AF3x

## Requirements

| What | Needed |
|---|---|
| Operating system | Linux |
| GPU | NVIDIA, compute capability ≥ 8.0 (e.g. A100, H100, L40S, RTX 3090/4090), driver supporting CUDA 12 |
| RAM | ≥ 64 GB recommended (MSA search of long sequences) |
| Disk | ~1 TB for the genetic databases (SSD recommended) |
| Tools | `git`, `wget`, `zstd`, `make`, a C/C++ compiler, zlib headers (e.g. `sudo apt install git wget zstd build-essential zlib1g-dev`) |
| Package manager | [uv](https://docs.astral.sh/uv/getting-started/installation/) (Option A) or conda/mamba (Option B) |

## 1. Get the code

```sh
git clone https://github.com/KosinskiLab/af3x.git
cd af3x
```

## 2. Genetic databases

AF3x uses the AlphaFold 3 databases. **If you already have them for AlphaFold 3, reuse them** and skip this step.

Otherwise download them (about 252 GB, 630 GB unpacked) to a directory outside `af3x`:

```sh
./fetch_databases.sh <DB_DIR>
```

Details: [Obtaining genetic databases](../installation.md#obtaining-genetic-databases).

## 3. Model parameters

AF3x uses the unmodified AlphaFold 3 model parameters; AF3x does not distribute them. Download them from Google
(https://storage.googleapis.com/alphafold3/af3.bin.zst), subject to the
[model parameters terms of use](https://github.com/google-deepmind/alphafold3/blob/main/WEIGHTS_TERMS_OF_USE.md),
to a directory outside `af3x` (`<MODEL_DIR>` below). If you already have them for AlphaFold 3, reuse them.

## 4. Install

Option A or B: editable install, so crosslinkers you add work without reinstalling. Docker: see [below](#docker).

### Option A: uv

```sh
uv venv --python 3.12
source .venv/bin/activate
uv sync --frozen          # package versions from uv.lock
uv run build_data         # chemical components database
```

Then HMMER (step 5) with `PREFIX=$VIRTUAL_ENV`.

### Option B: conda

```sh
mamba create -n af3x -c conda-forge python=3.12 gxx zlib
conda activate af3x
export CPATH=$CONDA_PREFIX/include LIBRARY_PATH=$CONDA_PREFIX/lib   # zlib headers for the C++ build
uv export --frozen --no-hashes --no-emit-project --all-groups > /tmp/af3x_requirements.txt
pip install -r /tmp/af3x_requirements.txt
pip install --no-deps -e .     # compiles the C++ extension (~10 min)
build_data                     # chemical components database
```

Then HMMER (step 5) with `PREFIX=$CONDA_PREFIX`.

## 5. HMMER

HMMER 3.4 with the three patches from `docker/`: `--seq_limit` (less memory) and two speed-ups (jackhmmer ~3×
faster). Run from the `af3x` directory:

```sh
AF3X=$PWD
PREFIX=$VIRTUAL_ENV                          # Option A; $CONDA_PREFIX for Option B
cd $(mktemp -d)
wget http://eddylab.org/software/hmmer/hmmer-3.4.tar.gz
echo "ca70d94fd0cf271bd7063423aabb116d42de533117343a9b27a65c17ff06fbf3  hmmer-3.4.tar.gz" | sha256sum --check
tar zxf hmmer-3.4.tar.gz
for p in jackhmmer_seq_limit easel_fast_fasta_parsing jackhmmer_parallel_fasta; do
  patch -p0 < $AF3X/docker/$p.patch
done
cd hmmer-3.4
./configure --prefix=$PREFIX
make -j8 && make install && (cd easel && make install)
cd $AF3X
jackhmmer -h --seq_limit 1 > /dev/null && echo "patched jackhmmer OK"
```

An unpatched HMMER 3.4 (e.g. `sudo apt install hmmer`) also works, without these gains.

## 6. Check the installation

```sh
python run_alphafold_data_test.py      # CPU only
```

First crosslink prediction (the test input contains its MSAs, no databases needed):

```sh
python run_alphafold.py \
  --json_path=src/alphafold3/test_data/crosslinks/9G5K/9G5K_input.json \
  --output_dir=output \
  --model_dir=<MODEL_DIR> \
  --norun_data_pipeline
```

Full run with MSA search: drop `--norun_data_pipeline`, add `--db_dir=<DB_DIR>`. Next:
[Crosslink input](crosslink_input.md) and [Running and output](running.md). GPU settings (e.g. `XLA_FLAGS`): [performance notes](../performance.md).

## Docker

Steps 1–3, then the [AlphaFold 3 Docker instructions](../installation.md#building-the-docker-container-that-will-run-alphafold-3)
from the `af3x` directory. The image includes AF3x and the patched HMMER (steps 4–5 not needed).

- The image is called `alphafold3` (`docker build -t alphafold3 …`), but it contains AF3x. Use that name in the
  `docker run` commands, or build with `-t af3x` and use `af3x` instead.
- The install in the image is not editable: rebuild the image after adding a crosslinker.
