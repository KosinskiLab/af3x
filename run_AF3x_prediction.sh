#!/bin/bash

# A typical run takes couple of hours but may be much longer
#SBATCH --job-name=af3x_predict
#SBATCH --time=02:00:00

# Log files:
#SBATCH -e logs/af3_predict_%j_err.txt
#SBATCH -o logs/af3_predict_%j_out.txt

# QoS sets priority
#SBATCH --qos=normal

#SBATCH -p gpu-el8
# Lower end GPUs might be sufficient for pairwise screens:
#SBATCH -C "gpu=H100|gpu=A100|gpu=L40s"

# Reserve the entire GPU so no-one else slows you down
#SBATCH -G 1

# Limit the run to a single node
#SBATCH -N 1

# Adjust this depending on the node
#SBATCH -n 32
#SBATCH --mem-per-gpu 193028

module load GCCcore/12.3.0
module load Mamba
module load CUDA/12.6.2
source activate af3x

export XLA_FLAGS="--xla_gpu_enable_triton_gemm=false"
export XLA_PYTHON_CLIENT_PREALLOCATE=true
export XLA_CLIENT_MEM_FRACTION=0.95

# Uncomment if you want to use unified memory
# export TF_FORCE_UNIFIED_MEMORY='1'
# MAXRAM=$(echo `ulimit -m` '/ 1024.0' | bc)
# GPUMEM=$(nvidia-smi --query-gpu=memory.total --format=csv,noheader,nounits | tail -1)
# export XLA_CLIENT_MEM_FRACTION=$(echo "scale=3; $MAXRAM / $GPUMEM" | bc)

AF3_DIR=/g/kosinski/kosinski/devel/af3x
DB_DIR=/scratch/kosinski/AlphaFold3_DB

echo "Job started at: $(date)"
start_time=$(date +%s)

# Default values for the flags.
REMOVE_OVERLAPPING_CROSSLINKS=false
SAMPLE_CROSSLINK_COMBINATIONS=false
SAMPLE_CROSSLINK_COMBINATIONS_VALUE=0
FAST=false
NUM_SEEDS=""

# Parse command-line arguments.
# Expect two positional arguments: INPUT_JSON and OUTDIR.
# Optional flags: --remove_overlapping_crosslinks, --sample_crosslink_combinations VALUE, --fast, --num_seeds VALUE
if [ "$#" -lt 2 ]; then
  echo "Usage: $0 INPUT_JSON OUTDIR [--remove_overlapping_crosslinks] [--sample_crosslink_combinations VALUE] [--fast] [--num_seeds VALUE]"
  exit 1
fi

INPUT_JSON=$1
OUTDIR=$2
shift 2

while [ "$#" -gt 0 ]; do
  case "$1" in
    --remove_overlapping_crosslinks)
      REMOVE_OVERLAPPING_CROSSLINKS=true
      shift
      ;;
    --sample_crosslink_combinations)
      if [ -z "$2" ] || [[ "$2" == --* ]]; then
        echo "Error: --sample_crosslink_combinations requires a value."
        exit 1
      fi
      SAMPLE_CROSSLINK_COMBINATIONS=true
      SAMPLE_CROSSLINK_COMBINATIONS_VALUE=$2
      shift 2
      ;;
    --fast)
      FAST=true
      shift
      ;;
    --num_seeds)
      if [ -z "$2" ] || [[ "$2" == --* ]]; then
        echo "Error: --num_seeds requires a value."
        exit 1
      fi
      NUM_SEEDS=$2
      shift 2
      ;;
    *)
      echo "Unknown parameter: $1"
      exit 1
      ;;
  esac
done

CMD="python $AF3_DIR/run_alphafold.py \
    --json_path=$INPUT_JSON \
    --model_dir=/g/kosinski/kosinski/software/AlphaFold3/models \
    --output_dir=$OUTDIR \
    --db_dir=$DB_DIR \
    --jax_compilation_cache_dir=/scratch/kosinski/AlphaFold3_cache \
    --norun_data_pipeline"

if [ "$REMOVE_OVERLAPPING_CROSSLINKS" = true ]; then
    CMD="$CMD --remove_overlapping_crosslinks"
fi

if [ "$SAMPLE_CROSSLINK_COMBINATIONS" = true ]; then
    CMD="$CMD --sample_crosslink_combinations $SAMPLE_CROSSLINK_COMBINATIONS_VALUE"
fi

if [ "$FAST" = true ]; then
    CMD="$CMD --num_recycles=1 --num_diffusion_samples=1"
fi

if [ -n "$NUM_SEEDS" ]; then
    CMD="$CMD --num_seeds=$NUM_SEEDS"
fi

echo "Running command:"
echo "$CMD"

$CMD

echo "Job finished at: $(date)"
end_time=$(date +%s)
elapsed_time=$((end_time - start_time))
echo "Elapsed time: ${elapsed_time} seconds"
elapsed_minutes=$((elapsed_time / 60))
elapsed_hours=$((elapsed_minutes / 60))
echo "Elapsed time: ${elapsed_minutes} minutes"
echo "Elapsed time: ${elapsed_hours} hours"
