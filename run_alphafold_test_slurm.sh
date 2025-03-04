#!/bin/bash

#A typical run takes couple of hours but may be much longer
#SBATCH --job-name=af3_test
#SBATCH --time=01:00:00

#log files:
#SBATCH -e logs/run_alphafold_test_%j_err.txt
#SBATCH -o logs/run_alphafold_test_%j_out.txt

#qos sets priority
#SBATCH --qos=highest

#SBATCH -p gpu-el8
#lower end GPUs might be sufficient for pairwise screens:
#SBATCH -C "gpu=A100|gpu=L40s|gpu=H100"

#Reserve the entire GPU so no-one else slows you down
#SBATCH -G 1

#Limit the run to a single node
#SBATCH -N 1

#Adjust this depending on the node
#SBATCH -n 14
#SBATCH --mem-per-gpu 128789

module load GCCcore/12.3.0
module load Mamba
module load HMMER/3.4-gompi-2023a
module load CUDA/12.6.2
source activate af3x

export XLA_FLAGS="--xla_gpu_enable_triton_gemm=false"
export XLA_PYTHON_CLIENT_PREALLOCATE=true
export XLA_CLIENT_MEM_FRACTION=0.95

# Uncomment if you want to use unfied memory
# export TF_FORCE_UNIFIED_MEMORY='1'
# MAXRAM=$(echo `ulimit -m` '/ 1024.0'|bc)
# GPUMEM=`nvidia-smi --query-gpu=memory.total --format=csv,noheader,nounits|tail -1`
# export XLA_CLIENT_MEM_FRACTION=`echo "scale=3;$MAXRAM / $GPUMEM"|bc`

python run_alphafold_test.py