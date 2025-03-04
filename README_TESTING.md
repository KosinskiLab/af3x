sbatch run_alphafold_test_slurm.sh

sbatch run_AF3x_prediction.sh \
    src/alphafold3/test_data/crosslinks/4G3Y/4g3y_input.json \
    /scratch/kosinski/af3x/4G3Y_remove \
    --remove_overlapping_crosslinks \
    --num_seeds 20

sbatch run_AF3x_prediction.sh \
    src/alphafold3/test_data/crosslinks/4G3Y/4g3y_input.json \
    /scratch/kosinski/af3x/4G3Y_combs1 \
    --remove_overlapping_crosslinks \
    --sample_crosslink_combinations 1 \
    --fast

sbatch run_AF3x_prediction.sh \
    src/alphafold3/test_data/crosslinks/4G3Y/4g3y_input.json \
    /scratch/kosinski/af3x/4G3Y_combs2 \
    --remove_overlapping_crosslinks \
    --sample_crosslink_combinations 2 \
    --fast


sbatch run_AF3x_prediction.sh \
    src/alphafold3/test_data/crosslinks/8WW0/8WW0_data_disulfide.json \
    /scratch/kosinski/af3x/8WW0 \
    --num_seeds 20

sbatch run_AF3x_prediction.sh \
    src/alphafold3/test_data/crosslinks/9G5K/9G5K_input.json \
    /scratch/kosinski/af3x/9G5K \
    --num_seeds 20
