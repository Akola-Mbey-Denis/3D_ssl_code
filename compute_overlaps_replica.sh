#!/bin/bash
#SBATCH --job-name=replica_overlaps
#SBATCH --time=24:00:00
#SBATCH --cpus-per-task=1
#SBATCH --ntasks=1
#SBATCH --mem=50G
#SBATCH --array=0-17                # 18 tasks, each processes 1 scene
#SBATCH --output=replica_slogs/%A_%a.out
#SBATCH --error=replica_slogs/%A_%a.err
#SBATCH --account=pr_96_general

# ---------------------------
# Scene setup
# ---------------------------
TASK_ID=$SLURM_ARRAY_TASK_ID

START_IND=$TASK_ID
# END_IND=$TASK_ID   # exactly one scene
END_IND=$((TASK_ID + 1))

echo "Processing scene $START_IND (Task $TASK_ID)"

# ---------------------------
# Environment setup
# ---------------------------
# Activate environment
source  /projects/rps/dff6142/fouheylab/dma9300/miniconda3/etc/profile.d/conda.sh
conda activate mast3r

cd /projects/rps/dff6142/fouheylab/dma9300/3d_ssl_project
# ---------------------------
# Run preprocessing
# ---------------------------
python -m preprocess_datasets.replica_overlaps \
    --start_idx $START_IND \
    --end_idx $END_IND

echo "Done processing scenes from $START_IND to $END_IND"