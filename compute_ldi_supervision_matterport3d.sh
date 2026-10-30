#!/bin/bash
#SBATCH --job-name=matterport3d_ldi
#SBATCH --time=8:00:00
#SBATCH --cpus-per-task=1
#SBATCH --ntasks=1
#SBATCH --mem=32G
#SBATCH --array=0-89
#SBATCH --output=matterport_ldi/%A_%a.out
#SBATCH --error=matterport_ldi/%A_%a.err
#SBATCH --account=torch_pr_41_general

# ------------------------------------------------------
# TOTAL number of samples
# test set 2444 --train : 8624 , val --1479
TOTAL=90
NUM_TASKS=90

# Automatically compute samples per task (ceil division)
SAMPLES_PER_TASK=$(( (TOTAL + NUM_TASKS - 1) / NUM_TASKS ))

TASK_ID=${SLURM_ARRAY_TASK_ID}

# Compute start and end indices
START=$(( TASK_ID * SAMPLES_PER_TASK ))
END=$(( START + SAMPLES_PER_TASK - 1 ))

# Clamp END to total
if [ $END -ge $TOTAL ]; then
    END=$(( TOTAL - 1 ))
fi

# Skip empty tasks
if [ $START -ge $TOTAL ]; then
    echo "Task $TASK_ID has no samples, exiting."
    exit 0
fi

echo "CPU task $TASK_ID processing samples $START to $END"

# ---------------------------
# Activate environment
source /projects/rps/dff6142/fouheylab/dma9300/miniconda3/etc/profile.d/conda.sh
conda activate mast3r

cd /projects/rps/dff6142/fouheylab/dma9300/3d_ssl_project/

# Run preprocessing
python -m preprocess_ldis.preprocess_matterport_ldis \
    --max_depth 16.0 \
    --start_idx $START \
    --end_idx $END

echo "Task $TASK_ID done (samples $START to $END)"