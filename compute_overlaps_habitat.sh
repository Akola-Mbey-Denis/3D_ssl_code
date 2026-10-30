#!/bin/bash
#SBATCH --job-name=habitat3d_preprocess_job
#SBATCH --time=24:00:00
#SBATCH --cpus-per-task=1
#SBATCH --ntasks=1
#SBATCH --mem=50G
#SBATCH --array=0-100
#SBATCH --output=habitat_slogs/%A_%a.out
#SBATCH --error=habitat_slogs/%A_%a.err
#SBATCH --account=torch_pr_41_tandon_advanced

# ---------------------------
# Scene split setup
# ---------------------------
TOTAL_SCENES=910
TASK_ID=$SLURM_ARRAY_TASK_ID
NUM_TASKS=100
SCENES_PER_TASK=$(( TOTAL_SCENES / NUM_TASKS ))
REMAINDER=$(( TOTAL_SCENES % NUM_TASKS ))

if [ $TASK_ID -lt $REMAINDER ]; then
    START_IND=$(( TASK_ID * (SCENES_PER_TASK + 1) ))
    END_IND=$(( START_IND + SCENES_PER_TASK ))
else
    START_IND=$(( TASK_ID * SCENES_PER_TASK + REMAINDER ))
    END_IND=$(( START_IND + SCENES_PER_TASK - 1 ))
fi

echo "Processing scenes from $START_IND to $END_IND (Task $TASK_ID)"

# ---------------------------
# Conda setup (NON-INTERACTIVE SAFE)
# ---------------------------
source /projects/rps/dff6142/fouheylab/dma9300/miniconda3/etc/profile.d/conda.sh
conda activate mast3r

# ---------------------------
# Go to project root
# ---------------------------

cd /projects/rps/dff6142/fouheylab/dma9300/3d_ssl_project


# ---------------------------
# Run preprocessing
# ---------------------------
python -m preprocess_datasets.habitat_overlaps --start_idx $START_IND --end_idx $END_IND
echo "Done processing scenes from $START_IND to $END_IND"
