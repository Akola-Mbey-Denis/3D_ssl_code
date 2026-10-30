#!/bin/bash
#SBATCH --job-name=scannet_overlap_jobs
#SBATCH --time=12:00:00          # 3 days
#SBATCH --cpus-per-task=1        # Each task uses 1 CPU
#SBATCH --ntasks=1
#SBATCH --mem=50G                # Each CPU gets 50 GB memory
#SBATCH --array=0-199             # 50 jobs in total (0–49)
#SBATCH --output=scannet_slogs/%A_%a.out
#SBATCH --error=scannet_slogs/%A_%a.err
#SBATCH --account=torch_pr_41_tandon_advanced

# ---------------------------
# Scene split setup
# ---------------------------
TOTAL_SCENES=200
TASK_ID=$SLURM_ARRAY_TASK_ID
NUM_TASKS=200

SCENES_PER_TASK=$(( TOTAL_SCENES / NUM_TASKS ))      # integer division
REMAINDER=$(( TOTAL_SCENES % NUM_TASKS ))            # extra scenes

# Compute start and end indices
if [ $TASK_ID -lt $REMAINDER ]; then
    # First $REMAINDER tasks handle one extra scene
    START_IND=$(( TASK_ID * (SCENES_PER_TASK + 1) ))
    END_IND=$(( START_IND + SCENES_PER_TASK ))       # inclusive
else
    START_IND=$(( TASK_ID * SCENES_PER_TASK + REMAINDER ))
    END_IND=$(( START_IND + SCENES_PER_TASK ))  # inclusive
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
# Run preprocessing script
# ---------------------------
python -m preprocess_datasets.scannet_overlaps --start_idx $START_IND --end_idx $END_IND

echo "Done processing scenes from $START_IND to $END_IND"
