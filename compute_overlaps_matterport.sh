#!/bin/bash
#SBATCH --job-name=preprocess_mp3d
#SBATCH --time=12:00:00
#SBATCH --cpus-per-task=1
#SBATCH --ntasks=1
#SBATCH --mem=50G
#SBATCH --array=0-89               # 45 tasks: 0..44
#SBATCH --output=slogs_overlaps/%A_%a.out
#SBATCH --error=slogs_overlaps/%A_%a.err
#SBATCH --account=pr_96_general

TOTAL=90
SCENES_PER_TASK=1

TASK_ID=${SLURM_ARRAY_TASK_ID}

START=$(( TASK_ID * SCENES_PER_TASK ))
END=$(( START + SCENES_PER_TASK - 1 ))

# Clamp to TOTAL-1 for safety
if [ $END -ge $TOTAL ]; then
    END=$(( TOTAL - 1 ))
fi

echo "Task $TASK_ID processing scenes $START to $END"

# activate env
source /projects/rps/dff6142/fouheylab/dma9300/miniconda3/etc/profile.d/conda.sh
conda activate mast3r

# ---------------------------
# Go to project root
# ---------------------------
cd /projects/rps/dff6142/fouheylab/dma9300/3d_ssl_project



# Run preprocess
python -m preprocess_datasets.matterport_overlaps  --mode dump_overlaps --start_idx $START --end_idx $END

echo "Task $TASK_ID done (scenes $START to $END)"
