#!/bin/bash
#SBATCH --job-name=scannet_pp
#SBATCH --time=6:00:00
#SBATCH --cpus-per-task=4
#SBATCH --ntasks=1
#SBATCH --mem=50G
#SBATCH --array=0-9                 # 10 jobs × 25 scenes = 250 scenes
#SBATCH --output=scannetpp_dumps/%A_%a.out
#SBATCH --error=scannetpp_dumps/%A_%a.err
#SBATCH --account=torch_pr_41_courant

# Each job processes 25 scenes: [START_IND, END_IND)
START_IND=$(( SLURM_ARRAY_TASK_ID * 25 ))
END_IND=$(( START_IND + 25 ))

echo "Task $SLURM_ARRAY_TASK_ID: scenes [$START_IND, $END_IND)"

source /projects/rps/dff6142/fouheylab/dma9300/miniconda3/etc/profile.d/conda.sh
conda activate mast3r
cd /projects/rps/dff6142/fouheylab/dma9300/3d_ssl_project

python -m preprocess_datasets.preprocess_scannetpp \
    --scannetpp_dir /projects/rps/dff6142/fouheylab/jt4100/datasets/scannetpp \
    --output_dir /scratch/dma9300/scannetpp_preprocessed_v2 \
    --source iphone --num_images 2000 --num_scenes 0 \
    --start_idx "$START_IND" --end_idx "$END_IND"

echo "Done: scenes [$START_IND, $END_IND)"