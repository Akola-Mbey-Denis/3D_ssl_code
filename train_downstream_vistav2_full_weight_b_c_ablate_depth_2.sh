#!/bin/bash

GPU_TYPE="b200"
NUM_GPUS=2
MEM_OVERRIDE=""
EPOCHS=40

# ---------------------------
# Parse flags
# ---------------------------
# usage: ./submit_vista_downstream.sh -g b200 -n 2 [-m 512] [-e 50|60]
while getopts g:n:m:e: flag; do
    case "${flag}" in
        g) GPU_TYPE=${OPTARG};;
        n) NUM_GPUS=${OPTARG};;
        m) MEM_OVERRIDE=${OPTARG};;     # GB, optional
        e) EPOCHS=${OPTARG};;           # fine-tuning epochs (50 / 60)
    esac
done

if ! [[ "$NUM_GPUS" =~ ^[1-8]$ ]]; then
    echo "Error: Number of GPUs must be between 1 and 8."
    exit 1
fi

HOSTNAME=$(hostname)

# ---------------------------
# Set resources based on host
# ---------------------------
MAX_MEMORY=640          # GB

# Torch per-GPU share = node CPUs / GPUs and node memory / GPUs (more is rejected:
# "GPU job setup is not valid")
if [[ $HOSTNAME == *.arc-ts.umich.edu || $HOSTNAME == *.eecs.umich.edu ]]; then
    GPU_TYPE="A40"
    CPU_PER_GPU=16; MEM_PER_GPU=128
else
    case $GPU_TYPE in
        "h100")    CPU_PER_GPU=24; MEM_PER_GPU=384;;
        "h200")    CPU_PER_GPU=16; MEM_PER_GPU=256;;
        "b200")    CPU_PER_GPU=16; MEM_PER_GPU=384;;
        "a100")    CPU_PER_GPU=16; MEM_PER_GPU=128;;
        "l40s")    CPU_PER_GPU=32; MEM_PER_GPU=128;;
        *) echo "Unsupported GPU type: $GPU_TYPE"; exit 1;;
    esac
fi

CPUS=$((CPU_PER_GPU * NUM_GPUS))
MEMORY=$((MEM_PER_GPU * NUM_GPUS))
if [ -n "$MEM_OVERRIDE" ]; then
    MEMORY=$MEM_OVERRIDE
fi
if [ $MEMORY -gt $MAX_MEMORY ]; then
    MEMORY=$MAX_MEMORY
fi

# ---------------------------
# GPU-type specific SLURM options
# ---------------------------
# B200 nodes are only available in preemptible mode: the job can be stopped at any time and is
# put back in the queue (same job id). Training must therefore resume from the last checkpoint
# in --output_dir when it restarts; logs are appended instead of overwritten.
if [ "$GPU_TYPE" == "b200" ]; then
    GRES="gpu:${NUM_GPUS}"
    EXTRA_SBATCH=(--comment="preemption=yes;requeue=true" --requeue --open-mode=append)
else
    GRES="gpu:${GPU_TYPE}:${NUM_GPUS}"
    EXTRA_SBATCH=()
fi

SRC=/projects/rps/dff6142/fouheylab/dma9300/3d_ssl_project/preprocessed_data/training_lmdbs   # LMDBs are copied from here to node-local disk

# ---------------------------
# Debug output
# ---------------------------
echo "Submitting downstream fine-tuning job with:"
echo "GPU_TYPE = $GPU_TYPE"
echo "NUM_GPUS = $NUM_GPUS"
echo "CPUS = $CPUS"
echo "MEMORY = ${MEMORY}GB"
echo "GRES = $GRES"
echo "EPOCHS = $EPOCHS"
[ ${#EXTRA_SBATCH[@]} -gt 0 ] && echo "EXTRA = ${EXTRA_SBATCH[*]}"

# ---------------------------
# Submit SLURM job
# ---------------------------

sbatch \
  --account=torch_pr_41_courant \
  --time=24:00:00 \
  --mem=${MEMORY}G \
  --gres=${GRES} \
  --cpus-per-task=${CPUS} \
  --constraint=${GPU_TYPE} \
  --job-name="vista_v2downstream_${EPOCHS}ep_${GPU_TYPE}_${NUM_GPUS}gpu" \
  "${EXTRA_SBATCH[@]}" \
  --output=vista_v2downstream_training/%j.out \
  --error=vista_v2downstream_training/%j.err \
  --wrap="source /projects/rps/dff6142/fouheylab/dma9300/miniconda3/etc/profile.d/conda.sh && \
          conda activate mast3r && \
          cd /projects/rps/dff6142/fouheylab/dma9300/3d_ssl_project/ && \
          export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 && \
          export LMDB_DIR=/dev/shm/\${USER}_\${SLURM_JOB_ID} && \
          trap 'rm -rf \$LMDB_DIR' EXIT && \
          echo \"start: \$(date)  restart count: \${SLURM_RESTART_COUNT:-0}\" && \
          mkdir -p \$LMDB_DIR && df -h \$LMDB_DIR && \
          time cp ${SRC}/*.lmdb \$LMDB_DIR/ && \
          torchrun \
          --nproc_per_node=${NUM_GPUS} \
          --master_port=8985 \
          finetune_multitask.py \
          --train_dataset=\"Hypersim_NVS(split='train', overfit=False, load_depth=True, n_target_views=2, max_views=4, config_path='/projects/rps/dff6142/fouheylab/dma9300/3d_ssl_project/configs/hypersim_nvs.yaml') + Taskonomy_NVS(split='train', overfit=False, load_depth=True, n_target_views=2, max_views=4, config_path='/projects/rps/dff6142/fouheylab/dma9300/3d_ssl_project/configs/taskonomy_nvs.yaml') + Replica_NVS(split='train', overfit=False, load_depth=True, n_target_views=2, max_views=4, config_path='/projects/rps/dff6142/fouheylab/dma9300/3d_ssl_project/configs/replica_nvs.yaml') + Matterport3D_NVS(split='train', overfit=False, load_depth=True, n_target_views=2, max_views=4, config_path='/projects/rps/dff6142/fouheylab/dma9300/3d_ssl_project/configs/matterport3d_nvs.yaml') + 100_000 @ Habitat_HM3D_NVS(split='train', overfit=False, load_depth=True, n_target_views=2, max_views=4, config_path='/projects/rps/dff6142/fouheylab/dma9300/3d_ssl_project/configs/habitat_hm3d_nvs.yaml') + 100_000 @ ScanNet_NVS(split='train', overfit=False, load_depth=True, n_target_views=2, max_views=4, config_path='/projects/rps/dff6142/fouheylab/dma9300/3d_ssl_project/configs/scannet_nvs.yaml') + 70_000 @ Arkitscenes_NVS(split='train', overfit=False, load_depth=True, n_target_views=2, max_views=4, config_path='/projects/rps/dff6142/fouheylab/dma9300/3d_ssl_project/configs/arkitscene_nvs.yaml') + 100_000 @ ScanNetPP_NVS(split='train', overfit=False, load_depth=True, n_target_views=2, max_views=4, config_path='/projects/rps/dff6142/fouheylab/dma9300/3d_ssl_project/configs/scannetpp_nvs.yaml')\" \
          --test_dataset=\"Hypersim_NVS(split='val', overfit=False, load_depth=True, n_target_views=2, max_views=4, config_path='/projects/rps/dff6142/fouheylab/dma9300/3d_ssl_project/configs/hypersim_nvs.yaml') + Taskonomy_NVS(split='val', overfit=False, load_depth=True, n_target_views=2, max_views=4, config_path='/projects/rps/dff6142/fouheylab/dma9300/3d_ssl_project/configs/taskonomy_nvs.yaml') + Replica_NVS(split='val', overfit=False, load_depth=True, n_target_views=2, max_views=4, config_path='/projects/rps/dff6142/fouheylab/dma9300/3d_ssl_project/configs/replica_nvs.yaml') + Matterport3D_NVS(split='val', overfit=False, load_depth=True, n_target_views=2, max_views=4, config_path='/projects/rps/dff6142/fouheylab/dma9300/3d_ssl_project/configs/matterport3d_nvs.yaml') + Habitat_HM3D_NVS(split='val', overfit=False, load_depth=True, n_target_views=2, max_views=4, config_path='/projects/rps/dff6142/fouheylab/dma9300/3d_ssl_project/configs/habitat_hm3d_nvs.yaml') + ScanNet_NVS(split='val', overfit=False, load_depth=True, n_target_views=2, max_views=4, config_path='/projects/rps/dff6142/fouheylab/dma9300/3d_ssl_project/configs/scannet_nvs.yaml') + Arkitscenes_NVS(split='val', overfit=False, load_depth=True, n_target_views=2, max_views=4, config_path='/projects/rps/dff6142/fouheylab/dma9300/3d_ssl_project/configs/arkitscene_nvs.yaml') + ScanNetPP_NVS(split='val', overfit=False, load_depth=True, n_target_views=2, max_views=4, config_path='/projects/rps/dff6142/fouheylab/dma9300/3d_ssl_project/configs/scannetpp_nvs.yaml')\" \
          --model=\"vista_basev2_downstream()\" \
          --ckpt_path=\"/projects/rps/dff6142/fouheylab/dma9300/3d_ssl_project/checkpoints/vista_v2_full_b_loss_norm_loss_decoder_depth_2_base_model_ssl_all_datasets_model_224_x_224_correct_augmentations\" \
          --lr=2e-4 \
          --epochs=${EPOCHS} \
          --batch_size=128 \
          --accum_iter=1 \
          --save_freq=1 \
          --keep_freq=10 \
          --eval_freq=5 \
          --print_freq=100 \
          --output_dir=\"/projects/rps/dff6142/fouheylab/dma9300/3d_ssl_project/checkpoints/vistav2_full_weight_b_c_heads_decoder_depth_2_scene_norm_patch_tokens_pose_base_model_downstream_depth_camera_all_datasets_model_224_x_224_${EPOCHS}ep_latest/\""