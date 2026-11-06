#!/bin/bash

GPU_TYPE="b200"
NUM_GPUS=2
MEM_OVERRIDE=""
EPOCHS=140

# ---------------------------
# Parse flags
# ---------------------------
# usage: ./submit_poincar3_ldi_scannetpp.sh -g b200 -n 2 [-m 512] [-e 70]
# LDI probe (frozen SSL encoder + LayeredDPTHead) on ScanNet++ ONLY (train), validated on ScanNet++ val.
# Global batch 64 = NUM_GPUS x batch_size: ~158,293 ScanNet++ sequences / 64 = ~2,473 steps / epoch
# (check the "Train dataset length" in the log).
while getopts g:n:m:e: flag; do
    case "${flag}" in
        g) GPU_TYPE=${OPTARG};;
        n) NUM_GPUS=${OPTARG};;
        m) MEM_OVERRIDE=${OPTARG};;     # GB, optional
        e) EPOCHS=${OPTARG};;           # fine-tuning epochs (70)
    esac
done

if ! [[ "$NUM_GPUS" =~ ^[1-8]$ ]]; then
    echo "Error: Number of GPUs must be between 1 and 8."
    exit 1
fi

GLOBAL_BATCH=64
if (( GLOBAL_BATCH % NUM_GPUS != 0 )); then
    echo "Error: global batch ${GLOBAL_BATCH} is not divisible by ${NUM_GPUS} GPUs."
    exit 1
fi
BATCH_SIZE=$((GLOBAL_BATCH / NUM_GPUS))     # per GPU

HOSTNAME=$(hostname)

# ---------------------------
# Set resources based on host
# ---------------------------
MAX_MEMORY=640         # GB

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

SRC=/projects/rps/dff6142/fouheylab/dma9300/3d_ssl_project/preprocessed_data/ldi_lmdbs   # LDI LMDBs are copied from here to node-local disk
LMDB_GLOB="scannetpp_*.lmdb"   # only the ScanNet++ LMDBs are needed (check the file names in \$SRC)

# ---------------------------
# Debug output
# ---------------------------
echo "Submitting POINCAR3 LDI probe job (ScanNet++ only) with:"
echo "GPU_TYPE = $GPU_TYPE"
echo "NUM_GPUS = $NUM_GPUS"
echo "CPUS = $CPUS"
echo "MEMORY = ${MEMORY}GB"
echo "GRES = $GRES"
echo "EPOCHS = $EPOCHS"
echo "LMDBs = ${SRC}/${LMDB_GLOB}"
ls ${SRC}/${LMDB_GLOB} >/dev/null 2>&1 || { echo "Error: no ${LMDB_GLOB} in ${SRC}"; exit 1; }
echo "BATCH_SIZE = $BATCH_SIZE per GPU (global $GLOBAL_BATCH)"
[ ${#EXTRA_SBATCH[@]} -gt 0 ] && echo "EXTRA = ${EXTRA_SBATCH[*]}"

# ---------------------------
# Submit SLURM job
# ---------------------------
mkdir -p poincar3_ldi_training

sbatch \
  --account=torch_pr_41_courant \
  --time=36:00:00 \
  --mem=${MEMORY}G \
  --gres=${GRES} \
  --cpus-per-task=${CPUS} \
  --constraint=${GPU_TYPE} \
  --job-name="poincar3_ldi_scannetpp_${EPOCHS}ep_${GPU_TYPE}_${NUM_GPUS}gpu" \
  "${EXTRA_SBATCH[@]}" \
  --output=poincar3_ldi_scannetpptraining/%j.out \
  --error=poincar3_ldi_scannetpptraining/%j.err \
  --wrap="source /projects/rps/dff6142/fouheylab/dma9300/miniconda3/etc/profile.d/conda.sh && \
          conda activate mast3r && \
          cd /projects/rps/dff6142/fouheylab/dma9300/3d_ssl_project/ && \
          export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 && \
          export LMDB_DIR=/dev/shm/\${USER}_\${SLURM_JOB_ID} && \
          trap 'rm -rf \$LMDB_DIR' EXIT && \
          echo \"start: \$(date)  restart count: \${SLURM_RESTART_COUNT:-0}\" && \
          mkdir -p \$LMDB_DIR && df -h \$LMDB_DIR && \
          time cp ${SRC}/${LMDB_GLOB} \$LMDB_DIR/ && \
          torchrun \
          --nproc_per_node=${NUM_GPUS} \
          --master_port=8794 \
          finetune_ldi.py \
          --train_dataset=\"ScannetPP_LDI(split='train', config_path='/projects/rps/dff6142/fouheylab/dma9300/3d_ssl_project/configs/scannet_pp_ldi.yaml')\" \
          --test_dataset=\"ScannetPP_LDI(split='val', config_path='/projects/rps/dff6142/fouheylab/dma9300/3d_ssl_project/configs/scannet_pp_ldi.yaml')\" \
          --model=\"poincar3_ldi(num_layers=4)\" \
          --ckpt_path=\"/projects/rps/dff6142/fouheylab/dma9300/3d_ssl_project/checkpoints/TODO_POINCAR3_SSL_CHECKPOINT_FOLDER\" \
          --lr=1e-4 \
          --epochs=${EPOCHS} \
          --batch_size=64\
          --accum_iter=1 \
          --save_freq=1 \
          --keep_freq=10 \
          --eval_freq=5 \
          --print_freq=100 \
          --output_dir=\"/projects/rps/dff6142/fouheylab/dma9300/3d_ssl_project/checkpoints/poincar3_shared_dpt_prediction_features_base_model_downstream_ldi_scannetpp_focal_loss_only_bs64_model_224_x_224_${EPOCHS}ep/\""