#!/bin/bash

GPU_TYPE="h200"
NUM_GPUS=1
TIME_LIMIT="72:00:00"
MEMORY=512             # GB, fixed (not scaled with the GPU count)

# ---------------------------
# Parse flags
# ---------------------------
while getopts g:n: flag; do
    case "${flag}" in
        g) GPU_TYPE=${OPTARG};;
        n) NUM_GPUS=${OPTARG};;
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
# Torch per-GPU share = node CPUs / GPUs and node memory / GPUs. Memory above the
# per-GPU share (MEM_PER_GPU x NUM_GPUS) may be rejected with "GPU job setup is not valid";
# a warning is printed below in that case.
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
SHARE=$((MEM_PER_GPU * NUM_GPUS))
if [ $MEMORY -gt $SHARE ]; then
    echo "WARNING: ${MEMORY}GB is more than the ${SHARE}GB per-GPU share for ${NUM_GPUS}x ${GPU_TYPE};"
    echo "         if sbatch says 'GPU job setup is not valid', lower MEMORY or use more GPUs."
fi

SRC=/projects/rps/dff6142/fouheylab/dma9300/3d_ssl_project/preprocessed_data/training_lmdbs   # LMDBs are copied from here to node-local disk

# ---------------------------
# Debug output
# ---------------------------
echo "Submitting SSL training job with:"
echo "GPU_TYPE = $GPU_TYPE"
echo "NUM_GPUS = $NUM_GPUS"
echo "CPUS = $CPUS"
echo "MEMORY = ${MEMORY}GB"
echo "TIME = ${TIME_LIMIT}"

# ---------------------------
# Submit SLURM job
# ---------------------------

sbatch \
  --account=torch_pr_41_tandon_advanced \
  --time=${TIME_LIMIT} \
  --mem=${MEMORY}G \
  --gres=gpu:${GPU_TYPE}:${NUM_GPUS} \
  --cpus-per-task=${CPUS} \
  --constraint=${GPU_TYPE} \
  --job-name="muskie_ssl_${GPU_TYPE}_${NUM_GPUS}gpu" \
  --output=muskie_ssl_training/%j.out \
  --error=muskie_ssl_training/%j.err \
  --wrap="source /projects/rps/dff6142/fouheylab/dma9300/miniconda3/etc/profile.d/conda.sh && \
          conda activate mast3r && \
          cd /projects/rps/dff6142/fouheylab/dma9300/3d_ssl_project/ && \
          export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 && \
          export LMDB_DIR=/dev/shm/\${USER}_\${SLURM_JOB_ID} && \
          mkdir -p \$LMDB_DIR && df -h \$LMDB_DIR && \
          time cp ${SRC}/*.lmdb \$LMDB_DIR/ && \
          torchrun \
          --nproc_per_node=${NUM_GPUS} \
          --master_port=8394 \
          train_ssl_muskie.py \
          --train_dataset=\"Hypersim_NVS(split='train', overfit=False, n_target_views=2, max_views=6, config_path='/projects/rps/dff6142/fouheylab/dma9300/3d_ssl_project/configs/hypersim_nvs.yaml') + Taskonomy_NVS(split='train', overfit=False, n_target_views=2, max_views=6, config_path='/projects/rps/dff6142/fouheylab/dma9300/3d_ssl_project/configs/taskonomy_nvs.yaml') + Replica_NVS(split='train', overfit=False, n_target_views=2, max_views=6, config_path='/projects/rps/dff6142/fouheylab/dma9300/3d_ssl_project/configs/replica_nvs.yaml') + MegaDepth_NVS(split='train', overfit=False, n_target_views=2, max_views=6, config_path='/projects/rps/dff6142/fouheylab/dma9300/3d_ssl_project/configs/megadepth_nvs.yaml') + Matterport3D_NVS(split='train', overfit=False, n_target_views=2, max_views=6, config_path='/projects/rps/dff6142/fouheylab/dma9300/3d_ssl_project/configs/matterport3d_nvs.yaml') + 100_000 @ Habitat_HM3D_NVS(split='train', overfit=False, n_target_views=2, max_views=6, config_path='/projects/rps/dff6142/fouheylab/dma9300/3d_ssl_project/configs/habitat_hm3d_nvs.yaml') + 100_000 @ DL3DV_NVS(split='train', overfit=False, n_target_views=2, max_views=6, config_path='/projects/rps/dff6142/fouheylab/dma9300/3d_ssl_project/configs/dl3dv_nvs.yaml') + 50_000 @ Mapfree_NVS(split='train', overfit=False, n_target_views=2, max_views=6, config_path='/projects/rps/dff6142/fouheylab/dma9300/3d_ssl_project/configs/mapfree_nvs.yaml') + 100_000 @ ScanNet_NVS(split='train', overfit=False, n_target_views=2, max_views=6, config_path='/projects/rps/dff6142/fouheylab/dma9300/3d_ssl_project/configs/scannet_nvs.yaml')\" \
          --test_dataset=\"Hypersim_NVS(split='val', overfit=False, n_target_views=2, max_views=6, config_path='/projects/rps/dff6142/fouheylab/dma9300/3d_ssl_project/configs/hypersim_nvs.yaml') + Taskonomy_NVS(split='val', overfit=False, n_target_views=2, max_views=6, config_path='/projects/rps/dff6142/fouheylab/dma9300/3d_ssl_project/configs/taskonomy_nvs.yaml') + Replica_NVS(split='val', overfit=False, n_target_views=2, max_views=6,config_path='/projects/rps/dff6142/fouheylab/dma9300/3d_ssl_project/configs/replica_nvs.yaml') + MegaDepth_NVS(split='val', overfit=False, n_target_views=2, max_views=6, config_path='/projects/rps/dff6142/fouheylab/dma9300/3d_ssl_project/configs/megadepth_nvs.yaml') + Matterport3D_NVS(split='val', overfit=False, n_target_views=2, max_views=6, config_path='/projects/rps/dff6142/fouheylab/dma9300/3d_ssl_project/configs/matterport3d_nvs.yaml') + Habitat_HM3D_NVS(split='val', overfit=False, n_target_views=2, max_views=6, config_path='/projects/rps/dff6142/fouheylab/dma9300/3d_ssl_project/configs/habitat_hm3d_nvs.yaml') + DL3DV_NVS(split='val', overfit=False, n_target_views=2, max_views=6, config_path='/projects/rps/dff6142/fouheylab/dma9300/3d_ssl_project/configs/dl3dv_nvs.yaml') + Mapfree_NVS(split='val', overfit=False, n_target_views=2, max_views=6, config_path='/projects/rps/dff6142/fouheylab/dma9300/3d_ssl_project/configs/mapfree_nvs.yaml') + ScanNet_NVS(split='val', overfit=False, n_target_views=2, max_views=6, config_path='/projects/rps/dff6142/fouheylab/dma9300/3d_ssl_project/configs/scannet_nvs.yaml')\" \
          --model=\"muskie_base(num_ref=2)\" \
          --base_lr=2e-4 \
          --epochs=110 \
          --batch_size=128 \
          --accum_iter=2 \
          --save_freq=10 \
          --keep_freq=10 \
          --eval_freq=10 \
          --print_freq=100 \
          --output_dir=\"/projects/rps/dff6142/fouheylab/dma9300/3d_ssl_project/checkpoints/muskie_base_model_ssl_all_datasets_model_224_x_224/\" ; rm -rf \$LMDB_DIR"