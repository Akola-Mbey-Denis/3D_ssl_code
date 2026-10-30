#!/bin/bash
# 1-hour run with the REAL data setup, inside an interactive GPU session (request >= 320G memory):
#   bash test_interactive.sh
# Same datasets / subsets / val sets as the submitted job. Uses all GPUs of the session
# (torchrun if > 1). Log + checkpoints are kept in $OUT so you can inspect them afterwards.
set -e
cd /projects/rps/dff6142/fouheylab/dma9300/3d_ssl_project
SRC=/projects/rps/dff6142/fouheylab/dma9300/3d_ssl_project/preprocessed_data/training_lmdbs
C=/projects/rps/dff6142/fouheylab/dma9300/3d_ssl_project/configs
OUT=/projects/rps/dff6142/fouheylab/dma9300/3d_ssl_project/${USER}_realtest_out
RUN_SECONDS=3600        # stop after 1 hour
MODEL="mae_base()"

export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1
export LMDB_DIR=/dev/shm/${USER}_test
trap 'rm -rf $LMDB_DIR' EXIT          # free the RAM copy even on Ctrl-C / error (keeps $OUT)
mkdir -p $LMDB_DIR $OUT
time cp $SRC/*.lmdb $LMDB_DIR/
ls $LMDB_DIR

# train mix (same as the real job): small datasets in full, fixed subsets of the big ones
TRAIN_SETS="Hypersim_NVS:hypersim_nvs: Taskonomy_NVS:taskonomy_nvs: Replica_NVS:replica_nvs:
            MegaDepth_NVS:megadepth_nvs: Matterport3D_NVS:matterport3d_nvs:
            Habitat_HM3D_NVS:habitat_hm3d_nvs:100_000 DL3DV_NVS:dl3dv_nvs:100_000
            Mapfree_NVS:mapfree_nvs:50_000 ScanNet_NVS:scannet_nvs:100_000"
TRAIN=""; TEST=""
for d in $TRAIN_SETS; do
  IFS=: read cls cfg n <<< "$d"
  ARGS="overfit=False, n_target_views=2, max_views=6, config_path='$C/$cfg.yaml'"
  TRAIN+="${TRAIN:+ + }${n:+$n @ }$cls(split='train', $ARGS)"
  TEST+="${TEST:+ + }$cls(split='val', $ARGS)"                 # full val sets
done

NGPU=$(nvidia-smi -L | wc -l)
if [ "$NGPU" -gt 1 ]; then LAUNCH="torchrun --nproc_per_node=$NGPU --master_port=8499"; else LAUNCH="python"; fi
echo "GPUs: $NGPU  ->  $LAUNCH"

timeout --signal=INT $RUN_SECONDS $LAUNCH train_ssl_baselines.py \
  --train_dataset "$TRAIN" \
  --test_dataset "$TEST" \
  --model "$MODEL" --base_lr 2e-4 --epochs 110 --batch_size 128 --accum_iter 1 \
  --save_freq 1 --keep_freq 1 --eval_freq 1 --print_freq 50 \
  --output_dir "$OUT/" 2>&1 | tee $OUT/log.txt || true

echo "Done (stopped after ${RUN_SECONDS}s or finished). Log: $OUT/log.txt"
echo "Train speed (last lines):"; grep -E "Epoch: \[[0-9]+\] +\[" $OUT/log.txt | tail -3