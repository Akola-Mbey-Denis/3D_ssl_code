#!/bin/bash
# Check the LDI probe runs end to end before submitting, inside an interactive GPU session (request >= 320G memory):
#   bash test_ldi_interactive.sh            # all four models: mae mum muskie vista
#   bash test_ldi_interactive.sh vista      # one model
# Per model, two stages (same models / checkpoints / datasets / global batch 256 as the submit scripts):
#   1. smoke  overfit keys (100 sequences per dataset), 2 epochs, eval every epoch -> the full loop runs:
#             train, val, TensorBoard images, checkpoint save + resume files
#   2. speed  the REAL train / val sets for $RUN_SECONDS, then stopped -> step time, loss going down, memory
# Uses all GPUs of the session (torchrun if > 1). Logs + checkpoints are kept in $OUT/<model>_{smoke,speed}.
set -e
cd /projects/rps/dff6142/fouheylab/dma9300/3d_ssl_project
SRC=/projects/rps/dff6142/fouheylab/dma9300/3d_ssl_project/preprocessed_data/ldi_lmdbs
C=/projects/rps/dff6142/fouheylab/dma9300/3d_ssl_project/configs
CK=/projects/rps/dff6142/fouheylab/dma9300/3d_ssl_project/checkpoints
OUT=/projects/rps/dff6142/fouheylab/dma9300/3d_ssl_project/${USER}_ldi_test_out
RUN_SECONDS=${RUN_SECONDS:-600}        # speed stage: stop after 10 min per model
GLOBAL_BATCH=256
MODELS=${1:-all}
[ "$MODELS" == "all" ] && MODELS="mae mum muskie vista"

export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1
export LMDB_DIR=/dev/shm/${USER}_ldi_test
trap 'rm -rf $LMDB_DIR' EXIT          # free the RAM copy even on Ctrl-C / error (keeps $OUT)
mkdir -p $LMDB_DIR $OUT
time cp $SRC/*.lmdb $LMDB_DIR/
ls -lh $LMDB_DIR

# datasets: same as the submitted job (train = full train splits, val = fixed 2,000-sequence val subsets)
SETS="ScanNet_LDI:scannet_ldi Matterport3D_LDI:matterport3d_ldi ARKitScenes_LDI:arkitscene_ldi"
TRAIN=""; TEST=""; OVERFIT=""
for d in $SETS; do
  IFS=: read cls cfg <<< "$d"
  TRAIN+="${TRAIN:+ + }$cls(split='train', config_path='$C/$cfg.yaml')"
  TEST+="${TEST:+ + }$cls(split='val', config_path='$C/$cfg.yaml')"
  OVERFIT+="${OVERFIT:+ + }$cls(split='train', overfit=True, config_path='$C/$cfg.yaml')"
done

NGPU=$(nvidia-smi -L | wc -l)
if (( GLOBAL_BATCH % NGPU != 0 )); then echo "global batch $GLOBAL_BATCH not divisible by $NGPU GPUs"; exit 1; fi
BS=$((GLOBAL_BATCH / NGPU))
if [ "$NGPU" -gt 1 ]; then LAUNCH="torchrun --nproc_per_node=$NGPU --master_port=8499"; else LAUNCH="python"; fi
echo "GPUs: $NGPU  ->  $LAUNCH   batch $BS per GPU (global $GLOBAL_BATCH)"

run() {   # run <name> <model> <ckpt> <train> <test> <epochs> <eval_freq> <seconds>
  local dir=$OUT/$1
  rm -rf $dir && mkdir -p $dir                          # fresh: finetune_ldi.py would resume from checkpoint-last
  echo; echo "=================== $1 ==================="
  timeout --signal=INT $8 $LAUNCH finetune_ldi.py \
    --train_dataset "$4" \
    --test_dataset "$5" \
    --model "$2" --ckpt_path "$3" \
    --lr 2e-4 --epochs $6 --batch_size $BS --accum_iter 1 --num_workers 8 \
    --save_freq 1 --keep_freq 100 --eval_freq $7 --print_freq 20 \
    --output_dir "$dir/" 2>&1 | tee $dir/log.txt || true
}

declare -A STATUS
for m in $MODELS; do
  MODEL="${m}_base_ldi(num_layers=4)"
  CKPT=$CK/${m}_base_model_ssl_all_datasets_model_224_x_224_correct_augmentations
  if [ ! -f $CKPT/checkpoint-final.pth ]; then
    echo "!! $m: no $CKPT/checkpoint-final.pth (load_encoder reads that file) -- skipped"; STATUS[$m]="NO CKPT"; continue
  fi

  run ${m}_smoke "$MODEL" $CKPT "$OVERFIT" "$OVERFIT" 2 1 1800
  run ${m}_speed "$MODEL" $CKPT "$TRAIN" "$TEST" 70 5 $RUN_SECONDS

  s=$OUT/${m}_smoke/log.txt; p=$OUT/${m}_speed/log.txt
  ok=OK
  grep -q "Traceback" $s $p && ok="ERROR (Traceback)"
  [ -f $OUT/${m}_smoke/checkpoint-last.pth ] || ok="$ok, no checkpoint-last"
  grep -q "Test Epoch" $s || ok="$ok, no eval"
  STATUS[$m]=$ok

  echo; echo "----- $m summary -----"
  grep -m1 "Parameters:" $s || true
  grep -h "fixed keys\|LDI \[" $s | sort -u | head -6 || true
  echo "smoke eval:";  grep "Averaged stats" $s | tail -2 || true
  echo "speed (last lines):"; grep -E "Epoch: \[[0-9]+\] +\[" $p | tail -3 || true
  nvidia-smi --query-gpu=index,memory.used,memory.total --format=csv,noheader || true
done

echo; echo "=================== RESULT ==================="
for m in $MODELS; do echo "$m: ${STATUS[$m]}"; done
echo "logs: $OUT/<model>_{smoke,speed}/log.txt   TensorBoard: tensorboard --logdir $OUT"