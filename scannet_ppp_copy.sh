#!/bin/bash
#SBATCH --job-name=copy_scannetpp
#SBATCH --account=torch_pr_41_tandon_advanced
#SBATCH --time=6:00:00
#SBATCH --cpus-per-task=8
#SBATCH --mem=16G
#SBATCH --output=copy_scannetpp_%j.out
#SBATCH --error=copy_scannetpp_%j.err
#
# Randomly select scenes from the OFFICIAL ScanNet++ splits and copy them (plus splits/metadata) to DST:
#   80 from train, 10 from val, 10 from test (counts configurable below).
#
#   DRY_RUN=1 bash copy_scannetpp_subset.sh     # select + show sizes, copy nothing
#   sbatch copy_scannetpp_subset.sh             # copy (re-run to resume)
#
set -euo pipefail

SRC=/projects/rps/dff6142/fouheylab/jt4100/datasets/scannetpp
DST=/scratch/dma9300/scannetpp
N_TRAIN=${N_TRAIN:-80}
N_VAL=${N_VAL:-10}
N_TEST=${N_TEST:-10}
SEED=${SEED:-0}            # same seed -> same scenes
PARALLEL=${PARALLEL:-8}    # scenes copied at the same time
DRY_RUN=${DRY_RUN:-0}

# official split files (ScanNet++ <root>/splits/); first existing file in each list is used
TRAIN_SPLITS="nvs_sem_train.txt train.txt"
VAL_SPLITS="nvs_sem_val.txt val.txt"
TEST_SPLITS="nvs_test.txt sem_test.txt test.txt"

if [ -d "$SRC/data" ]; then SCENE_ROOT="$SRC/data"; else SCENE_ROOT="$SRC"; fi
REL=${SCENE_ROOT#"$SRC"}; REL=${REL#/}
mkdir -p "$DST/$REL"

find_split() {   # first existing split file among the candidates
    for f in $1; do [ -f "$SRC/splits/$f" ] && { echo "$SRC/splits/$f"; return; }; done
    echo "ERROR: none of [$1] found in $SRC/splits" >&2; ls "$SRC/splits" >&2; exit 1
}

select_split() {  # name, candidates, n  -> writes DST/selected_<name>_seed<SEED>.txt
    local name=$1 cands=$2 n=$3 out="$DST/selected_${1}_seed${SEED}.txt" file
    file=$(find_split "$cands")
    if [ -s "$out" ]; then
        echo "$name: reusing $out"
    else
        # only scenes listed in the official split AND present on disk
        tr -d '\r' < "$file" | grep -v '^\s*$' | sort -u \
            | while read -r s; do if [ -d "$SCENE_ROOT/$s" ]; then echo "$s"; fi; done \
            | shuf -n "$n" --random-source=<(yes "$SEED") > "$out"
    fi
    local avail; avail=$(tr -d '\r' < "$file" | grep -vc '^\s*$')
    echo "$name: $(wc -l < "$out") of $avail scenes from $(basename "$file") -> $out"
    if [ "$(wc -l < "$out")" -lt "$n" ]; then echo "  WARNING: fewer than $n scenes available on disk for $name"; fi
    return 0
}

select_split train "$TRAIN_SPLITS" "$N_TRAIN"
select_split val   "$VAL_SPLITS"   "$N_VAL"
select_split test  "$TEST_SPLITS"  "$N_TEST"

ALL="$DST/selected_all_seed${SEED}.txt"
cat "$DST"/selected_{train,val,test}_seed"${SEED}".txt > "$ALL"
dups=$(sort "$ALL" | uniq -d | wc -l)
if [ "$dups" -gt 0 ]; then echo "WARNING: $dups scenes appear in more than one split list"; fi

# sizes, so you can check the /scratch quota before copying
for name in train val test; do
    ( cd "$SCENE_ROOT" && xargs -a "$DST/selected_${name}_seed${SEED}.txt" du -sc 2>/dev/null | tail -1 \
        | awk -v n="$name" '{printf "%-5s size: %.1f GB\n", n, $1/1024/1024}' )
done
df -h "$DST" | tail -1 | awk '{print "Free on destination: " $4}'

if [ "$DRY_RUN" = "1" ]; then echo "DRY_RUN=1: nothing copied."; exit 0; fi

# small top-level folders (splits, metadata, ...) and files
for d in "$SRC"/*; do
    if [ "$d" = "$SCENE_ROOT" ]; then continue; fi
    if [ -f "$d" ]; then rsync -a "$d" "$DST/"; continue; fi
    if [ -d "$d" ] && [ "$(du -s "$d" | cut -f1)" -lt 10485760 ]; then   # only folders < 10 GB
        rsync -a "$d" "$DST/"
    fi
done

# copy scenes in parallel; rsync resumes on re-runs
export SCENE_ROOT DST REL
xargs -a "$ALL" -P "$PARALLEL" -I{} \
    bash -c 'rsync -a --partial "$SCENE_ROOT/{}" "$DST/$REL/" && echo "done {}"'

echo "Copied $(ls -1 "$DST/$REL" | grep -cxFf "$ALL") / $(wc -l < "$ALL") scenes to $DST/$REL"