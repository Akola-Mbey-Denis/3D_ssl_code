#!/bin/bash
#SBATCH --job-name=download_replica
#SBATCH --output=replica_%j.out
#SBATCH --error=replica_%j.err
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=40G
#SBATCH --time=08:00:00

set -euo pipefail

DEST="/scratch/$USER/replica_full"
ARCHIVE_DIR="$DEST/REPLICA_FULL"

mkdir -p "$DEST" "$ARCHIVE_DIR" "$DEST/assets"

echo "Downloading Replica to $DEST"
cd "$ARCHIVE_DIR"

for p in {a..q}; do
    wget --continue \
        "https://github.com/facebookresearch/Replica-Dataset/releases/download/v1.0/replica_v1_0.tar.gz.parta$p"
done

cat replica_v1_0.tar.gz.parta{..q} \
    | unpigz -p 1 \
    | tar -xf - -C "$DEST"

wget --continue \
    http://dl.fbaipublicfiles.com/habitat/Replica/additional_habitat_configs.zip \
    -O "$DEST/assets/additional_habitat_configs.zip"

unzip -qn "$DEST/assets/additional_habitat_configs.zip" -d "$DEST"

echo "Replica download and extraction completed: $DEST"
EOF
