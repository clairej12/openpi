#!/usr/bin/env bash
set -e

REPO="lerobot/droid_1.0.1"
OUT="/media/volume/droid_data/DROID/droid_1.0.1"
START_IDX=0
NFILES=200  # how many indices to grab from chunk-000

for i in $(seq $START_IDX $((START_IDX+NFILES-1))); do
  idx=$(printf "%03d" "$i")  # 000, 001, ...

  declare -a FILES=(
    "data/chunk-000/file-${idx}.parquet"
    "videos/observation.images.exterior_1_left/chunk-000/file-${idx}.mp4"
    "videos/observation.images.exterior_2_left/chunk-000/file-${idx}.mp4"
    "videos/observation.images.wrist_left/chunk-000/file-${idx}.mp4"
  )

  for file in "${FILES[@]}"; do
    local_path="${OUT}/${file}"
    if [[ -f "$local_path" ]]; then
      echo "Skipping existing file: $local_path"
      continue
    fi

    echo "Downloading: $file"
    huggingface-cli download \
      "$REPO" "$file" \
      --repo-type dataset \
      --local-dir "$OUT" \
      --local-dir-use-symlinks False || echo "⚠️  Missing or failed: $file"
  done
done