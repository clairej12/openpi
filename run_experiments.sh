#!/usr/bin/env bash
set -euo pipefail

# Override these per machine/session as needed.
DATA_DIR="${DATA_DIR:-/media/volume/droid_data/DROID/droid_1.0.1/}"
OUT_DIR="${OUT_DIR:-/media/volume/generated_data/pi0_output/droid/droid_1.0.1_lerobot}"
PORT="${PORT:-8124}"
MAX_EPISODES="${MAX_EPISODES:-200}"
SAMPLES_PER_STATE="${SAMPLES_PER_STATE:-200}"

uv run record_droid_sanity_check.py \
  --data_dir "$DATA_DIR" \
  --format lerobot \
  --max_episodes "$MAX_EPISODES" \
  --port "$PORT" \
  --prompt "" \
  --samples_per_state "$SAMPLES_PER_STATE" \
  --save_npz \
  --checkpoint_by_episode \
  --out_dir "$OUT_DIR" \
  --save_images false \
  --frame_stride 1 \
  --max_video_readers 8

# Optional metadata patching after rollout:
# uv run patch_task_metadata.py \
#   --metrics_csv "$OUT_DIR/all_multimodality/metrics_per_state.csv" \
#   --out_csv "$OUT_DIR/all_multimodality/metrics_per_state_patched.csv" \
#   --lerobot_root "$DATA_DIR" \
#   --summary_csv "$OUT_DIR/summary.csv" \
#   --summary_out_csv "$OUT_DIR/summary_patched.csv"
