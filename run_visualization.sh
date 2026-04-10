#!/usr/bin/env bash
set -euo pipefail

# Override these per machine/session as needed.
OUTDIR="${OUTDIR:-/media/volume/generated_data/pi0_output/droid/droid_1.0.1_lerobot}"
LEROBOT_ROOT="${LEROBOT_ROOT:-/media/volume/droid_data/DROID/droid_1.0.1/}"
THRESHOLD_MODE="${THRESHOLD_MODE:-cluster_count}"

# Optional preprocessing:
# python3 mass_droid_clustering.py \
#   --summary_csv "$OUTDIR/summary.csv" \
#   --actions_npz "$OUTDIR/actions.npz" \
#   --outdir "$OUTDIR/all_multimodality" \
#   --plot_top_n 10 \
#   --k_min 1 --k_max 10 \
#   --n_jobs 4 \
#   --best_metric variance_drop_ratio \
#   --k_selection auc \
#   --debug_wvar

python droid_gaussian_threshold.py \
  --metrics_csv "$OUTDIR/all_multimodality/metrics_per_state.csv" \
  --outdir "$OUTDIR/gaussian_threshold_out" \
  --actions_npz "$OUTDIR/actions.npz" \
  --gaussian_trials 5 \
  --gaussian_max_points 10000 \
  --gaussian_multiplier 0.75 \
  --threshold_mode "$THRESHOLD_MODE" \
  --plot_pass_top 200 \
  --ee_mode first3 \
  --lerobot_root "$LEROBOT_ROOT" \
  --lerobot_state_col observation.state
