# OUTDIR=droid_sanity
# uv run visualize_droid_sanity_check.py \
#     --cluster_scope all_actions_across_samples \
#     --summary_csv $OUTDIR/summary.csv \
#     --actions_npz $OUTDIR/actions.npz \
#     --outdir $OUTDIR/clusters \
#     --method minkowski --n_clusters 5 \
#     --cluster_target chunks \
#     --ee-mode first3


OUTDIR=droid_1.0.1_lerobot
# python3 mass_droid_clustering.py \
#   --summary_csv $OUTDIR/summary.csv \
#   --actions_npz $OUTDIR/actions.npz \
#   --outdir $OUTDIR/all_multimodality \
#   --plot_top_n 50 \
#   --k_min 2 --k_max 5 \
#   --n_jobs 4 \
#   --best_metric variance_drop_ratio
#   # --max_states <value> \
# #   --parallel_dtw \
# #   --method dtw \

python droid_gaussian_threshold.py \
  --metrics_csv $OUTDIR/all_multimodality/metrics_per_state.csv \
  --outdir $OUTDIR/gaussian_threshold_out \
  --actions_npz $OUTDIR/actions.npz \
  --gaussian_trials 5 \
  --gaussian_max_points 10000 \
  --gaussian_multiplier 0.75 \
  --plot_pass_top 50 \
  --ee_mode first3