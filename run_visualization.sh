# OUTDIR=droid_sanity
# uv run visualize_droid_sanity_check.py \
#     --cluster_scope all_actions_across_samples \
#     --summary_csv $OUTDIR/summary.csv \
#     --actions_npz $OUTDIR/actions.npz \
#     --outdir $OUTDIR/clusters \
#     --method minkowski --n_clusters 5 \
#     --cluster_target chunks \
#     --ee-mode first3


OUTDIR=droid_100_all
# python3 mass_droid_clustering.py \
#   --summary_csv $OUTDIR/summary.csv \
#   --actions_npz $OUTDIR/actions.npz \
#   --outdir $OUTDIR/all_multimodality \
#   --plot_top_n 10 \
#   --k_min 4 --k_max 6 \
#   --n_jobs 4 \
#   --max_states 6500
# #   --parallel_dtw \
# #   --method dtw \

python droid_gaussian_threshold.py \
  --metrics_csv $OUTDIR/all_multimodality/metrics_per_state.csv \
  --outdir $OUTDIR/gaussian_threshold_out \
  --actions_npz $OUTDIR/actions.npz \
  --gaussian_trials 5 \
  --gaussian_max_points 5000 \
  --gaussian_multiplier 0.5 \
  --plot_pass_top 10 \
  --ee_mode first3