# OUTDIR=droid_sanity
# uv run visualize_droid_sanity_check.py \
#     --cluster_scope all_actions_across_samples \
#     --summary_csv $OUTDIR/summary.csv \
#     --actions_npz $OUTDIR/actions.npz \
#     --outdir $OUTDIR/clusters \
#     --method minkowski --n_clusters 5 \
#     --cluster_target chunks \
#     --ee-mode first3


OUTDIR=/media/volume/generated_data/pi0_output/droid/droid_1.0.1_lerobot
python3 mass_droid_clustering.py \
  --summary_csv $OUTDIR/summary.csv \
  --actions_npz $OUTDIR/actions.npz \
  --outdir $OUTDIR/all_multimodality \
  --plot_top_n 50 \
  --k_min 2 --k_max 10 \
  --n_jobs 4 \
  --best_metric variance_drop_ratio \
  --k_selection auc
# optional: add `--max_states <value>` to the command above
# --parallel_dtw \
# --method dtw \

# python droid_gaussian_threshold.py \
#   --metrics_csv $OUTDIR/all_multimodality/metrics_per_state.csv \
#   --outdir $OUTDIR/gaussian_threshold_out \
#   --actions_npz $OUTDIR/actions.npz \
#   --gaussian_trials 5 \
#   --gaussian_max_points 10000 \
#   --gaussian_multiplier 0.75 \
#   --threshold_mode global_ratio \
#   --plot_pass_top 200 \
#   --ee_mode first3 \
#   --lerobot_root /media/volume/droid_data/DROID/droid_1.0.1/ \
#   --lerobot_state_col observation.state \
#   # --pca_global
#
# # Alternative thresholding mode (per-cluster separate-vs-rest pass count):
# # python droid_gaussian_threshold.py \
# #   --metrics_csv $OUTDIR/all_multimodality/metrics_per_state.csv \
# #   --outdir $OUTDIR/gaussian_threshold_out_cluster_count \
# #   --actions_npz $OUTDIR/actions.npz \
# #   --gaussian_trials 5 \
# #   --gaussian_multiplier 1.0 \
# #   --threshold_mode cluster_count \
# #   --cluster_compare lte \
# #   --cluster_pass_min 2

# python analyze_task_multimodality_trajectories.py \
#   --metrics_csv $OUTDIR/all_multimodality/metrics_per_state.csv \
#   --actions_npz $OUTDIR/actions.npz \
#   --outdir $OUTDIR/task_multimodality_plots \
#   --tasks close,flip,get,move,open,pick,pour,pull,push,put,remove,sweep,take,turn,use,move \
#   --traj_stat mean \
#   --episodes_per_band 3 \
#   --task_source both \
#   --lerobot_root /media/volume/droid_data/DROID/droid_1.0.1/ \
#   --lerobot_state_col observation.state \
#   --ks 2,4,6,8,10
