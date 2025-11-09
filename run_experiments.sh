# uv run record_trajectories.py \
#   --config pi0_aloha_sim \
#   --ckpt-dir gs://openpi-assets/checkpoints/pi0_aloha_sim \
#   --env-factory openpi.envs.aloha_sim:make_env \
#   --env-kwargs-json '{"render_mode":"rgb_array","seed":0}' \
#   --num-episodes 5 \
#   --max-episode-steps 200 \
#   --out-dir rollouts \
#   --save-video

# uv run record_droid_sanity_check.py \
#     --data_dir /media/volume/models_and_data/DROID/ \
#     --dataset_name droid_100 \
#     --split train \
#     --max_episodes 100 \
#     --port 8124 \
#     --prompt "" \
#     --samples_per_state 50 \
#     --save_npz \
#     --checkpoint_by_episode \
#     --out_dir droid_100_all \
#     --image_dir droid_images

# uv run record_droid_sanity_check.py \
#     --data_dir /media/volume/models_and_data/DROID/ \
#     --dataset_name droid_100 \
#     --split train \
#     --max_episodes 30 \
#     --port 8123 \
#     --prompt "" \
#     --samples_per_state 50 \
#     --save_npz \
#     --checkpoint_by_episode \
#     --out_dir droid_100_all \
#     --image_dir droid_images

# uv run record_droid_sanity_check.py \
#     --data_dir /media/volume/models_and_data/DROID/ \
#     --dataset_name droid_100 \
#     --split train \
#     --max_episodes 40 \
#     --port 8123 \
#     --prompt "" \
#     --samples_per_state 50 \
#     --save_npz \
#     --checkpoint_by_episode \
#     --out_dir droid_100_all \
#     --image_dir droid_images

# uv run record_droid_sanity_check.py \
#     --data_dir /media/volume/models_and_data/DROID/ \
#     --dataset_name droid_100 \
#     --split train \
#     --max_episodes 50 \
#     --port 8123 \
#     --prompt "" \
#     --samples_per_state 50 \
#     --save_npz \
#     --checkpoint_by_episode \
#     --out_dir droid_100_all \
#     --image_dir droid_images

# uv run record_droid_sanity_check.py \
#     --data_dir /media/volume/models_and_data/DROID/ \
#     --dataset_name droid_100 \
#     --split train \
#     --max_episodes 60 \
#     --port 8123 \
#     --prompt "" \
#     --samples_per_state 50 \
#     --save_npz \
#     --checkpoint_by_episode \
#     --out_dir droid_100_all \
#     --image_dir droid_images

# uv run record_droid_sanity_check.py \
#     --data_dir /media/volume/models_and_data/DROID/ \
#     --dataset_name droid_100 \
#     --split train \
#     --max_episodes 70 \
#     --port 8123 \
#     --prompt "" \
#     --samples_per_state 50 \
#     --save_npz \
#     --checkpoint_by_episode \
#     --out_dir droid_100_all \
#     --image_dir droid_images

# uv run record_droid_sanity_check.py \
#     --data_dir /media/volume/models_and_data/DROID/ \
#     --dataset_name droid_100 \
#     --split train \
#     --max_episodes 80 \
#     --port 8123 \
#     --prompt "" \
#     --samples_per_state 50 \
#     --save_npz \
#     --checkpoint_by_episode \
#     --out_dir droid_100_all \
#     --image_dir droid_images

# uv run record_droid_sanity_check.py \
#     --data_dir /media/volume/models_and_data/DROID/ \
#     --dataset_name droid_100 \
#     --split train \
#     --max_episodes 90 \
#     --port 8123 \
#     --prompt "" \
#     --samples_per_state 50 \
#     --save_npz \
#     --checkpoint_by_episode \
#     --out_dir droid_100_all \
#     --image_dir droid_images

# uv run record_droid_sanity_check.py \
#     --data_dir /media/volume/models_and_data/DROID/ \
#     --dataset_name droid_100 \
#     --split train \
#     --max_episodes 100 \
#     --port 8123 \
#     --prompt "" \
#     --samples_per_state 50 \
#     --save_npz \
#     --checkpoint_by_episode \
#     --out_dir droid_100_all \
#     --image_dir droid_100_images

uv run record_droid_sanity_check.py \
    --data_dir /media/volume/droid_data/DROID/droid_1.0.1/ \
    --format lerobot \
    --max_episodes 100 \
    --port 8124 \
    --prompt "" \
    --samples_per_state 50 \
    --save_npz \
    --checkpoint_by_episode \
    --out_dir droid_1.0.1_lerobot \
    --image_dir droid_1.0.1_images