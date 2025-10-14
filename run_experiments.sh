uv run record_trajectories.py \
  --config pi0_aloha_sim \
  --ckpt-dir gs://openpi-assets/checkpoints/pi0_aloha_sim \
  --env-factory openpi.envs.aloha_sim:make_env \
  --env-kwargs-json '{"render_mode":"rgb_array","seed":0}' \
  --num-episodes 5 \
  --max-episode-steps 200 \
  --out-dir rollouts \
  --save-video