
from openpi.training import config as _config
from openpi.policies import policy_config
from openpi.shared import download
from openpi.policies import droid_policy


config = _config.get_config("pi05_droid")
print(f"config: {config}")
checkpoint_dir = "/media/volume/models_and_data/openpi_models/openpi-assets/checkpoints/pi05_droid"
print(f"checkpoint_dir: {checkpoint_dir}")
# Create a trained policy.
policy = policy_config.create_trained_policy(config, checkpoint_dir)

# Run inference on a dummy example.
example = droid_policy.make_droid_example()
action_chunk = policy.infer(example)["actions"]
print(f"Actions shape: {action_chunk.shape}")
