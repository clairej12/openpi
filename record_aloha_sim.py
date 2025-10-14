# run_aloha_sim_and_record.py
import dataclasses
import enum
import logging
import pathlib
import socket
import time
import multiprocessing as mp
from typing import Optional

import tyro
import numpy as np

# --- your local modules ---
import examples.aloha_sim.env as _env
import examples.aloha_sim.saver as _saver

# OpenPi (server side)
from openpi.policies import policy as _policy
from openpi.policies import policy_config as _policy_config
from openpi.serving import websocket_policy_server
from openpi.training import config as _config

# OpenPi (client side)
from openpi_client import action_chunk_broker
from openpi_client import websocket_client_policy as _websocket_client_policy
from openpi_client.runtime import runtime as _runtime
from openpi_client.runtime.agents import policy_agent as _policy_agent
from openpi_client.runtime import subscriber as _subscriber


# -----------------------------
# Args & simple config helpers
# -----------------------------
class EnvMode(enum.Enum):
    ALOHA_SIM = "aloha_sim"


@dataclasses.dataclass
class Checkpoint:
    """Load a policy from a trained checkpoint."""
    config: str
    dir: str


DEFAULT_CHECKPOINT = {
    EnvMode.ALOHA_SIM: Checkpoint(
        config="pi0_aloha_sim",
        dir="/media/volume/models_and_data/openpi_models/openpi-assets/checkpoints/pi0_aloha_sim" # "gs://openpi-assets/checkpoints/pi0_aloha_sim",
    )
}


@dataclasses.dataclass
class Args:
    # Output
    out_dir: pathlib.Path = pathlib.Path("data/aloha_sim/")
    save_images: bool = False             # include RGB in npz (big files!)
    subsample_video: int = 1              # keep every k frames when writing mp4

    # Env / rollout
    task: str = "gym_aloha/AlohaInsertion-v0"
    seed: int = 42
    action_horizon: int = 10
    max_hz: int = 50
    num_episodes: Optional[int] = None    # None = run forever

    # Server (policy host)
    host: str = "127.0.0.1"
    port: int = 8000
    env_mode: EnvMode = EnvMode.ALOHA_SIM

    # Which checkpoint (defaults by env)
    ckpt_config: Optional[str] = None
    ckpt_dir: Optional[str] = None

    # Logging
    log_level: str = "INFO"


# --------------------------------
# Server process (OpenPi policy)
# --------------------------------
def _create_policy(ckpt: Checkpoint, default_prompt: Optional[str] = None) -> _policy.Policy:
    return _policy_config.create_trained_policy(
        _config.get_config(ckpt.config),
        ckpt.dir,
        default_prompt=default_prompt
    )


def _serve_policy_process(port: int, ckpt: Checkpoint):
    logging.basicConfig(level=logging.INFO, force=True)
    policy = _create_policy(ckpt)
    policy_metadata = policy.metadata

    hostname = socket.gethostname()
    local_ip = socket.gethostbyname(hostname)
    logging.info("Starting policy server (host: %s, ip: %s, port: %d)", hostname, local_ip, port)

    server = websocket_policy_server.WebsocketPolicyServer(
        policy=policy,
        host="0.0.0.0",
        port=port,
        metadata=policy_metadata,
    )
    server.serve_forever()


def _wait_for_server(host: str, port: int, timeout_s: float = 30.0):
    t0 = time.time()
    while time.time() - t0 < timeout_s:
        try:
            with socket.create_connection((host, port), timeout=1.0):
                return True
        except OSError:
            time.sleep(0.2)
    return False


# --------------------------------
# NPZ trajectory saver subscriber
# --------------------------------
class NpzSaver(_subscriber.Subscriber):
    """
    Saves per-episode trajectories into .npz files:
      - states:   (T, state_dim)
      - actions:  (T, action_dim)
      - rewards:  (T,)
      - (optional) images: list or uint8 array, if save_images=True
    """
    def __init__(self, out_dir: pathlib.Path, save_images: bool = False):
        out_dir.mkdir(parents=True, exist_ok=True)
        self._out_dir = out_dir
        self._save_images = save_images
        self._reset_buffers()

    def _reset_buffers(self):
        self._states = []
        self._actions = []
        self._rewards = []
        self._images = []

    def on_episode_start(self) -> None:
        self._reset_buffers()

    def on_step(self, observation: dict, action: dict) -> None:
        # obs/state shape from your AlohaSimEnvironment converter
        state = np.asarray(observation["state"])
        self._states.append(state)

        act = np.asarray(action["actions"]) if "actions" in action else np.asarray(action)
        self._actions.append(act)

        # reward is not in observation; Runtime will call on_reward
        if self._save_images:
            # observation["images"]["cam_high"] is [C,H,W] uint8
            img = np.transpose(observation["images"]["cam_high"], (1, 2, 0))  # [H,W,C]
            self._images.append(img)

    def on_reward(self, reward: float) -> None:
        self._rewards.append(float(reward))

    def on_episode_end(self) -> None:
        existing = list(self._out_dir.glob("traj_[0-9]*.npz"))
        next_idx = max([int(p.stem.split("_")[1]) for p in existing], default=-1) + 1
        out_path = self._out_dir / f"traj_{next_idx}.npz"

        states = np.stack(self._states, axis=0) if self._states else np.empty((0,))
        actions = np.stack(self._actions, axis=0) if self._actions else np.empty((0,))
        rewards = np.asarray(self._rewards, dtype=np.float32)

        save_dict = {
            "states": states,
            "actions": actions,
            "rewards": rewards,
        }
        if self._save_images:
            # Save as a ragged list (object) to avoid huge contiguous arrays; or stack if you prefer
            save_dict["images"] = np.array(self._images, dtype=object)

        logging.info(f"Saving trajectory to {out_path}")
        np.savez_compressed(out_path, **save_dict)


# --------------------------------
# Main runner
# --------------------------------
def main(args: Args):
    logging.basicConfig(level=getattr(logging, args.log_level.upper(), logging.INFO), force=True)

    # Resolve which checkpoint to use
    ckpt = Checkpoint(
        config=args.ckpt_config or DEFAULT_CHECKPOINT[args.env_mode].config,
        dir=args.ckpt_dir or DEFAULT_CHECKPOINT[args.env_mode].dir
    )

    # Start server in a subprocess
    server = mp.Process(target=_serve_policy_process, args=(args.port, ckpt), daemon=True)
    server.start()
    logging.info("Launched policy server (pid=%s). Waiting for readiness…", server.pid)

    ok = _wait_for_server(args.host, args.port, timeout_s=45.0)
    if not ok:
        server.terminate()
        server.join(timeout=5)
        raise RuntimeError(f"Policy server did not become ready on {args.host}:{args.port}")

    # Add the task name to the out directory
    args.out_dir = args.out_dir / args.task.replace("/", "_")
    # Build runtime: environment + client policy agent + subscribers
    out_video_dir = args.out_dir / "videos"
    out_traj_dir = args.out_dir / "trajectories"

    runtime = _runtime.Runtime(
        environment=_env.AlohaSimEnvironment(
            task=args.task,
            seed=args.seed
        ),
        agent=_policy_agent.PolicyAgent(
            policy=action_chunk_broker.ActionChunkBroker(
                policy=_websocket_client_policy.WebsocketClientPolicy(
                    host=args.host,
                    port=args.port
                ),
                action_horizon=args.action_horizon
            )
        ),
        subscribers=[
            _saver.VideoSaver(out_video_dir, subsample=args.subsample_video),
            NpzSaver(out_traj_dir, save_images=args.save_images)
        ],
        max_hz=args.max_hz
    )

    # Run episodes
    try:
        if args.num_episodes is None:
            logging.info("Running indefinitely (Ctrl-C to stop)…")
            runtime.run()
        else:
            logging.info("Running %d episode(s)…", args.num_episodes)
            # If your Runtime lacks an explicit “run_n” or “request_stop” interface,
            # the simplest approach is to loop num_episodes times and call run() each time
            # if it returns after a single episode. If run() loops forever, you may need
            # a small wrapper in Runtime that stops after an episode. Many setups already
            # stop and restart internally, so try the simple version first:
            for ep in range(args.num_episodes):
                logging.info("Episode %d/%d", ep + 1, args.num_episodes)
                runtime.run()
    except KeyboardInterrupt:
        logging.info("Interrupted by user.")
    finally:
        # Clean shutdown of the server
        if server.is_alive():
            server.terminate()
            server.join(timeout=5)
        logging.info("Done.")


if __name__ == "__main__":
    tyro.cli(main)