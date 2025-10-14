#!/usr/bin/env python3
"""
End-to-end, single-script evaluator:
- Starts Pi0 policy server (DROID) in-process (background thread).
- Loads RLDS DROID dataset via TFDS.
- Samples many steps; for each step:
    * extract left/exterior image + wrist image + proprio
    * resize/pad to 224x224 (to match client)
    * send request to server, receive action chunk
- Collects outputs and writes:
    * CSV summary (per-step mean/std, etc.)
    * NPZ with raw action chunks (optional)

Usage example:
  python rlds_pi0_batch_eval.py \
    --data_dir /path/to/tfds \
    --dataset_name <YOUR_RLDS_NAME> \
    --split train \
    --max_steps 500 \
    --port 8123 \
    --prompt "place the red block on the shelf" \
    --save_npz
"""

import argparse
import atexit
import logging
import socket
import threading
import time
from typing import Dict, Tuple, Any, List

import numpy as np
import pandas as pd
from PIL import Image
import tensorflow_datasets as tfds

# --- OpenPI policy (server-side) ---
from openpi.policies import policy as _policy
from openpi.policies import policy_config as _policy_config
from openpi.training import config as _config
from openpi.serving import websocket_policy_server

# --- OpenPI client (for requests) ---
from openpi_client import image_tools
from openpi_client import websocket_client_policy


# ---------------------------
# Policy bootstrap (DROID)
# ---------------------------

def create_droid_policy(
    checkpoint_config: str | None,
    checkpoint_dir: str | None,
    default_prompt: str | None,
) -> _policy.Policy:
    """
    Load a DROID Pi0 policy either from provided checkpoint (config+dir) or default mapping.
    """
    if checkpoint_config and checkpoint_dir:
        logging.info("Loading trained policy: config=%s dir=%s", checkpoint_config, checkpoint_dir)
        return _policy_config.create_trained_policy(
            _config.get_config(checkpoint_config),
            checkpoint_dir,
            default_prompt=default_prompt,
        )
    # Fallback default (mirrors your serve script's DEFAULT_CHECKPOINT for DROID)
    logging.info("Loading DEFAULT DROID policy (pi05_droid).")
    return _policy_config.create_trained_policy(
        _config.get_config("pi05_droid"),
        "/media/volume/models_and_data/openpi_models/openpi-assets/checkpoints/pi05_droid",
        default_prompt=default_prompt,
    )


class InProcessPolicyServer:
    """
    Spins up the websocket policy server in a background thread in the same process.
    """

    def __init__(self, policy: _policy.Policy, host: str = "0.0.0.0", port: int = 8000):
        self._policy = policy
        self._host = host
        self._port = port
        self._server = None
        self._thread = None

    def start(self):
        logging.info("Starting in-process policy server on %s:%d ...", self._host, self._port)
        self._server = websocket_policy_server.WebsocketPolicyServer(
            policy=self._policy,
            host=self._host,
            port=self._port,
            metadata=self._policy.metadata,
        )
        self._thread = threading.Thread(target=self._server.serve_forever, daemon=True)
        self._thread.start()
        # Small wait so the socket is ready
        time.sleep(0.5)
        logging.info("Policy server is up.")

    def stop(self):
        try:
            if self._server:
                logging.info("Shutting down policy server...")
                self._server.shutdown()
            if self._thread:
                self._thread.join(timeout=2.0)
        except Exception as e:
            logging.warning("Error shutting down policy server: %s", e)


# ---------------------------
# RLDS helpers
# ---------------------------

def _ensure_uint8_rgb(arr: np.ndarray, *, bgr_to_rgb: bool = False) -> np.ndarray:
    arr = np.asarray(arr)
    if arr.ndim >= 3 and arr.shape[-1] > 3:
        arr = arr[..., :3]  # drop alpha
    if arr.dtype != np.uint8:
        if arr.max() <= 1.0:
            arr = (arr * 255.0).clip(0, 255).astype(np.uint8)
        else:
            arr = arr.clip(0, 255).astype(np.uint8)
    if bgr_to_rgb:
        arr = arr[..., ::-1]
    return arr


def _extract_step_obs(step: Dict[str, Any]) -> Dict[str, Any]:
    # Many RLDS datasets put data under step['observation'].
    return step.get("observation", step)


def _pick_images_and_state(
    step: Dict[str, Any],
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, Dict[str, Any]]:
    """
    Try to pull 'left' (external) and 'wrist' images and (joint, gripper).
    Return (left_img_uint8, wrist_img_uint8, joint_vec, gripper_vec, debug_info)
    """
    obs = _extract_step_obs(step)
    debug = {"available_keys": list(obs.keys())}

    # Find image keys heuristically
    img_keys = [k for k, v in obs.items() if isinstance(v, np.ndarray) and v.ndim >= 3]
    k_left = None
    k_wrist = None
    for k in img_keys:
        lname = k.lower()
        if k_left is None and any(s in lname for s in ("left", "front", "exterior")):
            k_left = k
        if k_wrist is None and any(s in lname for s in ("wrist", "hand", "gripper")):
            k_wrist = k

    # Fallbacks if wrist missing
    if k_left is None and img_keys:
        k_left = img_keys[0]
    if k_wrist is None:
        k_wrist = k_left

    if k_left is None:
        raise KeyError(f"No image keys found in RLDS step. Keys: {list(obs.keys())}")

    left = _ensure_uint8_rgb(obs[k_left], bgr_to_rgb=False)
    wrist = _ensure_uint8_rgb(obs[k_wrist], bgr_to_rgb=False)

    # Proprio
    joint = None
    for key in ("joint_position", "joint_positions", "proprio", "state"):
        if key in obs and isinstance(obs[key], np.ndarray):
            j = obs[key].astype(np.float32).reshape(-1)
            if j.size > 0:
                joint = j
                break
    if joint is None:
        joint = np.zeros((7,), dtype=np.float32)  # safe default

    gripper = None
    for key in ("gripper_position", "gripper", "gripper_open"):
        if key in obs and isinstance(obs[key], np.ndarray):
            g = obs[key].astype(np.float32).reshape(-1)[:1]
            if g.size == 1:
                gripper = g
                break
    if gripper is None:
        gripper = np.zeros((1,), dtype=np.float32)

    debug["picked_keys"] = {"left": k_left, "wrist": k_wrist}
    return left, wrist, joint, gripper, debug


def _iter_rlds_steps(ds, max_steps: int):
    """
    Yield up to max_steps steps across episodes from a TFDS RLDS dataset.
    """
    count = 0
    for episode in tfds.as_numpy(ds):
        steps = episode["steps"]
        T = min(len(v) for v in steps.values())
        for t in range(T):
            step = {k: v[t] for k, v in steps.items()}
            yield episode, step
            count += 1
            if count >= max_steps:
                return


# ---------------------------
# Main experiment
# ---------------------------

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_dir", type=str, required=True, help="TFDS data dir where RLDS resides")
    ap.add_argument("--dataset_name", type=str, required=True, help="TFDS dataset name (RLDS export)")
    ap.add_argument("--split", type=str, default="train")
    ap.add_argument("--max_steps", type=int, default=200, help="Number of RLDS steps to probe")
    ap.add_argument("--port", type=int, default=8000, help="Policy server port")
    ap.add_argument("--host", type=str, default="127.0.0.1", help="Client connects to this host")
    ap.add_argument("--prompt", type=str, default="place the red block on the shelf")
    ap.add_argument("--checkpoint_config", type=str, default=None, help="e.g., pi05_droid")
    ap.add_argument("--checkpoint_dir", type=str, default=None, help="path to checkpoint dir")
    ap.add_argument("--save_npz", action="store_true", help="Save raw actions to npz")
    ap.add_argument("--out_prefix", type=str, default="pi0_rlds_eval")
    args = ap.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s: %(message)s")

    # 1) Create policy and start server in-process
    policy = create_droid_policy(args.checkpoint_config, args.checkpoint_dir, default_prompt=None)
    server = InProcessPolicyServer(policy, host="0.0.0.0", port=args.port)
    server.start()
    atexit.register(server.stop)

    # 2) Client connects to local server
    client = websocket_client_policy.WebsocketClientPolicy(args.host, args.port)

    # 3) Load RLDS dataset
    logging.info("Loading RLDS dataset %s (split=%s) from %s", args.dataset_name, args.split, args.data_dir)
    ds = tfds.load(args.dataset_name, data_dir=args.data_dir, split=args.split, shuffle_files=False)

    # 4) Iterate steps and collect predictions
    rows: List[Dict[str, Any]] = []
    raw_actions: List[np.ndarray] = []

    start = time.time()
    for idx, (_, step) in enumerate(_iter_rlds_steps(ds, args.max_steps)):
        try:
            left, wrist, joint, gripper, dbg = _pick_images_and_state(step)

            # Resize/pad like your robot client
            left_224 = image_tools.resize_with_pad(left, 224, 224)
            wrist_224 = image_tools.resize_with_pad(wrist, 224, 224)

            request = {
                "observation/exterior_image_1_left": left_224,
                "observation/wrist_image_left": wrist_224,
                "observation/joint_position": joint,
                "observation/gripper_position": gripper,
                "prompt": args.prompt,
            }

            resp = client.infer(request)  # -> {"actions": (10,8)}
            actions = resp["actions"]  # np.ndarray

            # Summaries for quick inspection
            mean_vec = actions.mean(axis=0)
            std_vec = actions.std(axis=0)
            rows.append({
                "step_index": idx,
                "picked_left_key": dbg["picked_keys"]["left"],
                "picked_wrist_key": dbg["picked_keys"]["wrist"],
                "actions_shape": str(actions.shape),
                **{f"mean_{i}": float(m) for i, m in enumerate(mean_vec)},
                **{f"std_{i}": float(s) for i, s in enumerate(std_vec)},
            })
            raw_actions.append(actions)

            if (idx + 1) % 20 == 0:
                logging.info("Processed %d/%d steps...", idx + 1, args.max_steps)

        except Exception as e:
            logging.warning("Step %d failed: %s", idx, e)
            continue

    elapsed = time.time() - start
    logging.info("Done. Processed %d steps in %.1fs (%.2f steps/s).",
                 len(rows), elapsed, len(rows) / max(elapsed, 1e-6))

    # 5) Save outputs
    df = pd.DataFrame(rows)
    csv_path = f"{args.out_prefix}_summary.csv"
    df.to_csv(csv_path, index=False)
    logging.info("Wrote %s", csv_path)

    if args.save_npz and raw_actions:
        # Stack to [N, 10, 8] (if chunk size differs, will fallback to object array)
        try:
            arr = np.stack(raw_actions, axis=0)
        except Exception:
            arr = np.array(raw_actions, dtype=object)
        npz_path = f"{args.out_prefix}_actions.npz"
        np.savez_compressed(npz_path, actions=arr)
        logging.info("Wrote %s", npz_path)

    # Clean shutdown
    server.stop()


if __name__ == "__main__":
    main()