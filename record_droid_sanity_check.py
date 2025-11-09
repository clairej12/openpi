#!/usr/bin/env python3
import os
os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"

import argparse, atexit, logging, threading, time, os, re, glob
from typing import Dict, Tuple, Any, List, Optional
import numpy as np
import pandas as pd
from PIL import Image

# --- OpenPI (server + client) ---
from openpi.policies import policy as _policy
from openpi.policies import policy_config as _policy_config
from openpi.training import config as _config
from openpi.serving import websocket_policy_server
from openpi_client import image_tools
from openpi_client import websocket_client_policy

# try to import imageio for lerobot video reading
try:
    import imageio.v2 as imageio
except Exception:
    imageio = None

tf = None
tfds = None
# =========================================================
# shared helpers (same as your RLDS version)
# =========================================================

def robust_infer(client, request, host, port, max_retries=3, backoff=2.0):
    """Call client.infer with reconnection + backoff. Returns (resp, client)."""
    last_err = None
    for attempt in range(max_retries):
        try:
            resp = client.infer(request)
            return resp, client
        except Exception as e:
            last_err = e
            logging.warning("infer attempt %d failed: %s", attempt + 1, e)
            try:
                client.close()
            except Exception:
                pass
            time.sleep(backoff ** attempt)
            client = websocket_client_policy.WebsocketClientPolicy(host, port)
    raise RuntimeError(f"infer failed after retries: {last_err}")


def create_droid_policy(checkpoint_config: Optional[str], checkpoint_dir: Optional[str], default_prompt: Optional[str]) -> _policy.Policy:
    if checkpoint_config and checkpoint_dir:
        logging.info("Loading trained policy: config=%s dir=%s", checkpoint_config, checkpoint_dir)
        return _policy_config.create_trained_policy(_config.get_config(checkpoint_config), checkpoint_dir, default_prompt=default_prompt)
    logging.info("Loading DEFAULT DROID policy (pi05_droid).")
    return _policy_config.create_trained_policy(
        _config.get_config("pi05_droid"),
        "/media/volume/models_and_data/openpi_models/openpi-assets/checkpoints/pi05_droid",
        default_prompt=default_prompt,
    )


class InProcessPolicyServer:
    def __init__(self, policy: _policy.Policy, host: str = "0.0.0.0", port: int = 8000):
        self._policy = policy
        self._host = host
        self._port = port
        self._server = None
        self._thread = None

    def start(self):
        logging.info("Starting in-process policy server on %s:%d ...", self._host, self._port)
        self._server = websocket_policy_server.WebsocketPolicyServer(
            policy=self._policy, host=self._host, port=self._port, metadata=self._policy.metadata
        )
        self._thread = threading.Thread(target=self._server.serve_forever, daemon=True)
        self._thread.start()
        time.sleep(0.5)
        logging.info("Policy server is up.")

    def stop(self):
        try:
            if self._server:
                logging.info("Stopping policy server...")
            if self._thread:
                self._thread.join(timeout=2.0)
        except Exception as e:
            logging.warning("Error shutting down policy server: %s", e)


def _ensure_uint8_rgb(arr: np.ndarray, *, bgr_to_rgb: bool = False) -> np.ndarray:
    arr = np.asarray(arr)
    if arr.ndim >= 3 and arr.shape[-1] > 3:
        arr = arr[..., :3]
    if arr.dtype != np.uint8:
        if arr.max() <= 1.0:
            arr = (arr * 255.0).clip(0, 255).astype(np.uint8)
        else:
            arr = arr.clip(0, 255).astype(np.uint8)
    if bgr_to_rgb:
        arr = arr[..., ::-1]
    return arr


def _stack_or_object(arr_list: List[np.ndarray]) -> np.ndarray:
    if not arr_list:
        return np.array([], dtype=object)
    try:
        return np.stack(arr_list, axis=0)
    except Exception:
        return np.array(arr_list, dtype=object)


def _flush_episode_checkpoint(out_dir: str, episode_id: str, rows: List[Dict[str, Any]], actions: List[np.ndarray]):
    ep_root = os.path.join(out_dir, "episodes", episode_id)
    os.makedirs(ep_root, exist_ok=True)
    if rows:
        pd.DataFrame(rows).to_csv(os.path.join(ep_root, "summary.csv"), index=False)
    if actions:
        arr = _stack_or_object(actions)
        np.savez_compressed(os.path.join(ep_root, "actions.npz"), actions=arr)
    logging.info("Checkpointed episode %s (%d states).", episode_id, len(rows))


def _flush_shard_checkpoint(out_dir: str, shard_idx: int, rows: List[Dict[str, Any]], actions: List[np.ndarray], prefix: str = "shard"):
    shards_dir = os.path.join(out_dir, "shards")
    os.makedirs(shards_dir, exist_ok=True)
    if rows:
        pd.DataFrame(rows).to_csv(os.path.join(shards_dir, f"{prefix}_{shard_idx:05d}_summary.csv"), index=False)
    if actions:
        arr = _stack_or_object(actions)
        np.savez_compressed(os.path.join(shards_dir, f"{prefix}_{shard_idx:05d}_actions.npz"), actions=arr)
    logging.info("Wrote shard %d with %d steps.", shard_idx, len(rows))


# =========================================================
# RLDS / TFDS iterator (your previous logic)
# =========================================================
def _iter_rlds_steps(episode):
    """Yield (t_in_ep, step_np) from an RLDS-style episode from TFDS."""
    steps_obj = episode["steps"]
    try:
        for t, step in enumerate(tfds.as_numpy(steps_obj)):
            yield t, step
        return
    except Exception:
        pass

    # fallback for dict-of-arrays
    try:
        ep_np = tfds.as_numpy(episode)
        steps_np = ep_np["steps"]
        if isinstance(steps_np, dict):
            T = min(len(v) for v in steps_np.values())
            for t in range(T):
                yield t, {k: v[t] for k, v in steps_np.items()}
            return
    except Exception:
        pass

    raise TypeError(f"Unsupported 'steps' type: {type(episode['steps'])}")


# =========================================================
# LeRobot iterator
# =========================================================
def _discover_lerobot_parquets(data_root: str):
    """Yield (chunk_idx, file_idx, parquet_path) for all available parquet files."""
    pattern = os.path.join(data_root, "data", "chunk-*", "file-*.parquet")
    for path in sorted(glob.glob(pattern)):
        # .../chunk-000/file-001.parquet
        m_chunk = re.search(r"chunk-(\d+)", path)
        m_file = re.search(r"file-(\d+)\.parquet$", path)
        if not m_chunk or not m_file:
            continue
        chunk_idx = int(m_chunk.group(1))
        file_idx = int(m_file.group(1))
        yield chunk_idx, file_idx, path


def _load_lerobot_frame_from_video(video_path: str, frame_idx: int):
    """Return an RGB uint8 image for given frame_idx or None if missing/unreadable."""
    if imageio is None:
        return None
    if not os.path.exists(video_path):
        return None
    try:
        # open and read specific frame
        reader = imageio.get_reader(video_path)
        img = reader.get_data(frame_idx)
        reader.close()
        return _ensure_uint8_rgb(img)
    except Exception as e:
        logging.warning("Failed to read frame %d from %s: %s", frame_idx, video_path, e)
        return None


def _iter_lerobot_steps(data_root: str):
    """
    Yield dicts that look like RLDS steps from the LeRobot layout:
      {
        "episode_id": str,
        "episode_index": int,
        "t_in_episode": int,
        "observation": {
            ...images...,
            ...state...
        },
        "task": ... (if present)
      }
    We assume each parquet row corresponds to one frame, and matching mp4 files
    live under videos/<video_key>/chunk-XXX/file-XXX.mp4.
    """
    # possible video keys from the metadata you pasted
    video_keys = [
        "observation.images.exterior_1_left",
        "observation.images.wrist_left",
        "observation.images.exterior_2_left",
    ]

    for chunk_idx, file_idx, parquet_path in _discover_lerobot_parquets(data_root):
        df = pd.read_parquet(parquet_path)
        # per this dataset, each row is 1 frame
        for row_idx, row in df.iterrows():
            # episode_id/index in metadata
            episode_index = int(row.get("episode_index", 0))
            frame_index = int(row.get("frame_index", row_idx))
            # we mimic your previous episode_id style
            episode_id = f"ep{episode_index:06d}"

            # build observation dict
            obs = {}

            # add state-like fields if present
            # the dataset actually gives several variants; we'll prefer the 8-dim one if present
            if "observation.state" in df.columns:
                state = np.array(row["observation.state"], dtype=np.float32).reshape(-1)
                obs["state"] = state
                # split into joint + gripper if we can
                if state.size >= 7:
                    obs["joint_position"] = state[:7].astype(np.float32)
                if state.size >= 8:
                    obs["gripper_position"] = state[7:8].astype(np.float32)
            else:
                # fallback: separate columns
                if "observation.state.joint_position" in df.columns:
                    obs["joint_position"] = np.array(row["observation.state.joint_position"], dtype=np.float32).reshape(-1)
                if "observation.state.gripper_position" in df.columns:
                    obs["gripper_position"] = np.array(row["observation.state.gripper_position"], dtype=np.float32).reshape(-1)

            # language/instruction fields
            if "language_instruction" in df.columns:
                obs["language_instruction"] = row["language_instruction"]
            if "language_instruction_2" in df.columns:
                obs["language_instruction_2"] = row["language_instruction_2"]
            if "language_instruction_3" in df.columns:
                obs["language_instruction_3"] = row["language_instruction_3"]

            # now try to load images from videos
            at_least_one_image = False
            for vk in video_keys:
                video_path = os.path.join(
                    data_root,
                    "videos",
                    vk,
                    f"chunk-{chunk_idx:03d}",
                    f"file-{file_idx:03d}.mp4",
                )
                img = _load_lerobot_frame_from_video(video_path, row_idx)
                if img is not None:
                    # shorten key names to match your image picking code expectations
                    short_k = vk.split(".")[-1]  # e.g. "exterior_1_left"
                    obs[short_k] = img
                    at_least_one_image = True

            if not at_least_one_image:
                # user said: "we should skip those that appear in the metadata but don't have video"
                continue

            # yield in a shape similar to RLDS step
            step = {
                "episode_id": episode_id,
                "episode_index": episode_index,
                "t_in_episode": frame_index,   # it's okay if it's per-ep frame index
                "observation": obs,
            }
            yield episode_id, episode_index, frame_index, step


# =========================================================
# main
# =========================================================

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_dir", type=str, required=True, help="Root dir for dataset (TFDS data_dir or LeRobot root)")
    ap.add_argument("--dataset_name", type=str, default=None, help="TFDS dataset name (for RLDS mode)")
    ap.add_argument("--split", type=str, default="train")

    ap.add_argument("--format", type=str, default="auto", choices=["auto", "rlds", "lerobot"],
                    help="Dataset layout. auto=try rlds if dataset_name is set, else lerobot")

    ap.add_argument("--max_episodes", type=int, default=10)
    ap.add_argument("--port", type=int, default=8000)
    ap.add_argument("--host", type=str, default="127.0.0.1")
    ap.add_argument("--prompt", type=str, default="place the red block on the shelf")
    ap.add_argument("--checkpoint_config", type=str, default=None)
    ap.add_argument("--checkpoint_dir", type=str, default=None)
    ap.add_argument("--save_npz", action="store_true")
    ap.add_argument("--out_dir", type=str, default="droid_sanity")
    ap.add_argument("--samples_per_state", type=int, default=1)

    ap.add_argument("--checkpoint_by_episode", action="store_true")
    ap.add_argument("--checkpoint_every_steps", type=int, default=0)
    ap.add_argument("--shard_prefix", type=str, default="shard")

    ap.add_argument("--overwrite_images", action="store_true")
    ap.add_argument("--image_dir", type=str, default="droid_images")

    args = ap.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s: %(message)s")

    # 1) start policy server
    policy = create_droid_policy(args.checkpoint_config, args.checkpoint_dir, default_prompt=None)
    server = InProcessPolicyServer(policy, host="0.0.0.0", port=args.port)
    server.start()
    atexit.register(server.stop)

    # 2) client (we lazily create in lerobot/RLDS loop)
    client = None

    # 3) figure out format
    if args.format == "auto":
        if args.dataset_name:
            data_format = "rlds"
        else:
            data_format = "lerobot"
    else:
        data_format = args.format

    os.makedirs(args.out_dir, exist_ok=True)
    os.makedirs(args.image_dir, exist_ok=True)

    # --- load existing summary to build skip set ---
    existing_pairs = set()
    summary_csv_path = os.path.join(args.out_dir, "summary.csv")
    if os.path.exists(summary_csv_path):
        try:
            _old_df = pd.read_csv(summary_csv_path)
            if {"episode_id", "t_in_episode"} <= set(_old_df.columns):
                _old_df["episode_id"] = _old_df["episode_id"].astype(str)
                _old_df = _old_df.dropna(subset=["t_in_episode"])
                _old_df["t_in_episode"] = _old_df["t_in_episode"].astype(int)
                existing_pairs = {(eid, t) for eid, t in zip(_old_df["episode_id"], _old_df["t_in_episode"])}
                logging.info("Loaded %d existing (episode_id, t_in_episode) pairs for dedup.", len(existing_pairs))
        except Exception as e:
            logging.warning("Failed reading existing summary.csv: %s", e)

    # global buffers
    rows_global: List[Dict[str, Any]] = []
    raw_actions_global: List[np.ndarray] = []
    episodes_seen: Dict[str, Dict[str, Any]] = {}

    # per-episode
    rows_ep: List[Dict[str, Any]] = []
    actions_ep: List[np.ndarray] = []

    # shard
    shard_rows: List[Dict[str, Any]] = []
    shard_actions: List[np.ndarray] = []
    shard_idx = 0

    start = time.time()
    g_idx = -1
    episodes_processed = 0

    # =========================================================
    # iterate data
    # =========================================================
    if data_format == "rlds":
        import tensorflow_datasets as tfds
        import tensorflow as tf
        # load TFDS
        logging.info("Loading RLDS dataset %s from %s", args.dataset_name, args.data_dir)
        _ = tfds.builder(args.dataset_name, data_dir=args.data_dir).info
        ds = tfds.load(args.dataset_name, data_dir=args.data_dir, split=args.split, shuffle_files=False)

        for ep_idx, episode in enumerate(ds):
            if episodes_processed >= args.max_episodes:
                break

            # instruction accumulators
            instr_accum = {
                "language_instruction": set(),
                "language_instruction_2": set(),
                "language_instruction_3": set(),
            }
            rows_ep.clear()
            actions_ep.clear()

            episode_id = None
            task_name = None
            instruction_any = None

            for t_in_ep, step_np in _iter_rlds_steps(episode):
                g_idx += 1
                # infer meta
                meta = step_np  # already numpy-like
                # try to get episode id etc. (keep your old logic here if you want)
                # for RLDS we can just synthesize
                if episode_id is None:
                    episode_id = f"ep{ep_idx:06d}"

                pair = (episode_id, int(t_in_ep))
                if pair in existing_pairs:
                    logging.info("Skipping episode %s step %d (already in summary.csv).", episode_id, t_in_ep)
                    continue

                if client is None:
                    client = websocket_client_policy.WebsocketClientPolicy(args.host, args.port)

                # your old image/state picker should still work for rlds here
                # but I’ll re-use your logic quickly
                obs = step_np.get("observation", step_np)
                # find image keys
                img_keys = [k for k, v in obs.items() if isinstance(v, np.ndarray) and v.ndim >= 3]
                if not img_keys:
                    continue
                # pick first as left, second (or same) as wrist
                left_key = img_keys[0]
                wrist_key = img_keys[1] if len(img_keys) > 1 else left_key
                left = _ensure_uint8_rgb(obs[left_key])
                wrist = _ensure_uint8_rgb(obs[wrist_key])

                ep_dir = os.path.join(args.image_dir, episode_id)
                os.makedirs(ep_dir, exist_ok=True)
                left_path = os.path.join(ep_dir, f"state_{t_in_ep:06d}_left.jpg")
                wrist_path = os.path.join(ep_dir, f"state_{t_in_ep:06d}_wrist.jpg")

                if args.overwrite_images or not os.path.exists(left_path):
                    Image.fromarray(left).save(left_path, quality=90)
                if args.overwrite_images or not os.path.exists(wrist_path):
                    Image.fromarray(wrist).save(wrist_path, quality=90)

                # get joint/gripper
                joint = obs.get("joint_position", np.zeros((7,), dtype=np.float32)).astype(np.float32).reshape(-1)
                gripper = obs.get("gripper_position", np.zeros((1,), dtype=np.float32)).astype(np.float32).reshape(-1)[:1]

                # run policy
                left_224 = image_tools.resize_with_pad(left, 224, 224)
                wrist_224 = image_tools.resize_with_pad(wrist, 224, 224)
                request = {
                    "observation/exterior_image_1_left": left_224,
                    "observation/wrist_image_left": wrist_224,
                    "observation/joint_position": joint,
                    "observation/gripper_position": gripper,
                    "prompt": args.prompt,
                }

                K = max(1, args.samples_per_state)
                chunks = []
                for _ in range(K):
                    resp, client = robust_infer(client, request, args.host, args.port)
                    chunks.append(resp["actions"])
                try:
                    chunks_arr = np.stack(chunks, axis=0)
                except Exception:
                    chunks_arr = np.array(chunks, dtype=object)

                m_arr = np.asarray(chunks_arr)
                mean_vec = m_arr.mean(axis=0).mean(axis=0)
                std_vec = m_arr.std(axis=0).std(axis=0)

                row_dict = {
                    "global_step_index": g_idx,
                    "episode_index": ep_idx,
                    "episode_id": episode_id,
                    "task_name": None,
                    "instruction": None,
                    "t_in_episode": t_in_ep,
                    "picked_left_key": left_key,
                    "picked_wrist_key": wrist_key,
                    "left_img_path": os.path.relpath(left_path, args.out_dir),
                    "wrist_img_path": os.path.relpath(wrist_path, args.out_dir),
                    "samples_per_state": K,
                    "actions_shape": str(np.shape(chunks_arr)),
                    **{f"mean_{i}": float(m) for i, m in enumerate(mean_vec)},
                    **{f"std_{i}": float(s) for i, s in enumerate(std_vec)},
                }

                rows_global.append(row_dict)
                raw_actions_global.append(chunks_arr)
                rows_ep.append(row_dict)
                actions_ep.append(chunks_arr)

                if args.checkpoint_every_steps > 0:
                    shard_rows.append(row_dict)
                    shard_actions.append(chunks_arr)
                    if len(shard_rows) >= args.checkpoint_every_steps:
                        shard_idx += 1
                        _flush_shard_checkpoint(args.out_dir, shard_idx, shard_rows, shard_actions, prefix=args.shard_prefix)
                        shard_rows.clear()
                        shard_actions.clear()

            episodes_seen[episode_id] = {
                "episode_index": ep_idx,
                "episode_id": episode_id,
                "task_name": "",
                "instruction_any": "",
                "num_states_collected": len(rows_ep),
            }

            if args.checkpoint_by_episode:
                _flush_episode_checkpoint(args.out_dir, episode_id, rows_ep, actions_ep)

            episodes_processed += 1

    else:
        # =========================================
        # LeRobot mode
        # =========================================
        print("Processing LeRobot dataset from", args.data_dir)
        current_ep_id = None
        rows_ep = []
        actions_ep = []

        for (episode_id, episode_index, frame_index, step) in _iter_lerobot_steps(args.data_dir):
            # if we’ve moved to a new episode, flush the previous one (if requested)
            if current_ep_id is not None and episode_id != current_ep_id:
                print(f"Finished episode {current_ep_id} ({len(rows_ep)} states).")
                if args.checkpoint_by_episode:
                    _flush_episode_checkpoint(args.out_dir, current_ep_id, rows_ep, actions_ep)
                rows_ep = []
                actions_ep = []

            # stop condition: count distinct episodes
            if episode_id not in episodes_seen and len(episodes_seen) >= args.max_episodes:
                break

            current_ep_id = episode_id  # track

            g_idx += 1
            t_in_ep = frame_index
            pair = (episode_id, int(t_in_ep))
            if pair in existing_pairs:
                logging.info("Skipping episode %s step %d (already in summary.csv).", episode_id, t_in_ep)
                continue

            if client is None:
                client = websocket_client_policy.WebsocketClientPolicy(args.host, args.port)

            obs = step["observation"]
            # pick images: prefer exterior_1_left, then wrist_left, then exterior_2_left
            img_candidates = []
            for k in ["exterior_1_left", "wrist_left", "exterior_2_left"]:
                if k in obs and isinstance(obs[k], np.ndarray):
                    img_candidates.append(k)
            if not img_candidates:
                continue

            left_key = img_candidates[0]
            wrist_key = img_candidates[1] if len(img_candidates) > 1 else left_key
            left = _ensure_uint8_rgb(obs[left_key])
            wrist = _ensure_uint8_rgb(obs[wrist_key])

            joint = obs.get("joint_position", np.zeros((7,), dtype=np.float32)).astype(np.float32).reshape(-1)
            gripper = obs.get("gripper_position", np.zeros((1,), dtype=np.float32)).astype(np.float32).reshape(-1)[:1]

            # save images
            ep_dir = os.path.join(args.image_dir, episode_id)
            os.makedirs(ep_dir, exist_ok=True)
            left_path = os.path.join(ep_dir, f"state_{t_in_ep:06d}_left.jpg")
            wrist_path = os.path.join(ep_dir, f"state_{t_in_ep:06d}_wrist.jpg")

            if args.overwrite_images or not os.path.exists(left_path):
                Image.fromarray(left).save(left_path, quality=90)
            if args.overwrite_images or not os.path.exists(wrist_path):
                Image.fromarray(wrist).save(wrist_path, quality=90)

            # run policy
            left_224 = image_tools.resize_with_pad(left, 224, 224)
            wrist_224 = image_tools.resize_with_pad(wrist, 224, 224)
            request = {
                "observation/exterior_image_1_left": left_224,
                "observation/wrist_image_left": wrist_224,
                "observation/joint_position": joint,
                "observation/gripper_position": gripper,
                "prompt": args.prompt,
            }

            K = max(1, args.samples_per_state)
            chunks = []
            for _ in range(K):
                resp, client = robust_infer(client, request, args.host, args.port)
                chunks.append(resp["actions"])
            try:
                chunks_arr = np.stack(chunks, axis=0)
            except Exception:
                chunks_arr = np.array(chunks, dtype=object)

            m_arr = np.asarray(chunks_arr)
            mean_vec = m_arr.mean(axis=0).mean(axis=0)
            std_vec = m_arr.std(axis=0).std(axis=0)

            row_dict = {
                "global_step_index": g_idx,
                "episode_index": episode_index,
                "episode_id": episode_id,
                "task_name": "",
                "instruction": obs.get("language_instruction", ""),
                "t_in_episode": t_in_ep,
                "picked_left_key": left_key,
                "picked_wrist_key": wrist_key,
                "left_img_path": os.path.relpath(left_path, args.out_dir),
                "wrist_img_path": os.path.relpath(wrist_path, args.out_dir),
                "samples_per_state": K,
                "actions_shape": str(np.shape(chunks_arr)),
                **{f"mean_{i}": float(m) for i, m in enumerate(mean_vec)},
                **{f"std_{i}": float(s) for i, s in enumerate(std_vec)},
            }

            # global
            rows_global.append(row_dict)
            raw_actions_global.append(chunks_arr)

            # per-episode
            rows_ep.append(row_dict)
            actions_ep.append(chunks_arr)

            # track episode
            if episode_id not in episodes_seen:
                episodes_seen[episode_id] = {
                    "episode_index": episode_index,
                    "episode_id": episode_id,
                    "task_name": "",
                    "instruction_any": obs.get("language_instruction", ""),
                    "num_states_collected": 0,
                }
            episodes_seen[episode_id]["num_states_collected"] += 1

            # shards
            if args.checkpoint_every_steps > 0:
                shard_rows.append(row_dict)
                shard_actions.append(chunks_arr)
                if len(shard_rows) >= args.checkpoint_every_steps:
                    shard_idx += 1
                    _flush_shard_checkpoint(args.out_dir, shard_idx, shard_rows, shard_actions, prefix=args.shard_prefix)
                    shard_rows.clear()
                    shard_actions.clear()

        # after the loop, flush the last episode we were working on
        if args.checkpoint_by_episode and current_ep_id is not None and rows_ep:
            _flush_episode_checkpoint(args.out_dir, current_ep_id, rows_ep, actions_ep)
    
    # =========================================================
    # final flush / save
    # =========================================================
    if args.checkpoint_every_steps > 0 and (shard_rows or shard_actions):
        shard_idx += 1
        _flush_shard_checkpoint(args.out_dir, shard_idx, shard_rows, shard_actions, prefix=args.shard_prefix)

    elapsed = time.time() - start
    logging.info("Done. Processed %d episodes (%d states) in %.1fs.",
                 len(episodes_seen), len(rows_global), elapsed)

    # write global summary.csv (append + dedup)
    df = pd.DataFrame(rows_global)
    csv_path = os.path.join(args.out_dir, "summary.csv")
    if os.path.exists(csv_path):
        try:
            old_df = pd.read_csv(csv_path)
        except Exception as e:
            logging.warning("Could not read existing summary.csv (%s); creating new.", e)
            old_df = pd.DataFrame()
        combined = pd.concat([old_df, df], ignore_index=True)
        if {"episode_id", "t_in_episode"} <= set(combined.columns):
            combined["episode_id"] = combined["episode_id"].astype(str)
            combined = combined.dropna(subset=["t_in_episode"])
            combined["t_in_episode"] = combined["t_in_episode"].astype(int)
            combined = combined.drop_duplicates(subset=["episode_id", "t_in_episode"], keep="first")
        combined.to_csv(csv_path, index=False)
        logging.info("Appended to %s (now %d rows).", csv_path, len(combined))
    else:
        df.to_csv(csv_path, index=False)
        logging.info("Wrote %s", csv_path)

    # write episodes.csv
    ep_df = pd.DataFrame(sorted(episodes_seen.values(), key=lambda d: d["episode_index"]))
    ep_df.to_csv(os.path.join(args.out_dir, "episodes.csv"), index=False)
    logging.info("Wrote episodes.csv")

    # write actions
    if args.save_npz and raw_actions_global:
        new_arr = _stack_or_object(raw_actions_global)
        npz_path = os.path.join(args.out_dir, "actions.npz")
        if os.path.exists(npz_path):
            try:
                old = np.load(npz_path, allow_pickle=True)
                old_arr = old.get("actions")
            except Exception as e:
                logging.warning("Could not read existing actions.npz (%s); writing new.", e)
                old_arr = None

            if old_arr is None or old_arr.size == 0:
                combined_arr = new_arr
            elif new_arr.size == 0:
                combined_arr = old_arr
            else:
                try:
                    combined_arr = np.concatenate([old_arr, new_arr], axis=0)
                except Exception:
                    combined_arr = _stack_or_object(list(old_arr) + list(new_arr))
            np.savez_compressed(npz_path, actions=combined_arr)
            logging.info("Appended to %s (new length ~ %d).", npz_path, len(combined_arr))
        else:
            np.savez_compressed(npz_path, actions=new_arr)
            logging.info("Wrote %s", npz_path)

    server.stop()


if __name__ == "__main__":
    main()
# #!/usr/bin/env python3
# import argparse, atexit, logging, threading, time, os, socket
# from typing import Dict, Tuple, Any, List, Optional
# import numpy as np
# import pandas as pd
# from PIL import Image
# import tensorflow_datasets as tfds
# import tensorflow as tf
# import collections.abc as cabc

# # --- OpenPI (server + client) ---
# from openpi.policies import policy as _policy
# from openpi.policies import policy_config as _policy_config
# from openpi.training import config as _config
# from openpi.serving import websocket_policy_server
# from openpi_client import image_tools
# from openpi_client import websocket_client_policy

# # ---------------------------
# # Policy bootstrap (DROID)
# # ---------------------------

# def robust_infer(client, request, host, port, max_retries=3, backoff=2.0):
#     last_err = None
#     for attempt in range(max_retries):
#         try:
#             resp = client.infer(request)
#             return resp, client
#         except Exception as e:
#             last_err = e
#             logging.warning("infer attempt %d failed: %s", attempt+1, e)
#             try: client.close()
#             except Exception: pass
#             time.sleep((backoff ** attempt))
#             client = websocket_client_policy.WebsocketClientPolicy(host, port)
#     raise RuntimeError(f"infer failed after retries: {last_err}")

# def create_droid_policy(checkpoint_config: Optional[str], checkpoint_dir: Optional[str], default_prompt: Optional[str]) -> _policy.Policy:
#     if checkpoint_config and checkpoint_dir:
#         logging.info("Loading trained policy: config=%s dir=%s", checkpoint_config, checkpoint_dir)
#         return _policy_config.create_trained_policy(_config.get_config(checkpoint_config), checkpoint_dir, default_prompt=default_prompt)
#     logging.info("Loading DEFAULT DROID policy (pi05_droid).")
#     return _policy_config.create_trained_policy(
#         _config.get_config("pi05_droid"),
#         "/media/volume/models_and_data/openpi_models/openpi-assets/checkpoints/pi05_droid",
#         default_prompt=default_prompt,
#     )

# class InProcessPolicyServer:
#     def __init__(self, policy: _policy.Policy, host: str = "0.0.0.0", port: int = 8000):
#         self._policy = policy; self._host = host; self._port = port
#         self._server = None; self._thread = None
#     def start(self):
#         logging.info("Starting in-process policy server on %s:%d ...", self._host, self._port)
#         self._server = websocket_policy_server.WebsocketPolicyServer(
#             policy=self._policy, host=self._host, port=self._port, metadata=self._policy.metadata
#         )
#         self._thread = threading.Thread(target=self._server.serve_forever, daemon=True); self._thread.start()
#         time.sleep(0.5); logging.info("Policy server is up.")
#     def stop(self):
#         try:
#             if self._server: logging.info("Stopping policy server...")
#             if self._thread: self._thread.join(timeout=2.0)
#         except Exception as e:
#             logging.warning("Error shutting down policy server: %s", e)

# # ---------------------------
# # RLDS helpers
# # ---------------------------
# def _ensure_uint8_rgb(arr: np.ndarray, *, bgr_to_rgb: bool = False) -> np.ndarray:
#     arr = np.asarray(arr)
#     if arr.ndim >= 3 and arr.shape[-1] > 3: arr = arr[..., :3]
#     if arr.dtype != np.uint8:
#         if arr.max() <= 1.0: arr = (arr * 255.0).clip(0, 255).astype(np.uint8)
#         else: arr = arr.clip(0, 255).astype(np.uint8)
#     if bgr_to_rgb: arr = arr[..., ::-1]
#     return arr

# def _extract_step_obs(step: Dict[str, Any]) -> Dict[str, Any]:
#     return step.get("observation", step)

# def _pick_images_and_state(step: Dict[str, Any]) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, Dict[str, Any]]:
#     obs = _extract_step_obs(step); debug = {"available_keys": list(obs.keys())}
#     img_keys = [k for k, v in obs.items() if isinstance(v, np.ndarray) and v.ndim >= 3]
#     k_left = k_wrist = None
#     for k in img_keys:
#         lname = k.lower()
#         if k_left is None and any(s in lname for s in ("left", "front", "exterior")): k_left = k
#         if k_wrist is None and any(s in lname for s in ("wrist", "hand", "gripper")): k_wrist = k
#     if k_left is None and img_keys: k_left = img_keys[0]
#     if k_wrist is None: k_wrist = k_left
#     if k_left is None: raise KeyError(f"No image keys found in RLDS step. Keys: {list(obs.keys())}")
#     left = _ensure_uint8_rgb(obs[k_left], bgr_to_rgb=False)
#     wrist = _ensure_uint8_rgb(obs[k_wrist], bgr_to_rgb=False)

#     joint = None
#     for key in ("joint_position", "joint_positions", "proprio", "state"):
#         if key in obs and isinstance(obs[key], np.ndarray):
#             j = obs[key].astype(np.float32).reshape(-1)
#             if j.size > 0: joint = j; break
#     if joint is None: joint = np.zeros((7,), dtype=np.float32)

#     gripper = None
#     for key in ("gripper_position", "gripper", "gripper_open"):
#         if key in obs and isinstance(obs[key], np.ndarray):
#             g = obs[key].astype(np.float32).reshape(-1)[:1]
#             if g.size == 1: gripper = g; break
#     if gripper is None: gripper = np.zeros((1,), dtype=np.float32)

#     debug["picked_keys"] = {"left": k_left, "wrist": k_wrist}
#     return left, wrist, joint, gripper, debug

# def _to_numpy_tree(x):
#     return tf.nest.map_structure(lambda t: t.numpy() if hasattr(t, "numpy") else t, x)

# def _infer_episode_meta(episode: Optional[Dict[str, Any]], step_np: Dict[str, Any]) -> Dict[str, Any]:
#     meta = {"episode_id": None, "task_name": None, "instruction": None}
#     if episode is not None:
#         for k in ("episode_id", "id", "episode_uid"):
#             if k in episode: meta["episode_id"] = str(episode[k]); break
#         for k in ("task", "task_name", "language_instruction", "instruction", "goal", "description"):
#             if k in episode:
#                 val = episode[k]
#                 meta["task_name"] = str(val) if meta["task_name"] is None else meta["task_name"]
#                 meta["instruction"] = str(val) if meta["instruction"] is None else meta["instruction"]
#         if "episode_metadata" in episode and isinstance(episode["episode_metadata"], dict):
#             em = episode["episode_metadata"]
#             for k in ("episode_id","id","episode_uid"):
#                 if meta["episode_id"] is None and k in em: meta["episode_id"] = str(em[k])
#             for k in ("task","task_name","language_instruction","instruction","goal","description"):
#                 if k in em:
#                     if meta["task_name"] is None: meta["task_name"] = str(em[k])
#                     if meta["instruction"] is None: meta["instruction"] = str(em[k])
#     obs = step_np.get("observation", step_np)
#     for k in ("episode_id","episode_uid","id"):
#         if meta["episode_id"] is None and k in step_np: meta["episode_id"] = str(step_np[k])
#         if meta["episode_id"] is None and k in obs: meta["episode_id"] = str(obs[k])
#     for k in ("task","task_name","language_instruction","instruction","goal","description"):
#         if meta["task_name"] is None and k in step_np: meta["task_name"] = str(step_np[k])
#         if meta["task_name"] is None and k in obs: meta["task_name"] = str(obs[k])
#         if meta["instruction"] is None and k in step_np: meta["instruction"] = str(step_np[k])
#         if meta["instruction"] is None and k in obs: meta["instruction"] = str(obs[k])
#     if meta["task_name"] and meta["task_name"].strip() == "": meta["task_name"] = None
#     if meta["instruction"] and meta["instruction"].strip() == "": meta["instruction"] = None
#     return meta

# def _decode_bytes(s):
#     if isinstance(s, (bytes, bytearray)):
#         try:
#             return s.decode("utf-8", errors="ignore")
#         except Exception:
#             return str(s)
#     return s

# def _collect_episode_instruction_fields(step_np: Dict[str, Any], accum: Dict[str, set]):
#     """Accumulate instruction variants from a numpy-ified step dict."""
#     obs = step_np.get("observation", step_np)
#     for key in ("language_instruction", "language_instruction_2", "language_instruction_3"):
#         if key in step_np:
#             v = _decode_bytes(step_np[key])
#             if isinstance(v, str) and v.strip(): accum[key].add(v.strip())
#         if key in obs:
#             v = _decode_bytes(obs[key])
#             if isinstance(v, str) and v.strip(): accum[key].add(v.strip())

# # ---------------------------
# # Checkpoint helpers
# # ---------------------------
# def _stack_or_object(arr_list: List[np.ndarray]) -> np.ndarray:
#     if not arr_list: return np.array([], dtype=object)
#     try:
#         return np.stack(arr_list, axis=0)
#     except Exception:
#         return np.array(arr_list, dtype=object)

# def _flush_episode_checkpoint(out_dir: str, episode_id: str, rows: List[Dict[str, Any]], actions: List[np.ndarray]):
#     ep_root = os.path.join(out_dir, "episodes", episode_id)
#     os.makedirs(ep_root, exist_ok=True)
#     if rows:
#         ep_df = pd.DataFrame(rows)
#         ep_df.to_csv(os.path.join(ep_root, "summary.csv"), index=False)
#     if actions:
#         arr = _stack_or_object(actions)
#         np.savez_compressed(os.path.join(ep_root, "actions.npz"), actions=arr)
#     logging.info("Checkpointed episode %s (%d states).", episode_id, len(rows))

# def _flush_shard_checkpoint(out_dir: str, shard_idx: int, rows: List[Dict[str, Any]], actions: List[np.ndarray], prefix: str="shard"):
#     shards_dir = os.path.join(out_dir, "shards"); os.makedirs(shards_dir, exist_ok=True)
#     if rows:
#         shard_df = pd.DataFrame(rows)
#         shard_df.to_csv(os.path.join(shards_dir, f"{prefix}_{shard_idx:05d}_summary.csv"), index=False)
#     if actions:
#         arr = _stack_or_object(actions)
#         np.savez_compressed(os.path.join(shards_dir, f"{prefix}_{shard_idx:05d}_actions.npz"), actions=arr)
#     logging.info("Wrote shard %d with %d steps.", shard_idx, len(rows))

# # ---------------------------
# # Main
# # ---------------------------
# def main():
#     ap = argparse.ArgumentParser()
#     ap.add_argument("--data_dir", type=str, required=True)
#     ap.add_argument("--dataset_name", type=str, required=True)
#     ap.add_argument("--split", type=str, default="train")
#     ap.add_argument("--max_episodes", type=int, default=10, help="Process up to this many episodes (instead of steps).")
#     ap.add_argument("--port", type=int, default=8000)
#     ap.add_argument("--host", type=str, default="127.0.0.1")
#     ap.add_argument("--prompt", type=str, default="place the red block on the shelf")
#     ap.add_argument("--checkpoint_config", type=str, default=None)
#     ap.add_argument("--checkpoint_dir", type=str, default=None)
#     ap.add_argument("--save_npz", action="store_true")
#     ap.add_argument("--out_dir", type=str, default="droid_sanity")
#     ap.add_argument("--samples_per_state", type=int, default=1)

#     # Checkpointing knobs
#     ap.add_argument("--checkpoint_by_episode", action="store_true",
#                     help="Write summary.csv and actions.npz at the end of each episode.")
#     ap.add_argument("--checkpoint_every_steps", type=int, default=0,
#                     help="If >0, write shard checkpoints every N processed steps.")
#     ap.add_argument("--shard_prefix", type=str, default="shard", help="Prefix for shard filenames.")

#     # Overwrite behavior for saved images
#     ap.add_argument("--overwrite_images", action="store_true",
#                     help="If set, overwrite existing per-state images. By default, steps with existing images are skipped.")
#     ap.add_argument("--image_dir", type=str, default="droid_images", help="Directory to save per-state images.")

#     args = ap.parse_args()
#     logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s: %(message)s")

#     # 1) Start policy server
#     policy = create_droid_policy(args.checkpoint_config, args.checkpoint_dir, default_prompt=None)
#     server = InProcessPolicyServer(policy, host="0.0.0.0", port=args.port)
#     server.start()
#     atexit.register(server.stop)

#     # 2) Client
#     client = None

#     # 3) Data
#     logging.info("Loading RLDS dataset %s (split=%s) from %s", args.dataset_name, args.split, args.data_dir)
#     _ = tfds.builder(args.dataset_name, data_dir=args.data_dir).info
#     ds = tfds.load(args.dataset_name, data_dir=args.data_dir, split=args.split, shuffle_files=False)

#     # 4) Iterate by EPISODE
#     rows_global: List[Dict[str, Any]] = []
#     raw_actions_global: List[np.ndarray] = []
#     episodes_seen: Dict[str, Dict[str, Any]] = {}

#     # Buffers for episode-level checkpointing
#     rows_ep: List[Dict[str, Any]] = []
#     actions_ep: List[np.ndarray] = []

#     # Buffers for shard checkpointing
#     shard_rows: List[Dict[str, Any]] = []
#     shard_actions: List[np.ndarray] = []
#     shard_idx = 0

#     os.makedirs(args.out_dir, exist_ok=True)
#     img_root = args.image_dir
#     os.makedirs(img_root, exist_ok=True)

#     # --- NEW: load existing summary.csv and build (episode_id, t_in_episode) skip set ---
#     existing_pairs = set()
#     summary_csv_path = os.path.join(args.out_dir, "summary.csv")
#     if os.path.exists(summary_csv_path):
#         try:
#             _old_df = pd.read_csv(summary_csv_path)
#             if {"episode_id", "t_in_episode"} <= set(_old_df.columns):
#                 # normalize types to be safe
#                 _old_df["episode_id"] = _old_df["episode_id"].astype(str)
#                 # coerce t_in_episode to int where possible; drop NaNs
#                 _old_df = _old_df.dropna(subset=["t_in_episode"])
#                 _old_df["t_in_episode"] = _old_df["t_in_episode"].astype(int)
#                 existing_pairs = {(eid, t) for eid, t in zip(_old_df["episode_id"], _old_df["t_in_episode"])}
#                 logging.info("Loaded %d existing (episode_id, t_in_episode) pairs for dedup.", len(existing_pairs))
#             else:
#                 logging.warning("Existing summary.csv lacks required columns; cannot dedup by pair.")
#         except Exception as e:
#             logging.warning("Failed reading existing summary.csv: %s", e)

#     start = time.time()
#     g_idx = -1
#     episodes_processed = 0

#     for ep_idx, episode in enumerate(ds):
#         print(f"Processing episode #{ep_idx}...")
#         if episodes_processed >= args.max_episodes:
#             break

#         # Initialize per-episode instruction accumulators
#         instr_accum = {
#             "language_instruction": set(),
#             "language_instruction_2": set(),
#             "language_instruction_3": set(),
#         }

#         # Try to infer basic meta from the first step we see
#         episode_id = None
#         task_name = None
#         instruction_any = None

#         # Steps may be a nested dataset or dict-of-arrays
#         def _iter_steps_numpy(episode):
#             """
#             Yield (t_in_ep, step_np) for a single episode.
#             Supports both:
#             - tf.data iterable episodes: episode["steps"] is a Dataset
#             - dict-of-arrays episodes (rare)
#             """
#             # Try iterable dataset first (RLDS standard)
#             steps_obj = episode["steps"]
#             try:
#                 # This makes each step a pure-numpy dict/tree
#                 for t, step in enumerate(tfds.as_numpy(steps_obj)):
#                     yield t, step
#                 return
#             except Exception:
#                 pass  # fall through to dict-of-arrays case

#             # Fallback: some datasets pack steps as dict of numpy arrays
#             try:
#                 ep_np = tfds.as_numpy(episode)
#                 steps_np = ep_np["steps"]
#                 if isinstance(steps_np, dict):
#                     T = min(len(v) for v in steps_np.values())
#                     for t in range(T):
#                         yield t, {k: v[t] for k, v in steps_np.items()}
#                     return
#             except Exception:
#                 pass

#             # If we get here, we don't know how to iterate this structure
#             raise TypeError(f"Unsupported 'steps' type: {type(episode['steps'])}")

#         rows_ep.clear(); actions_ep.clear()

#         ep_np = None 
#         for t_in_ep, step_np in _iter_steps_numpy(episode):
#             g_idx += 1
#             if g_idx % 100 == 0:
#                 print(f"  Global step #{g_idx} (episode #{ep_idx} step #{t_in_ep})...")
#             meta = _infer_episode_meta(ep_np, step_np)  # unchanged signature; it will use step_np

#             # Episode meta from this step
#             meta = _infer_episode_meta(ep_np, step_np)
#             episode_id = episode_id or (meta["episode_id"] or f"ep{ep_idx:06d}")
#             task_name = task_name or meta["task_name"]
#             instruction_any = instruction_any or meta["instruction"]

#             # --- NEW: skip if (episode_id, t_in_ep) already processed ---
#             # Normalize types for lookup
#             _pair = (str(episode_id or ""), int(t_in_ep))
#             if _pair in existing_pairs:
#                 print(f"  Skipping episode {episode_id} step {t_in_ep} (already in summary.csv).")
#                 logging.info("Skipping episode %s step %d (already in summary.csv).", episode_id, t_in_ep)
#                 continue

#             if client is None:
#                 client = websocket_client_policy.WebsocketClientPolicy(args.host, args.port)
#             # Accumulate instruction fields found in this step
#             _collect_episode_instruction_fields(step_np, instr_accum)

#             try:
#                 # Extract obs
#                 left, wrist, joint, gripper, dbg = _pick_images_and_state(step_np)

#                 # Paths for per-state images
#                 ep_dir = os.path.join(img_root, episode_id)
#                 os.makedirs(ep_dir, exist_ok=True)
#                 left_path = os.path.join(ep_dir, f"state_{t_in_ep:06d}_left.jpg")
#                 wrist_path = os.path.join(ep_dir, f"state_{t_in_ep:06d}_wrist.jpg")

#                 left_exists = os.path.exists(left_path)
#                 wrist_exists = os.path.exists(wrist_path)

#                 # Skip re-writing
#                 if args.overwrite_images or not left_exists:
#                     Image.fromarray(left).save(left_path, quality=90)
#                 if args.overwrite_images or not wrist_exists:
#                     Image.fromarray(wrist).save(wrist_path, quality=90)

#                 # Resize/pad for policy
#                 left_224 = image_tools.resize_with_pad(left, 224, 224)
#                 wrist_224 = image_tools.resize_with_pad(wrist, 224, 224)

#                 request = {
#                     "observation/exterior_image_1_left": left_224,
#                     "observation/wrist_image_left": wrist_224,
#                     "observation/joint_position": joint,
#                     "observation/gripper_position": gripper,
#                     "prompt": args.prompt,
#                 }

#                 K = max(1, args.samples_per_state)
#                 chunks = []
#                 for _k in range(K):
#                     resp, client = robust_infer(client, request, args.host, args.port)  # {"actions": (T, A)}
#                     chunks.append(resp["actions"])
#                 try:
#                     chunks_arr = np.stack(chunks, axis=0)   # (K,T,A)
#                 except Exception:
#                     chunks_arr = np.array(chunks, dtype=object)

#                 # Summaries
#                 m_arr = np.asarray(chunks_arr)
#                 mean_vec = m_arr.mean(axis=0).mean(axis=0)
#                 std_vec  = m_arr.std(axis=0).std(axis=0)

#                 row_dict = {
#                     "global_step_index": g_idx,
#                     "episode_index": ep_idx,
#                     "episode_id": episode_id,
#                     "task_name": task_name,
#                     "instruction": instruction_any,
#                     "t_in_episode": t_in_ep,
#                     "picked_left_key": dbg["picked_keys"]["left"],
#                     "picked_wrist_key": dbg["picked_keys"]["wrist"],
#                     "left_img_path": os.path.relpath(left_path, args.out_dir),
#                     "wrist_img_path": os.path.relpath(wrist_path, args.out_dir),
#                     "samples_per_state": K,
#                     "actions_shape": str(np.shape(chunks_arr)),
#                     **{f"mean_{i}": float(m) for i, m in enumerate(mean_vec)},
#                     **{f"std_{i}": float(s) for i, s in enumerate(std_vec)},
#                 }

#                 rows_global.append(row_dict)
#                 raw_actions_global.append(chunks_arr)
#                 rows_ep.append(row_dict)
#                 actions_ep.append(chunks_arr)

#                 if args.checkpoint_every_steps > 0:
#                     shard_rows.append(row_dict)
#                     shard_actions.append(chunks_arr)
#                     if len(shard_rows) >= args.checkpoint_every_steps:
#                         shard_idx += 1
#                         _flush_shard_checkpoint(args.out_dir, shard_idx, shard_rows, shard_actions, prefix=args.shard_prefix)
#                         shard_rows.clear(); shard_actions.clear()

#             except Exception as e:
#                 logging.warning("Episode %s step %d failed: %s", episode_id, t_in_ep, e)
#                 continue

#         # End of episode: finalize episode record + checkpoint if requested
#         episodes_processed += 1

#         # Canonicalize episode instruction fields
#         def _first_or_empty(s): 
#             return next(iter(s)) if s else ""
#         ep_instr_1 = _first_or_empty(instr_accum["language_instruction"])
#         ep_instr_2 = _first_or_empty(instr_accum["language_instruction_2"])
#         ep_instr_3 = _first_or_empty(instr_accum["language_instruction_3"])
#         ep_instr_all = " || ".join([s for s in sorted(
#             list(instr_accum["language_instruction"] | instr_accum["language_instruction_2"] | instr_accum["language_instruction_3"])
#         ) if s])

#         episodes_seen[episode_id] = {
#             "episode_index": ep_idx,
#             "episode_id": episode_id,
#             "task_name": task_name or "",
#             "instruction_any": instruction_any or ep_instr_1 or ep_instr_2 or ep_instr_3,
#             "language_instruction": ep_instr_1,
#             "language_instruction_2": ep_instr_2,
#             "language_instruction_3": ep_instr_3,
#             "instruction_all_unique": ep_instr_all,
#             "num_states_collected": len(rows_ep),
#         }

#         if args.checkpoint_by_episode:
#             _flush_episode_checkpoint(args.out_dir, episode_id, rows_ep, actions_ep)

#         logging.info("Finished episode %s (#%d). Episodes processed: %d/%d",
#                      episode_id, ep_idx, episodes_processed, args.max_episodes)

#     # Final shard flush
#     if args.checkpoint_every_steps > 0 and (shard_rows or shard_actions):
#         shard_idx += 1
#         _flush_shard_checkpoint(args.out_dir, shard_idx, shard_rows, shard_actions, prefix=args.shard_prefix)

#     elapsed = time.time() - start
#     logging.info("Done. Processed %d episodes (%d states) in %.1fs.",
#                  episodes_processed, len(rows_global), elapsed)

#     # 5) Save global outputs (final snapshot)
#     df = pd.DataFrame(rows_global)

#     csv_path = os.path.join(args.out_dir, "summary.csv")
#     if os.path.exists(csv_path):
#         try:
#             old_df = pd.read_csv(csv_path)
#         except Exception as e:
#             logging.warning("Could not read existing summary.csv (%s); creating new.", e)
#             old_df = pd.DataFrame()
#         # Append then drop duplicates on (episode_id, t_in_episode)
#         combined = pd.concat([old_df, df], ignore_index=True)
#         # If either column missing, just save combined (best effort)
#         if {"episode_id", "t_in_episode"} <= set(combined.columns):
#             combined["episode_id"] = combined["episode_id"].astype(str)
#             combined = combined.dropna(subset=["t_in_episode"])
#             combined["t_in_episode"] = combined["t_in_episode"].astype(int)
#             combined = combined.drop_duplicates(subset=["episode_id", "t_in_episode"], keep="first")
#         combined.to_csv(csv_path, index=False)
#         logging.info("Appended to %s (now %d rows).", csv_path, len(combined))
#     else:
#         df.to_csv(csv_path, index=False)
#         logging.info("Wrote %s", csv_path)

#     ep_df = pd.DataFrame(sorted(episodes_seen.values(), key=lambda d: d["episode_index"]))
#     ep_csv = os.path.join(args.out_dir, "episodes.csv")
#     ep_df.to_csv(ep_csv, index=False); logging.info("Wrote %s", ep_csv)

#     if args.save_npz and raw_actions_global:
#         new_arr = _stack_or_object(raw_actions_global)
#         npz_path = os.path.join(args.out_dir, "actions.npz")

#         if os.path.exists(npz_path):
#             try:
#                 old = np.load(npz_path, allow_pickle=True)
#                 old_arr = old.get("actions")
#             except Exception as e:
#                 logging.warning("Could not read existing actions.npz (%s); writing new.", e)
#                 old_arr = None

#             if old_arr is None or old_arr.size == 0:
#                 combined_arr = new_arr
#             elif new_arr.size == 0:
#                 combined_arr = old_arr
#             else:
#                 # Try to concatenate; if shapes differ, fall back to object stacking
#                 try:
#                     combined_arr = np.concatenate([old_arr, new_arr], axis=0)
#                 except Exception:
#                     combined_arr = _stack_or_object(list(old_arr) + list(new_arr))

#             np.savez_compressed(npz_path, actions=combined_arr)
#             logging.info("Appended to %s (new length ~ %d).", npz_path, len(combined_arr))
#         else:
#             np.savez_compressed(npz_path, actions=new_arr)
#             logging.info("Wrote %s", npz_path)

#     server.stop()

# if __name__ == "__main__":
#     main()
