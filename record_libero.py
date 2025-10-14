# run_libero_end2end_record.py
import collections
import dataclasses
import logging
import math
import pathlib
from typing import Optional, Deque, List, Dict, Any

import imageio
import numpy as np
import tyro
import tqdm

# LIBERO
from libero.libero import benchmark
from libero.libero import get_libero_path
from libero.libero.envs import OffScreenRenderEnv

# OpenPI (load policy in-process; no websocket)
from openpi.policies import policy as _policy
from openpi.policies import policy_config as _policy_config
from openpi.training import config as _config

# ---- Simple image helpers (avoid extra deps) ----
def _resize_with_pad(img: np.ndarray, out_h: int, out_w: int) -> np.ndarray:
    """Resize with letterbox padding to (out_h, out_w). Uses cv2 if available else numpy fallback."""
    try:
        import cv2
        h, w = img.shape[:2]
        scale = min(out_w / w, out_h / h)
        nh, nw = int(round(h * scale)), int(round(w * scale))
        resized = cv2.resize(img, (nw, nh), interpolation=cv2.INTER_LINEAR)
        canvas = np.zeros((out_h, out_w, 3), dtype=resized.dtype)
        top = (out_h - nh) // 2
        left = (out_w - nw) // 2
        canvas[top:top+nh, left:left+nw] = resized
        return canvas
    except Exception:
        # Very simple nearest-neighbor-ish fallback
        h, w = img.shape[:2]
        scale = min(out_w / w, out_h / h)
        nh, nw = max(1, int(h * scale)), max(1, int(w * scale))
        # crude resize
        yi = (np.linspace(0, h - 1, nh)).astype(int)
        xi = (np.linspace(0, w - 1, nw)).astype(int)
        resized = img[np.ix_(yi, xi)]
        canvas = np.zeros((out_h, out_w, 3), dtype=resized.dtype)
        top = (out_h - nh) // 2
        left = (out_w - nw) // 2
        canvas[top:top+nh, left:left+nw] = resized
        return canvas

def _to_uint8(img: np.ndarray) -> np.ndarray:
    if img.dtype == np.uint8:
        return img
    img = np.clip(img, 0, 255)
    return img.astype(np.uint8)

# ---- Constants ----
LIBERO_DUMMY_ACTION = [0.0] * 6 + [-1.0]  # do-nothing/close gripper used during settling
LIBERO_ENV_RESOLUTION = 256               # camera render size used in training

@dataclasses.dataclass
class Args:
    # Checkpoint (policy) to load
    ckpt_config: str = "pi05_libero"     # OpenPI training config
    ckpt_dir: str = "gs://openpi-assets/checkpoints/pi05_libero"  # or local path

    # Suite & rollouts
    task_suite_name: str = "libero_spatial"     # libero_spatial, libero_object, libero_goal, libero_10, libero_90
    num_trials_per_task: int = 5                 # rollouts per task
    seed: int = 7                                # base seed (used for env seeding)
    replan_steps: int = 5                        # action-chunk horizon
    resize_size: int = 224                       # policy image input size
    num_steps_wait: int = 10                     # wait steps at episode start (objects settle)

    # Output
    out_dir: pathlib.Path = pathlib.Path("data/libero_e2e")
    save_images: bool = False                    # store images inside npz (can be large)
    video_fps: int = 10

    # Logging
    log_level: str = "INFO"

def _create_policy(ckpt_config: str, ckpt_dir: str) -> _policy.Policy:
    """Load a trained policy directly (no server)."""
    return _policy_config.create_trained_policy(
        _config.get_config(ckpt_config),
        ckpt_dir,
        default_prompt=None
    )

def _quat2axisangle(quat: np.ndarray) -> np.ndarray:
    """
    Robosuite-style quaternion (x, y, z, w) -> axis-angle (3,)
    """
    q = quat.copy()
    if q[3] > 1.0: q[3] = 1.0
    if q[3] < -1.0: q[3] = -1.0
    den = math.sqrt(max(1.0 - q[3] * q[3], 0.0))
    if math.isclose(den, 0.0):
        return np.zeros(3, dtype=np.float32)
    return (q[:3] * 2.0 * math.acos(q[3])) / den

def _get_libero_env(task, resolution: int, seed: int):
    """Init OffScreenRenderEnv for a given LIBERO task, return (env, task_description)."""
    task_description = task.language
    task_bddl_file = pathlib.Path(get_libero_path("bddl_files")) / task.problem_folder / task.bddl_file
    env_args = {
        "bddl_file_name": task_bddl_file,
        "camera_heights": resolution,
        "camera_widths": resolution
    }
    env = OffScreenRenderEnv(**env_args)
    env.seed(seed)
    return env, task_description

def _policy_infer_chunk(
    policy: _policy.Policy,
    obs_image: np.ndarray,
    wrist_image: np.ndarray,
    state_vec: np.ndarray,
    prompt: str,
) -> Dict[str, Any]:
    """
    Call policy directly. Expected to return {"actions": np.ndarray[T, action_dim], ...}.
    """
    element = {
        "observation/image": obs_image,         # HxWxC uint8
        "observation/wrist_image": wrist_image, # HxWxC uint8
        "observation/state": state_vec,         # (dim,)
        "prompt": prompt,
    }
    # Most OpenPI policies expose a `predict_action` or `__call__`. The server packs
    # this into "infer". Here we call `predict_action` and return a similar dict.
    # If your policy exposes a different method, adjust below.
    out = policy.predict_action(element)  # returns dict with "actions" or similar
    # Normalize expected structure
    if "actions" not in out:
        # Some policies might return {"action": (T,dim)} singular. Normalize.
        if "action" in out:
            out["actions"] = out["action"]
    return out

def _choose_max_steps(task_suite_name: str) -> int:
    if task_suite_name == "libero_spatial":
        return 220
    elif task_suite_name == "libero_object":
        return 280
    elif task_suite_name == "libero_goal":
        return 300
    elif task_suite_name == "libero_10":
        return 520
    elif task_suite_name == "libero_90":
        return 400
    else:
        raise ValueError(f"Unknown task suite: {task_suite_name}")

def _ensure_dirs(base: pathlib.Path, task_name: str):
    task_segment = task_name.replace(" ", "_")
    traj_dir = base / "trajectories" / task_segment
    vid_dir = base / "videos" / task_segment
    traj_dir.mkdir(parents=True, exist_ok=True)
    vid_dir.mkdir(parents=True, exist_ok=True)
    return traj_dir, vid_dir

def _save_traj_npz(
    out_path: pathlib.Path,
    states: List[np.ndarray],
    actions: List[np.ndarray],
    rewards: List[float],
    images: Optional[List[np.ndarray]],
    meta: Dict[str, Any],
    save_images: bool,
):
    states_arr = np.stack(states, axis=0) if states else np.empty((0,), dtype=np.float32)
    actions_arr = np.stack(actions, axis=0) if actions else np.empty((0,), dtype=np.float32)
    rewards_arr = np.asarray(rewards, dtype=np.float32)

    payload = {
        "states": states_arr,
        "actions": actions_arr,
        "rewards": rewards_arr,
        "meta": meta,
    }
    if save_images and images is not None:
        # store as object array to avoid huge contiguous allocation (or stack if preferred)
        payload["images"] = np.array(images, dtype=object)

    logging.info(f"Saving trajectory: {out_path}")
    np.savez_compressed(out_path, **payload)

def _save_video_mp4(out_path: pathlib.Path, frames: List[np.ndarray], fps: int):
    if not frames:
        return
    logging.info(f"Saving video: {out_path}")
    imageio.mimwrite(out_path, [np.asarray(f) for f in frames], fps=fps)

def main(args: Args):
    logging.basicConfig(level=getattr(logging, args.log_level.upper(), logging.INFO), force=True)

    # Load benchmark suite & policy
    logging.info("Loading LIBERO benchmark: %s", args.task_suite_name)
    benchmark_dict = benchmark.get_benchmark_dict()
    task_suite = benchmark_dict[args.task_suite_name]()
    num_tasks = task_suite.n_tasks
    max_steps = _choose_max_steps(args.task_suite_name)

    logging.info("Loading policy: config=%s dir=%s", args.ckpt_config, args.ckpt_dir)
    policy = _create_policy(args.ckpt_config, args.ckpt_dir)

    # Output base
    args.out_dir.mkdir(parents=True, exist_ok=True)

    total_episodes = 0
    total_successes = 0

    # Iterate all tasks
    for task_id in tqdm.tqdm(range(num_tasks), desc="Tasks"):
        # Task & initial states
        task = task_suite.get_task(task_id)
        init_states = task_suite.get_task_init_states(task_id)
        env, task_desc = _get_libero_env(task, LIBERO_ENV_RESOLUTION, args.seed)

        # Prepare output dirs
        traj_dir, vid_dir = _ensure_dirs(args.out_dir, task_desc)

        task_eps = 0
        task_success = 0

        for ep_idx in tqdm.tqdm(range(args.num_trials_per_task), desc=f"Task {task_id}", leave=False):
            # Reset env & set fixed initial state for comparability
            env.reset()
            obs = env.set_init_state(init_states[ep_idx % len(init_states)])

            # Buffers
            frames: List[np.ndarray] = []
            states: List[np.ndarray] = []
            actions: List[np.ndarray] = []
            rewards: List[float] = []

            # Short horizon action plan
            plan: Deque[np.ndarray] = collections.deque()

            # Episode loop
            t = 0
            done = False

            # Wait to let objects stabilize
            while t < args.num_steps_wait:
                obs, reward, done, info = env.step(LIBERO_DUMMY_ACTION)
                t += 1
                if done:
                    break

            # Roll
            while (not done) and (t < max_steps + args.num_steps_wait):
                # Preprocess images: rotate 180° to match training; then resize/pad; to uint8
                img = np.ascontiguousarray(obs["agentview_image"][::-1, ::-1])
                wrist = np.ascontiguousarray(obs["robot0_eye_in_hand_image"][::-1, ::-1])
                img = _to_uint8(_resize_with_pad(img, args.resize_size, args.resize_size))
                wrist = _to_uint8(_resize_with_pad(wrist, args.resize_size, args.resize_size))
                frames.append(img)

                # Build low-dim state
                state_vec = np.concatenate((
                    obs["robot0_eef_pos"],                 # (3,)
                    _quat2axisangle(obs["robot0_eef_quat"]),  # (3,)
                    obs["robot0_gripper_qpos"]             # (1,)
                ), dtype=np.float32)

                # Replan if no actions queued
                if not plan:
                    out = _policy_infer_chunk(policy, img, wrist, state_vec, str(task_desc))
                    if "actions" not in out:
                        raise RuntimeError("Policy did not return 'actions' in inference output.")
                    chunk = out["actions"]
                    # Expect shape (T, action_dim) or list of length T
                    if isinstance(chunk, list):
                        chunk = np.asarray(chunk)
                    if chunk.ndim == 1:
                        chunk = chunk[None, :]  # (1, action_dim)

                    if chunk.shape[0] < args.replan_steps:
                        raise RuntimeError(
                            f"Policy returned only {chunk.shape[0]} steps but replan_steps={args.replan_steps}"
                        )
                    for k in range(args.replan_steps):
                        plan.append(chunk[k])

                act = plan.popleft()
                actions.append(np.asarray(act, dtype=np.float32))
                states.append(state_vec)

                obs, reward, done, info = env.step(act.tolist())
                rewards.append(float(reward))
                t += 1

            # Episode bookkeeping
            success = bool(done)
            total_episodes += 1
            task_eps += 1
            if success:
                total_successes += 1
                task_success += 1

            # Save outputs for this episode
            suffix = "success" if success else "failure"
            base_name = f"task{task_id:02d}_ep{ep_idx:03d}_{suffix}"

            # Video
            _save_video_mp4(vid_dir / f"{base_name}.mp4", frames, fps=args.video_fps)

            # Trajectory
            meta = {
                "task_id": task_id,
                "task_description": task_desc,
                "episode_index": ep_idx,
                "success": success,
                "max_steps": max_steps,
                "num_steps_wait": args.num_steps_wait,
                "replan_steps": args.replan_steps,
                "resize_size": args.resize_size,
                "seed": args.seed,
            }
            _save_traj_npz(
                traj_dir / f"{base_name}.npz",
                states=states,
                actions=actions,
                rewards=rewards,
                images=frames if args.save_images else None,
                meta=meta,
                save_images=args.save_images
            )

            logging.info(
                f"[Task {task_id}] Episode {ep_idx+1}/{args.num_trials_per_task} "
                f"=> success={success} | task_succ={task_success}/{task_eps} "
                f"({(task_success / max(task_eps,1)) * 100:.1f}%) "
                f"| total_succ={total_successes}/{total_episodes} "
                f"({(total_successes / max(total_episodes,1)) * 100:.1f}%)"
            )

        logging.info(
            f"[Task {task_id}] Final task success rate: "
            f"{(task_success / max(task_eps,1)):.3f}"
        )

    logging.info(
        "Done. Overall success: %d/%d (%.1f%%)",
        total_successes, total_episodes,
        (total_successes / max(total_episodes, 1)) * 100.0
    )

if __name__ == "__main__":
    tyro.cli(main)