#!/usr/bin/env python3
import os
import argparse
import logging
import numpy as np


def _stack_or_object(arr_list):
    """Same idea as in your original script."""
    if not arr_list:
        return np.array([], dtype=object)
    try:
        return np.stack(arr_list, axis=0)
    except Exception:
        return np.array(arr_list, dtype=object)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--root",
        required=True,
        help="Directory that contains the 'episodes' folder (i.e. root/episodes/ep000000/...)",
    )
    ap.add_argument(
        "--out",
        default=None,
        help="Path to output actions.npz (default: <root>/actions.npz)",
    )
    args = ap.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s: %(message)s")

    episodes_dir = os.path.join(args.root, "episodes")
    if not os.path.isdir(episodes_dir):
        raise SystemExit(f"episodes directory not found: {episodes_dir}")

    out_path = args.out or os.path.join(args.root, "actions.npz")

    all_arrays = []

    # iterate episodes in order
    for name in sorted(os.listdir(episodes_dir)):
        ep_dir = os.path.join(episodes_dir, name)
        if not os.path.isdir(ep_dir):
            continue
        npz_path = os.path.join(ep_dir, "actions.npz")
        if not os.path.exists(npz_path):
            logging.warning("episode %s has no actions.npz, skipping", name)
            continue

        try:
            data = np.load(npz_path, allow_pickle=True)
            arr = data.get("actions")
            if arr is None:
                logging.warning("actions.npz in %s has no 'actions' key, skipping", npz_path)
                continue
            all_arrays.append(arr)
            logging.info("loaded %s (%s, len=%s)", npz_path, type(arr), getattr(arr, "shape", None))
        except Exception as e:
            logging.warning("failed to load %s: %s", npz_path, e)
            continue

    if not all_arrays:
        logging.info("no episode actions found, nothing to write")
        return

    # try to mimic your "append or fall back to object" behavior
    if len(all_arrays) == 1:
        combined = all_arrays[0]
    else:
        # first try straight concat along 0
        try:
            combined = np.concatenate(all_arrays, axis=0)
        except Exception:
            # fall back to object stacking (your script does this)
            flat = []
            for arr in all_arrays:
                # if arr is already 1D list-like of steps, just extend
                try:
                    flat.extend(list(arr))
                except Exception:
                    flat.append(arr)
            combined = _stack_or_object(flat)

    np.savez_compressed(out_path, actions=combined)
    logging.info("wrote combined actions to %s (len ~ %d)", out_path, len(combined))


if __name__ == "__main__":
    main()