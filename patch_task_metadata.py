#!/usr/bin/env python3
import argparse, os, re, glob
import numpy as np
import pandas as pd

def patch_table(csv_in, meta_df, csv_out, table_name="table"):
    df = pd.read_csv(csv_in)

    # Ensure columns exist
    for col in ["task_name", "instruction"]:
        if col not in df.columns:
            df[col] = ""

    df["episode_id"] = df["episode_id"].astype(str)
    df["t_in_episode"] = pd.to_numeric(df["t_in_episode"], errors="coerce").astype("Int64")

    meta_df = meta_df.copy()
    meta_df["episode_id"] = meta_df["episode_id"].astype(str)
    meta_df["t_in_episode"] = pd.to_numeric(meta_df["t_in_episode"], errors="coerce").astype("Int64")

    merged = df.merge(
        meta_df,
        on=["episode_id", "t_in_episode"],
        how="left",
        suffixes=("", "_meta")
    )

    def _fill(col):
        base = merged[col].fillna("").astype(str)
        meta = merged[f"{col}_meta"].fillna("").astype(str)
        return np.where(base.str.strip() == "", meta, base)

    merged["task_name"] = _fill("task_name")
    merged["instruction"] = _fill("instruction")

    merged.drop(columns=[c for c in merged.columns if c.endswith("_meta")], inplace=True)
    merged.to_csv(csv_out, index=False)

    tn_empty = (merged["task_name"].str.strip() == "").mean()
    ins_empty = (merged["instruction"].str.strip() == "").mean()
    print(f"[{table_name}] wrote {csv_out}")
    print(f"[{table_name}] empty task_name frac={tn_empty:.3f}, instruction frac={ins_empty:.3f}")


# -------- LeRobot helpers (same conventions as your experiment) --------
def _discover_lerobot_parquets(data_root: str):
    pattern = os.path.join(data_root, "data", "chunk-*", "file-*.parquet")
    for path in sorted(glob.glob(pattern)):
        m_chunk = re.search(r"chunk-(\d+)", path)
        m_file = re.search(r"file-(\d+)\.parquet$", path)
        if not m_chunk or not m_file:
            continue
        yield int(m_chunk.group(1)), int(m_file.group(1)), path

def build_lerobot_meta_map(data_root: str, max_rows_per_parquet=None):
    """
    Returns DataFrame with columns:
      episode_id, t_in_episode, task_name, instruction
    """
    recs = []
    for _, _, parquet_path in _discover_lerobot_parquets(data_root):
        df = pd.read_parquet(parquet_path)

        if max_rows_per_parquet is not None and len(df) > max_rows_per_parquet:
            df = df.iloc[:max_rows_per_parquet].copy()

        # expected columns in your iterator
        ep_idx = df.get("episode_index", pd.Series([0]*len(df))).astype(int)
        frame_idx = df.get("frame_index", pd.Series(range(len(df)))).astype(int)

        # instruction candidates
        instr_cols = [c for c in ["language_instruction", "language_instruction_2", "language_instruction_3"] if c in df.columns]
        if instr_cols:
            instr = df[instr_cols].astype(str).apply(
                lambda r: next((x for x in r.values if isinstance(x, str) and x.strip() and x.strip().lower() != "nan"), ""),
                axis=1
            )
        else:
            instr = pd.Series([""] * len(df))

        # task_name: if dataset has something explicit, use it; else keep blank
        # (Many LeRobot sets don’t have task_name separate from instruction.)
        task = df["task"].astype(str) if "task" in df.columns else pd.Series([""] * len(df))

        for i in range(len(df)):
            episode_id = f"ep{int(ep_idx.iloc[i]):06d}"
            recs.append({
                "episode_id": episode_id,
                "t_in_episode": int(frame_idx.iloc[i]),
                "task_name": "" if task is None else str(task.iloc[i]) if str(task.iloc[i]).strip().lower() != "nan" else "",
                "instruction": str(instr.iloc[i]) if str(instr.iloc[i]).strip().lower() != "nan" else "",
            })

    meta_df = pd.DataFrame(recs)
    # keep last non-empty info if duplicates appear
    meta_df["instruction"] = meta_df["instruction"].fillna("")
    meta_df["task_name"] = meta_df["task_name"].fillna("")
    meta_df.sort_values(["episode_id", "t_in_episode"], inplace=True)

    # aggregate duplicates: prefer non-empty
    def _pick_nonempty(series):
        for x in series[::-1]:
            if isinstance(x, str) and x.strip():
                return x
        return ""

    meta_df = meta_df.groupby(["episode_id", "t_in_episode"], as_index=False).agg({
        "task_name": _pick_nonempty,
        "instruction": _pick_nonempty
    })
    return meta_df


# -------- RLDS patch (optional) --------
def build_rlds_meta_map(tfds_data_dir: str, dataset_name: str, split: str, max_episodes=None):
    """
    Reads TFDS RLDS dataset and extracts per-step instruction/task if present.
    DOES NOT run policy inference.
    """
    import tensorflow_datasets as tfds

    recs = []
    ds = tfds.load(dataset_name, data_dir=tfds_data_dir, split=split, shuffle_files=False)

    def _decode(v):
        if isinstance(v, (bytes, bytearray)):
            try:
                return v.decode("utf-8", errors="ignore")
            except Exception:
                return str(v)
        return v

    for ep_i, episode in enumerate(ds):
        if max_episodes is not None and ep_i >= max_episodes:
            break

        episode_id = f"ep{ep_i:06d}"

        # iterate steps robustly
        steps_obj = episode["steps"]
        for t, step in enumerate(tfds.as_numpy(steps_obj)):
            obs = step.get("observation", step)

            # candidates for instruction fields
            cand = []
            for k in ["language_instruction", "language_instruction_2", "language_instruction_3",
                      "instruction", "task_name", "task", "goal", "description"]:
                if k in step: cand.append(_decode(step[k]))
                if k in obs:  cand.append(_decode(obs[k]))

            cand = [c for c in cand if isinstance(c, str) and c.strip() and c.strip().lower() != "nan"]
            instruction = cand[0] if cand else ""

            # RLDS often doesn’t have separate task_name; keep blank unless explicit
            task_name = ""
            for k in ["task_name", "task"]:
                v = step.get(k, None)
                if isinstance(v, (bytes, bytearray)): v = _decode(v)
                if isinstance(v, str) and v.strip():
                    task_name = v.strip()
                    break

            recs.append({
                "episode_id": episode_id,
                "t_in_episode": int(t),
                "task_name": task_name,
                "instruction": instruction
            })

    return pd.DataFrame(recs)


def patch_metrics(metrics_csv, meta_df, out_csv):
    patch_table(metrics_csv, meta_df, out_csv, table_name="metrics")

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--metrics_csv", required=True)
    ap.add_argument("--out_csv", required=True)

    ap.add_argument("--lerobot_root", default=None, help="If set, patch from LeRobot parquets")
    ap.add_argument("--max_rows_per_parquet", type=int, default=None)

    ap.add_argument("--tfds_data_dir", default=None, help="If set, also patch from RLDS TFDS")
    ap.add_argument("--dataset_name", default=None)
    ap.add_argument("--split", default="train")
    ap.add_argument("--max_episodes", type=int, default=None)
    ap.add_argument("--summary_csv", default=None)
    ap.add_argument("--summary_out_csv", default=None)


    args = ap.parse_args()

    meta_parts = []

    if args.lerobot_root:
        meta_parts.append(build_lerobot_meta_map(args.lerobot_root, args.max_rows_per_parquet))

    if args.tfds_data_dir and args.dataset_name:
        meta_parts.append(build_rlds_meta_map(args.tfds_data_dir, args.dataset_name, args.split, args.max_episodes))

    if not meta_parts:
        raise SystemExit("Provide --lerobot_root and/or (--tfds_data_dir + --dataset_name) to build metadata map.")

    meta_df = pd.concat(meta_parts, ignore_index=True)
    # dedup across sources: prefer non-empty
    meta_df["instruction"] = meta_df["instruction"].fillna("")
    meta_df["task_name"] = meta_df["task_name"].fillna("")
    meta_df.sort_values(["episode_id", "t_in_episode"], inplace=True)

    def _pick_nonempty(series):
        for x in series[::-1]:
            if isinstance(x, str) and x.strip():
                return x
        return ""

    meta_df = meta_df.groupby(["episode_id", "t_in_episode"], as_index=False).agg({
        "task_name": _pick_nonempty,
        "instruction": _pick_nonempty
    })

    patch_metrics(args.metrics_csv, meta_df, args.out_csv)

    if args.summary_csv:
        if not args.summary_out_csv:
            raise SystemExit("--summary_out_csv required when --summary_csv is set")

        patch_table(
            args.summary_csv,
            meta_df,
            args.summary_out_csv,
            table_name="summary"
        )


if __name__ == "__main__":
    main()