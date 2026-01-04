#!/usr/bin/env python3
"""
analyze_task_multimodality_trajectories.py

Goal
----
For selected tasks (e.g. pour/remove/pick), analyze multimodality
(measured by variance_drop_ratio) at the state level, and produce:

A) Distribution plots (COMPARATIVE across tasks in the same plot):
   1) Per-trajectory summary distribution (violin + jitter points):
        For each episode trajectory, summarize state-level multimodality
        with a statistic (mean/max/median/p80/etc). Plot distribution per task.
   2) Multimodality distribution over time:
        For each time step t, across trajectories for that task, plot
        median with IQR ribbon (25-75%). (All tasks overlaid.)
   3) Variance of multimodality over time:
        For each time step t, compute Var across trajectories (task-wise),
        and plot over time (all tasks overlaid).

B) Episode 3D visualizations (per task, across low/mid/high bands):
   * Trajectory of state "anchors" in 3D action space, colored by variance_drop_ratio.
   * Ghost action chunk trajectories at each state (semi-transparent).

We use:
  - metrics_per_state.csv (from multimodality clustering script)
  - actions.npz          (the same actions array used there)

Outputs
-------
Under --outdir:
  - task_episode_summary.csv
  - plots/
      - traj_summary_violin_<stat>.png
      - ratio_over_time_median_iqr.png
      - ratio_variance_over_time.png
  - episode_plots/<task>/
      - episode_<episode_id>_band_<band>.html  (interactive)

Usage
-----
Example:
  python analyze_task_multimodality_trajectories.py \
      --metrics_csv $OUTDIR/all_multimodality/metrics_per_state.csv \
      --actions_npz $OUTDIR/actions.npz \
      --outdir $OUTDIR/task_multimodality_plots \
      --tasks pour,remove,pick \
      --traj_stat max \
      --episodes_per_band 3 \
      --task_source both
"""

import argparse
import os
import sys
import numpy as np
import pandas as pd

from scipy.spatial.distance import cdist
from sklearn.decomposition import PCA

import matplotlib
import matplotlib.pyplot as plt
import plotly.graph_objects as go


# ----------------------------------------------------------------------
# Embeddings
# ----------------------------------------------------------------------

def embed_state_vector(action_vecs, mode="first3"):
    """
    action_vecs: (N, A) -> (N, 3)
    """
    X = np.asarray(action_vecs, dtype=float)
    if X.ndim != 2:
        raise ValueError("action_vecs must be 2D (N, A)")
    if mode == "first3":
        if X.shape[1] < 3:
            pad = np.zeros((X.shape[0], 3 - X.shape[1]))
            return np.hstack([X, pad])[:, :3]
        return X[:, :3]
    else:
        pca = PCA(n_components=3, random_state=0)
        return pca.fit_transform(X)


def embed_chunk_first3(chunk):
    """
    chunk: (T, A) -> (T, 3)
    """
    arr = np.asarray(chunk, dtype=float)
    if arr.ndim != 2:
        raise ValueError("chunk must be 2D (T, A)")
    T, A = arr.shape
    if A >= 3:
        return arr[:, :3]
    pad = np.zeros((T, 3 - A), dtype=float)
    return np.hstack([arr, pad])


# ----------------------------------------------------------------------
# Plotly helpers
# ----------------------------------------------------------------------

viridis = matplotlib.colormaps.get_cmap("viridis")


def ratio_to_rgba(ratio, r_min, r_max, alpha=1.0):
    if r_max > r_min:
        norm = (ratio - r_min) / (r_max - r_min)
    else:
        norm = 0.5
    norm = float(np.clip(norm, 0.0, 1.0))
    r, g, b, _ = viridis(norm)
    return f"rgba({int(r*255)}, {int(g*255)}, {int(b*255)}, {alpha})"


def plot_episode_trajectory_with_ghosts(
    episode_id,
    band,
    task_label,
    ep_df,
    actions_arr,
    global_r_min,
    global_r_max,
    out_html,
    max_chunks_per_state=5,
    mode="first3",
):
    """
    Uses boundary-consistent "state anchors":
      - first state anchor = START of its own chunk(s)
      - later state anchor = END of previous state's chunk(s)
    """
    ep_df = ep_df.sort_values("t_in_episode").reset_index(drop=True)

    state_indices = ep_df["state_index"].astype(int).to_numpy()
    t_steps = ep_df["t_in_episode"].to_numpy()
    ratios = ep_df["variance_drop_ratio"].to_numpy()

    def state_anchor_from_chunks(arr, which="start"):
        arr = np.asarray(arr, dtype=float)
        if arr.ndim == 3:
            if which == "start":
                pts = arr[:, 0, :]
            else:
                pts = arr[:, -1, :]
            return pts.mean(axis=0)
        elif arr.ndim == 2:
            return arr[0, :] if which == "start" else arr[-1, :]
        else:
            raise ValueError(f"Unsupported action shape: {arr.shape}")

    anchors = []
    for i, s in enumerate(state_indices):
        if i == 0:
            anchors.append(state_anchor_from_chunks(actions_arr[s], which="start"))
        else:
            s_prev = state_indices[i - 1]
            anchors.append(state_anchor_from_chunks(actions_arr[s_prev], which="end"))

    anchors = np.vstack(anchors)
    xyz_states = embed_state_vector(anchors, mode=mode)

    fig = go.Figure()

    fig.add_trace(go.Scatter3d(
        x=xyz_states[:, 0],
        y=xyz_states[:, 1],
        z=xyz_states[:, 2],
        mode="lines",
        line=dict(width=3),
        opacity=0.45,
        showlegend=False,
        name="trajectory",
    ))

    fig.add_trace(go.Scatter3d(
        x=xyz_states[:, 0],
        y=xyz_states[:, 1],
        z=xyz_states[:, 2],
        mode="markers",
        marker=dict(
            size=5,
            color=ratios,
            colorscale="Viridis",
            cmin=global_r_min,
            cmax=global_r_max,
            colorbar=dict(title="variance_drop_ratio"),
        ),
        text=[f"task={task_label}<br>t={t}<br>ratio={r:.4f}" for t, r in zip(t_steps, ratios)],
        hoverinfo="text",
        name="states",
    ))

    # Ghost trajectories
    for idx, s in enumerate(state_indices):
        arr = np.asarray(actions_arr[s], dtype=float)
        ratio_s = float(ratios[idx])
        color_s = ratio_to_rgba(ratio_s, global_r_min, global_r_max, alpha=1.0)

        if arr.ndim == 3:
            Kc = arr.shape[0]
            chunk_indices = np.arange(Kc)
            if Kc > max_chunks_per_state:
                rng = np.random.RandomState(0 + int(s))
                chunk_indices = rng.choice(Kc, size=max_chunks_per_state, replace=False)

            for ck in chunk_indices:
                chunk_xyz = embed_chunk_first3(arr[ck])
                fig.add_trace(go.Scatter3d(
                    x=chunk_xyz[:, 0],
                    y=chunk_xyz[:, 1],
                    z=chunk_xyz[:, 2],
                    mode="lines",
                    line=dict(color=color_s, width=1),
                    opacity=0.15,
                    showlegend=False,
                    hoverinfo="none",
                ))

        elif arr.ndim == 2:
            chunk_xyz = embed_chunk_first3(arr)
            fig.add_trace(go.Scatter3d(
                x=chunk_xyz[:, 0],
                y=chunk_xyz[:, 1],
                z=chunk_xyz[:, 2],
                mode="lines",
                line=dict(color=color_s, width=1),
                opacity=0.15,
                showlegend=False,
                hoverinfo="none",
            ))
        else:
            sys.stderr.write(f"[warn] episode {episode_id}: state {s} unsupported action shape {arr.shape}\n")

    fig.update_layout(
        title=f"Task={task_label} | Episode {episode_id} ({band}) | states colored by variance_drop_ratio",
        scene=dict(xaxis_title="X", yaxis_title="Y", zaxis_title="Z"),
        margin=dict(l=0, r=0, b=0, t=40),
    )

    fig.write_html(out_html)
    print(f"  wrote {out_html}")


# ----------------------------------------------------------------------
# Task labeling helpers
# ----------------------------------------------------------------------

def normalize_tasks(tasks_csv):
    # allow "pour, remove ,Pick" etc
    return [t.strip().lower() for t in tasks_csv.split(",") if t.strip()]


def task_match_label(row, tasks, source="both"):
    """
    Returns the matched task keyword (string) or None.
    If multiple match, returns the first in tasks list (stable priority).
    """
    tn = str(row.get("task_name", ""))
    instr = str(row.get("instruction", ""))

    hay = ""
    if source == "task_name":
        hay = tn
    elif source == "instruction":
        hay = instr
    else:
        hay = f"{tn} {instr}"

    h = hay.lower()
    for t in tasks:
        if t in h:
            return t
    return None


# ----------------------------------------------------------------------
# Distribution plotting
# ----------------------------------------------------------------------

def compute_traj_stat(values, stat):
    x = np.asarray(values, dtype=float)
    x = x[~np.isnan(x)]
    if x.size == 0:
        return np.nan
    if stat == "mean":
        return float(np.mean(x))
    if stat == "max":
        return float(np.max(x))
    if stat == "median":
        return float(np.median(x))
    if stat == "p80":
        return float(np.quantile(x, 0.80))
    if stat == "p90":
        return float(np.quantile(x, 0.90))
    if stat == "min":
        return float(np.min(x))
    raise ValueError(f"Unknown traj_stat: {stat}")


def violin_with_jitter(ax, groups, labels, title, ylabel):
    """
    groups: list of 1D arrays
    """
    positions = np.arange(1, len(groups) + 1)
    ax.violinplot(groups, positions=positions, showmedians=True, showextrema=True)

    # jittered points (matplotlib default color cycle will apply per call if we vary,
    # but to keep it simple and comparable, just use default marker.)
    rng = np.random.RandomState(0)
    for i, vals in enumerate(groups, start=1):
        vals = np.asarray(vals)
        if vals.size == 0:
            continue
        xj = i + 0.06 * rng.randn(vals.size)
        ax.scatter(xj, vals, s=12, alpha=0.5)

    ax.set_xticks(positions)
    ax.set_xticklabels(labels, rotation=25, ha="right")
    ax.set_title(title)
    ax.set_ylabel(ylabel)
    ax.set_xlabel("Task")


def plot_over_time_median_iqr(out_png, time_stats_df, tasks, k_fixed):
    """
    time_stats_df columns:
      task, t_norm, median, q25, q75, n

    Plot: median line + IQR ribbon.
    Cleaner: no big dots; use small markers only when data are sparse.
    """
    plt.figure(figsize=(9, 5))
    ax = plt.gca()

    for task in tasks:
        sub = time_stats_df[time_stats_df["task"] == task].sort_values("t_norm")
        if sub.empty:
            continue

        t = sub["t_norm"].to_numpy()
        med = sub["median"].to_numpy()
        q25 = sub["q25"].to_numpy()
        q75 = sub["q75"].to_numpy()

        # If there are many points, skip markers; if few, tiny markers help.
        use_markers = (len(t) <= 25)
        marker = "." if use_markers else None
        ms = 3 if use_markers else None

        line = ax.plot(
            t, med,
            linewidth=2,
            marker=marker,
            markersize=ms,
            label=task,
        )
        color = line[0].get_color()
        ax.fill_between(t, q25, q75, alpha=0.18, color=color)

    ax.set_title(f"State multimodality over time (median ± IQR), k={k_fixed}")
    ax.set_xlabel("normalized time (0 → start, 1 → end)")
    ax.set_ylabel("variance_drop_ratio")
    ax.set_xlim(0.0, 1.0)
    ax.grid(True, alpha=0.25)
    ax.legend(frameon=False)
    plt.tight_layout()
    plt.savefig(out_png, dpi=150)
    plt.close()


def plot_over_time_variance(out_png, time_var_df, tasks, k_fixed):
    """
    time_var_df columns:
      task, t_norm, var, n
    Cleaner: no big dots; tiny markers only when sparse.
    """
    plt.figure(figsize=(9, 5))
    ax = plt.gca()

    for task in tasks:
        sub = time_var_df[time_var_df["task"] == task].sort_values("t_norm")
        if sub.empty:
            continue

        t = sub["t_norm"].to_numpy()
        v = sub["var"].to_numpy()

        use_markers = (len(t) <= 25)
        marker = "." if use_markers else None
        ms = 3 if use_markers else None

        ax.plot(
            t, v,
            linewidth=2,
            marker=marker,
            markersize=ms,
            label=task,
        )

    ax.set_title(f"Variance of multimodality over time, k={k_fixed}")
    ax.set_xlabel("normalized time (0 → start, 1 → end)")
    ax.set_ylabel("Var(variance_drop_ratio)")
    ax.set_xlim(0.0, 1.0)
    ax.grid(True, alpha=0.25)
    ax.legend(frameon=False)
    plt.tight_layout()
    plt.savefig(out_png, dpi=150)
    plt.close()


# ----------------------------------------------------------------------
# main
# ----------------------------------------------------------------------

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--metrics_csv", type=str, required=True)
    ap.add_argument("--actions_npz", type=str, required=True)
    ap.add_argument("--outdir", type=str, required=True)

    ap.add_argument("--tasks", type=str, default="pour",
                    help="Comma-separated task keywords to match (case-insensitive). "
                         "Example: pour,remove,pick")
    ap.add_argument("--task_source", type=str, default="both",
                    choices=["both", "task_name", "instruction"],
                    help="Where to match task keywords.")

    ap.add_argument("--traj_stat", type=str, default="max",
                    choices=["mean", "max", "median", "p80", "p90", "min"],
                    help="How to summarize per-trajectory multimodality across its states.")

    ap.add_argument("--episodes_per_band", type=int, default=3,
                    help="Max number of episodes to visualize per band (low/mid/high) per task.")
    ap.add_argument("--embedding_mode", type=str, default="first3",
                    choices=["first3", "pca"])
    ap.add_argument("--max_t", type=int, default=None,
                    help="Optional cap on t_in_episode for time plots.")
    ap.add_argument(
        "--k",
        type=int,
        required=True,
        help="Number of clusters k to analyze multimodality at (must match metrics_csv)"
    )
    args = ap.parse_args()

    os.makedirs(args.outdir, exist_ok=True)
    plot_dir = os.path.join(args.outdir, "plots")
    os.makedirs(plot_dir, exist_ok=True)
    episode_plot_root = os.path.join(args.outdir, "episode_plots")
    os.makedirs(episode_plot_root, exist_ok=True)

    tasks = normalize_tasks(args.tasks)
    if not tasks:
        raise ValueError("--tasks parsed to empty list")

    print(f"Tasks: {tasks} (matching source={args.task_source})")

    metrics_df = pd.read_csv(args.metrics_csv)
    actions_data = np.load(args.actions_npz, allow_pickle=True)
    actions_arr = actions_data["actions"]

    required_cols = {"state_index", "episode_id", "t_in_episode", "task_name", "instruction",
                     "total_variance", "variance_drop"}
    missing = required_cols - set(metrics_df.columns)
    if missing:
        raise ValueError(f"metrics_csv missing required columns: {missing}")

    # ratio
    tv = metrics_df["total_variance"].to_numpy()
    vd = metrics_df["variance_drop"].to_numpy()
    with np.errstate(divide="ignore", invalid="ignore"):
        metrics_df["variance_drop_ratio"] = np.where(tv > 0, vd / tv, np.nan)
    
    # -------------------------------
    # FIX k FOR ANALYSIS
    # -------------------------------
    k_fixed = int(args.k)
    metrics_df = metrics_df[metrics_df["k"] == k_fixed].copy()

    if metrics_df.empty:
        raise ValueError(f"No rows found with k={k_fixed} in metrics_csv")

    print(f"Analyzing multimodality at fixed k={k_fixed}")

    # label each row with a matched task keyword
    metrics_df["task_label"] = metrics_df.apply(
        lambda r: task_match_label(r, tasks, source=args.task_source),
        axis=1
    )

    df = metrics_df[metrics_df["task_label"].notna()].copy()
    if df.empty:
        print("No rows matched any tasks. Nothing to do.")
        return

    # ensure numeric t
    df["t_in_episode"] = pd.to_numeric(df["t_in_episode"], errors="coerce")
    df = df[df["t_in_episode"].notna()].copy()
    df["t_in_episode"] = df["t_in_episode"].astype(int)

    if args.max_t is not None:
        df = df[df["t_in_episode"] <= int(args.max_t)].copy()

    # --------------------------------------------------------------
    # Normalize time to [0, 1] PER EPISODE so trajectories are comparable
    # We then bin normalized time into a fixed grid for aggregation.
    # --------------------------------------------------------------
    df["t_in_episode"] = df["t_in_episode"].astype(int)

    # per-episode max timestep (avoid divide by 0 for length-1 episodes)
    ep_tmax = df.groupby("episode_id")["t_in_episode"].transform("max").astype(float)
    df["t_norm"] = np.where(ep_tmax > 0, df["t_in_episode"] / ep_tmax, 0.0)

    # Bin normalized time so tasks overlay cleanly even with different lengths
    N_TIME_BINS = 50  # adjust if you want smoother/coarser curves
    df["t_bin"] = np.clip(
        np.round(df["t_norm"] * (N_TIME_BINS - 1)).astype(int),
        0,
        N_TIME_BINS - 1
    )
    df["t_norm_bin"] = df["t_bin"] / float(N_TIME_BINS - 1)

    # global color scale across ALL selected tasks (for comparable 3D)
    valid_all = df["variance_drop_ratio"].to_numpy()
    valid_all = valid_all[~np.isnan(valid_all)]
    if valid_all.size == 0:
        print("No valid variance_drop_ratio values after filtering.")
        return
    global_r_min = float(np.min(valid_all))
    global_r_max = float(np.max(valid_all))
    print(f"Global ratio range across selected tasks: [{global_r_min:.4f}, {global_r_max:.4f}]")

    # ------------------------------------------------------------------
    # 1) Per-trajectory summary distribution (violin + jitter)
    # ------------------------------------------------------------------
    traj_rows = []
    for (task, ep_id), g in df.groupby(["task_label", "episode_id"]):
        vals = g["variance_drop_ratio"].to_numpy()
        stat_val = compute_traj_stat(vals, args.traj_stat)
        n_states = len(g)
        traj_rows.append({
            "task": task,
            "episode_id": ep_id,
            "n_states": n_states,
            f"{args.traj_stat}_ratio": stat_val,
            "max_ratio": compute_traj_stat(vals, "max"),
            "mean_ratio": compute_traj_stat(vals, "mean"),
            "median_ratio": compute_traj_stat(vals, "median"),
        })

    traj_df = pd.DataFrame(traj_rows)
    traj_df.to_csv(os.path.join(args.outdir, "task_episode_summary.csv"), index=False)
    print(f"Wrote: {os.path.join(args.outdir, 'task_episode_summary.csv')}")

    # build violin groups in task order (robust: keep groups/labels aligned)
    pairs = []
    for t in tasks:
        sub_df = traj_df[traj_df["task"] == t].copy()
        vals = sub_df[f"{args.traj_stat}_ratio"].dropna().to_numpy()
        if vals.size == 0:
            continue
        n_traj = sub_df["episode_id"].nunique()
        pairs.append((f"{t} (n={n_traj})", vals))

    if pairs:
        labels, groups = zip(*pairs)   # perfectly aligned
        labels = list(labels)
        groups = list(groups)

        out_violin = os.path.join(plot_dir, f"traj_summary_violin_{args.traj_stat}.png")
        plt.figure(figsize=(max(7, 1.2 * len(groups)), 5))
        ax = plt.gca()
        violin_with_jitter(
            ax,
            groups=groups,
            labels=labels,
            title=f"Per-trajectory multimodality summary ({args.traj_stat}) at k={k_fixed}",
            ylabel=f"{args.traj_stat}(variance_drop_ratio over states in trajectory)",
        )
        plt.tight_layout()
        plt.savefig(out_violin, dpi=150)
        plt.close()
        print(f"Wrote: {out_violin}")
    else:
        print("[warn] No data for violin plot.")

    # ------------------------------------------------------------------
    # 2) Multimodality distribution over normalized time (median + IQR)
    # ------------------------------------------------------------------
    time_stats = []
    for (task, tb), g in df.groupby(["task_label", "t_bin"]):
        vals = g["variance_drop_ratio"].to_numpy()
        vals = vals[~np.isnan(vals)]
        if vals.size == 0:
            continue

        # x-axis value is the normalized bin center
        t_norm = float(g["t_norm_bin"].iloc[0])

        time_stats.append({
            "task": task,
            "t_norm": t_norm,
            "n": int(vals.size),
            "median": float(np.median(vals)),
            "q25": float(np.quantile(vals, 0.25)),
            "q75": float(np.quantile(vals, 0.75)),
        })
    time_stats_df = pd.DataFrame(time_stats)

    if not time_stats_df.empty:
        out_time = os.path.join(plot_dir, "ratio_over_time_median_iqr.png")
        plot_over_time_median_iqr(out_time, time_stats_df, tasks=tasks, k_fixed=k_fixed)
        print(f"Wrote: {out_time}")
    else:
        print("[warn] No data for time distribution plot.")

    # ------------------------------------------------------------------
    # 3) Variance in multimodality over normalized time (across trajectories)
    # ------------------------------------------------------------------
    time_var = []
    for (task, tb), g in df.groupby(["task_label", "t_bin"]):
        vals = g["variance_drop_ratio"].to_numpy()
        vals = vals[~np.isnan(vals)]
        if vals.size < 2:
            continue

        t_norm = float(g["t_norm_bin"].iloc[0])

        time_var.append({
            "task": task,
            "t_norm": t_norm,
            "n": int(vals.size),
            "var": float(np.var(vals, ddof=1)),
        })
    time_var_df = pd.DataFrame(time_var)

    if not time_var_df.empty:
        out_var = os.path.join(plot_dir, "ratio_variance_over_time.png")
        plot_over_time_variance(out_var, time_var_df, tasks=tasks, k_fixed=k_fixed)
        print(f"Wrote: {out_var}")
    else:
        print("[warn] No data for variance-over-time plot.")

    # ------------------------------------------------------------------
    # Episode 3D plots per task: choose episodes across low/mid/high bands
    # ------------------------------------------------------------------
    for task in tasks:
        sub_traj = traj_df[traj_df["task"] == task].copy()
        if sub_traj.empty:
            print(f"[info] Task '{task}': no trajectories found.")
            continue

        # banding based on the same chosen stat, to reflect "trajectory multimodality"
        col = f"{args.traj_stat}_ratio"
        sub_traj = sub_traj[sub_traj[col].notna()].copy()
        if sub_traj.empty:
            print(f"[info] Task '{task}': no valid trajectories for banding.")
            continue

        q_low = sub_traj[col].quantile(0.33)
        q_high = sub_traj[col].quantile(0.66)

        def band_fn(x):
            if x <= q_low:
                return "low"
            if x >= q_high:
                return "high"
            return "mid"

        sub_traj["band"] = sub_traj[col].apply(band_fn)

        task_episode_plot_dir = os.path.join(episode_plot_root, task)
        os.makedirs(task_episode_plot_dir, exist_ok=True)

        print(f"\nTask '{task}': band thresholds on {col}: low <= {q_low:.4f}, high >= {q_high:.4f}")

        for band in ["high", "mid", "low"]:
            band_df = sub_traj[sub_traj["band"] == band].copy()
            if band_df.empty:
                continue

            if band == "high":
                band_df.sort_values(col, ascending=False, inplace=True)
            elif band == "low":
                band_df.sort_values(col, ascending=True, inplace=True)
            else:
                # choose "representative" mids near the median
                m = band_df[col].median()
                band_df["dist"] = (band_df[col] - m).abs()
                band_df.sort_values("dist", ascending=True, inplace=True)

            selected = band_df.head(args.episodes_per_band)

            print(f"  Band '{band}': plotting {len(selected)} episode(s)")
            for r in selected.itertuples():
                ep_id = r.episode_id
                ep_states = df[(df["task_label"] == task) & (df["episode_id"] == ep_id)].copy()
                if ep_states.empty:
                    continue

                out_html = os.path.join(task_episode_plot_dir, f"episode_{str(ep_id)}_band_{band}.html")

                plot_episode_trajectory_with_ghosts(
                    episode_id=ep_id,
                    band=band,
                    task_label=task,
                    ep_df=ep_states,
                    actions_arr=actions_arr,
                    global_r_min=global_r_min,
                    global_r_max=global_r_max,
                    out_html=out_html,
                    max_chunks_per_state=5,
                    mode=args.embedding_mode,
                )

    print("\nDone.")


if __name__ == "__main__":
    main()