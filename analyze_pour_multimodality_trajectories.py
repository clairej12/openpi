#!/usr/bin/env python3
"""
analyze_pour_multimodality_trajectories.py

Goal
----
For "Pour" tasks, find episodes (training trajectories) that show different
levels of multimodality (measured via variance_drop_ratio), and visualize:

  * The whole training trajectory for an episode in 3D action space, where
    each STATE is a point, colored by its variance_drop_ratio.
  * The "action chunks" at each state as semi-transparent "ghost trajectories"
    emanating around those states.

We use:
  - metrics_per_state.csv (from your multimodality clustering script)
  - actions.npz         (the same actions array used there)

Assumptions
-----------
- metrics_per_state.csv has at least these columns:
    state_index, episode_id, t_in_episode, task_name,
    total_variance, variance_drop

- actions.npz contains:
    actions: an array such that actions[state_index] is either:
        (K_chunks, T, A) : multi-chunk trajectories of length T in R^A
        (T, A)           : time steps, which we treat as 1-step chunks

- "Pour tasks" are identified by task_name containing the substring "pour"
  (case-insensitive).

- "Multimodality" per state is measured as:
    variance_drop_ratio = variance_drop / total_variance

- "Trajectory" = all states in a given episode_id, ordered by t_in_episode.

- For ghost trajectories, we plot each chunk in action space using the first
  three action dimensions (with zero padding if A < 3).

Outputs
-------
- outdir/pour_episode_summary.csv
    Per-episode summary stats for Pour tasks, including max variance_drop_ratio
    and a "band" label: low / mid / high.

- outdir/episode_plots/episode_<episode_id>_band_<band>.html
    Interactive Plotly 3D plots for selected episodes across the distribution.

Usage
-----
Example:

  python analyze_pour_multimodality_trajectories.py \
      --metrics_csv $OUTDIR/all_multimodality/metrics_per_state.csv \
      --actions_npz $OUTDIR/actions.npz \
      --outdir $OUTDIR/pour_multimodality_plots \
      --episodes_per_band 3 \
      --task_keyword Pour
"""

import argparse
import os
import sys
import numpy as np
import pandas as pd

from scipy.spatial.distance import cdist
from sklearn.decomposition import PCA

import matplotlib
import plotly.graph_objects as go


# ----------------------------------------------------------------------
# Minkowski variance helpers (same notion as your clustering script)
# ----------------------------------------------------------------------

def _as_2d(x):
    x = np.asarray(x)
    if x.ndim == 1:
        return x.reshape(-1, 1).astype(float, copy=False)
    if x.ndim >= 3:
        x = x.reshape(x.shape[0], -1)
    return x.astype(float, copy=False)


def symmetrized_l2_minkowski_traj(X, Y):
    """
    Trajectory-level distance between X and Y.

    X: (T_x, A)
    Y: (T_y, A)

    d1^2 = sum_i min_j ||x_i - y_j||^2
    d2^2 = sum_j min_i ||y_j - x_i||^2
    d   = sqrt( (d1^2 + d2^2) / 2 )
    """
    X = _as_2d(X)
    Y = _as_2d(Y)
    if X.shape[0] == 0 and Y.shape[0] == 0:
        return 0.0
    if X.shape[0] == 0 or Y.shape[0] == 0:
        return np.inf

    dists_sq = cdist(X, Y, metric="sqeuclidean")  # (T_x, T_y)
    d1_sq = np.sum(np.min(dists_sq, axis=1))      # sum_i min_j ||x_i - y_j||^2
    d2_sq = np.sum(np.min(dists_sq, axis=0))      # sum_j min_i ||y_j - x_i||^2
    return float(np.sqrt(0.5 * (d1_sq + d2_sq)))


def compute_minkowski_distance_matrix(trajectories):
    """
    trajectories: list of arrays, each (T_i, A) = one action chunk (trajectory).
    Returns D[i,j] = symmetrized_l2_minkowski_traj(traj_i, traj_j).
    """
    N = len(trajectories)
    D = np.zeros((N, N), dtype=float)
    for i in range(N):
        for j in range(i + 1, N):
            d = symmetrized_l2_minkowski_traj(trajectories[i], trajectories[j])
            D[i, j] = D[j, i] = d
    return D


def _minkowski_variance_from_D(D):
    """
    D: (N, N) array of symmetrized Minkowski distances between chunks.

    NOTE: Here we use a simple surrogate consistent with your earlier scripts:
      Var := 0.5 * E[d^2]
    """
    D = np.asarray(D, float)
    if D.size == 0:
        return 0.0
    term1 = float(np.mean(D ** 2))          # E[d^2]
    return 0.5 * term1


def total_variance_minkowski(D):
    """Total variance of the set of action chunks, in Minkowski distance space."""
    return _minkowski_variance_from_D(D)


# ----------------------------------------------------------------------
# Embeddings
# ----------------------------------------------------------------------

def embed_state_vector(mean_action_vecs, mode="first3"):
    """
    mean_action_vecs: (N_states, A) array of per-state mean action vectors
                      (averaged over chunks/time).

    Returns: xyz: (N_states, 3) 3D embedding.
    """
    X = np.asarray(mean_action_vecs, dtype=float)
    if X.ndim != 2:
        raise ValueError("mean_action_vecs must be 2D (N_states, A)")

    if mode == "first3":
        if X.shape[1] < 3:
            pad = np.zeros((X.shape[0], 3 - X.shape[1]))
            return np.hstack([X, pad])[:, :3]
        return X[:, :3]
    else:  # PCA fallback if user wants
        pca = PCA(n_components=3, random_state=0)
        Z = pca.fit_transform(X)
        return Z


def embed_chunk_first3(chunk):
    """
    chunk: (T, A) -> (T, 3) using first 3 dimensions, zero-padding if needed.
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


def ratio_to_color_hex(ratio, r_min, r_max):
    """
    Map a variance_drop_ratio to a color using Viridis.
    """
    if r_max > r_min:
        norm = (ratio - r_min) / (r_max - r_min)
    else:
        norm = 0.5  # degenerate case: all ratios equal
    norm = float(np.clip(norm, 0.0, 1.0))
    r, g, b, _ = viridis(norm)
    return f"rgba({int(r*255)}, {int(g*255)}, {int(b*255)}, 1.0)"


def plot_episode_trajectory_with_ghosts(
    episode_id,
    band,
    ep_df,
    actions_arr,
    global_r_min,
    global_r_max,
    out_html,
    max_chunks_per_state=5,
    mode="first3",
):
    """
    ep_df: subset of metrics_df for a single episode_id, sorted by t_in_episode.
           Must contain columns: state_index, t_in_episode, variance_drop_ratio.

    actions_arr: actions[state_index] -> either (Kc, T, A) or (T, A)

    We:
      - build a per-state embedding (mean over all actions at that state)
      - plot the trajectory of states in 3D (in order of t_in_episode)
      - color the states by variance_drop_ratio
      - overlay ghost action-chunk trajectories at each state, in a faint color
        corresponding to that state's ratio.
    """
    # Ensure sorted by t_in_episode
    ep_df = ep_df.sort_values("t_in_episode").reset_index(drop=True)

    state_indices = ep_df["state_index"].astype(int).to_numpy()
    t_steps = ep_df["t_in_episode"].to_numpy()
    ratios = ep_df["variance_drop_ratio"].to_numpy()

    def state_anchor_from_chunks(arr, which="start"):
        """
        arr: (Kc, Tc, A) or (Tc, A)
        which: "start" or "end"
        returns: (A,) anchor vector
        """
        arr = np.asarray(arr, dtype=float)
        if arr.ndim == 3:
            # (Kc, Tc, A)
            if which == "start":
                pts = arr[:, 0, :]      # (Kc, A)
            else:
                pts = arr[:, -1, :]     # (Kc, A)
            return pts.mean(axis=0)
        elif arr.ndim == 2:
            # (Tc, A) treat as one chunk
            if which == "start":
                return arr[0, :]
            else:
                return arr[-1, :]
        else:
            raise ValueError(f"Unsupported action shape: {arr.shape}")


    # ---- Build per-state anchor vectors (boundary-consistent) ----
    anchors = []
    for i, s in enumerate(state_indices):
        if i == 0:
            # first state: beginning of its next action chunk(s)
            anchors.append(state_anchor_from_chunks(actions_arr[s], which="start"))
        else:
            # later states: endpoint of previous state's action chunk(s)
            s_prev = state_indices[i - 1]
            anchors.append(state_anchor_from_chunks(actions_arr[s_prev], which="end"))

    anchors = np.vstack(anchors)  # (N_states, A)
    xyz_states = embed_state_vector(anchors, mode=mode)

    # Build Plotly figure
    fig = go.Figure()

    # Main trajectory line (grey)
    fig.add_trace(go.Scatter3d(
        x=xyz_states[:, 0],
        y=xyz_states[:, 1],
        z=xyz_states[:, 2],
        mode="lines",
        line=dict(color="rgba(150,150,150,0.5)", width=3),
        showlegend=False,
        name="trajectory",
    ))

    # State markers colored by variance_drop_ratio
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
            colorbar=dict(title="variance drop ratio"),
        ),
        text=[f"t={t}, ratio={r:.3f}" for t, r in zip(t_steps, ratios)],
        hoverinfo="text",
        name="states",
    ))

    # Ghost trajectories: chunks at each state
    for idx, s in enumerate(state_indices):
        arr = np.asarray(actions_arr[s], dtype=float)
        ratio_s = float(ratios[idx])
        color_s = ratio_to_color_hex(ratio_s, global_r_min, global_r_max)

        if arr.ndim == 3:
            # (Kc, Tc, A)
            Kc, Tc, A = arr.shape
            chunk_indices = np.arange(Kc)
            if Kc > max_chunks_per_state:
                # sample a few chunks for readability
                rng = np.random.RandomState(0 + int(s))
                chunk_indices = rng.choice(Kc, size=max_chunks_per_state, replace=False)

            for ck in chunk_indices:
                chunk = arr[ck]  # (Tc, A)
                chunk_xyz = embed_chunk_first3(chunk)

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
            # treat as a single chunk
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
            sys.stderr.write(f"[warn] episode {episode_id}: state {s} has unsupported action shape {arr.shape}; skipping ghost chunks.\n")

    fig.update_layout(
        title=f"Episode {episode_id} ({band} multimodality) – states colored by variance_drop_ratio",
        scene=dict(
            xaxis_title="X",
            yaxis_title="Y",
            zaxis_title="Z",
        ),
        margin=dict(l=0, r=0, b=0, t=40),
    )

    fig.write_html(out_html)
    print(f"  wrote {out_html}")


# ----------------------------------------------------------------------
# main
# ----------------------------------------------------------------------

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--metrics_csv", type=str, required=True,
                    help="metrics_per_state.csv from multimodality clustering run")
    ap.add_argument("--actions_npz", type=str, required=True,
                    help="actions.npz used in the clustering run")
    ap.add_argument("--outdir", type=str, required=True,
                    help="output directory for summary + plots")
    ap.add_argument("--task_keyword", type=str, default="Pour",
                    help="substring to identify the task (case-insensitive), e.g. 'Pour'")
    ap.add_argument("--episodes_per_band", type=int, default=3,
                    help="max number of episodes to visualize per band (low/mid/high)")
    ap.add_argument("--embedding_mode", type=str, default="first3",
                    choices=["first3", "pca"],
                    help="how to embed state mean action vectors into 3D")
    args = ap.parse_args()

    os.makedirs(args.outdir, exist_ok=True)
    episode_plot_dir = os.path.join(args.outdir, "episode_plots")
    os.makedirs(episode_plot_dir, exist_ok=True)

    # Load data
    metrics_df = pd.read_csv(args.metrics_csv)
    actions_data = np.load(args.actions_npz, allow_pickle=True)
    actions_arr = actions_data["actions"]

    required_cols = {"state_index", "episode_id", "t_in_episode", "task_name",
                     "total_variance", "variance_drop"}
    missing = required_cols - set(metrics_df.columns)
    if missing:
        raise ValueError(f"metrics_csv is missing required columns: {missing}")

    # Compute variance_drop_ratio (if not already present)
    if "variance_drop_ratio" not in metrics_df.columns:
        tv = metrics_df["total_variance"].to_numpy()
        vd = metrics_df["variance_drop"].to_numpy()
        with np.errstate(divide="ignore", invalid="ignore"):
            ratio = np.where(tv > 0, vd / tv, np.nan)
        metrics_df["variance_drop_ratio"] = ratio

    # Filter to Pour tasks
    task_kw = args.task_keyword.lower()
    print(f"Searching for {task_kw} tasks in task_name OR instruction...")

    # Need to cast to string in case the column contains NaN or non-string values.
    task_str = metrics_df["task_name"].astype(str)
    instr_str = metrics_df["instruction"].astype(str)

    pour_mask = (
        task_str.str.contains(task_kw, case=False, na=False)
        |
        instr_str.str.contains(task_kw, case=False, na=False)
    )

    pour_df = metrics_df[pour_mask]

    if pour_df.empty:
        print(f"No rows found with task_name or instruction containing '{args.task_keyword}'. Nothing to do.")
        return
    else:
        print(f"Found {len(pour_df)} {task_kw} rows.")

    # Get global min/max ratio over Pour states (for consistent color scaling)
    valid_ratios = pour_df["variance_drop_ratio"].to_numpy()
    valid_ratios = valid_ratios[~np.isnan(valid_ratios)]
    if valid_ratios.size == 0:
        print("No valid variance_drop_ratio values for Pour tasks.")
        return

    global_r_min = float(np.min(valid_ratios))
    global_r_max = float(np.max(valid_ratios))
    print(f"Global variance_drop_ratio range for Pour tasks: [{global_r_min:.4f}, {global_r_max:.4f}]")

    # Per-episode summary over Pour tasks
    grouped = pour_df.groupby("episode_id")
    summary_rows = []
    for ep_id, g in grouped:
        ratios_ep = g["variance_drop_ratio"].to_numpy()
        ratios_valid = ratios_ep[~np.isnan(ratios_ep)]
        if ratios_valid.size == 0:
            continue
        n_states = len(g)
        max_ratio = float(np.max(ratios_valid))
        mean_ratio = float(np.mean(ratios_valid))
        p80 = float(np.quantile(ratios_valid, 0.80))
        summary_rows.append({
            "episode_id": ep_id,
            "n_states": n_states,
            "max_ratio": max_ratio,
            "mean_ratio": mean_ratio,
            "p80_ratio": p80,
        })

    if not summary_rows:
        print("No episodes with valid variance_drop_ratio for Pour tasks.")
        return

    summary_df = pd.DataFrame(summary_rows)

    # Define bands in terms of max_ratio (you could switch to p80_ratio if desired)
    q_low = summary_df["max_ratio"].quantile(0.33)
    q_high = summary_df["max_ratio"].quantile(0.66)

    def _band(max_r):
        if max_r <= q_low:
            return "low"
        elif max_r >= q_high:
            return "high"
        else:
            return "mid"

    summary_df["band"] = summary_df["max_ratio"].apply(_band)

    summary_csv = os.path.join(args.outdir, "pour_episode_summary.csv")
    summary_df.to_csv(summary_csv, index=False)
    print(f"Wrote per-episode summary to {summary_csv}")
    print(f"Band thresholds (based on max_ratio): low <= {q_low:.4f}, high >= {q_high:.4f}")

    # For each band, pick up to episodes_per_band episodes (sorted by max_ratio)
    for band in ["high", "mid", "low"]:
        band_df = summary_df[summary_df["band"] == band].copy()
        if band_df.empty:
            print(f"No episodes in band '{band}'.")
            continue

        if band == "high":
            band_df.sort_values("max_ratio", ascending=False, inplace=True)
        elif band == "low":
            band_df.sort_values("max_ratio", ascending=True, inplace=True)
        else:  # mid
            # sort by mean_ratio near median
            median_mid = band_df["max_ratio"].median()
            band_df["dist_from_mid"] = (band_df["max_ratio"] - median_mid).abs()
            band_df.sort_values("dist_from_mid", ascending=True, inplace=True)

        selected_episodes = band_df.head(args.episodes_per_band)

        print(f"\nBand '{band}': plotting up to {len(selected_episodes)} episodes...")
        for row in selected_episodes.itertuples():
            ep_id = row.episode_id
            ep_subset = pour_df[pour_df["episode_id"] == ep_id].copy()

            # ensure we have t_in_episode and state_index
            if ep_subset["t_in_episode"].isna().all():
                print(f"  skipping episode {ep_id}: all t_in_episode are NaN.")
                continue

            out_html = os.path.join(
                episode_plot_dir,
                f"episode_{str(ep_id)}_band_{band}.html",
            )

            plot_episode_trajectory_with_ghosts(
                episode_id=ep_id,
                band=band,
                ep_df=ep_subset,
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