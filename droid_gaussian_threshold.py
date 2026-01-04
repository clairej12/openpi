#!/usr/bin/env python3
"""
postprocess_gaussian_threshold.py

Read the clustered/metric outputs from the main spectral script and filter
(state, k) combinations whose variance_drop_ratio beats a Gaussian baseline
(for the same (num_points, action_dim, k) and matched trajectory structure).

Assumptions:
- You already ran the main script and you have:
    outdir/metrics_per_state.csv
    outdir/per_state/state_000000.npz  (etc.)
    actions_npz  (same one used for clustering)

- Per-state npz files contain: rows (list of dicts) and best (dict) – as in your script.

We:
1) load metrics
2) compute Gaussian *reference* baselines per (num_points, action_dim, k),
   using synthetic trajectory "chunks" constructed as:
       w ~ mixture of N(mu_c, I), c in {0..k-1}
       a[t] = mu_c + v_scales[t] * eps_t
   where:
       - T and A (per_step_dim) are inferred from a real state's actions
         with that combo,
       - v_t is the empirical variance at time t from that real state,
       - v_T is the variance at the final time step.
   We then run Minkowski + spectral clustering on these Gaussian chunks and
   compute the average variance_drop_ratio = (tv - wvar) / tv.
3) compute variance_drop_ratio for the real states
4) apply threshold: ratio_actual >= multiplier * ratio_baseline
5) save passing rows
6) produce:
   - per-k violin plots of variance_drop_ratio by path type, with baseline ratio
   - plots of baseline ratios vs k across all combos
   - Plotly 3D interactive visualizations for the first --plot_pass_top passes
"""

import argparse
import os
import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib

from sklearn.cluster import SpectralClustering
from sklearn.decomposition import PCA
from scipy.spatial.distance import cdist

import plotly.graph_objects as go
from plotly.subplots import make_subplots
from matplotlib.lines import Line2D
from plotly.colors import qualitative

# ------------------------------------------------------------
# Minkowski distance & variance helpers (chunk-based)
# ------------------------------------------------------------

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
    Here we use a simplified form:
      Var := 0.5 * E[d^2]
    where expectations are empirical over the N chunks.
    """
    D = np.asarray(D, float)
    if D.size == 0:
        return 0.0
    term1 = float(np.mean(D ** 2))          # E[d^2]
    return 0.5 * term1


def total_variance_minkowski(D):
    """Total variance of the set of action chunks, in Minkowski distance space."""
    return _minkowski_variance_from_D(D)


def weighted_incluster_variance_minkowski(D, labels):
    """
    D: (N, N) Minkowski distance matrix between chunks.
    labels: cluster labels per chunk (length N).

    Computes size-weighted average of cluster variances, with variance
    defined via _minkowski_variance_from_D on each cluster submatrix.
    """
    labels = np.asarray(labels)
    N = D.shape[0]
    if N == 0:
        return 0.0

    wvar = 0.0
    for c in np.unique(labels):
        idx = np.where(labels == c)[0]
        if idx.size == 0:
            continue
        Dc = D[np.ix_(idx, idx)]
        var_c = _minkowski_variance_from_D(Dc)
        wvar += (idx.size / N) * var_c
    return float(wvar)


# ------------------------------------------------------------
# Plotly-based 3D plotting helpers
# ------------------------------------------------------------

def to_xyz(actions, mode="first3", pca_model=None, kinematic_map=None):
    X = np.asarray(actions, dtype=float)
    if mode == "first3":
        if X.shape[1] < 3:
            pad = np.zeros((X.shape[0], 3 - X.shape[1]))
            return np.hstack([X, pad])[:, :3]
        return X[:, :3]
    elif mode == "custom":
        if kinematic_map is None:
            raise ValueError("custom mode requires kinematic_map")
        return np.asarray(kinematic_map(X), dtype=float)
    else:  # "pca"
        if pca_model is None:
            pca_model = PCA(n_components=3, random_state=0)
            Z = pca_model.fit_transform(X)
        else:
            Z = pca_model.transform(X)
        return Z


viridis = matplotlib.colormaps.get_cmap("viridis")


def cval_to_hex(cval):
    r, g, b, _ = viridis(cval)
    return f"rgb({int(r*255)}, {int(g*255)}, {int(b*255)})"


def plot_actions_xyz(
    xyz,
    labels,
    index_rows,
    html_path="",
    title="",
    point_size=2,
    line_width=4.0,
    line_alpha=0.35,
    # NEW:
    chunk_start_xyz=None,          # dict: chunk_id -> (3,) start point from real states
    ghost_visible="true",          # "legendonly" | "true" | "false"
):
    """
    3D scatter of points (actions) + togglable ghost trajectories (lines),
    and start markers that come from real DROID states.
    """
    xyz = np.asarray(xyz)
    labels = np.asarray(labels).reshape(-1)

    if labels.shape[0] != xyz.shape[0]:
        raise ValueError("labels length mismatch")
    if index_rows is None or len(index_rows) != xyz.shape[0]:
        raise ValueError("index_rows length mismatch with xyz")

    sample_idx = np.array([r[1] for r in index_rows], dtype=int)  # chunk id per point
    action_idx = np.array([r[2] for r in index_rows], dtype=int)  # time id per point

    # Consistent color map on clusters
    uniq = np.unique(labels)
    uniq_sorted = np.sort(uniq)
    palette = qualitative.Dark24 or ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728",
                                     "#9467bd", "#8c564b", "#e377c2", "#7f7f7f",
                                     "#bcbd22", "#17becf"]
    color_map = {lab: palette[i % len(palette)] for i, lab in enumerate(uniq_sorted)}

    fig = go.Figure()

    # --- build per-chunk ordered polylines + dominant cluster per chunk ---
    chunk_ids = np.unique(sample_idx)
    chunk_dom = {}
    chunk_poly = {}  # chunk_id -> (T,3) points in time order
    for cid in chunk_ids:
        m = (sample_idx == cid)
        if not np.any(m):
            continue
        order = np.argsort(action_idx[m])
        pts = xyz[m][order]
        labs = labels[m][order]
        if pts.shape[0] == 0:
            continue
        vals, counts = np.unique(labs, return_counts=True)
        dom = int(vals[np.argmax(counts)])
        chunk_dom[cid] = dom
        chunk_poly[cid] = pts

    # --- ghost trajectories: ONE trace per cluster so it’s legend-togglable ---
    # (Plotly can’t do multi-color lines inside one trace; grouping by cluster keeps toggles useful.)
    if ghost_visible == "legendonly":
        vis = "legendonly"
        show_line_legend = True
    elif ghost_visible == "false":
        vis = False
        show_line_legend = False
    else:
        vis = True
        show_line_legend = False

    # --- per-cluster markers + ghost trajectories tied together in legend groups ---
    for lab in uniq_sorted:
        mask = (labels == lab)
        if not np.any(mask):
            continue

        color_hex = color_map[lab]
        legend_group = f"cluster-{lab}"

        # Markers (points) for this cluster
        fig.add_trace(go.Scatter3d(
            x=xyz[mask, 0], y=xyz[mask, 1], z=xyz[mask, 2],
            mode="markers",
            marker=dict(
                size=point_size,
                color=color_hex,
                opacity=0.95,
            ),
            name=f"cluster {lab}",
            legendgroup=legend_group,
            legendgrouptitle_text=f"cluster {lab}",
            showlegend=True,
            visible=True,
        ))

        # aggregate all chunk polylines for this cluster into one trace using NaN separators
        xs, ys, zs = [], [], []
        used_any = False
        for cid, dom in chunk_dom.items():
            if dom != lab:
                continue
            pts = chunk_poly[cid]
            if pts.shape[0] < 2:
                continue
            xs.extend(pts[:, 0].tolist() + [np.nan])
            ys.extend(pts[:, 1].tolist() + [np.nan])
            zs.extend(pts[:, 2].tolist() + [np.nan])
            used_any = True

        if used_any:
            fig.add_trace(go.Scatter3d(
                x=xs, y=ys, z=zs,
                mode="lines",
                line=dict(width=line_width, color=color_hex),
                opacity=line_alpha,
                name=f"cluster {lab} trajectories",
                visible=vis,
                legendgroup=legend_group,
                showlegend=show_line_legend,  # gives a toggle when requested
            ))

    # --- start markers: REAL DROID states (if provided) ---
    if chunk_start_xyz is not None:
        starts_by_cluster = {}
        connect_by_cluster = {}
        start_legend_shown = False
        for cid, dom in chunk_dom.items():
            if cid not in chunk_start_xyz:
                continue
            p_start = np.asarray(chunk_start_xyz[cid], dtype=float).reshape(3,)
            starts_by_cluster.setdefault(dom, set()).add(tuple(p_start.tolist()))

            if cid in chunk_poly:
                p_first = chunk_poly[cid][0]
                xs, ys, zs = connect_by_cluster.setdefault(dom, ([], [], []))
                xs.extend([p_start[0], p_first[0], np.nan])
                ys.extend([p_start[1], p_first[1], np.nan])
                zs.extend([p_start[2], p_first[2], np.nan])

        for dom, pts_set in starts_by_cluster.items():
            pts = np.array(list(pts_set))
            color_hex = color_map[dom]
            legend_group = f"cluster-{dom}"
            fig.add_trace(go.Scatter3d(
                x=pts[:, 0], y=pts[:, 1], z=pts[:, 2],
                mode="markers",
                marker=dict(
                    size=9,
                    symbol="diamond",
                    color=color_hex,
                    opacity=0.98,
                    line=dict(width=2, color="#000000"),
                ),
                name="start state (DROID)",
                legendgroup=legend_group,
                showlegend=not start_legend_shown,
            ))
            start_legend_shown = True

        for dom, (xs, ys, zs) in connect_by_cluster.items():
            color_hex = color_map[dom]
            legend_group = f"cluster-{dom}"
            fig.add_trace(go.Scatter3d(
                x=xs, y=ys, z=zs,
                mode="lines",
                line=dict(width=3, color=color_hex, dash="dot"),
                opacity=0.6,
                name="start to first action",
                legendgroup=legend_group,
                showlegend=False,
            ))

    # --- end markers (optional; still based on action trajectory endpoints) ---
    ends_by_cluster = {}
    for cid, dom in chunk_dom.items():
        pts = chunk_poly[cid]
        ends_by_cluster.setdefault(dom, []).append(pts[-1])
    for dom, pts in ends_by_cluster.items():
        pts = np.stack(pts, axis=0)
        color_hex = color_map[dom]
        legend_group = f"cluster-{dom}"
        fig.add_trace(go.Scatter3d(
            x=pts[:, 0], y=pts[:, 1], z=pts[:, 2],
            mode="markers",
            marker=dict(
                size=3,
                symbol="x",
                color=color_hex,
                opacity=0.9,
            ),
            name="chunk endpoints",
            legendgroup=legend_group,
            showlegend=False,
        ))

    fig.update_layout(
        title=title,
        scene=dict(xaxis_title="X", yaxis_title="Y", zaxis_title="Z"),
        margin=dict(l=0, r=0, b=0, t=40),
        legend=dict(itemsizing="constant"),
    )

    fig.write_html(html_path)

def plot_per_cluster_panels(xyz, per_point_labels, index_rows, n_clusters,
                            html_path, title_prefix="State",
                            point_size=2, line_width=4.0, line_alpha=0.6):
    """
    Plotly multi-panel (subplots) 3D visualization: one scene per cluster.
    """
    xyz = np.asarray(xyz)
    labels = np.asarray(per_point_labels).reshape(-1)
    assert xyz.shape[0] == labels.shape[0] == len(index_rows)

    sample_idx = np.array([r[1] for r in index_rows], dtype=int)
    action_idx = np.array([r[2] for r in index_rows], dtype=int)

    uniq_clusters = np.unique(labels)
    C = int(max(n_clusters, uniq_clusters.max() + 1))

    cols = min(4, C)
    rows = int(np.ceil(C / cols))

    fig = make_subplots(
        rows=rows,
        cols=cols,
        specs=[[{"type": "scene"} for _ in range(cols)] for _ in range(rows)],
        subplot_titles=[f"Cluster {ci}" for ci in range(C)],
    )

    # assign each chunk to its dominant cluster
    chunk_ids = np.unique(sample_idx)
    chunk_to_cluster = {}
    for k in chunk_ids:
        m = (sample_idx == k)
        if not np.any(m):
            continue
        vals, counts = np.unique(labels[m], return_counts=True)
        chunk_to_cluster[k] = int(vals[np.argmax(counts)])

    for ci in range(C):
        row_i = ci // cols + 1
        col_i = ci % cols + 1
        chunks_in_ci = [k for k, lab in chunk_to_cluster.items() if lab == ci]
        if not chunks_in_ci:
            continue

        for k in chunks_in_ci:
            m = (sample_idx == k)
            pts = xyz[m]
            t_ids = action_idx[m]
            order = np.argsort(t_ids)
            pts = pts[order]
            if pts.shape[0] == 0:
                continue

            # points
            fig.add_trace(
                go.Scatter3d(
                    x=pts[:, 0],
                    y=pts[:, 1],
                    z=pts[:, 2],
                    mode="markers",
                    marker=dict(size=point_size),
                    opacity=0.95,
                    showlegend=False,
                ),
                row=row_i,
                col=col_i,
            )

            # line
            if pts.shape[0] >= 2:
                fig.add_trace(
                    go.Scatter3d(
                        x=pts[:, 0],
                        y=pts[:, 1],
                        z=pts[:, 2],
                        mode="lines",
                        line=dict(width=line_width),
                        opacity=line_alpha,
                        showlegend=False,
                    ),
                    row=row_i,
                    col=col_i,
                )

            # start / end markers
            p0 = pts[0]
            p1 = pts[-1]
            fig.add_trace(
                go.Scatter3d(
                    x=[p0[0]],
                    y=[p0[1]],
                    z=[p0[2]],
                    mode="markers",
                    marker=dict(symbol="circle-open", size=3, line=dict(width=1)),
                    showlegend=False,
                ),
                row=row_i,
                col=col_i,
            )
            fig.add_trace(
                go.Scatter3d(
                    x=[p1[0]],
                    y=[p1[1]],
                    z=[p1[2]],
                    mode="markers",
                    marker=dict(symbol="x", size=3),
                    showlegend=False,
                ),
                row=row_i,
                col=col_i,
            )

    fig.update_layout(
        title=f"{title_prefix}: per-cluster chunk views",
        margin=dict(l=0, r=0, b=0, t=40),
    )

    fig.write_html(html_path)
# ------------------------------------------------------------
# Gaussian mixture trajectory generation + clustering
# ------------------------------------------------------------

def _cluster_chunks_spectral_from_minkowski(trajectories, k_cluster, sigma=None, random_state=0):
    """
    Returns:
      labels: (N,) predicted cluster label per chunk
      D_mink: (N,N) Minkowski distance matrix
    """
    D_mink = compute_minkowski_distance_matrix(trajectories)
    if D_mink.shape[0] < k_cluster or k_cluster < 1:
        return None, D_mink

    pos = D_mink[D_mink > 0]
    sigma_used = float(np.median(pos)) if (sigma is None and pos.size) else (float(sigma) if sigma is not None else 1.0)

    A = np.exp(-D_mink ** 2 / (2.0 * sigma_used ** 2))
    np.fill_diagonal(A, 1.0)

    cl = SpectralClustering(
        n_clusters=k_cluster,
        affinity="precomputed",
        assign_labels="kmeans",
        random_state=random_state,
    )
    labels = cl.fit_predict(A)
    return labels, D_mink


def plot_gaussian_baseline_gt_vs_pred(
    out_png,
    endpoints,         # (N, per_step_dim) or any per-chunk embedding input
    gt_modes,          # (N,)
    pred_labels,       # (N,)
    title="Gaussian baseline",
):
    """
    Marker shape = ground truth mixture mode
    Color         = predicted cluster label

    Uses PCA->2D on endpoints for visualization.
    """
    endpoints = np.asarray(endpoints, dtype=float)
    gt_modes = np.asarray(gt_modes, dtype=int)
    pred_labels = np.asarray(pred_labels, dtype=int)

    # PCA to 2D
    Z = PCA(n_components=2, random_state=0).fit_transform(endpoints)

    # marker shapes for GT
    gt_markers = ["o", "s", "^", "D", "P", "X", "v", "<", ">"]
    uniq_gt = np.unique(gt_modes)
    uniq_pred = np.unique(pred_labels)

    # colors for predicted clusters
    cmap = plt.get_cmap("tab10")
    pred_to_color = {lab: cmap(int(lab) % 10) for lab in uniq_pred}

    plt.figure(figsize=(7.5, 6))
    ax = plt.gca()

    for g in uniq_gt:
        m = (gt_modes == g)
        if not np.any(m):
            continue

        # within this GT mode, color by predicted label
        for lab in uniq_pred:
            mm = m & (pred_labels == lab)
            if not np.any(mm):
                continue
            ax.scatter(
                Z[mm, 0], Z[mm, 1],
                s=45,
                marker=gt_markers[int(g) % len(gt_markers)],
                c=[pred_to_color[lab]],
                alpha=0.85,
                edgecolors="none",
            )

    ax.set_title(title)
    ax.set_xlabel("PCA-1")
    ax.set_ylabel("PCA-2")
    ax.grid(True, alpha=0.25)

    # legend: GT shapes
    gt_handles = [
        Line2D([0], [0], marker=gt_markers[int(g) % len(gt_markers)], color="k",
               linestyle="None", markersize=8, label=f"GT mode {int(g)}")
        for g in uniq_gt
    ]

    # legend: Pred colors
    pred_handles = [
        Line2D([0], [0], marker="o", color=pred_to_color[lab],
               linestyle="None", markersize=8, label=f"Pred cluster {int(lab)}")
        for lab in uniq_pred
    ]

    # two legends (stacked) placed outside the plot so they don't cover points
    leg1 = ax.legend(
        handles=gt_handles,
        title="Ground truth (shape)",
        loc="upper left",
        bbox_to_anchor=(1.02, 1.0),
        frameon=False,
        borderaxespad=0.0,
    )
    ax.add_artist(leg1)
    ax.legend(
        handles=pred_handles,
        title="Predicted (color)",
        loc="upper left",
        bbox_to_anchor=(1.02, 0.55),
        frameon=False,
        borderaxespad=0.0,
    )

    plt.tight_layout(rect=[0, 0, 0.8, 1])  # leave room on the right for legends
    plt.savefig(out_png, dpi=150)
    plt.close()

# ------------------------------------------------------------
# read per-state file written by the clustering script
# ------------------------------------------------------------
def load_per_state_file(path):
    data = np.load(path, allow_pickle=True)
    rows = list(data["rows"].tolist())
    best = None
    if "best" in data and data["best"].size > 0:
        first = data["best"][0]
        if isinstance(first, dict):
            best = first
        else:
            best = first.item()
    return rows, best

import re, glob

def _discover_lerobot_parquets(data_root: str):
    """
    Yield parquet paths under data_root. Tries common LeRobot DROID layouts:
      - data/chunk-*/file-*.parquet
      - data/chunk-*/.parquet
      - data/*.parquet
    Falls back to a recursive **/*.parquet search if nothing matches.
    """
    patterns = [
        os.path.join(data_root, "data", "chunk-*", "file-*.parquet"),
        os.path.join(data_root, "data", "chunk-*", "*.parquet"),
        os.path.join(data_root, "data", "*.parquet"),
    ]
    seen = set()
    for pattern in patterns:
        for path in sorted(glob.glob(pattern)):
            if path in seen:
                continue
            seen.add(path)
            yield path

    if seen:
        return

    # Fallback: recurse if the dataset is organized differently.
    for path in sorted(glob.glob(os.path.join(data_root, "**", "*.parquet"), recursive=True)):
        if path in seen:
            continue
        seen.add(path)
        yield path

def _parse_episode_id_to_index(ep_id: str) -> int:
    # expects "ep000123" etc.
    if ep_id is None:
        raise ValueError("episode_id is None")
    m = re.match(r"ep(\d+)", str(ep_id).strip())
    if not m:
        raise ValueError(f"Bad episode_id format: {ep_id} (expected ep######)")
    return int(m.group(1))

def _pick_first_existing_col(df, candidates):
    for c in candidates:
        if c in df.columns:
            return c
    return None

def load_lerobot_states_for_keys(
    lerobot_root: str,
    needed_keys,                # set of (episode_index:int, frame_index:int)
    state_col_candidates=None,  # list of column names to try
    max_rows_per_parquet=None,
    frame_tolerance: int = 0,   # allow +/- this many frames as fallback
):
    """
    Returns:
      state_map[(episode_index, frame_index)] = np.ndarray shape (S,)
    Only loads rows needed for visualization (top passes). If frame_tolerance>0,
    will match the closest available frame within +/- tolerance for an episode.
    """
    def _coerce_to_int(arr):
        out = []
        for v in arr:
            if pd.isna(v):
                out.append(None)
                continue
            try:
                out.append(int(v))
                continue
            except Exception:
                pass
            m = re.search(r"(\d+)", str(v))
            out.append(int(m.group(1)) if m else None)
        return np.array([(-1 if v is None else int(v)) for v in out], dtype=int)

    if state_col_candidates is None:
        # DROID/LeRobot commonly has one of these; adjust if you know exact column.
        state_col_candidates = [
            "observation.state",
            "observation/robot_state",
            "robot_state",
            "state",
            "observation.qpos",
            "qpos",
        ]

    needed_keys = set((int(e), int(t)) for (e, t) in needed_keys)
    needed_eps = set(e for (e, _) in needed_keys)
    needed_by_ep = {}
    for (e, t) in needed_keys:
        needed_by_ep.setdefault(int(e), set()).add(int(t))

    state_map = {}
    if not needed_keys:
        return state_map

    parquets = list(_discover_lerobot_parquets(lerobot_root))
    if not parquets:
        sys.stderr.write(f"[lerobot][warn] no parquet files found under {lerobot_root}\n")

    for parquet_path in parquets:
        df = pd.read_parquet(parquet_path)
        if max_rows_per_parquet is not None and len(df) > max_rows_per_parquet:
            df = df.iloc[:max_rows_per_parquet].copy()

        ep_col = _pick_first_existing_col(df, ["episode_index", "episode_id", "episode", "episode_uid"])
        fr_col = _pick_first_existing_col(df, ["frame_index", "frame_id", "frame", "timestep", "step_index", "step_id"])
        if ep_col is None or fr_col is None:
            continue

        # quick prune: only episodes we need
        eps = pd.Series(_coerce_to_int(df[ep_col].to_numpy()), index=df.index)
        keep_ep = eps.isin(list(needed_eps))
        if not keep_ep.any():
            continue
        df = df.loc[keep_ep].copy()

        st_col = _pick_first_existing_col(df, state_col_candidates)
        if st_col is None:
            # no usable state column in this parquet
            continue

        # iterate only over needed frames for those episodes
        ep_idx = pd.Series(_coerce_to_int(df[ep_col].to_numpy()), index=df.index).to_numpy()
        fr_idx = pd.Series(_coerce_to_int(df[fr_col].to_numpy()), index=df.index).to_numpy()
        st_vals = df[st_col].to_numpy()

        for i in range(len(df)):
            key = (int(ep_idx[i]), int(fr_idx[i]))
            ep = key[0]
            fr = key[1]
            # exact match
            if key in needed_keys and key not in state_map:
                v = st_vals[i]
                arr = np.asarray(v, dtype=float).reshape(-1)
                state_map[key] = arr
                continue

            # tolerant match (closest within +/- frame_tolerance)
            if frame_tolerance > 0 and ep in needed_by_ep:
                targets = needed_by_ep[ep]
                # quick reject if no target near this frame
                near = [t for t in targets if abs(fr - t) <= frame_tolerance]
                if not near:
                    continue
                # choose nearest target frame
                target = sorted(near, key=lambda t: abs(fr - t))[0]
                target_key = (ep, target)
                if target_key in state_map:
                    continue
                v = st_vals[i]
                arr = np.asarray(v, dtype=float).reshape(-1)
                state_map[target_key] = arr

        if len(state_map) >= len(needed_keys):
            break

    return state_map

# ------------------------------------------------------------
# Gaussian *trajectory* baseline for ratio with mixture + variance profile
# ------------------------------------------------------------

def chunk_embedding_from_trajectory(traj: np.ndarray, method: str = "mean") -> np.ndarray:
    """
    traj: (T, A)
    returns: (A,) for mean/final, or (T*A,) for flatten
    """
    traj = np.asarray(traj, dtype=float)
    if method == "mean":
        return traj.mean(axis=0)
    if method == "final":
        return traj[-1]
    if method == "flatten":
        return traj.reshape(-1)
    raise ValueError(f"Unknown embedding method: {method}")

def _generate_gaussian_mixture_trajectories_iid_per_timestep(
    num_chunks, per_step_dim, chunk_len, n_modes, v_scales, rng,
    sep=3.0,  # mode separation in std units
):
    """
    Each chunk i:
      choose mode c
      for each t: a[t] = mu_c + v_scales[t] * eps_t,  eps_t ~ N(0, I)
    Returns:
      trajectories: list of (T, A)
      gt_modes: (N,) component id per chunk
    """
    v_scales = np.asarray(v_scales, dtype=float)
    if v_scales.shape[0] != chunk_len:
        if v_scales.shape[0] > chunk_len:
            v_scales = v_scales[:chunk_len]
        else:
            v_scales = np.pad(v_scales, (0, chunk_len - v_scales.shape[0]), mode="edge")

    # scalar means spaced sep apart (broadcasted to all dims, like your original)
    if n_modes <= 1:
        means = np.array([0.0], dtype=float)
        n_modes_eff = 1
    else:
        A = per_step_dim
        d_target = 6.0      # try 4.0, 6.0, 8.0
        sep = d_target / np.sqrt(A)

        means = (np.arange(n_modes, dtype=float) - (n_modes - 1) / 2.0) * sep
        n_modes_eff = n_modes

    trajectories = []
    gt_modes = np.zeros(num_chunks, dtype=int)

    for i in range(num_chunks):
        c = int(rng.randint(0, n_modes_eff))
        mu = float(means[c])

        # iid Gaussian noise per timestep
        eps = rng.randn(chunk_len, per_step_dim)  # (T, A)
        alpha = np.linspace(0.0, 1.0, chunk_len)[:, None]  # (T,1), alpha[0]=0
        traj = (alpha * mu) + (v_scales[:, None] * eps)
        
        trajectories.append(traj.astype(float))
        gt_modes[i] = c

    return trajectories, gt_modes

def gaussian_baseline_ratio(num_chunks,
                            per_step_dim,
                            chunk_len,
                            k_cluster,
                            n_modes,
                            v_scales,
                            n_trials=30,
                            rng=None,
                            sigma=None):
    """
    Generate from an n_modes Gaussian mixture (endpoint modes),
    but cluster with k_cluster (spectral clustering).
    """
    if rng is None:
        rng = np.random.RandomState(0)

    v_scales = np.asarray(v_scales, dtype=float)
    if chunk_len <= 0:
        return 0.0
    if v_scales.shape[0] != chunk_len:
        if v_scales.shape[0] > chunk_len:
            v_scales = v_scales[:chunk_len]
        else:
            v_scales = np.pad(v_scales, (0, chunk_len - v_scales.shape[0]), mode="edge")

    ratios = []
    for _ in range(n_trials):
        trajectories, _gt = _generate_gaussian_mixture_trajectories_iid_per_timestep(
            num_chunks=num_chunks,
            per_step_dim=per_step_dim,
            chunk_len=chunk_len,
            n_modes=n_modes,
            v_scales=v_scales,
            rng=rng,
            sep=3.0,
        )

        D_mink = compute_minkowski_distance_matrix(trajectories)
        tv = total_variance_minkowski(D_mink)
        if tv <= 0:
            continue

        pos = D_mink[D_mink > 0]
        sigma_used = float(np.median(pos)) if (sigma is None and pos.size) else (sigma or 1.0)
        A = np.exp(-D_mink ** 2 / (2.0 * sigma_used ** 2))
        np.fill_diagonal(A, 1.0)

        if num_chunks < k_cluster or k_cluster < 1:
            continue

        try:
            cl = SpectralClustering(
                n_clusters=k_cluster,
                affinity="precomputed",
                assign_labels="kmeans",
                random_state=0,
            )
            labels = cl.fit_predict(A)
        except Exception:
            continue

        wvar = weighted_incluster_variance_minkowski(D_mink, labels)
        ratios.append((tv - wvar) / tv)

    return float(np.mean(ratios)) if ratios else 0.0

# ------------------------------------------------------------
# main
# ------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--metrics_csv", type=str, required=True,
                    help="metrics_per_state.csv from the clustering run")
    ap.add_argument("--actions_npz", type=str, required=True,
                    help="same actions_npz used during clustering")
    ap.add_argument("--outdir", type=str, required=True,
                    help="where to write filtered CSV + plots")
    ap.add_argument("--per_state_dir", type=str, default=None,
                    help="directory containing per_state/*.npz; "
                         "defaults to dirname(metrics_csv)/per_state")
    ap.add_argument("--gaussian_trials", type=int, default=30)
    ap.add_argument("--gaussian_max_points", type=int, default=3000)  # kept for API compatibility, not used now
    ap.add_argument("--gaussian_multiplier", type=float, default=1.0,
                    help="threshold = multiplier * gaussian_baseline_ratio")
    ap.add_argument("--plot_pass_top", type=int, default=10,
                    help="visualize first N passing rows")
    ap.add_argument("--sigma", type=float, default=None,
                    help="sigma for RBF kernel; if None, use median heuristic")
    ap.add_argument("--ee_mode", type=str, default="first3",
                    choices=["first3", "pca", "custom"])
    ap.add_argument("--lerobot_root", type=str, default=None,
                    help="LeRobot DROID dataset root (contains data/chunk-*/file-*.parquet)")
    ap.add_argument("--lerobot_state_col", type=str, default=None,
                    help="If set, use this parquet column as the state vector (overrides candidates).")
    ap.add_argument("--lerobot_max_rows_per_parquet", type=int, default=None,
                    help="Optional debug cap per parquet read.")
    ap.add_argument("--lerobot_frame_tolerance", type=int, default=0,
                    help="Allow +/- this many frames when looking up a state. Useful if "
                         "metrics t_in_episode is offset from parquet frame_index.")
    ap.add_argument("--ghost_visible", type=str, default="true",
                    choices=["true", "legendonly", "false"])
    args = ap.parse_args()

    os.makedirs(args.outdir, exist_ok=True)
    if args.per_state_dir is None:
        args.per_state_dir = os.path.join(os.path.dirname(args.metrics_csv), "per_state")

    metrics_df = pd.read_csv(args.metrics_csv)
    actions_data = np.load(args.actions_npz, allow_pickle=True)
    actions_arr = actions_data["actions"]

    # --------------------------------------------------------
    # Compute real variance_drop_ratio for all rows
    # --------------------------------------------------------
    tv = metrics_df["total_variance"].to_numpy()
    vd = metrics_df["variance_drop"].to_numpy()
    with np.errstate(divide="ignore", invalid="ignore"):
        ratio = np.where(tv > 0, vd / tv, np.nan)
    metrics_df["variance_drop_ratio"] = ratio

    # --------------------------------------------------------
    # Build (num_points, action_dim, k) -> (chunk_len, per_step_dim, v_scales)
    # using a representative state for each combo
    # --------------------------------------------------------
    combo_to_params = {}
    for _, row in metrics_df.iterrows():
        Np = int(row["num_points"])
        d = int(row["action_dim"])
        k = int(row["k"])
        state_idx = int(row["state_index"])
        key = (Np, d, k)
        if key in combo_to_params:
            continue

        arr = np.asarray(actions_arr[state_idx])
        if arr.ndim == 3:
            # (Kc, Tc, A)
            Kc, Tc, A = arr.shape
            chunk_len = Tc
            per_step_dim = A

            # compute per-time-step variance over all chunks and dims
            v_t = []
            for t in range(Tc):
                slice_t = arr[:, t, :].reshape(-1)
                v_t.append(np.var(slice_t))
            v_t = np.asarray(v_t, dtype=float)

            # guard against all-zero variance
            v_T = v_t[-1] if v_t[-1] > 0 else (np.max(v_t) if np.max(v_t) > 0 else 1.0)
            # v_t is a variance; we need STD-ratio so that Var(a_t)= (scale_t^2)*Var(w)
            # and Var(w)=1 at final step, so scale_T must be 1.
            v_scales = np.sqrt(v_t / v_T)

            # ---- DEBUG PRINT: per time-step variance and scales ----
            print(
                f"[combo params] (num_points={Np}, action_dim={d}, k={k}) "
                f"state={state_idx}, chunk_len={chunk_len}, per_step_dim={per_step_dim}"
            )
            print(f"  v_t (variance):   {np.array2string(v_t, precision=4)}")
            print(f"  std_scales sqrt(v_t/v_T): {np.array2string(v_scales, precision=4)}")

        elif arr.ndim == 2:
            # (T, A); in the main script, each time step is a 1-step trajectory
            T_steps, A = arr.shape
            chunk_len = 1
            per_step_dim = A
            v_scales = np.array([1.0], dtype=float)

            print(
                f"[combo params] (num_points={Np}, action_dim={d}, k={k}) "
                f"state={state_idx}, chunk_len={chunk_len}, per_step_dim={per_step_dim}"
            )
            print("  2D actions: using single-step chunk; v_scales=[1.0]")

        else:
            sys.stderr.write(
                f"[warn] state {state_idx}: unsupported action shape {arr.shape}; "
                "skipping this combo for baseline.\n"
            )
            continue

        combo_to_params[key] = (chunk_len, per_step_dim, v_scales)

    print(f"Found {len(combo_to_params)} unique (num_points, action_dim, k) combos for Gaussian baseline")

    rng = np.random.RandomState(0)
    combo_to_baseline_ratio = {}
    baseline_records = []  # for plotting baselines vs k

    # --------------------------------------------------------
    # Compute Gaussian baseline variance_drop_ratio per combo
    # --------------------------------------------------------
    baseline_dir = os.path.join(args.outdir, "gaussian_baseline_plots")
    os.makedirs(baseline_dir, exist_ok=True)

    # put near the top of main(), after baseline_dir is defined
    viz_dir = os.path.join(baseline_dir, "baseline_cluster_viz")
    os.makedirs(viz_dir, exist_ok=True)
    VIZ_MAX_EXAMPLES = 12
    viz_count = 0

    baseline_records = []  # for plotting baselines vs k with multiple n

    # Cover all k values present in the data so baselines exist for the plotted ks.
    MAX_K = max(10, int(metrics_df["k"].max()))
    MAX_N = 10

    combo_to_baseline_ratio = {}  # (Np, d, k, n) -> ratio
    combo_to_threshold_ratio = {} # (Np, d, k) -> ratio used for threshold (default n=k)

    for (Np, d, k), (chunk_len, per_step_dim, v_scales) in combo_to_params.items():
        if k < 1 or k > MAX_K:
            continue

        for n_modes in range(1, MAX_N + 1):
            base_ratio = gaussian_baseline_ratio(
                num_chunks=Np,
                per_step_dim=per_step_dim,
                chunk_len=chunk_len,
                k_cluster=k,
                n_modes=n_modes,
                v_scales=v_scales,
                n_trials=args.gaussian_trials,
                rng=rng,
                sigma=args.sigma,
            )
            combo_to_baseline_ratio[(Np, d, k, n_modes)] = base_ratio
            baseline_records.append({
                "num_points": Np,
                "action_dim": d,
                "k": k,
                "n_modes": n_modes,
                "baseline_ratio": base_ratio,
            })

            # viz (must be INSIDE the n_modes loop)
            if viz_count < VIZ_MAX_EXAMPLES and (n_modes == k):
                rng_viz = np.random.RandomState(123 + 1000 * k + 10 * n_modes + (Np % 97))

                trajectories, gt_modes = _generate_gaussian_mixture_trajectories_iid_per_timestep(
                    num_chunks=Np,
                    per_step_dim=per_step_dim,
                    chunk_len=chunk_len,
                    n_modes=n_modes,
                    v_scales=v_scales,
                    rng=rng_viz,
                    sep=3.0,
                )

                pred_labels, _ = _cluster_chunks_spectral_from_minkowski(
                    trajectories,
                    k_cluster=k,
                    sigma=args.sigma,
                    random_state=0,
                )

                if pred_labels is not None:
                    EMBED_METHOD = "mean"
                    embed = np.stack(
                        [chunk_embedding_from_trajectory(tr, EMBED_METHOD) for tr in trajectories],
                        axis=0
                    )

                    out_png = os.path.join(
                        viz_dir,
                        f"gtshape_predcolor_{EMBED_METHOD}_N{Np}_d{d}_k{k}_n{n_modes}.png"
                    )

                    plot_gaussian_baseline_gt_vs_pred(
                        out_png,
                        endpoints=embed,
                        gt_modes=gt_modes,
                        pred_labels=pred_labels,
                        title=f"Gaussian baseline | N={Np} d={d} | GT n={n_modes} (shape) | spectral k={k} (color)",
                    )
                    viz_count += 1

        # threshold baseline should be set OUTSIDE viz, always:
        combo_to_threshold_ratio[(Np, d, k)] = combo_to_baseline_ratio[(Np, d, k, k)]
    # --------------------------------------------------------
    # Apply threshold using variance_drop_ratio
    # --------------------------------------------------------
    baseline_list = []
    pass_mask = []
    ratios = metrics_df["variance_drop_ratio"].to_numpy()

    for idx, row in metrics_df.iterrows():
        key = (int(row["num_points"]), int(row["action_dim"]), int(row["k"]))
        base_ratio = combo_to_threshold_ratio.get(key, 0.0)
        baseline_list.append(base_ratio)
        actual_ratio = ratios[idx]
        passes = (not np.isnan(actual_ratio)) and (actual_ratio >= args.gaussian_multiplier * base_ratio)
        pass_mask.append(passes)

    metrics_df["gaussian_baseline_ratio"] = baseline_list
    metrics_df["pass_gaussian"] = pass_mask

    pass_df = metrics_df[metrics_df["pass_gaussian"] == True].copy()  # noqa: E712
    # sort by "how much it beats" the threshold (in ratio space)
    pass_df["beat_margin"] = pass_df["variance_drop_ratio"] - \
        (args.gaussian_multiplier * pass_df["gaussian_baseline_ratio"])
    pass_df.sort_values("beat_margin", ascending=False, inplace=True)

    pass_csv = os.path.join(args.outdir, "gaussian_passes.csv")
    metrics_df.to_csv(os.path.join(args.outdir, "metrics_with_gaussian.csv"), index=False)
    pass_df.to_csv(pass_csv, index=False)
    print(f"Saved filtered passes to: {pass_csv}")
    print(f"{len(pass_df)} state+k rows passed the Gaussian-based threshold.")

    # --------------------------------------------------------
    # Plot all Gaussian baselines vs k (with k clearly indicated)
    # --------------------------------------------------------

    if baseline_records:
        base_df = pd.DataFrame(baseline_records)

        # 1) Scatter of all baseline ratios vs k (each combo)
        plt.figure(figsize=(7, 5))
        plt.scatter(
            base_df["k"].astype(int),
            base_df["baseline_ratio"],
            s=20,
            alpha=0.7,
        )
        plt.xlabel("k (number of clusters)")
        plt.ylabel("Gaussian baseline variance_drop_ratio")
        plt.title("Gaussian baseline variance_drop_ratio vs k (all combos)")
        plt.grid(True, alpha=0.3)
        out_path_scatter = os.path.join(baseline_dir, "baseline_ratio_vs_k_scatter.png")
        plt.tight_layout()
        plt.savefig(out_path_scatter, dpi=150)
        plt.close()
        print(f"Saved baseline scatter vs k to {os.path.abspath(out_path_scatter)}")

        # 2) Mean baseline ratio per k (line plot)
        base_df = pd.DataFrame(baseline_records)

        # mean baseline ratio for each (k, n_modes)
        mean_kn = base_df.groupby(["k", "n_modes"])["baseline_ratio"].mean().reset_index()

        plt.figure(figsize=(8, 5))
        for n_modes in sorted(base_df["n_modes"].astype(int).unique().tolist()):
            sub = mean_kn[mean_kn["n_modes"] == n_modes].sort_values("k")
            if sub.empty:
                continue
            plt.plot(
                sub["k"].astype(int),
                sub["baseline_ratio"],
                marker="o",
                linewidth=2,
                label=f"n={n_modes} modes",
            )

        plt.xlabel("k (spectral clustering clusters)")
        plt.ylabel("Mean Gaussian baseline variance_drop_ratio")
        plt.title("Gaussian baseline variance_drop_ratio vs k, for different mixture modes n")
        k_ticks = sorted(base_df["k"].astype(int).unique().tolist())
        plt.xticks(k_ticks)
        plt.grid(True, alpha=0.3)
        plt.legend(frameon=False)

        out_path_mean = os.path.join(baseline_dir, "baseline_ratio_vs_k_mean.png")
        plt.tight_layout()
        plt.savefig(out_path_mean, dpi=150)
        plt.close()
        print(f"Saved mean baseline plot to {os.path.abspath(out_path_mean)}")

    # --------------------------------------------------------
    # Aggregated violin plots: variance_drop_ratio by path type per k
    # --------------------------------------------------------

    # Choose a source column for path type
    # Prefer non-empty task_name; otherwise fall back to instruction
    def _choose_name(row):
        tn = row.get("task_name", "")
        if isinstance(tn, str) and tn.strip():
            return tn
        instr = row.get("instruction", "")
        if isinstance(instr, str) and instr.strip():
            return instr
        return ""

    def _get_path_type_from_row(row):
        name = _choose_name(row)
        s = name.strip()
        if not s:
            return "unknown"
        return s.split()[0]

    # Add path_type column to both full metrics and passes
    metrics_df["path_type"] = metrics_df.apply(_get_path_type_from_row, axis=1)
    pass_df["path_type"] = pass_df.apply(_get_path_type_from_row, axis=1)

    violin_dir = os.path.join(args.outdir, "gaussian_task_violins")
    os.makedirs(violin_dir, exist_ok=True)

    unique_ks = sorted(metrics_df["k"].unique().tolist())
    for k_val in unique_ks:
        # rows that passed Gaussian threshold for this k
        metrics_k = metrics_df[metrics_df["k"] == k_val].copy()
        if metrics_k.empty:
            continue

        path_types = sorted(metrics_k["path_type"].unique().tolist())
        if not path_types:
            continue

        data = []
        labels_pt = []
        ns = []  # for n annotations
        for pt in path_types:
            vals = metrics_k.loc[metrics_k["path_type"] == pt, "variance_drop_ratio"].dropna().values
            if vals.size == 0:
                continue
            data.append(vals)
            labels_pt.append(pt)
            ns.append(vals.size)

        if not data:
            continue

        metrics_k = metrics_df[metrics_df["k"] == k_val]
        # --- compute baseline lines for all n_modes computed upstream, clustering with this k_val ---
        combos = (
            metrics_k[["num_points", "action_dim"]]
            .drop_duplicates()
            .astype(int)
            .to_records(index=False)
        )

        baseline_lines = {}  # n_modes -> mean baseline ratio for this k
        print(f"[baseline] k={k_val}: combos={list(combos)}")
        n_modes_values = sorted(set([key[3] for key in combo_to_baseline_ratio.keys() if key[2] == int(k_val)]))
        for n_modes in n_modes_values:
            vals = []
            for (Np, d) in combos:
                key = (int(Np), int(d), int(k_val), int(n_modes))
                if key in combo_to_baseline_ratio:
                    vals.append(combo_to_baseline_ratio[key])
            baseline_lines[n_modes] = float(np.mean(vals)) if vals else float("nan")
        print(f"[baseline] k={k_val}: baseline_lines={baseline_lines}")

        # --- make the violin plot as before ---
        x_positions = np.arange(1, len(data) + 1)

        plt.figure(figsize=(max(7, 0.8 * len(labels_pt)), 5))
        _ = plt.violinplot(
            data,
            positions=x_positions,
            showmeans=False,
            showmedians=True,
            showextrema=True,
        )

        plt.xticks(x_positions, labels_pt, rotation=45, ha="right")
        plt.ylabel("Variance drop ratio")
        plt.xlabel("Path type (first word of task_name/instruction)")
        plt.title(f"Variance drop ratio by path type | k={k_val}")

        # optional: overlay passed points
        # pass_k = pass_df[pass_df["k"] == k_val]
        # rng = np.random.RandomState(0)
        # for i, pt in enumerate(labels_pt, start=1):
        #     vals_pass = pass_k.loc[pass_k["path_type"] == pt, "variance_drop_ratio"].dropna().values
        #     if vals_pass.size == 0:
        #         continue
        #     xj = i + 0.06 * rng.randn(vals_pass.size)
        #     plt.scatter(xj, vals_pass, s=10, alpha=0.5)

        # --- overlay baseline lines for available n_modes ---
        linestyles = {1: ":", 2: "--", 3: "-.", 4: (0, (3, 1, 1, 1)), 5: "-", 6: (0, (5, 2)), 7: (0, (1, 1)), 8: (0, (3, 5, 1, 5)), 9: (0, (2, 2, 2, 2)), 10: (0, (5, 1, 1, 1, 1, 1))}  # distinct but simple
        for n_modes in n_modes_values:
            b = baseline_lines.get(n_modes, float("nan"))
            if np.isnan(b):
                continue
            plt.axhline(
                b,
                linestyle=linestyles.get(n_modes, "--"),
                linewidth=1.5,
                alpha=0.9,
                label=f"n={n_modes};k={k_val};mean={b:.3f}",
            )
        
        # Only add a legend if there are labeled artists to avoid empty-legend warnings.
        handles, labels = plt.gca().get_legend_handles_labels()
        if labels:
            plt.legend(
                handles,
                labels,
                frameon=False,
                fontsize=8,
                loc="center left",
                bbox_to_anchor=(1.02, 0.5),  # push legend outside axes
            )

        plt.tight_layout(rect=[0, 0, 0.82, 1])  # reserve space for legend

        out_path = os.path.join(violin_dir, f"pathtype_violin_k{k_val:02d}.png")
        plt.savefig(out_path, dpi=150)
        plt.close()
        print(f"Saved violin plot for k={k_val} to {os.path.abspath(out_path)}")

    # --------------------------------------------------------
    # visualize first N passes with Plotly 3D
    # --------------------------------------------------------
    plot_dir = os.path.join(args.outdir, "gaussian_threshold_plots")
    os.makedirs(plot_dir, exist_ok=True)

    # helper to load cached best labels for a state (if available)
    def maybe_load_best_for_state(state_idx):
        path = os.path.join(args.per_state_dir, f"state_{state_idx:06d}.npz")
        if not os.path.exists(path):
            return None
        _, best = load_per_state_file(path)
        return best

    top_passes = pass_df.head(args.plot_pass_top)
    print(f"Plotting first {len(top_passes)} passing rows...")

    # ---- Preload DROID states for the top passes (only if --lerobot_root provided) ----
    lerobot_state_map = {}
    state_col_candidates = None
    if args.lerobot_state_col:
        state_col_candidates = [args.lerobot_state_col]

    if args.lerobot_root is not None:
        if "episode_id" not in top_passes.columns or "t_in_episode" not in top_passes.columns:
            raise ValueError("To use --lerobot_root, metrics_df must contain episode_id and t_in_episode "
                            "(run your patcher first).")

        needed_keys = set()
        for r in top_passes.itertuples():
            ep_i = _parse_episode_id_to_index(getattr(r, "episode_id"))
            t_i = int(getattr(r, "t_in_episode"))
            needed_keys.add((ep_i, t_i))

        lerobot_state_map = load_lerobot_states_for_keys(
            args.lerobot_root,
            needed_keys=needed_keys,
            state_col_candidates=state_col_candidates,
            max_rows_per_parquet=args.lerobot_max_rows_per_parquet,
            frame_tolerance=max(0, int(args.lerobot_frame_tolerance or 0)),
        )
        print(f"[lerobot] loaded {len(lerobot_state_map)}/{len(needed_keys)} requested (episode_index,frame_index) states")

    for rank, row in enumerate(top_passes.itertuples(), start=1):
        state_idx = int(row.state_index)
        k = int(row.k)
        print(f"  plotting pass #{rank}: state={state_idx} k={k}")

        acts_raw = actions_arr[state_idx]
        arr = np.asarray(acts_raw)

        # ---------- reconstruct chunk structure ----------
        if arr.ndim == 3:
            # arr: (Kc, Tc, A) = (num_chunks, timesteps, action_dim_per_step)
            Kc, Tc, A = arr.shape
            traj_matrix = arr.reshape(Kc * Tc, A)        # per-point features
            chunk_ids = np.repeat(np.arange(Kc), Tc)     # length Kc*Tc
            time_ids = np.tile(np.arange(Tc), Kc)        # length Kc*Tc
        elif arr.ndim == 2:
            # treat each time step as its own "chunk" of length 1
            T_steps, A = arr.shape
            Kc, Tc = T_steps, 1
            traj_matrix = arr.reshape(Kc * Tc, A)
            chunk_ids = np.arange(Kc)                    # each point is its own chunk
            time_ids = np.zeros(Kc, dtype=int)
        else:
            sys.stderr.write(f"[warn] skipping state {state_idx}: unsupported shape {arr.shape}\n")
            continue

        # index_rows: one entry per point (state_idx, chunk_id, time_id)
        index_rows = [
            (state_idx, int(c), int(t))
            for c, t in zip(chunk_ids, time_ids)
        ]

        # ---------- try to reuse cached best if it matches this k ----------
        cached = maybe_load_best_for_state(state_idx)

        if cached is not None and int(cached.get("k", -1)) == k and cached.get("labels") is not None:
            # cached labels are per-chunk
            chunk_labels_cached = np.asarray(cached["labels"])
            used_idx = cached.get("idx", None)

            if used_idx is not None and len(used_idx) > 0:
                used_idx = np.asarray(used_idx, dtype=int)
                # build mapping from original chunk index -> label
                chunk_label_for_orig = np.full(Kc, -1, dtype=int)
                for local_idx, orig_idx in enumerate(used_idx):
                    if 0 <= orig_idx < Kc:
                        chunk_label_for_orig[orig_idx] = chunk_labels_cached[local_idx]

                # keep only points whose chunks were actually clustered
                mask = chunk_label_for_orig[chunk_ids] >= 0
                if not np.any(mask):
                    sys.stderr.write(
                        f"[warn] No points to plot for state {state_idx} after applying used_idx.\n"
                    )
                    continue

                X = traj_matrix[mask]
                ir = [index_rows[j] for j in np.where(mask)[0]]
                labels_per_point = chunk_label_for_orig[chunk_ids[mask]]
            else:
                # no subsampling: chunk_labels length should be Kc
                if chunk_labels_cached.shape[0] != Kc:
                    sys.stderr.write(
                        f"[warn] state {state_idx}: expected {Kc} chunk labels, got {chunk_labels_cached.shape[0]}; "
                        "falling back to reclustering.\n"
                    )
                    cached = None  # force recluster below
                else:
                    X = traj_matrix
                    ir = index_rows
                    labels_per_point = chunk_labels_cached[chunk_ids]
        else:
            cached = None  # ensure we don't try to use partially

        # ---------- if no usable cache, recluster at this k on points ----------
        if cached is None:
            X = traj_matrix
            ir = index_rows
            # simple Euclidean RBF affinity on points (for visualization only)
            D = cdist(X, X, metric="euclidean")
            pos = D[D > 0]
            sigma_used = float(np.median(pos)) if (args.sigma is None and pos.size) else (args.sigma or 1.0)
            A_mat = np.exp(-D ** 2 / (2.0 * sigma_used ** 2))
            np.fill_diagonal(A_mat, 1.0)

            cl = SpectralClustering(
                n_clusters=k,
                affinity="precomputed",
                assign_labels="kmeans",
                random_state=0,
            )
            labels_per_point = cl.fit_predict(A_mat)

        # ---------- now X, labels_per_point, ir all have matching length ----------
        xyz = to_xyz(X, mode=args.ee_mode)

        subdir = os.path.join(
            plot_dir, f"pass{rank:02d}_state_{state_idx:06d}_k{k:02d}"
        )
        os.makedirs(subdir, exist_ok=True)
        overview_html = os.path.join(subdir, "overview.html")
        percluster_html = os.path.join(subdir, "percluster.html")

        title = (
            f"state={state_idx} k={k} | "
            f"ratio={row.variance_drop_ratio:.4f} | "
            f"gauss={row.gaussian_baseline_ratio:.4f} | "
            f"margin={row.beat_margin:.4f}"
        )
        
        chunk_start_xyz = None
        if args.lerobot_root is not None:
            ep_i = _parse_episode_id_to_index(getattr(row, "episode_id"))
            t_i = int(getattr(row, "t_in_episode"))
            key = (ep_i, t_i)
            if key not in lerobot_state_map:
                print(f"[lerobot][warn] missing state for {key} (ep={ep_i}, t={t_i}); no start marker")
            else:
                svec = lerobot_state_map[key]  # (S,)

                # project to 3D
                sxyz = to_xyz(svec.reshape(1, -1), mode="first3")[0]

                # IMPORTANT: one real DROID state per plotted state_idx.
                # We mark it as the "origin" for ALL chunks (same start marker).
                chunk_start_xyz = {int(c): np.asarray(sxyz, float) for c in range(Kc)}

        plot_actions_xyz(
            xyz,
            labels_per_point,
            ir,
            html_path=overview_html,
            title=title,
            point_size=2,
            line_width=4.0,
            line_alpha=0.25,
            chunk_start_xyz=chunk_start_xyz,
            ghost_visible=args.ghost_visible,
        )
        n_cls = len(np.unique(labels_per_point))
        plot_per_cluster_panels(
            xyz,
            labels_per_point,
            ir,
            n_clusters=n_cls,
            html_path=percluster_html,
            title_prefix=f"state={state_idx} k={k}",
            point_size=2,
            line_width=4.0,
            line_alpha=0.55,
        )

    print(f"Plots written under: {os.path.abspath(plot_dir)}")


if __name__ == "__main__":
    main()
