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
       w ~ N(0, I_A), a_t = (t / (T-1)) * w,  t = 0..T-1
   where T and A are inferred from a real state's actions with that combo.
   We then run Minkowski + spectral clustering on these Gaussian chunks and
   compute the average variance_drop_ratio = (tv - wvar) / tv.
3) compute variance_drop_ratio for the real states
4) apply threshold: ratio_actual >= multiplier * ratio_baseline
5) save passing rows
6) produce:
   - per-k violin plots of variance_drop_ratio by path type, with baseline ratio
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
    Implements:
      Var := E[d^2] + 2 (E[d])^2 - 2 E_x[ (E_{x'}[d(x,x')])^2 ]
    where expectations are empirical over the N chunks.
    """
    D = np.asarray(D, float)
    if D.size == 0:
        return 0.0
    term1 = float(np.mean(D ** 2))          # E[d^2]
    mean_d = float(np.mean(D))             # E[d]
    row_means = D.mean(axis=1)             # E_{x'}[d(x,x')]
    term3 = float(np.mean(row_means ** 2)) # E_x[(E_{x'}[d])^2]
    # return term1 + 2.0 * (mean_d ** 2) - 2.0 * term3
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

def plot_actions_xyz(xyz, labels, index_rows, html_path="", title="",
                     point_size=2, line_width=4.0, line_alpha=0.6):
    """
    Plotly 3D scatter + trajectory lines for all points,
    with chunk lines + endpoints colored by the cluster they belong to.
    """
    xyz = np.asarray(xyz)
    labels = np.asarray(labels).reshape(-1)
    if labels.shape[0] != xyz.shape[0]:
        raise ValueError("labels length mismatch")
    if index_rows is None or len(index_rows) != xyz.shape[0]:
        raise ValueError("index_rows length mismatch with xyz")

    sample_idx = np.array([r[1] for r in index_rows], dtype=int)
    action_idx = np.array([r[2] for r in index_rows], dtype=int)

    # Consistent color map
    uniq = np.unique(labels)
    uniq_sorted = np.sort(uniq)
    color_map = {lab: i / max(1, len(uniq_sorted)-1) for i, lab in enumerate(uniq_sorted)}
    # These remain normalized [0,1], Plotly Viridis will interpret automatically

    fig = go.Figure()

    # --- main scatter ---
    fig.add_trace(go.Scatter3d(
        x=xyz[:, 0],
        y=xyz[:, 1],
        z=xyz[:, 2],
        mode="markers",
        marker=dict(
            size=point_size,
            color=[color_map[l] for l in labels],
            colorscale="Viridis",
            showscale=True,
        ),
        name="points",
        opacity=0.95,
    ))

    # --- chunk-level trajectories ---
    for s in np.unique(sample_idx):
        m = (sample_idx == s)
        order = np.argsort(action_idx[m])
        pts = xyz[m][order]
        labs = labels[m][order]

        if pts.shape[0] < 2:
            continue

        # dominant cluster
        vals, counts = np.unique(labs, return_counts=True)
        dom = vals[np.argmax(counts)]
        cval = color_map[dom]
        color_hex = cval_to_hex(cval)

        # trajectory line
        fig.add_trace(go.Scatter3d(
            x=pts[:, 0],
            y=pts[:, 1],
            z=pts[:, 2],
            mode="lines",
            line=dict(
                width=line_width,
                color=color_hex,     # exact cluster color
            ),
            opacity=line_alpha,
            showlegend=False,
        ))

        # start marker
        p0 = pts[0]
        fig.add_trace(go.Scatter3d(
            x=[p0[0]], y=[p0[1]], z=[p0[2]],
            mode="markers",
            marker=dict(
                symbol="circle-open",
                size=3,
                line=dict(width=2, color=color_hex),
                color=color_hex,
            ),
            showlegend=False,
        ))

        # end marker
        p1 = pts[-1]
        fig.add_trace(go.Scatter3d(
            x=[p1[0]], y=[p1[1]], z=[p1[2]],
            mode="markers",
            marker=dict(
                symbol="x",
                size=3,
                color=color_hex,
            ),
            showlegend=False,
        ))

    fig.update_layout(
        title=title,
        scene=dict(
            xaxis_title="X",
            yaxis_title="Y",
            zaxis_title="Z",
        ),
        margin=dict(l=0, r=0, b=0, t=40),
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


# ------------------------------------------------------------
# Gaussian *trajectory* baseline for ratio
# ------------------------------------------------------------
def gaussian_baseline_ratio(num_chunks,
                            per_step_dim,
                            chunk_len,
                            k,
                            n_trials=30,
                            rng=None,
                            sigma=None):
    """
    For each trial:
      - Generate num_chunks synthetic trajectories (chunks) of length chunk_len in R^{per_step_dim}:
            w ~ N(0, I)
            a_t = (t / (chunk_len - 1)) * w,   t = 0..chunk_len-1
        (if chunk_len == 1, use just w as a single step)
      - Compute Minkowski distance matrix between chunks
      - Compute total variance tv
      - Build Gaussian affinity from D using the same median-sigma logic
      - Run spectral clustering with n_clusters = k
      - Compute weighted in-cluster variance wvar
      - Compute variance_drop_ratio = (tv - wvar) / tv
    Return the average variance_drop_ratio over trials.
    """
    if rng is None:
        rng = np.random.RandomState(0)

    ratios = []
    for _ in range(n_trials):
        trajectories = []
        # build synthetic Gaussian trajectories
        for _c in range(num_chunks):
            w = rng.randn(per_step_dim)
            if chunk_len <= 1:
                traj = w.reshape(1, per_step_dim)
            else:
                t_grid = np.arange(chunk_len, dtype=float)
                t_grid = t_grid / max(chunk_len - 1, 1.0)
                traj = (t_grid[:, None] * w[None, :])
            trajectories.append(traj)

        # Minkowski distance matrix
        D_mink = compute_minkowski_distance_matrix(trajectories)
        tv = total_variance_minkowski(D_mink)
        if tv <= 0:
            continue

        # build affinity from D_mink
        pos = D_mink[D_mink > 0]
        sigma_used = float(np.median(pos)) if (sigma is None and pos.size) else (sigma or 1.0)
        A = np.exp(-D_mink ** 2 / (2.0 * sigma_used ** 2))
        np.fill_diagonal(A, 1.0)

        if num_chunks < k or k < 1:
            continue
        try:
            cl = SpectralClustering(
                n_clusters=k,
                affinity="precomputed",
                assign_labels="kmeans",
                random_state=0,
            )
            labels = cl.fit_predict(A)
        except Exception:
            continue

        wvar = weighted_incluster_variance_minkowski(D_mink, labels)
        drop = tv - wvar
        if tv > 0:
            ratios.append(drop / tv)

    if not ratios:
        return 0.0
    return float(np.mean(ratios))


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
    # Build (num_points, action_dim, k) -> (chunk_len, per_step_dim)
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
        elif arr.ndim == 2:
            # (T, A); in the main script, each time step is a 1-step trajectory
            T, A = arr.shape
            chunk_len = 1
            per_step_dim = A
        else:
            sys.stderr.write(
                f"[warn] state {state_idx}: unsupported action shape {arr.shape}; "
                "skipping this combo for baseline.\n"
            )
            continue

        combo_to_params[key] = (chunk_len, per_step_dim)

    print(f"Found {len(combo_to_params)} unique (num_points, action_dim, k) combos for Gaussian baseline")

    rng = np.random.RandomState(0)
    combo_to_baseline_ratio = {}

    # --------------------------------------------------------
    # Compute Gaussian baseline variance_drop_ratio per combo
    # --------------------------------------------------------
    for (Np, d, k), (chunk_len, per_step_dim) in combo_to_params.items():
        base_ratio = gaussian_baseline_ratio(
            num_chunks=Np,
            per_step_dim=per_step_dim,
            chunk_len=chunk_len,
            k=k,
            n_trials=args.gaussian_trials,
            rng=rng,
            sigma=args.sigma,
        )
        combo_to_baseline_ratio[(Np, d, k)] = base_ratio

    # --------------------------------------------------------
    # Apply threshold using variance_drop_ratio
    # --------------------------------------------------------
    baseline_list = []
    pass_mask = []
    ratios = metrics_df["variance_drop_ratio"].to_numpy()

    for idx, row in metrics_df.iterrows():
        key = (int(row["num_points"]), int(row["action_dim"]), int(row["k"]))
        base_ratio = combo_to_baseline_ratio.get(key, 0.0)
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
        pass_k = pass_df[pass_df["k"] == k_val]
        if pass_k.empty:
            continue

        path_types = sorted(pass_k["path_type"].unique().tolist())
        if not path_types:
            continue

        # collect variance_drop_ratio distributions per path type
        data = []
        labels_pt = []
        for pt in path_types:
            vals = pass_k.loc[pass_k["path_type"] == pt, "variance_drop_ratio"].dropna().values
            if vals.size == 0:
                continue
            data.append(vals)
            labels_pt.append(pt)

        if not data:
            continue

        metrics_k = metrics_df[metrics_df["k"] == k_val]
        if metrics_k.empty:
            baseline_mean = float("nan")
        else:
            baseline_mean = float(metrics_k["gaussian_baseline_ratio"].mean())

        x_positions = np.arange(1, len(labels_pt) + 1)

        plt.figure(figsize=(max(7, 0.8 * len(labels_pt)), 5))
        vp = plt.violinplot(data, positions=x_positions, showmeans=False, showmedians=True, showextrema=True)

        plt.xticks(x_positions, labels_pt, rotation=45, ha="right")
        plt.ylabel("Variance drop ratio (passing rows)")
        plt.xlabel("Path type (first word of task_name/instruction)")
        plt.title(f"Variance drop ratio by path type | k={k_val}")

        if not np.isnan(baseline_mean):
            plt.axhline(
                baseline_mean,
                linestyle="--",
                linewidth=1.5,
                label=f"Gaussian baseline ratio (mean={baseline_mean:.3f})",
            )
            plt.legend(frameon=False)

        plt.tight_layout()
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

        plot_actions_xyz(
            xyz,
            labels_per_point,
            ir,
            html_path=overview_html,
            title=title,
            point_size=2,
            line_width=4.0,
            line_alpha=0.4,
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