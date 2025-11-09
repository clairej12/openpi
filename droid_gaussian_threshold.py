#!/usr/bin/env python3
"""
postprocess_gaussian_threshold.py

Read the clustered/metric outputs from the main spectral script and filter
(state, k) combinations whose variance_drop beats a Gaussian baseline
(for the same N, d, k). Then visualize the first N passes using the same
3D plots as the clustering script.

Assumptions:
- You already ran the main script and you have:
    outdir/metrics_per_state.csv
    outdir/per_state/state_000000.npz  (etc.)
    actions_npz  (same one used for clustering)
- Per-state npz files contain: rows (list of dicts) and best (dict) – as in your script.

We:
1) load metrics
2) compute Gaussian baselines per (num_points, action_dim, k)
3) apply threshold: actual_drop >= multiplier * baseline_drop
4) save passing rows
5) replot first --plot_pass_top of them
"""

import argparse, os, sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.cluster import SpectralClustering
from sklearn.decomposition import PCA
from scipy.spatial.distance import cdist

# ------------------------------------------------------------
# plotting helpers (borrowed from your clustering script)
# ------------------------------------------------------------
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401


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
    else:  # pca
        if pca_model is None:
            pca_model = PCA(n_components=3, random_state=0)
            Z = pca_model.fit_transform(X)
        else:
            Z = pca_model.transform(X)
        return Z


def plot_actions_xyz(xyz, labels, index_rows, png_path="", title="",
                     point_size=1, line_width=1.0, line_alpha=0.6,
                     line_color_mode="by_dominant_cluster",
                     start_marker_size=10, end_marker_size=20):
    import matplotlib.lines as mlines
    fig = plt.figure(figsize=(7, 6))
    ax = fig.add_subplot(111, projection="3d")
    xyz = np.asarray(xyz)
    labels = np.asarray(labels).reshape(-1)
    if labels.shape[0] != xyz.shape[0]:
        raise ValueError("labels length mismatch")
    scatter = ax.scatter(xyz[:, 0], xyz[:, 1], xyz[:, 2], c=labels,
                         s=point_size, alpha=0.95)
    start_h = end_h = None
    if index_rows is not None and len(index_rows) == xyz.shape[0]:
        sample_idx = np.array([r[1] for r in index_rows], dtype=int)
        action_idx = np.array([r[2] for r in index_rows], dtype=int)
        unique_samples = np.unique(sample_idx)
        for s in unique_samples:
            m = (sample_idx == s)
            order = np.argsort(action_idx[m])
            pts = xyz[m][order]
            labs = labels[m][order]
            if pts.shape[0] >= 2:
                if line_color_mode == "by_dominant_cluster":
                    vals, counts = np.unique(labs, return_counts=True)
                    dom = vals[np.argmax(counts)]
                    line_color = scatter.cmap(scatter.norm(dom))
                else:
                    line_color = "k"
                ax.plot(pts[:, 0], pts[:, 1], pts[:, 2],
                        linewidth=line_width, alpha=line_alpha, color=line_color)
            p0, p1 = pts[0], pts[-1]
            ax.scatter([p0[0]], [p0[1]], [p0[2]], s=start_marker_size,
                       c="none", edgecolor="k", linewidths=1.0, marker="o")
            ax.scatter([p1[0]], [p1[1]], [p1[2]], s=end_marker_size, c="k",
                       marker="x", linewidths=1.5)
        start_h = mlines.Line2D([], [], color="k", marker="o", markersize=6,
                                markerfacecolor="none", linestyle="None", label="chunk start")
        end_h = mlines.Line2D([], [], color="k", marker="x", markersize=7,
                              linestyle="None", label="chunk end")
    legend_items = [mlines.Line2D([], [], color=scatter.cmap(scatter.norm(lab)),
                                  marker="s", linestyle="None", markersize=6,
                                  label=f"cluster {lab}") for lab in np.unique(labels)]
    if start_h is not None:
        legend_items.append(start_h)
    if end_h is not None:
        legend_items.append(end_h)
    if legend_items:
        ax.legend(handles=legend_items, loc="best", frameon=False, fontsize=9)
    ax.set_title(title)
    ax.set_xlabel("X"); ax.set_ylabel("Y"); ax.set_zlabel("Z")
    # equal-ish axes
    x_limits = ax.get_xlim3d(); y_limits = ax.get_ylim3d(); z_limits = ax.get_zlim3d()
    x_range = abs(x_limits[1] - x_limits[0])
    y_range = abs(y_limits[1] - y_limits[0])
    z_range = abs(z_limits[1] - z_limits[0])
    max_range = max([x_range, y_range, z_range])
    xm, ym, zm = np.mean(x_limits), np.mean(y_limits), np.mean(z_limits)
    ax.set_xlim3d([xm - max_range / 2, xm + max_range / 2])
    ax.set_ylim3d([ym - max_range / 2, ym + max_range / 2])
    ax.set_zlim3d([zm - max_range / 2, zm + max_range / 2])
    fig.tight_layout()
    fig.savefig(png_path, dpi=150)
    plt.close(fig)


def plot_per_cluster_panels(xyz, per_point_labels, index_rows, n_clusters,
                            png_path, title_prefix="State",
                            point_size=3, line_width=1.0, line_alpha=0.6):
    import matplotlib.lines as mlines
    xyz = np.asarray(xyz)
    labels = np.asarray(per_point_labels).reshape(-1)
    assert xyz.shape[0] == labels.shape[0] == len(index_rows)
    sample_idx = np.array([r[1] for r in index_rows], dtype=int)
    action_idx = np.array([r[2] for r in index_rows], dtype=int)
    uniq_clusters = np.unique(labels)
    C = int(max(n_clusters, uniq_clusters.max() + 1))
    cols = min(C, 4)
    rows = int(np.ceil(C / cols))
    fig = plt.figure(figsize=(5 * cols, 4.5 * rows))

    def _equal(ax):
        xlim = ax.get_xlim3d(); ylim = ax.get_ylim3d(); zlim = ax.get_zlim3d()
        xr = abs(xlim[1] - xlim[0]); yr = abs(ylim[1] - ylim[0]); zr = abs(zlim[1] - zlim[0])
        mr = max(xr, yr, zr)
        xm = np.mean(xlim); ym = np.mean(ylim); zm = np.mean(zlim)
        ax.set_xlim3d([xm - mr / 2, xm + mr / 2])
        ax.set_ylim3d([ym - mr / 2, ym + mr / 2])
        ax.set_zlim3d([zm - mr / 2, zm + mr / 2])

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
        ax = fig.add_subplot(rows, cols, ci + 1, projection="3d")
        ax.set_title(f"Cluster {ci}")
        ax.set_xlabel("X"); ax.set_ylabel("Y"); ax.set_zlabel("Z")
        chunks_in_ci = [k for k, lab in chunk_to_cluster.items() if lab == ci]
        if not chunks_in_ci:
            ax.text(0.5, 0.5, 0.5, "No chunks", transform=ax.transAxes,
                    ha="center", va="center")
            continue
        for k in chunks_in_ci:
            m = (sample_idx == k)
            pts = xyz[m]
            ax.scatter(pts[:, 0], pts[:, 1], pts[:, 2], s=point_size, alpha=0.95)
        for k in chunks_in_ci:
            m = (sample_idx == k)
            order = np.argsort(action_idx[m])
            pts = xyz[m][order]
            if pts.shape[0] >= 2:
                ax.plot(pts[:, 0], pts[:, 1], pts[:, 2],
                        linewidth=line_width, alpha=line_alpha)
            p0, p1 = pts[0], pts[-1]
            ax.scatter([p0[0]], [p0[1]], [p0[2]], s=36, c="none",
                       edgecolor="k", linewidths=1.0, marker="o")
            ax.scatter([p1[0]], [p1[1]], [p1[2]], s=48, c="k",
                       marker="x", linewidths=1.5)
        _equal(ax)

    start_h = mlines.Line2D([], [], color="k", marker="o", markersize=6,
                            markerfacecolor="none", linestyle="None", label="chunk start")
    end_h = mlines.Line2D([], [], color="k", marker="x", markersize=7,
                          linestyle="None", label="chunk end")
    fig.legend(handles=[start_h, end_h], loc="upper right", frameon=False)
    fig.suptitle(f"{title_prefix}: per-cluster chunk views", y=0.995)
    fig.tight_layout(rect=[0, 0.00, 1, 0.96])
    fig.savefig(png_path, dpi=150)
    plt.close(fig)


# ------------------------------------------------------------
# helper to rebuild affinity (same logic as main script)
# ------------------------------------------------------------
def build_affinity_for_points(X, method="minkowski", sigma=None,
                              dtw_window=None, random_state=0,
                              parallel_dtw=False, max_workers=None):
    X = np.asarray(X, dtype=float)
    N = X.shape[0]
    if method == "minkowski":
        D = cdist(X, X, metric="euclidean")
        pos = D[D > 0]
        sigma_used = float(np.median(pos)) if (sigma is None and pos.size) else (sigma or 1.0)
        A = np.exp(-D ** 2 / (2 * sigma_used ** 2))
        np.fill_diagonal(A, 1.0)
        return A, D, sigma_used
    # if someone passes dtw here, we could add it, but for plotting top 10 it's fine
    raise NotImplementedError("DTW branch not wired for postprocess script")


# ------------------------------------------------------------
# read per-state file written by the clustering script
# ------------------------------------------------------------
def load_per_state_file(path):
    data = np.load(path, allow_pickle=True)
    rows = list(data["rows"].tolist())
    best = None
    if "best" in data and data["best"].size > 0:
        first = data["best"][0]
        # depending on how it was saved, first is either dict or 0d object
        if isinstance(first, dict):
            best = first
        else:  # numpy object scalar
            best = first.item()
    return rows, best


# ------------------------------------------------------------
# Gaussian baseline for arbitrary k (including non powers of 2)
# ------------------------------------------------------------
def gaussian_baseline_drop(num_points, action_dim, k,
                           n_trials=30, max_points_sim=3000, rng=None):
    """
    For each trial:
      - sample N x d from N(0, I)
      - recursively split the cluster with the largest variance (by dim)
        until we get k clusters (works for k=4,5,6,...)
      - compute variance_drop := global_var - weighted_cluster_var
    Return the average variance_drop over trials.
    """
    if rng is None:
        rng = np.random.RandomState(0)
    N = min(num_points, max_points_sim)
    drops = []
    for _ in range(n_trials):
        X = rng.randn(N, action_dim)
        clusters = [X]
        # keep splitting the noisiest cluster until we have k
        while len(clusters) < k:
            # pick cluster with largest mean variance
            vars_ = [c.var(axis=0).mean() for c in clusters]
            idx = int(np.argmax(vars_))
            C = clusters.pop(idx)
            if C.shape[0] <= 1:
                # can't split, put it back and break
                clusters.append(C)
                break
            # split along dim with biggest variance
            dim = int(np.argmax(C.var(axis=0)))
            thr = np.median(C[:, dim])
            left = C[C[:, dim] <= thr]
            right = C[C[:, dim] > thr]
            if left.size == 0 or right.size == 0:
                # fallback: random split
                perm = rng.permutation(C.shape[0])
                m = C.shape[0] // 2
                left = C[perm[:m]]
                right = C[perm[m:]]
            clusters.append(left)
            clusters.append(right)
            # if we overshot (rare), merge two smallest
            if len(clusters) > k:
                sizes = [c.shape[0] for c in clusters]
                a = int(np.argmin(sizes))
                tmp = clusters.pop(a)
                b = int(np.argmin([c.shape[0] for c in clusters]))
                other = clusters.pop(b)
                merged = np.vstack([tmp, other])
                clusters.append(merged)
        global_var = X.var(axis=0).mean()
        within = 0.0
        for c in clusters:
            if c.shape[0] == 0:
                continue
            var_c = c.var(axis=0).mean()
            within += (c.shape[0] / N) * var_c
        drop = global_var - within
        drops.append(drop)
    return float(np.mean(drops))


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
    ap.add_argument("--gaussian_max_points", type=int, default=3000)
    ap.add_argument("--gaussian_multiplier", type=float, default=0.5,
                    help="threshold = multiplier * gaussian_baseline_drop")
    ap.add_argument("--plot_pass_top", type=int, default=10,
                    help="visualize first N passing rows")
    ap.add_argument("--method", type=str, default="minkowski",
                    choices=["minkowski"],
                    help="used only when we have to recluster a non-best-k")
    ap.add_argument("--sigma", type=float, default=None)
    ap.add_argument("--ee_mode", type=str, default="first3",
                    choices=["first3", "pca", "custom"])
    args = ap.parse_args()

    os.makedirs(args.outdir, exist_ok=True)
    if args.per_state_dir is None:
        args.per_state_dir = os.path.join(os.path.dirname(args.metrics_csv), "per_state")

    metrics_df = pd.read_csv(args.metrics_csv)
    actions_data = np.load(args.actions_npz, allow_pickle=True)
    actions_arr = actions_data["actions"]

    # collect unique (N, d, k)
    combos = set()
    for _, row in metrics_df.iterrows():
        Np = int(row["num_points"])
        d = int(row["action_dim"])
        k = int(row["k"])
        combos.add((Np, d, k))

    print(f"Found {len(combos)} unique (num_points, action_dim, k) combos for Gaussian baseline")

    rng = np.random.RandomState(0)
    combo_to_baseline = {}
    for (Np, d, k) in combos:
        base = gaussian_baseline_drop(
            Np,
            d,
            k,
            n_trials=args.gaussian_trials,
            max_points_sim=args.gaussian_max_points,
            rng=rng,
        )
        combo_to_baseline[(Np, d, k)] = base

    # apply threshold
    baseline_list = []
    pass_mask = []
    for _, row in metrics_df.iterrows():
        key = (int(row["num_points"]), int(row["action_dim"]), int(row["k"]))
        base = combo_to_baseline[key]
        baseline_list.append(base)
        actual = float(row["variance_drop"])
        passes = actual >= args.gaussian_multiplier * base
        pass_mask.append(passes)
    metrics_df["gaussian_baseline_drop"] = baseline_list
    metrics_df["pass_gaussian"] = pass_mask

    pass_df = metrics_df[metrics_df["pass_gaussian"] == True].copy()  # noqa: E712
    # sort by "how much it beats" the threshold
    pass_df["beat_margin"] = pass_df["variance_drop"] - \
        (args.gaussian_multiplier * pass_df["gaussian_baseline_drop"])
    pass_df.sort_values("beat_margin", ascending=False, inplace=True)

    pass_csv = os.path.join(args.outdir, "gaussian_passes.csv")
    metrics_df.to_csv(os.path.join(args.outdir, "metrics_with_gaussian.csv"), index=False)
    pass_df.to_csv(pass_csv, index=False)
    print(f"Saved filtered passes to: {pass_csv}")
    print(f"{len(pass_df)} state+k rows passed the Gaussian-based threshold.")

    # --------------------------------------------------------
    # visualize first N passes
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
        # reconstruct X and index_rows exactly like clustering script
        if acts_raw.ndim == 3:
            Kc, Tc, A = acts_raw.shape
            X_full = acts_raw.reshape(Kc * Tc, A)
            index_rows = [(state_idx, kk, t) for kk in range(Kc) for t in range(Tc)]
        elif acts_raw.ndim == 2:
            Tc, A = acts_raw.shape
            X_full = acts_raw
            index_rows = [(state_idx, 0, t) for t in range(Tc)]
        else:
            sys.stderr.write(f"[warn] skipping state {state_idx}: unsupported shape {acts_raw.shape}\n")
            continue

        # try to reuse cached best if it matches this k
        cached = maybe_load_best_for_state(state_idx)
        if cached is not None and int(cached.get("k", -1)) == k and cached.get("labels") is not None:
            labels = np.asarray(cached["labels"])
            used_idx = cached.get("idx", None)
            X = X_full
            ir = index_rows
            if used_idx is not None and len(used_idx) > 0:
                X = X_full[used_idx]
                ir = [index_rows[j] for j in used_idx.tolist()]
        else:
            # recluster at this k
            X = X_full
            ir = index_rows
            A_mat, _, _ = build_affinity_for_points(
                X,
                method=args.method,
                sigma=args.sigma,
            )
            cl = SpectralClustering(
                n_clusters=k,
                affinity="precomputed",
                assign_labels="kmeans",
                random_state=0,
            )
            labels = cl.fit_predict(A_mat)

        xyz = to_xyz(X, mode=args.ee_mode)

        subdir = os.path.join(plot_dir, f"pass{rank:02d}_state_{state_idx:06d}_k{k:02d}")
        os.makedirs(subdir, exist_ok=True)
        overview_png = os.path.join(subdir, "overview.png")
        percluster_png = os.path.join(subdir, "percluster.png")

        title = (f"state={state_idx} k={k} | "
                 f"var_drop={row.variance_drop:.4f} | "
                 f"gauss={row.gaussian_baseline_drop:.4f} | "
                 f"margin={row.beat_margin:.4f}")

        plot_actions_xyz(
            xyz, labels, ir,
            png_path=overview_png,
            title=title,
            point_size=1,
            line_width=1.0,
            line_alpha=0.4,
        )
        n_cls = len(np.unique(labels))
        plot_per_cluster_panels(
            xyz, labels, ir, n_clusters=n_cls,
            png_path=percluster_png,
            title_prefix=f"state={state_idx} k={k}",
            point_size=1,
            line_width=1.2,
            line_alpha=0.55,
        )

    print(f"Plots written under: {os.path.abspath(plot_dir)}")


if __name__ == "__main__":
    main()