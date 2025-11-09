#!/usr/bin/env python3
"""
analyze_action_multimodality.py
Spectral version, parallelized, fast Euclidean affinity, and per-state checkpointing.

New in this version:
- For every state i, we write:  outdir/per_state/state_{i:06d}.npz
- On rerun, if that file exists, we just load it and skip recomputing.
- At the end we aggregate ALL per-state files and still rank/plot top states.
"""

import argparse, os, sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.metrics import silhouette_score
from sklearn.decomposition import PCA
from sklearn.cluster import SpectralClustering
from scipy.spatial.distance import cdist
from concurrent.futures import ProcessPoolExecutor, as_completed
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401

# ------------------------------------------------------------------------------------
# Globals for worker processes (set once by initializer)
# ------------------------------------------------------------------------------------
_G_ACTIONS = None
_G_META = None
_G_ARGS = None


def _init_worker(actions_arr, meta_records, args_dict):
    """Initializer for each worker; stores big arrays in globals to avoid re-pickling."""
    global _G_ACTIONS, _G_META, _G_ARGS
    _G_ACTIONS = actions_arr
    _G_META = meta_records
    _G_ARGS = args_dict


# ------------------------------------------------------------------------------------
# distance + spectral clustering helpers
# ------------------------------------------------------------------------------------

def pairwise_sym_minkowski_from_points(X):
    """
    Build NxN symmetrized-Minkowski distances for already-flattened action points.

    In your current setup each sample is just one flattened vector, so this is just
    Euclidean pairwise — but we keep the function so we can swap in a true
    sym-minkowski later.
    """
    X = np.asarray(X, float)
    D = cdist(X, X, metric="euclidean")
    return D


def total_variance_minkowski(D):
    """
    D: (N, N) pairwise distances in your "Minkowski" sense.
    We'll use average squared distance as a dispersion measure.
    """
    if D.size == 0:
        return 0.0
    return float(np.mean(D ** 2))


def weighted_incluster_variance_minkowski(D, labels):
    """
    D: (N, N)
    labels: cluster labels
    Weighted average of within-cluster squared distances.
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
        var_c = np.mean(Dc ** 2)
        wvar += (idx.size / N) * var_c
    return float(wvar)


def _as_2d(x):
    x = np.asarray(x)
    if x.ndim == 1:
        return x.reshape(-1, 1).astype(float, copy=False)
    if x.ndim >= 3:
        x = x.reshape(x.shape[0], -1)
    return x.astype(float, copy=False)


def dtw_distance_band(X, Y, window=None):
    X = _as_2d(X)
    Y = _as_2d(Y)
    len_x, len_y = X.shape[0], Y.shape[0]
    if len_x == 0 and len_y == 0:
        return 0.0
    if len_x == 0 or len_y == 0:
        return np.inf
    if window is None:
        window = max(len_x, len_y) // 10
    window = int(window)
    w = max(window, abs(len_x - len_y))
    prev = np.full(len_y + 1, np.inf)
    curr = np.full(len_y + 1, np.inf)
    prev[0] = 0.0
    for i in range(1, len_x + 1):
        curr.fill(np.inf)
        j_start = max(1, i - w)
        j_end = min(len_y, i + w)
        xi = X[i - 1]
        for j in range(j_start, j_end + 1):
            dist = np.dot(xi - Y[j - 1], xi - Y[j - 1])
            curr[j] = dist + min(curr[j - 1], prev[j], prev[j - 1])
        prev, curr = curr, prev
    return float(np.sqrt(prev[len_y]))


_TRAJ = None


def _init_dtw_worker(trajectories):  # noqa
    global _TRAJ
    _TRAJ = trajectories


def _pair_dtw(args):  # noqa
    i, j, win = args
    return i, j, dtw_distance_band(_TRAJ[i], _TRAJ[j], window=win)


def compute_dtw_matrix(trajectories, window=None, parallelize=False, max_workers=None):
    from concurrent.futures import ProcessPoolExecutor as PPE
    N = len(trajectories)
    D = np.zeros((N, N), dtype=float)
    if not parallelize or N < 64:
        for i in range(N):
            for j in range(i + 1, N):
                d = dtw_distance_band(trajectories[i], trajectories[j], window=window)
                D[i, j] = D[j, i] = d
        return D
    args = [(i, j, window) for i in range(N) for j in range(i + 1, N)]
    with PPE(max_workers=max_workers, initializer=_init_dtw_worker, initargs=(trajectories,)) as ex:
        for i, j, d in ex.map(_pair_dtw, args, chunksize=32):
            D[i, j] = D[j, i] = d
    return D


def build_affinity_for_points(
    X,
    method="minkowski",
    sigma=None,
    dtw_window=None,
    random_state=0,
    parallel_dtw=False,
    max_workers=None,
):
    """
    Build NxN affinity from X (N x d) fast when we can.
    """
    X = np.asarray(X, dtype=float)
    N = X.shape[0]

    if method == "minkowski":
        # Fast path: we treat flat actions as points and just do Euclidean.
        D = cdist(X, X, metric="euclidean")
        pos = D[D > 0]
        sigma_used = float(np.median(pos)) if (sigma is None and pos.size) else (sigma or 1.0)
        A = np.exp(-D ** 2 / (2 * sigma_used ** 2))
        np.fill_diagonal(A, 1.0)
        return A, D, sigma_used

    # DTW path
    trajs_points = [X[n:n + 1, :] for n in range(N)]
    D = compute_dtw_matrix(
        trajs_points,
        window=dtw_window,
        parallelize=parallel_dtw,
        max_workers=max_workers,
    )
    pos = D[D > 0]
    sigma_used = float(np.median(pos)) if (sigma is None and pos.size) else (sigma or 1.0)
    A = np.exp(-D ** 2 / (2.0 * sigma_used ** 2))
    np.fill_diagonal(A, 1.0)
    return A, D, sigma_used


# ------------------------------------------------------------------------------------
# plotting helpers
# ------------------------------------------------------------------------------------
def to_points(acts):
    acts = np.asarray(acts)
    if acts.ndim == 2:
        return acts.astype(float, copy=False)
    elif acts.ndim == 3:
        K, T, A = acts.shape
        return acts.reshape(K * T, A).astype(float, copy=False)
    else:
        raise ValueError(f"Unsupported actions shape: {acts.shape}")


def to_xyz(actions, mode="pca", pca_model=None, kinematic_map=None):
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
    else:
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
    scatter = ax.scatter(xyz[:, 0], xyz[:, 1], xyz[:, 2], c=labels, s=point_size, alpha=0.95)
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
            ax.scatter([p0[0]], [p0[1]], [p0[2]], s=start_marker_size, c="none",
                       edgecolor="k", linewidths=1.0, marker="o")
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
    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.set_zlabel("Z")
    x_limits = ax.get_xlim3d()
    y_limits = ax.get_ylim3d()
    z_limits = ax.get_zlim3d()
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


def plot_per_cluster_panels(xyz, per_point_labels, index_rows, n_clusters, png_path, title_prefix="State",
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
        xlim = ax.get_xlim3d()
        ylim = ax.get_ylim3d()
        zlim = ax.get_zlim3d()
        xr = abs(xlim[1] - xlim[0])
        yr = abs(ylim[1] - ylim[0])
        zr = abs(zlim[1] - zlim[0])
        mr = max(xr, yr, zr)
        xm = np.mean(xlim)
        ym = np.mean(ylim)
        zm = np.mean(zlim)
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
        ax.set_xlabel("X")
        ax.set_ylabel("Y")
        ax.set_zlabel("Z")
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


# ------------------------------------------------------------------------------------
# per-state worker (now with checkpoint)
# ------------------------------------------------------------------------------------

def _state_file_path(per_state_dir, i):
    return os.path.join(per_state_dir, f"state_{i:06d}.npz")


def _load_state_file(path):
    data = np.load(path, allow_pickle=True)
    rows_arr = data["rows"]
    rows = list(rows_arr.tolist())

    best_arr = data["best"]

    if best_arr.size == 0:
        best = None
    else:
        # usually this is a single dict saved in a 1-element object array
        b = best_arr[0]
        # if it's already a dict, just use it
        if isinstance(b, dict):
            best = b
        else:
            # in case it’s some 0-d object array or something weird
            try:
                best = b.item()
            except AttributeError:
                best = b

    return rows, best


def _save_state_file(path, rows, best):
    tmp_path = path + ".tmp"   # ".../state_000123.npz.tmp"
    with open(tmp_path, "wb") as f:
        np.savez_compressed(
            f,
            rows=np.array(rows, dtype=object),
            best=np.array([] if best is None else [best], dtype=object),
        )
    os.replace(tmp_path, path)


def _process_state(i):
    """
    Run the whole clustering pipeline on state i.
    If per-state file exists, load and return it.
    Otherwise compute, save, and return.
    """
    actions_arr = _G_ACTIONS
    meta_records = _G_META
    args = _G_ARGS
    per_state_dir = args["per_state_dir"]

    state_path = _state_file_path(per_state_dir, i)
    if os.path.exists(state_path):
        # already computed
        rows, best_cache = _load_state_file(state_path)
        return rows, best_cache

    acts_raw = actions_arr[i]
    try:
        X = to_points(acts_raw)
    except Exception as e:
        sys.stderr.write(f"[skip] state {i}: {e}\n")
        rows = []
        _save_state_file(state_path, rows, None)
        return rows, None

    # Optional subsample
    used_idx = None
    if args["sample_cap"] is not None and X.shape[0] > args["sample_cap"]:
        rng = np.random.RandomState(args["random_state"] + i)
        used_idx = rng.choice(X.shape[0], args["sample_cap"], replace=False)
        X = X[used_idx]

    # metadata
    if i < len(meta_records):
        row_meta = meta_records[i]
        episode_id = row_meta.get("episode_id", f"ep{i:06d}")
        t_in_episode = int(row_meta.get("t_in_episode", -1))
        task_name = str(row_meta.get("task_name", ""))
        instruction = str(row_meta.get("instruction", ""))
    else:
        episode_id = f"ep{i:06d}"
        t_in_episode = -1
        task_name = ""
        instruction = ""

    # build affinity (fast), also build Minkowski distances for metrics
    try:
        A, _, _ = build_affinity_for_points(
            X,
            method=args["method"],
            sigma=args["sigma"],
            dtw_window=args["dtw_window"],
            random_state=args["random_state"],
            parallel_dtw=args["parallel_dtw"],
            max_workers=args["max_workers"],
        )
        D_mink = pairwise_sym_minkowski_from_points(X)
        tv = total_variance_minkowski(D_mink)
    except Exception as e:
        sys.stderr.write(f"[skip] state {i}: affinity failed ({e})\n")
        rows = []
        _save_state_file(state_path, rows, None)
        return rows, None

    rows = []
    best = {
        "score": None,
        "k": None,
        "wvar": None,
        "drop": None,
        "r2": None,
        "ch": None,
        "sil": None,
        "labels": None,
        "used_idx": used_idx,
    }

    for k in range(args["k_min"], args["k_max"] + 1):
        if X.shape[0] < k or k < 1:
            continue
        try:
            cl = SpectralClustering(
                n_clusters=k,
                affinity="precomputed",
                assign_labels="kmeans",
                random_state=args["random_state"],
            )
            labels = cl.fit_predict(A)
        except Exception as e:
            sys.stderr.write(f"[warn] state {i}: spectral failed for k={k}: {e}\n")
            continue

        # use Minkowski-based variance
        wvar = weighted_incluster_variance_minkowski(D_mink, labels)
        drop = tv - wvar
        tss = tv
        bss = drop

        r2 = 0.0 if tss <= 0 else (bss / tss)

        Np = X.shape[0]
        ch = np.nan
        if k > 1 and Np > k and wvar > 0:
            ch = (bss / (k - 1)) / (wvar / (Np - k))

        # silhouette on Euclidean
        sil = np.nan
        try:
            if Np >= 10 and len(np.unique(labels)) > 1:
                if args["silhouette_sample_cap"] and Np > args["silhouette_sample_cap"]:
                    rng = np.random.RandomState(args["random_state"] + i + k)
                    idx = rng.choice(Np, args["silhouette_sample_cap"], replace=False)
                    sil = silhouette_score(X[idx], labels[idx], metric="euclidean")
                else:
                    sil = silhouette_score(X, labels, metric="euclidean")
        except Exception:
            sil = np.nan

        rows.append({
            "state_index": i,
            "episode_id": episode_id,
            "t_in_episode": t_in_episode,
            "task_name": task_name,
            "instruction": instruction,
            "num_points": int(X.shape[0]),
            "action_dim": int(X.shape[1]),
            "k": int(k),
            "total_variance": float(tss),
            "weighted_incluster_variance": float(wvar),
            "variance_drop": float(drop),
            "r2": float(r2),
            "calinski_harabasz": float(ch),
            "silhouette": float(sil),
            "best_k": False,
        })

        metric_value = {
            "variance_drop": drop,
            "r2": r2,
            "ch": ch,
            "silhouette": sil,
        }[args["best_metric"]]
        cmp_val = -np.inf if (metric_value is None or np.isnan(metric_value)) else metric_value

        if (best["score"] is None) or (cmp_val > best["score"] + 1e-12) or \
           (abs(cmp_val - best["score"]) <= 1e-12 and (best["k"] is None or k < best["k"])):
            best.update({
                "score": cmp_val,
                "k": k,
                "wvar": wvar,
                "drop": drop,
                "r2": r2,
                "ch": ch,
                "sil": sil,
                "labels": labels.copy(),
            })

    best_cache = None
    if best["k"] is not None:
        rows.append({
            "state_index": i,
            "episode_id": episode_id,
            "t_in_episode": t_in_episode,
            "task_name": task_name,
            "instruction": instruction,
            "num_points": int(X.shape[0]),
            "action_dim": int(X.shape[1]),
            "k": int(best["k"]),
            "total_variance": float(tv),
            "weighted_incluster_variance": float(best["wvar"]),
            "variance_drop": float(best["drop"]),
            "r2": float(best["r2"]),
            "calinski_harabasz": float(best["ch"]),
            "silhouette": float(best["sil"]),
            "best_k": True,
        })
        best_cache = {
            "state_index": i,
            "labels": best["labels"],
            "idx": None if used_idx is None else used_idx.copy(),
            "k": int(best["k"]),
        }

    # save to disk so we can resume later
    _save_state_file(state_path, rows, best_cache)

    return rows, best_cache


# ------------------------------------------------------------------------------------
# main
# ------------------------------------------------------------------------------------

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--summary_csv", type=str, required=True)
    ap.add_argument("--actions_npz", type=str, required=True)
    ap.add_argument("--outdir", type=str, default="multimodality_out")

    ap.add_argument("--k_min", type=int, default=5)
    ap.add_argument("--k_max", type=int, default=8)
    ap.add_argument("--random_state", type=int, default=0)
    ap.add_argument("--top_n", type=int, default=50)
    ap.add_argument("--sample_cap", type=int, default=None)
    ap.add_argument("--best_metric", type=str,
                    choices=["variance_drop", "r2", "ch", "silhouette"], default="ch")
    ap.add_argument("--silhouette_sample_cap", type=int, default=5000)

    ap.add_argument("--method", type=str, choices=["minkowski", "dtw"], default="minkowski")
    ap.add_argument("--sigma", type=float, default=None)
    ap.add_argument("--dtw_window", type=int, default=None)
    ap.add_argument("--parallel_dtw", action="store_true")
    ap.add_argument("--max_workers", type=int, default=None)

    ap.add_argument("--plot_top_n", type=int, default=20)
    ap.add_argument("--ee_mode", type=str, choices=["pca", "first3", "custom"], default="first3")

    ap.add_argument("--n_jobs", type=int, default=1,
                    help="Number of processes to use for per-state clustering (>=1).")

    # NEW: limit how many states to process
    ap.add_argument("--max_states", type=int, default=None,
                    help="Process at most this many states (by index from 0).")

    args = ap.parse_args()
    os.makedirs(args.outdir, exist_ok=True)
    per_state_dir = os.path.join(args.outdir, "per_state")
    os.makedirs(per_state_dir, exist_ok=True)

    # load data
    meta_df = pd.read_csv(args.summary_csv)
    data = np.load(args.actions_npz, allow_pickle=True)
    actions_arr = data["actions"]
    N_states = len(actions_arr)

    if len(meta_df) != N_states:
        sys.stderr.write(f"[warn] summary_csv rows ({len(meta_df)}) != actions entries ({N_states}); proceeding by index.\n")

    # limit here
    if args.max_states is not None:
        N_effective = min(N_states, args.max_states)
    else:
        N_effective = N_states

    meta_records = meta_df.to_dict(orient="records")

    worker_args = dict(
        k_min=args.k_min,
        k_max=args.k_max,
        random_state=args.random_state,
        sample_cap=args.sample_cap,
        best_metric=args.best_metric,
        silhouette_sample_cap=args.silhouette_sample_cap,
        method=args.method,
        sigma=args.sigma,
        dtw_window=args.dtw_window,
        parallel_dtw=args.parallel_dtw,
        max_workers=args.max_workers,
        per_state_dir=per_state_dir,
    )

    print(f"Processing {N_effective} states (out of {N_states}) with {args.n_jobs} worker(s)...")

    best_labels_cache = {}

    if args.n_jobs == 1:
        _init_worker(actions_arr, meta_records, worker_args)
        for i in range(N_effective):
            if i % 50 == 0:
                print(f"  state {i}/{N_effective}...")
            _, best_i = _process_state(i)
            if best_i is not None:
                best_labels_cache[best_i["state_index"]] = best_i
    else:
        from concurrent.futures import ProcessPoolExecutor, as_completed
        with ProcessPoolExecutor(max_workers=args.n_jobs,
                                 initializer=_init_worker,
                                 initargs=(actions_arr, meta_records, worker_args)) as ex:
            futures = {ex.submit(_process_state, i): i for i in range(N_effective)}
            for fut in as_completed(futures):
                i = futures[fut]
                if i % 50 == 0:
                    print(f"  state {i}/{N_effective}...")
                rows_i, best_i = fut.result()
                if best_i is not None:
                    best_labels_cache[best_i["state_index"]] = best_i

    # --------------------------------------------------
    # Aggregation: only gather the ones we actually processed
    # --------------------------------------------------
    all_rows = []
    for i in range(N_effective):   # <-- use N_effective here
        state_path = _state_file_path(per_state_dir, i)
        if not os.path.exists(state_path):
            continue
        rows_i, _ = _load_state_file(state_path)
        all_rows.extend(rows_i)

    if not all_rows:
        print("No rows computed — check inputs.")
        return

    metrics_df = pd.DataFrame(all_rows)
    best_df = metrics_df[metrics_df["best_k"] == True].copy()  # noqa: E712

    sort_col = {
        "variance_drop": "variance_drop",
        "r2": "r2",
        "ch": "calinski_harabasz",
        "silhouette": "silhouette",
    }[args.best_metric]
    best_df.sort_values([sort_col], ascending=False, inplace=True)

    metrics_csv = os.path.join(args.outdir, "metrics_per_state.csv")
    best_csv = os.path.join(args.outdir, "top_states.csv")
    metrics_df.to_csv(metrics_csv, index=False)
    best_df.head(args.top_n).to_csv(best_csv, index=False)
    print(f"\nSaved:\n- {os.path.abspath(metrics_csv)}\n- {os.path.abspath(best_csv)}")

    print(f"\nTop states (best-k) by {sort_col}:")
    cols_to_show = [
        "state_index", "episode_id", "t_in_episode", "k", "num_points",
        "total_variance", "weighted_incluster_variance",
        "variance_drop", "r2", "calinski_harabasz", "silhouette"
    ]
    print(best_df[cols_to_show].head(args.top_n).to_string(index=False))

    # ---------- plots ----------
    hist_path = os.path.join(args.outdir, "variance_hist.png")
    plt.figure(figsize=(7, 5))
    plt.hist(best_df["weighted_incluster_variance"].values, bins=40)
    plt.xlabel("Weighted in-cluster variance (best k per state)")
    plt.ylabel("Count of states")
    plt.title("Histogram of weighted in-cluster variance (best-k)")
    plt.tight_layout()
    plt.savefig(hist_path, dpi=150)
    plt.close()
    print(f"\nSaved histogram: {os.path.abspath(hist_path)}")

    # per-k histos
    eval_df = metrics_df[metrics_df["best_k"] == False].copy()
    ks = sorted(eval_df["k"].unique().tolist())
    all_vals = eval_df["weighted_incluster_variance"].values
    if all_vals.size:
        bins = np.histogram_bin_edges(all_vals, bins=40)

        overlay_path = os.path.join(args.outdir, "variance_hist_by_k_overlay.png")
        plt.figure(figsize=(8, 6))
        for k in ks:
            vals = eval_df.loc[eval_df["k"] == k, "weighted_incluster_variance"].values
            if vals.size == 0:
                continue
            plt.hist(vals, bins=bins, histtype="step", linewidth=1.5, label=f"k={k}")
        plt.xlabel("Weighted in-cluster variance (per specific k)")
        plt.ylabel("Count of states")
        plt.title("Per-k histogram (overlay)")
        plt.legend(frameon=False)
        plt.tight_layout()
        plt.savefig(overlay_path, dpi=150)
        plt.close()
        print(f"Saved per-k overlay histogram: {os.path.abspath(overlay_path)}")

        grid_path = os.path.join(args.outdir, "variance_hist_by_k_grid.png")
        n = len(ks)
        cols = min(4, n)
        rows = int(np.ceil(n / cols))
        fig, axes = plt.subplots(rows, cols, figsize=(5 * cols, 4 * rows),
                                 squeeze=False, sharex=True, sharey=True)
        for idx, k in enumerate(ks):
            r, c = divmod(idx, cols)
            ax = axes[r][c]
            vals = eval_df.loc[eval_df["k"] == k, "weighted_incluster_variance"].values
            ax.hist(vals, bins=bins)
            ax.set_title(f"k={k}")
            ax.set_xlabel("weighted in-cluster var")
            ax.set_ylabel("count")
        for idx in range(n, rows * cols):
            r, c = divmod(idx, cols)
            axes[r][c].axis("off")
        fig.suptitle("Per-k histograms (weighted in-cluster variance)", y=0.995)
        fig.tight_layout(rect=[0, 0.02, 1, 0.97])
        fig.savefig(grid_path, dpi=150)
        plt.close(fig)
        print(f"Saved per-k grid histogram: {os.path.abspath(grid_path)}")

        drop_overlay_path = os.path.join(args.outdir, "variance_drop_by_k_overlay.png")
        plt.figure(figsize=(8, 6))
        all_drop_vals = eval_df["variance_drop"].values
        drop_bins = np.histogram_bin_edges(all_drop_vals, bins=40)
        for k in ks:
            vals = eval_df.loc[eval_df["k"] == k, "variance_drop"].values
            if vals.size == 0:
                continue
            plt.hist(vals, bins=drop_bins, histtype="step", linewidth=1.5, label=f"k={k}")
        plt.xlabel("Variance drop = total var - weighted in-cluster var (per specific k)")
        plt.ylabel("Count of states")
        plt.title("Per-k histogram of variance drop (overlay)")
        plt.legend(frameon=False)
        plt.tight_layout()
        plt.savefig(drop_overlay_path, dpi=150)
        plt.close()
        print(f"Saved per-k variance-drop overlay: {os.path.abspath(drop_overlay_path)}")

    # ========================== Top-N visualizations ==========================
    top_plot_dir = os.path.join(args.outdir, "top_plots")
    os.makedirs(top_plot_dir, exist_ok=True)

    # we need actions_arr again here (already loaded)
    top_rows = best_df.head(args.plot_top_n).reset_index(drop=True)
    print(f"\nRendering cluster plots for top {len(top_rows)} states (by {sort_col})...")

    for rank, row in enumerate(top_rows.itertuples()):
        state_idx = int(row.state_index)
        ep_id = str(row.episode_id)
        t_in_ep = int(row.t_in_episode)
        best_k = int(row.k)

        acts_raw = actions_arr[state_idx]
        if acts_raw.ndim == 3:
            Kc, Tc, A = acts_raw.shape
            traj_matrix = acts_raw.reshape(Kc * Tc, A)
            index_rows = [(state_idx, k, t) for k in range(Kc) for t in range(Tc)]
        elif acts_raw.ndim == 2:
            Tc, A = acts_raw.shape
            traj_matrix = acts_raw
            index_rows = [(state_idx, 0, t) for t in range(Tc)]
        else:
            sys.stderr.write(f"[warn] Skipping state {state_idx}: unsupported shape {acts_raw.shape}\n")
            continue

        X = traj_matrix.astype(float, copy=False)

        cache = best_labels_cache.get(state_idx, None)
        if cache is None or cache.get("labels") is None:
            # rebuild affinity & spectral once at best_k
            A, _, _ = build_affinity_for_points(
                X,
                method=args.method,
                sigma=args.sigma,
                dtw_window=args.dtw_window,
                random_state=args.random_state,
                parallel_dtw=args.parallel_dtw,
                max_workers=args.max_workers,
            )
            cl = SpectralClustering(
                n_clusters=best_k,
                affinity="precomputed",
                assign_labels="kmeans",
                random_state=args.random_state,
            )
            labels_for_plot = cl.fit_predict(A)
            used_idx_for_plot = None
        else:
            labels_for_plot = cache["labels"]
            if cache["idx"] is not None:
                used_idx_for_plot = cache["idx"]
                X = X[used_idx_for_plot]
                index_rows = [index_rows[j] for j in used_idx_for_plot.tolist()]
            else:
                used_idx_for_plot = None

        xyz = to_xyz(X, mode=args.ee_mode)
        sub = os.path.join(top_plot_dir, f"rank{rank + 1:02d}_ep{ep_id}_state_{t_in_ep:06d}")
        os.makedirs(sub, exist_ok=True)

        title = (f"ep={ep_id} state={t_in_ep} | best-k={best_k} | "
                 f"{args.best_metric}={getattr(row, sort_col):.4f}")
        overview_png = os.path.join(sub, "overview.png")
        percluster_png = os.path.join(sub, "percluster.png")

        plot_actions_xyz(
            xyz, labels_for_plot, index_rows,
            png_path=overview_png, title=title,
            point_size=1, line_width=1.0, line_alpha=0.4,
            line_color_mode="by_dominant_cluster",
        )
        n_cls = len(np.unique(labels_for_plot))
        plot_per_cluster_panels(
            xyz, labels_for_plot, index_rows, n_clusters=n_cls,
            png_path=percluster_png, title_prefix=f"ep={ep_id} state={t_in_ep}",
            point_size=1, line_width=1.2, line_alpha=0.55,
        )

        idx_to_save = np.array([], dtype=int) if used_idx_for_plot is None else used_idx_for_plot
        np.savez_compressed(
            os.path.join(args.outdir, f"best_labels_state_{state_idx:06d}.npz"),
            labels=labels_for_plot, idx=idx_to_save, k=best_k,
        )

    print(f"Top-N plots written under: {os.path.abspath(top_plot_dir)}")


if __name__ == "__main__":
    main()