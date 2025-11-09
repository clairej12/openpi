#!/usr/bin/env python3
import argparse, os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.cluster import SpectralClustering
from sklearn.decomposition import PCA
from scipy.spatial.distance import cdist
from concurrent.futures import ProcessPoolExecutor

# ---------- distance + clustering ----------
def _as_2d(x):
    x = np.asarray(x)
    if x.ndim == 1: return x.reshape(-1, 1).astype(float, copy=False)
    if x.ndim >= 3: x = x.reshape(x.shape[0], -1)
    return x.astype(float, copy=False)

def symmetrized_l2_minkowski(X, Y):
    X = _as_2d(X); Y = _as_2d(Y)
    dists = cdist(X, Y, metric="sqeuclidean")
    d1_sq = np.sum(np.min(dists, axis=1))
    d2_sq = np.sum(np.min(dists, axis=0))
    return float(np.sqrt((d1_sq + d2_sq) / 2.0))

def compute_minkowski_distance_matrix(trajectories):
    N = len(trajectories); D = np.zeros((N, N), dtype=float)
    for i in range(N):
        for j in range(i+1, N):
            d = symmetrized_l2_minkowski(trajectories[i], trajectories[j])
            D[i, j] = D[j, i] = d
    return D

def spectral_cluster_minkowski(trajectories, n_clusters=3, sigma=None, random_state=0):
    D = compute_minkowski_distance_matrix(trajectories)
    pos = D[D > 0]
    if sigma is None: sigma = float(np.median(pos)) if pos.size else 1.0
    A = np.exp(-D**2 / (2 * sigma**2)); np.fill_diagonal(A, 1.0)
    cl = SpectralClustering(n_clusters=n_clusters, affinity="precomputed", assign_labels="kmeans", random_state=random_state)
    labels = cl.fit_predict(A)
    return labels, D, A

def dtw_distance_band(X, Y, window=None):
    X = _as_2d(X); Y = _as_2d(Y)
    len_x, len_y = X.shape[0], Y.shape[0]
    if len_x == 0 and len_y == 0: return 0.0
    if len_x == 0 or len_y == 0:  return np.inf
    if window is None: window = max(len_x, len_y) // 10
    window = int(window); w = max(window, abs(len_x - len_y))
    prev = np.full(len_y + 1, np.inf); curr = np.full(len_y + 1, np.inf); prev[0] = 0.0
    for i in range(1, len_x + 1):
        curr.fill(np.inf); j_start = max(1, i - w); j_end = min(len_y, i + w); xi = X[i - 1]
        for j in range(j_start, j_end + 1):
            dist = np.dot(xi - Y[j - 1], xi - Y[j - 1])
            curr[j] = dist + min(curr[j - 1], prev[j], prev[j - 1])
        prev, curr = curr, prev
    return float(np.sqrt(prev[len_y]))

_TRAJ = None
def _init_worker(trajectories):  # noqa
    global _TRAJ; _TRAJ = trajectories
def _pair_dtw(args):  # noqa
    i, j, win = args; return i, j, dtw_distance_band(_TRAJ[i], _TRAJ[j], window=win)

def compute_dtw_matrix(trajectories, window=None, parallelize=False, max_workers=None):
    N = len(trajectories); D = np.zeros((N, N), dtype=float)
    if not parallelize or N < 64:
        for i in range(N):
            for j in range(i + 1, N):
                d = dtw_distance_band(trajectories[i], trajectories[j], window=window)
                D[i, j] = D[j, i] = d
        return D
    args = [(i, j, window) for i in range(N) for j in range(i + 1, N)]
    with ProcessPoolExecutor(max_workers=max_workers, initializer=_init_worker, initargs=(trajectories,)) as ex:
        for i, j, d in ex.map(_pair_dtw, args, chunksize=32):
            D[i, j] = D[j, i] = d
    return D

def spectral_cluster_dtw(trajectories, n_clusters=3, sigma=None, window=None, random_state=0, parallelize=False, max_workers=None):
    D = compute_dtw_matrix(trajectories, window=window, parallelize=parallelize, max_workers=max_workers)
    pos = D[D > 0]; sigma = float(np.median(pos)) if (sigma is None and pos.size) else (sigma or 1.0)
    A = np.exp(-D**2 / (2.0 * sigma**2)); np.fill_diagonal(A, 1.0)
    cl = SpectralClustering(n_clusters=n_clusters, affinity="precomputed", assign_labels="kmeans", random_state=random_state)
    labels = cl.fit_predict(A)
    return labels, D, A

# ---------- viz helpers ----------
def to_xyz(actions, mode="pca", pca_model=None, kinematic_map=None):
    X = np.asarray(actions, dtype=float)
    if mode == "first3":
        if X.shape[1] < 3:
            pad = np.zeros((X.shape[0], 3 - X.shape[1]))
            return np.hstack([X, pad])[:, :3]
        return X[:, :3]
    elif mode == "custom":
        if kinematic_map is None: raise ValueError("custom mode requires kinematic_map")
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
                     line_color_mode="per_sample",
                     start_marker_size=10, end_marker_size=20):
    import numpy as np, matplotlib.lines as mlines
    from matplotlib.cm import get_cmap
    fig = plt.figure(figsize=(7, 6))
    ax = fig.add_subplot(111, projection="3d")
    xyz = np.asarray(xyz); labels = np.asarray(labels).reshape(-1)
    if labels.shape[0] != xyz.shape[0]: raise ValueError("labels length mismatch")
    scatter = ax.scatter(xyz[:,0], xyz[:,1], xyz[:,2], c=labels, s=point_size, alpha=0.95)
    start_h = end_h = None
    if index_rows is not None and len(index_rows) == xyz.shape[0]:
        sample_idx = np.array([r[1] for r in index_rows], dtype=int)
        action_idx = np.array([r[2] for r in index_rows], dtype=int)
        unique_samples = np.unique(sample_idx)
        if line_color_mode == "per_sample":
            cmap = get_cmap("tab20", len(unique_samples))
            sample_to_color = {s: cmap(i % cmap.N) for i, s in enumerate(unique_samples)}
        else:
            sample_to_color = {}
        for s in unique_samples:
            m = (sample_idx == s)
            order = np.argsort(action_idx[m])
            pts = xyz[m][order]; labs = labels[m][order]
            if pts.shape[0] >= 2:
                if line_color_mode == "by_dominant_cluster":
                    vals, counts = np.unique(labs, return_counts=True)
                    dom = vals[np.argmax(counts)]
                    line_color = scatter.cmap(scatter.norm(dom))
                else:
                    line_color = sample_to_color[s]
                ax.plot(pts[:,0], pts[:,1], pts[:,2], linewidth=line_width, alpha=line_alpha, color=line_color)
            p0, p1 = pts[0], pts[-1]
            ax.scatter([p0[0]],[p0[1]],[p0[2]], s=start_marker_size, c="none", edgecolor="k", linewidths=1.0, marker="o")
            ax.scatter([p1[0]],[p1[1]],[p1[2]], s=end_marker_size, c="k", marker="x", linewidths=1.5)
        import matplotlib.lines as mlines
        start_h = mlines.Line2D([], [], color="k", marker="o", markersize=6, markerfacecolor="none", linestyle="None", label="chunk start")
        end_h   = mlines.Line2D([], [], color="k", marker="x", markersize=7, linestyle="None", label="chunk end")
    legend_items = [mlines.Line2D([], [], color=scatter.cmap(scatter.norm(lab)), marker="s", linestyle="None", markersize=6, label=f"cluster {lab}") for lab in np.unique(labels)]
    if start_h is not None: legend_items.append(start_h)
    if end_h   is not None: legend_items.append(end_h)
    if legend_items: ax.legend(handles=legend_items, loc="best", frameon=False, fontsize=9)
    ax.set_title(title); ax.set_xlabel("X"); ax.set_ylabel("Y"); ax.set_zlabel("Z")
    x_limits = ax.get_xlim3d(); y_limits = ax.get_ylim3d(); z_limits = ax.get_zlim3d()
    x_range = abs(x_limits[1]-x_limits[0]); y_range = abs(y_limits[1]-y_limits[0]); z_range = abs(z_limits[1]-z_limits[0])
    max_range = max([x_range,y_range,z_range])
    xm, ym, zm = np.mean(x_limits), np.mean(y_limits), np.mean(z_limits)
    ax.set_xlim3d([xm-max_range/2, xm+max_range/2])
    ax.set_ylim3d([ym-max_range/2, ym+max_range/2])
    ax.set_zlim3d([zm-max_range/2, zm+max_range/2])
    fig.tight_layout(); fig.savefig(png_path, dpi=150); plt.close(fig)

def plot_per_cluster_panels(xyz, per_point_labels, index_rows, n_clusters, png_path, title_prefix="State",
                            point_size=3, line_width=1.0, line_alpha=0.6):
    import numpy as np, matplotlib.lines as mlines
    from matplotlib.cm import get_cmap
    xyz = np.asarray(xyz); labels = np.asarray(per_point_labels).reshape(-1)
    assert xyz.shape[0] == labels.shape[0] == len(index_rows)
    sample_idx = np.array([r[1] for r in index_rows], dtype=int)
    action_idx = np.array([r[2] for r in index_rows], dtype=int)
    uniq_clusters = np.unique(labels); C = int(max(n_clusters, uniq_clusters.max() + 1))
    cmap = plt.get_cmap("tab10", C)
    chunk_ids = np.unique(sample_idx); chunk_to_cluster = {}
    for k in chunk_ids:
        m = (sample_idx == k)
        if not np.any(m): continue
        vals, counts = np.unique(labels[m], return_counts=True)
        chunk_to_cluster[k] = int(vals[np.argmax(counts)])
    cols = min(C, 4); rows = int(np.ceil(C / cols))
    fig = plt.figure(figsize=(5*cols, 4.5*rows))
    def _equal(ax):
        xlim = ax.get_xlim3d(); ylim = ax.get_ylim3d(); zlim = ax.get_zlim3d()
        xr = abs(xlim[1]-xlim[0]); yr=abs(ylim[1]-ylim[0]); zr=abs(zlim[1]-zlim[0])
        mr = max(xr, yr, zr)
        xm=np.mean(xlim); ym=np.mean(ylim); zm=np.mean(zlim)
        ax.set_xlim3d([xm-mr/2, xm+mr/2]); ax.set_ylim3d([ym-mr/2, ym+mr/2]); ax.set_zlim3d([zm-mr/2, zm+mr/2])
    for ci in range(C):
        ax = fig.add_subplot(rows, cols, ci+1, projection="3d")
        ax.set_title(f"Cluster {ci}"); ax.set_xlabel("X"); ax.set_ylabel("Y"); ax.set_zlabel("Z")
        chunks_in_ci = [k for k, lab in chunk_to_cluster.items() if lab == ci]
        if not chunks_in_ci:
            ax.text(0.5, 0.5, 0.5, "No chunks", transform=ax.transAxes, ha="center", va="center"); continue
        color = cmap(ci)
        for k in chunks_in_ci:
            m = (sample_idx == k)
            pts = xyz[m]
            ax.scatter(pts[:,0], pts[:,1], pts[:,2], c=[color], s=point_size, alpha=0.95)
        for k in chunks_in_ci:
            m = (sample_idx == k)
            order = np.argsort(action_idx[m]); pts = xyz[m][order]
            if pts.shape[0] >= 2: ax.plot(pts[:,0], pts[:,1], pts[:,2], color=color, linewidth=line_width, alpha=line_alpha)
            p0, p1 = pts[0], pts[-1]
            ax.scatter([p0[0]],[p0[1]],[p0[2]], s=36, c="none", edgecolor="k", linewidths=1.0, marker="o")
            ax.scatter([p1[0]],[p1[1]],[p1[2]], s=48, c="k", marker="x", linewidths=1.5)
        _equal(ax)
    start_h = mlines.Line2D([], [], color="k", marker="o", markersize=6, markerfacecolor="none", linestyle="None", label="chunk start")
    end_h   = mlines.Line2D([], [], color="k", marker="x", markersize=7, linestyle="None", label="chunk end")
    fig.legend(handles=[start_h, end_h], loc="upper right", frameon=False)
    fig.suptitle(f"{title_prefix}: per-cluster chunk views", y=0.995)
    fig.tight_layout(rect=[0, 0.00, 1, 0.96]); fig.savefig(png_path, dpi=150); plt.close(fig)

# ---------- small utils ----------
def _clean_byteslike(x):
    if isinstance(x, str) and x.startswith("b'") and x.endswith("'"):
        try:
            return bytes(x[2:-1], 'utf-8').decode('unicode_escape').encode('latin1', 'backslashreplace').decode('utf-8')
        except Exception:
            return x[2:-1]
    return x

def _episode_id_from_row(row):
    ep = row.get("episode_id", None)
    if pd.isna(ep) or ep is None:
        if "episode_index" in row and not pd.isna(row["episode_index"]):
            return f"ep{int(row['episode_index']):06d}"
        return "ep_unknown"
    if isinstance(ep, (int, np.integer)):
        return f"ep{int(ep):06d}"
    ep = str(ep)
    ep = _clean_byteslike(ep)
    return ep

def _state_slug(t_in_episode):
    try:
        t = int(t_in_episode)
    except Exception:
        t = 0
    return f"state_{t:06d}"

def _load_per_state_csv(csv_path):
    df = pd.read_csv(csv_path)
    # infer dims
    a_cols = [c for c in df.columns if c.startswith("a")]
    action_dim = len(a_cols)
    num_actions = len(df)
    left_img = df["left_img_path"].iloc[0] if "left_img_path" in df.columns and len(df) else None
    wrist_img = df["wrist_img_path"].iloc[0] if "wrist_img_path" in df.columns and len(df) else None
    return df, num_actions, action_dim, left_img, wrist_img

# ---------- main ----------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--summary_csv", type=str, required=True)
    ap.add_argument("--actions_npz", type=str, required=True)
    ap.add_argument("--outdir", type=str, default="clusters_out")
    ap.add_argument("--method", type=str, choices=["minkowski","dtw"], default="minkowski")
    ap.add_argument("--n_clusters", type=int, default=3)
    ap.add_argument("--sigma", type=float, default=None)
    ap.add_argument("--dtw_window", type=int, default=None)
    ap.add_argument("--parallel_dtw", action="store_true")
    ap.add_argument("--max_workers", type=int, default=None)
    ap.add_argument("--ee-mode", type=str, choices=["pca","first3","custom"], default="pca")
    ap.add_argument("--random_state", type=int, default=0)
    ap.add_argument("--cluster_scope", choices=["chunk_only","first_action_across_samples","all_actions_across_samples"], default="chunk_only")
    ap.add_argument("--cluster_target", choices=["actions","chunks"], default="actions")
    # NEW: default do NOT overwrite; allow opt-in overwrite
    ap.add_argument("--overwrite_results", action="store_true",
                    help="If set, recompute and overwrite existing per-state CSV/plots.")
    args = ap.parse_args()

    os.makedirs(args.outdir, exist_ok=True)

    # Load provenance + actions
    df = pd.read_csv(args.summary_csv)
    data = np.load(args.actions_npz, allow_pickle=True)
    actions_arr = data["actions"]

    rows_meta = []
    N = len(actions_arr)
    per_episode_meta = {}

    for i in range(N):
        acts_raw = actions_arr[i]
        acts = (np.asarray(acts_raw, dtype=object)
                if isinstance(actions_arr, np.ndarray) and actions_arr.dtype == object
                else np.asarray(acts_raw, dtype=float))

        row = df.iloc[i]
        episode_id  = _episode_id_from_row(row)
        task_name   = _clean_byteslike(row.get("task_name", ""))
        instruction = _clean_byteslike(row.get("instruction", ""))
        t_in_ep     = int(row.get("t_in_episode", -1))
        left_img    = row.get("left_img_path", None)
        wrist_img   = row.get("wrist_img_path", None)

        ep_dir = os.path.join(args.outdir, episode_id)
        os.makedirs(ep_dir, exist_ok=True)
        state_slug = _state_slug(t_in_ep)

        png_path   = os.path.join(ep_dir, f"{state_slug}_{args.cluster_target}_{args.method}.png")
        panel_path = os.path.join(ep_dir, f"{state_slug}_{args.cluster_target}_{args.method}_percluster.png")
        csv_path   = os.path.join(ep_dir, f"{state_slug}_{args.method}.csv")

        have_csv   = os.path.exists(csv_path)
        have_png   = os.path.exists(png_path)
        have_panel = os.path.exists(panel_path)

        # Fast path: results already exist and overwrite is off
        if have_csv and have_png and have_panel and not args.overwrite_results:
            # Index existing results into summaries without recompute
            print(f"Skipping state {i} ({episode_id} t={t_in_ep}): results exist.")
            df_state, num_actions, action_dim, left_img_old, wrist_img_old = _load_per_state_csv(csv_path)
            row_meta = {
                "episode_id": episode_id,
                "t_in_episode": t_in_ep,
                "task_name": task_name,
                "instruction": instruction,
                "state_global_index": i,
                "num_actions": int(num_actions),
                "action_dim": int(action_dim),
                "plot_path": os.path.abspath(png_path),
                "percluster_plot_path": os.path.abspath(panel_path),
                "clusters_csv": os.path.abspath(csv_path),
                "left_img_path": left_img_old if left_img_old is not None else left_img,
                "wrist_img_path": wrist_img_old if wrist_img_old is not None else wrist_img,
            }
            rows_meta.append(row_meta)
            per_episode_meta.setdefault(episode_id, []).append(row_meta)
            continue

        # If CSV exists (overwrite off) but a plot is missing, rebuild the plots from CSV (no recluster).
        if have_csv and not args.overwrite_results and (not have_png or not have_panel):
            print(f"Rebuilding plots for state {i} ({episode_id} t={t_in_ep}) from existing CSV.")
            df_state = pd.read_csv(csv_path)
            a_cols = [c for c in df_state.columns if c.startswith("a")]
            traj_matrix = df_state[a_cols].values
            per_point_labels = df_state["cluster"].values.astype(int)
            # rebuild index_rows
            if "sample_index" in df_state.columns and "action_in_chunk" in df_state.columns:
                index_rows = [(i, int(si), int(ai)) for si, ai in zip(df_state["sample_index"].values,
                                                                      df_state["action_in_chunk"].values)]
            else:
                index_rows = [(i, 0, t) for t in range(traj_matrix.shape[0])]

            xyz = to_xyz(traj_matrix, mode=args.ee_mode)
            title = (f"{episode_id} {state_slug} task={task_name} "
                     f"target={args.cluster_target}, N_points={traj_matrix.shape[0]}, "
                     f"method={args.method}")

            if not have_png:
                plot_actions_xyz(xyz, per_point_labels, index_rows, png_path=png_path, title=title,
                                 point_size=1, line_width=1.0, line_alpha=0.4, line_color_mode="by_dominant_cluster")
            if not have_panel:
                n_cls = len(np.unique(per_point_labels))
                plot_per_cluster_panels(xyz, per_point_labels, index_rows, n_clusters=n_cls,
                                        png_path=panel_path, title_prefix=f"{episode_id} {state_slug}",
                                        point_size=1, line_width=1.2, line_alpha=0.55)

            num_actions = traj_matrix.shape[0]
            action_dim = traj_matrix.shape[1]
            left_img_old = df_state["left_img_path"].iloc[0] if "left_img_path" in df_state.columns else left_img
            wrist_img_old = df_state["wrist_img_path"].iloc[0] if "wrist_img_path" in df_state.columns else wrist_img

            row_meta = {
                "episode_id": episode_id,
                "t_in_episode": t_in_ep,
                "task_name": task_name,
                "instruction": instruction,
                "state_global_index": i,
                "num_actions": int(num_actions),
                "action_dim": int(action_dim),
                "plot_path": os.path.abspath(png_path),
                "percluster_plot_path": os.path.abspath(panel_path),
                "clusters_csv": os.path.abspath(csv_path),
                "left_img_path": left_img_old,
                "wrist_img_path": wrist_img_old,
            }
            rows_meta.append(row_meta)
            per_episode_meta.setdefault(episode_id, []).append(row_meta)
            continue

        # Otherwise: compute clustering (either brand new, or overwrite requested)
        if acts.ndim == 2:
            T, A = acts.shape
            if args.cluster_target == "chunks":
                traj_matrix = acts
                index_rows = [(i, 0, t) for t in range(T)]
                per_point_labels = np.zeros(T, dtype=int)
            else:
                traj_matrix = acts
                index_rows = [(i, 0, t) for t in range(T)]
                trajs_points = [traj_matrix[n:n+1, :] for n in range(traj_matrix.shape[0])]
                if args.method == "minkowski":
                    point_labels, D, A_mat = spectral_cluster_minkowski(
                        trajs_points, n_clusters=args.n_clusters,
                        sigma=args.sigma, random_state=args.random_state)
                else:
                    point_labels, D, A_mat = spectral_cluster_dtw(
                        trajs_points, n_clusters=args.n_clusters,
                        sigma=args.sigma, window=args.dtw_window,
                        random_state=args.random_state,
                        parallelize=args.parallel_dtw, max_workers=args.max_workers)
                per_point_labels = np.asarray(point_labels, dtype=int)

        elif acts.ndim == 3:
            K, T, A = acts.shape
            if args.cluster_target == "chunks":
                trajs_chunks = [acts[k, :, :] for k in range(K)]
                if args.method == "minkowski":
                    chunk_labels, D, A_mat = spectral_cluster_minkowski(
                        trajs_chunks, n_clusters=args.n_clusters,
                        sigma=args.sigma, random_state=args.random_state)
                else:
                    chunk_labels, D, A_mat = spectral_cluster_dtw(
                        trajs_chunks, n_clusters=args.n_clusters,
                        sigma=args.sigma, window=args.dtw_window,
                        random_state=args.random_state,
                        parallelize=args.parallel_dtw, max_workers=args.max_workers)
                traj_matrix = acts.reshape(K*T, A)
                index_rows  = [(i, k, t) for k in range(K) for t in range(T)]
                per_point_labels = np.repeat(np.asarray(chunk_labels, dtype=int), T)
            else:
                traj_matrix = acts.reshape(K*T, A)
                index_rows  = [(i, k, t) for k in range(K) for t in range(T)]
                trajs_points = [traj_matrix[n:n+1, :] for n in range(traj_matrix.shape[0])]
                if args.method == "minkowski":
                    point_labels, D, A_mat = spectral_cluster_minkowski(
                        trajs_points, n_clusters=args.n_clusters,
                        sigma=args.sigma, random_state=args.random_state)
                else:
                    point_labels, D, A_mat = spectral_cluster_dtw(
                        trajs_points, n_clusters=args.n_clusters,
                        sigma=args.sigma, window=args.dtw_window,
                        random_state=args.random_state,
                        parallelize=args.parallel_dtw, max_workers=args.max_workers)
                per_point_labels = np.asarray(point_labels, dtype=int)
        else:
            raise ValueError(f"Unsupported actions shape for state {i}: {acts.shape}")

        # Project + plots (save under episode folder, using local state slug)
        xyz = to_xyz(traj_matrix, mode=args.ee_mode)
        title = (f"{episode_id} {state_slug} task={task_name} "
                 f"target={args.cluster_target}, N_points={traj_matrix.shape[0]}, "
                 f"method={args.method}, n_clusters={args.n_clusters}")

        plot_actions_xyz(xyz, per_point_labels, index_rows, png_path=png_path, title=title,
                         point_size=1, line_width=1.0, line_alpha=0.4, line_color_mode="by_dominant_cluster")

        n_cls = len(np.unique(per_point_labels))
        plot_per_cluster_panels(xyz, per_point_labels, index_rows, n_clusters=n_cls,
                                png_path=panel_path, title_prefix=f"{episode_id} {state_slug}",
                                point_size=1, line_width=1.2, line_alpha=0.55)

        # Per-state expanded CSV (save under episode folder)
        per_state = pd.DataFrame({
            "episode_id": episode_id,
            "task_name": task_name,
            "instruction": instruction,
            "t_in_episode": t_in_ep,
            "state_global_index": i,
            "sample_index": [r[1] for r in index_rows],
            "action_in_chunk": [r[2] for r in index_rows],
            **{f"a{j}": traj_matrix[:, j] for j in range(traj_matrix.shape[1])},
            "cluster": per_point_labels.astype(int),
            "left_img_path": left_img,
            "wrist_img_path": wrist_img,
        })
        per_state.to_csv(csv_path, index=False)

        # Meta row for global summary
        row_meta = {
            "episode_id": episode_id,
            "t_in_episode": t_in_ep,
            "task_name": task_name,
            "instruction": instruction,
            "state_global_index": i,
            "num_actions": int(traj_matrix.shape[0]),
            "action_dim": int(traj_matrix.shape[1]),
            "plot_path": os.path.abspath(png_path),
            "percluster_plot_path": os.path.abspath(panel_path),
            "clusters_csv": os.path.abspath(csv_path),
            "left_img_path": left_img,
            "wrist_img_path": wrist_img,
        }
        rows_meta.append(row_meta)
        per_episode_meta.setdefault(episode_id, []).append(row_meta)

    # Global summary
    global_summary = pd.DataFrame(rows_meta).sort_values(["episode_id","t_in_episode"])
    global_summary.to_csv(os.path.join(args.outdir, "summary_per_state.csv"), index=False)

    # Per-episode summaries
    for ep_id, items in per_episode_meta.items():
        ep_df = pd.DataFrame(items).sort_values("t_in_episode")
        ep_df.to_csv(os.path.join(args.outdir, ep_id, "episode_summary.csv"), index=False)

    print(f"Done.\n- Global summary: {os.path.abspath(os.path.join(args.outdir, 'summary_per_state.csv'))}\n- Per-episode summaries in each {args.outdir}/<episode_id>/")

if __name__ == "__main__":
    main()

# #!/usr/bin/env python3
# """
# analyze_actions_per_state.py

# Reads:
#   - <prefix>_summary.csv  (from your batch eval script)
#   - <prefix>_actions.npz  (optional raw actions; recommended)

# For each starting state (row i):
#   - Extracts the full action chunk actions[i] of shape (T_i, A)
#   - Clusters the T_i actions into K clusters using:
#       * spectral clustering + Minkowski-inspired distance  OR
#       * spectral clustering + DTW distance
#   - Plots a 3D scatter of the actions colored by cluster
#     (projection to interpretable space chosen by --ee-mode)

# Outputs:
#   - <outdir>/state_<i>_clusters.csv        (action_index -> cluster)
#   - <outdir>/state_<i>_plot.png            (3D scatter colored by cluster)
#   - <outdir>/summary_per_state.csv         (one row per state with meta)

# Usage:
#   uv run analyze_actions_per_state.py \
#     --summary_csv droid_sanity/summary.csv \
#     --actions_npz droid_sanity/actions.npz \
#     --outdir droid_sanity/clusters \
#     --method minkowski --n_clusters 8 \
#     --ee-mode first3
# """

# import argparse
# import os
# import numpy as np
# import pandas as pd
# import matplotlib.pyplot as plt

# from sklearn.cluster import SpectralClustering
# from sklearn.decomposition import PCA
# from scipy.spatial.distance import cdist
# from concurrent.futures import ProcessPoolExecutor

# # -------------------------
# # Distance + clustering code (as provided, with a few guards)
# # -------------------------

# def _as_2d(x):
#     x = np.asarray(x)
#     if x.ndim == 1:
#         return x.reshape(-1, 1).astype(float, copy=False)
#     if x.ndim >= 3:  # flatten extras; keep time axis
#         x = x.reshape(x.shape[0], -1)
#     return x.astype(float, copy=False)

# def symmetrized_l2_minkowski(X, Y):
#     """
#     X, Y: arrays of shape (len_X, dim), (len_Y, dim)
#     Returns: scalar distance
#     """
#     X = _as_2d(X); Y = _as_2d(Y)
#     dists = cdist(X, Y, metric="sqeuclidean")  # shape (len_X, len_Y)
#     d1_sq = np.sum(np.min(dists, axis=1))  # min over j for each x_i
#     d2_sq = np.sum(np.min(dists, axis=0))  # min over i for each y_j
#     return float(np.sqrt((d1_sq + d2_sq) / 2.0))

# def compute_minkowski_distance_matrix(trajectories):
#     """
#     trajectories: list of arrays of shape (len_i, dim)
#     Returns: NxN distance matrix
#     """
#     N = len(trajectories)
#     D = np.zeros((N, N), dtype=float)
#     for i in range(N):
#         for j in range(i+1, N):
#             d = symmetrized_l2_minkowski(trajectories[i], trajectories[j])
#             D[i, j] = D[j, i] = d
#     return D

# def spectral_cluster_minkowski(trajectories, n_clusters=3, sigma=None, random_state=0):
#     D = compute_minkowski_distance_matrix(trajectories)
#     # choose sigma as median distance if not provided (robust)
#     pos = D[D > 0]
#     if sigma is None:
#         sigma = float(np.median(pos)) if pos.size else 1.0
#     # convert to affinity
#     A = np.exp(-D**2 / (2 * sigma**2))
#     np.fill_diagonal(A, 1.0)
#     clustering = SpectralClustering(
#         n_clusters=n_clusters,
#         affinity="precomputed",
#         assign_labels="kmeans",
#         random_state=random_state
#     )
#     labels = clustering.fit_predict(A)
#     return labels, D, A

# def dtw_distance_band(X, Y, window=None):
#     """
#     DTW with Sakoe–Chiba band.
#     X: (len_x, dim), Y: (len_y, dim)
#     window: None -> max(len_x, len_y)//10
#     Returns scalar DTW distance (L2).
#     """
#     X = _as_2d(X); Y = _as_2d(Y)
#     len_x, len_y = X.shape[0], Y.shape[0]
#     if len_x == 0 and len_y == 0: return 0.0
#     if len_x == 0 or len_y == 0:  return np.inf

#     if window is None:
#         window = max(len_x, len_y) // 10
#     window = int(window)
#     w = max(window, abs(len_x - len_y))

#     prev = np.full(len_y + 1, np.inf)
#     curr = np.full(len_y + 1, np.inf)
#     prev[0] = 0.0

#     for i in range(1, len_x + 1):
#         curr.fill(np.inf)
#         j_start = max(1, i - w)
#         j_end   = min(len_y, i + w)
#         xi = X[i - 1]
#         for j in range(j_start, j_end + 1):
#             dist = np.dot(xi - Y[j - 1], xi - Y[j - 1])  # squared L2
#             curr[j] = dist + min(curr[j - 1], prev[j], prev[j - 1])
#         prev, curr = curr, prev

#     total_sq = prev[len_y]
#     return float(np.sqrt(total_sq))

# # Parallel DTW (kept for completeness; per-state T is usually small so serial is fine)
# _TRAJ = None
# def _init_worker(trajectories):  # noqa
#     global _TRAJ
#     _TRAJ = trajectories
# def _pair_dtw(args):  # noqa
#     i, j, win = args
#     return i, j, dtw_distance_band(_TRAJ[i], _TRAJ[j], window=win)

# def compute_dtw_matrix(trajectories, window=None, parallelize=False, max_workers=None):
#     N = len(trajectories)
#     D = np.zeros((N, N), dtype=float)
#     if not parallelize or N < 64:
#         for i in range(N):
#             for j in range(i + 1, N):
#                 d = dtw_distance_band(trajectories[i], trajectories[j], window=window)
#                 D[i, j] = D[j, i] = d
#         return D
#     args = [(i, j, window) for i in range(N) for j in range(i + 1, N)]
#     with ProcessPoolExecutor(max_workers=max_workers, initializer=_init_worker, initargs=(trajectories,)) as ex:
#         for i, j, d in ex.map(_pair_dtw, args, chunksize=32):
#             D[i, j] = D[j, i] = d
#     return D

# def spectral_cluster_dtw(trajectories, n_clusters=3, sigma=None, window=None, random_state=0, parallelize=False, max_workers=None):
#     D = compute_dtw_matrix(trajectories, window=window, parallelize=parallelize, max_workers=max_workers)
#     pos = D[D > 0]
#     sigma = float(np.median(pos)) if (sigma is None and pos.size) else (sigma or 1.0)
#     A = np.exp(-D**2 / (2.0 * sigma**2))
#     np.fill_diagonal(A, 1.0)
#     clustering = SpectralClustering(
#         n_clusters=n_clusters,
#         affinity="precomputed",
#         assign_labels="kmeans",
#         random_state=random_state
#     )
#     labels = clustering.fit_predict(A)
#     return labels, D, A

# # -------------------------
# # Visualization helpers
# # -------------------------

# def _labels_for_points(cluster_target, acts, point_labels, chunk_labels, index_rows):
#     """
#     Returns a 1-D array of length N_points (same as xyz.shape[0]) with
#     a cluster id per point.
#     - cluster_target == 'actions': point_labels already match points
#     - cluster_target == 'chunks': expand chunk_labels to points using index_rows
#     """
#     import numpy as np

#     # If we clustered individual points, we already have one label per point
#     if cluster_target == "actions":
#         lab = np.asarray(point_labels).reshape(-1)
#         return lab

#     # Else: clustered whole chunks -> need to expand K labels across their T points
#     if index_rows is None:
#         raise ValueError("index_rows is required to expand chunk labels for per-point coloring.")

#     # index_rows: [(state_idx, sample_idx, action_in_chunk), ...] aligned with points order
#     sample_idx = np.array([r[1] for r in index_rows], dtype=int)
#     # chunk_labels is length K, indexed by sample_idx (k). Expand per point:
#     lab = np.asarray([chunk_labels[k] for k in sample_idx], dtype=int)
#     return lab

# def to_xyz(actions, mode="pca", pca_model=None, kinematic_map=None):
#     """
#     Map actions (T, A) -> (T, 3) for plotting.
#     mode:
#       - "pca": fit PCA (or use provided pca_model) to project to 3D
#       - "first3": take first 3 columns (assumes they correspond to Δx,Δy,Δz or similar)
#       - "custom": call kinematic_map(actions) -> (T,3)
#     """
#     X = np.asarray(actions, dtype=float)
#     if mode == "first3":
#         if X.shape[1] < 3:
#             # pad with zeros if fewer than 3 dims
#             pad = np.zeros((X.shape[0], 3 - X.shape[1]))
#             return np.hstack([X, pad])[:, :3]
#         return X[:, :3]
#     elif mode == "custom":
#         if kinematic_map is None:
#             raise ValueError("mode='custom' requires kinematic_map callable returning (T,3)")
#         return np.asarray(kinematic_map(X), dtype=float)
#     else:  # PCA
#         if pca_model is None:
#             pca_model = PCA(n_components=3, random_state=0)
#             Z = pca_model.fit_transform(X)
#         else:
#             Z = pca_model.transform(X)
#         return Z

# def _set_axes_equal(ax):
#     # Equal aspect for 3D so lines aren't visually distorted
#     import numpy as np
#     x_limits = ax.get_xlim3d()
#     y_limits = ax.get_ylim3d()
#     z_limits = ax.get_zlim3d()
#     x_range = abs(x_limits[1] - x_limits[0])
#     y_range = abs(y_limits[1] - y_limits[0])
#     z_range = abs(z_limits[1] - z_limits[0])
#     max_range = max([x_range, y_range, z_range])
#     x_mid = np.mean(x_limits); y_mid = np.mean(y_limits); z_mid = np.mean(z_limits)
#     ax.set_xlim3d([x_mid - max_range/2, x_mid + max_range/2])
#     ax.set_ylim3d([y_mid - max_range/2, y_mid + max_range/2])
#     ax.set_zlim3d([z_mid - max_range/2, z_mid + max_range/2])

# def plot_per_cluster_panels(
#     xyz,                    # (N_points, 3)
#     per_point_labels,       # (N_points,) label per point
#     index_rows,             # [(state_idx, sample_idx, action_in_chunk)] aligned with xyz rows
#     n_clusters,             # total clusters used in this state (len(np.unique(per_point_labels)))
#     png_path,               # output path
#     title_prefix="State",
#     point_size=3,
#     line_width=1.0,
#     line_alpha=0.6,
# ):
#     """
#     Makes a multi-panel figure with one 3D subplot per cluster.
#     Each subplot shows ONLY the chunks assigned to that cluster,
#     with small per-point scatter (colored by the cluster's color) + a
#     thin polyline connecting actions within each chunk, and start/end markers.
#     """

#     import numpy as np
#     import matplotlib.pyplot as plt
#     import matplotlib.lines as mlines
#     from matplotlib.cm import get_cmap

#     xyz = np.asarray(xyz)
#     labels = np.asarray(per_point_labels).reshape(-1)
#     assert xyz.shape[0] == labels.shape[0] == len(index_rows), "xyz/labels/index_rows mismatch"

#     # Map points -> (sample_idx, action_in_chunk)
#     sample_idx = np.array([r[1] for r in index_rows], dtype=int)
#     action_idx = np.array([r[2] for r in index_rows], dtype=int)
#     uniq_clusters = np.unique(labels)
#     C = int(max(n_clusters, uniq_clusters.max() + 1))  # guard for possibly missing cluster ids
#     # fixed colormap for clusters
#     cmap = plt.get_cmap("tab10", C)

#     # For chunk selection per cluster:
#     # If you ran --cluster_target chunks, all points from a chunk share the same label (because you expanded).
#     # If you ran --cluster_target actions, infer a chunk's cluster by majority vote over its points.
#     chunk_ids = np.unique(sample_idx)
#     chunk_to_cluster = {}
#     for k in chunk_ids:
#         m = (sample_idx == k)
#         if not np.any(m):
#             continue
#         vals, counts = np.unique(labels[m], return_counts=True)
#         chunk_to_cluster[k] = int(vals[np.argmax(counts)])

#     # Subplot grid (rows x cols)
#     cols = min(C, 4)
#     rows = int(np.ceil(C / cols))
#     fig = plt.figure(figsize=(5*cols, 4.5*rows))

#     # Helper to equalize axes
#     def _equalize(ax):
#         xlim = ax.get_xlim3d(); ylim = ax.get_ylim3d(); zlim = ax.get_zlim3d()
#         xr = abs(xlim[1]-xlim[0]); yr = abs(ylim[1]-ylim[0]); zr = abs(zlim[1]-zlim[0])
#         mr = max(xr, yr, zr)
#         xm = np.mean(xlim); ym = np.mean(ylim); zm = np.mean(zlim)
#         ax.set_xlim3d([xm - mr/2, xm + mr/2])
#         ax.set_ylim3d([ym - mr/2, ym + mr/2])
#         ax.set_zlim3d([zm - mr/2, zm + mr/2])

#     for ci in range(C):
#         ax = fig.add_subplot(rows, cols, ci + 1, projection="3d")
#         ax.set_title(f"Cluster {ci}")
#         ax.set_xlabel("X"); ax.set_ylabel("Y"); ax.set_zlabel("Z")

#         # Select chunks whose assigned cluster == ci
#         chunks_in_ci = [k for k, lab in chunk_to_cluster.items() if lab == ci]
#         if not chunks_in_ci:
#             ax.text(0.5, 0.5, 0.5, "No chunks", transform=ax.transAxes, ha="center", va="center")
#             continue

#         # Plot all points from those chunks
#         color = cmap(ci)
#         # Scatter (small points)
#         for k in chunks_in_ci:
#             m = (sample_idx == k)
#             pts = xyz[m]
#             ax.scatter(pts[:,0], pts[:,1], pts[:,2], c=[color], s=point_size, alpha=0.95)

#         # Polylines + start/end markers per chunk
#         for k in chunks_in_ci:
#             m = (sample_idx == k)
#             order = np.argsort(action_idx[m])
#             pts = xyz[m][order]
#             if pts.shape[0] >= 2:
#                 ax.plot(pts[:,0], pts[:,1], pts[:,2], color=color, linewidth=line_width, alpha=line_alpha)
#             # start (open circle), end (X)
#             p0 = pts[0]; p1 = pts[-1]
#             ax.scatter([p0[0]],[p0[1]],[p0[2]], s=36, c="none", edgecolor="k", linewidths=1.0, marker="o")
#             ax.scatter([p1[0]],[p1[1]],[p1[2]], s=48, c="k", marker="x", linewidths=1.5)

#         _equalize(ax)

#     # Add a legend stub for start/end markers once
#     start_h = mlines.Line2D([], [], color="k", marker="o", markersize=6,
#                             markerfacecolor="none", linestyle="None", label="chunk start")
#     end_h   = mlines.Line2D([], [], color="k", marker="x", markersize=7,
#                             linestyle="None", label="chunk end")
#     fig.legend(handles=[start_h, end_h], loc="upper right", frameon=False)

#     fig.suptitle(f"{title_prefix}: per-cluster chunk views", y=0.995)
#     fig.tight_layout(rect=[0, 0.00, 1, 0.96])
#     fig.savefig(png_path, dpi=150)
#     plt.close(fig)

# def plot_actions_xyz(xyz, labels, index_rows, png_path="", title="",
#                      point_size=1, line_width=1.0, line_alpha=0.6,
#                      line_color_mode="per_sample",
#                      start_marker_size=10, end_marker_size=20):
#     """
#     xyz: (N_items, 3)
#     labels: (N_items,) per-point cluster ids
#     index_rows: list[(state_idx, sample_idx, action_in_chunk)] aligned with xyz rows
#     line_color_mode: "per_sample" (distinct line color per chunk) or "by_dominant_cluster"
#     start/end markers: start = filled circle with black edge; end = 'X' marker
#     """
#     import numpy as np
#     import matplotlib.pyplot as plt
#     from matplotlib.cm import get_cmap
#     import matplotlib.lines as mlines

#     xyz = np.asarray(xyz)
#     labels = np.asarray(labels).reshape(-1)

#     if labels.shape[0] != xyz.shape[0]:
#         raise ValueError(f"labels length {labels.shape[0]} != xyz rows {xyz.shape[0]}")

#     fig = plt.figure(figsize=(7, 6))
#     ax = fig.add_subplot(111, projection="3d")

#     # 1) scatter all points colored by cluster (small)
#     scatter = ax.scatter(xyz[:, 0], xyz[:, 1], xyz[:, 2],
#                          c=labels, s=point_size, alpha=0.95)

#     # 2) polylines + start/end markers per chunk
#     if index_rows is not None and len(index_rows) == xyz.shape[0]:
#         sample_idx = np.array([r[1] for r in index_rows], dtype=int)
#         action_idx = np.array([r[2] for r in index_rows], dtype=int)
#         unique_samples = np.unique(sample_idx)

#         # color for lines
#         if line_color_mode == "per_sample":
#             cmap = get_cmap("tab20", len(unique_samples))
#             sample_to_color = {s: cmap(i % cmap.N) for i, s in enumerate(unique_samples)}
#         else:
#             sample_to_color = {}

#         start_handles_added = False
#         end_handles_added = False

#         for s in unique_samples:
#             m = (sample_idx == s)
#             if not np.any(m):
#                 continue
#             order = np.argsort(action_idx[m])
#             pts = xyz[m][order]          # (T_s, 3)
#             labs = labels[m][order]      # (T_s,)

#             if pts.shape[0] >= 2:
#                 if line_color_mode == "by_dominant_cluster":
#                     vals, counts = np.unique(labs, return_counts=True)
#                     dom = vals[np.argmax(counts)]
#                     line_color = scatter.cmap(scatter.norm(dom))
#                 else:
#                     line_color = sample_to_color[s]

#                 ax.plot(pts[:, 0], pts[:, 1], pts[:, 2],
#                         linewidth=line_width, alpha=line_alpha, color=line_color)

#             # Start marker (first point)
#             p0 = pts[0]
#             ax.scatter([p0[0]], [p0[1]], [p0[2]],
#                        s=start_marker_size, c="none", edgecolor="k", linewidths=1.0, marker="o")
#             # End marker (last point)
#             p1 = pts[-1]
#             ax.scatter([p1[0]], [p1[1]], [p1[2]],
#                        s=end_marker_size, c="k", marker="x", linewidths=1.5)

#         # Legend entries for start/end markers
#         start_h = mlines.Line2D([], [], color="k", marker="o", markersize=6,
#                                 markerfacecolor="none", linestyle="None", label="chunk start")
#         end_h   = mlines.Line2D([], [], color="k", marker="x", markersize=7,
#                                 linestyle="None", label="chunk end")
#     else:
#         start_h = end_h = None

#     # 3) legend for clusters
#     legend_items = []
#     for lab in np.unique(labels):
#         legend_items.append(
#             mlines.Line2D([], [], color=scatter.cmap(scatter.norm(lab)), marker="s",
#                           linestyle="None", markersize=6, label=f"cluster {lab}")
#         )
#     if start_h is not None: legend_items.append(start_h)
#     if end_h   is not None: legend_items.append(end_h)
#     if legend_items:
#         ax.legend(handles=legend_items, loc="best", frameon=False, fontsize=9)

#     # 4) labels, equal axes, save
#     ax.set_title(title)
#     ax.set_xlabel("X"); ax.set_ylabel("Y"); ax.set_zlabel("Z")

#     # equal aspect
#     x_limits = ax.get_xlim3d(); y_limits = ax.get_ylim3d(); z_limits = ax.get_zlim3d()
#     x_range = abs(x_limits[1] - x_limits[0])
#     y_range = abs(y_limits[1] - y_limits[0])
#     z_range = abs(z_limits[1] - z_limits[0])
#     max_range = max([x_range, y_range, z_range])
#     x_mid = np.mean(x_limits); y_mid = np.mean(y_limits); z_mid = np.mean(z_limits)
#     ax.set_xlim3d([x_mid - max_range/2, x_mid + max_range/2])
#     ax.set_ylim3d([y_mid - max_range/2, y_mid + max_range/2])
#     ax.set_zlim3d([z_mid - max_range/2, z_mid + max_range/2])

#     fig.tight_layout()
#     fig.savefig(png_path, dpi=150)
#     plt.close(fig)
# # -------------------------
# # Main
# # -------------------------

# def main():
#     ap = argparse.ArgumentParser()
#     ap.add_argument("--summary_csv", type=str, required=True)
#     ap.add_argument("--actions_npz", type=str, required=True)
#     ap.add_argument("--outdir", type=str, default="clusters_out")
#     ap.add_argument("--method", type=str, choices=["minkowski", "dtw"], default="minkowski")
#     ap.add_argument("--n_clusters", type=int, default=3)
#     ap.add_argument("--sigma", type=float, default=None)        # similarity bandwidth; if None -> median
#     ap.add_argument("--dtw_window", type=int, default=None)     # Sakoe–Chiba band
#     ap.add_argument("--parallel_dtw", action="store_true")
#     ap.add_argument("--max_workers", type=int, default=None)
#     ap.add_argument("--ee-mode", type=str, choices=["pca","first3","custom"], default="pca")
#     ap.add_argument("--random_state", type=int, default=0)
#     ap.add_argument("--cluster_scope",
#                 choices=["chunk_only","first_action_across_samples","all_actions_across_samples"],
#                 default="chunk_only")
#     ap.add_argument("--cluster_target", choices=["actions","chunks"], default="actions",
#                 help="Cluster individual actions (points) or whole chunks (trajectories).")
#     args = ap.parse_args()

#     os.makedirs(args.outdir, exist_ok=True)

#     # Load data
#     df = pd.read_csv(args.summary_csv)
#     data = np.load(args.actions_npz, allow_pickle=True)
#     actions_arr = data["actions"]            # shape (N, T, A) or object array list of (T_i, A)

#     # Summary we’ll write at the end
#     rows_meta = []

#     N = len(actions_arr)
#     for i in range(N):
#         # Normalize to (T,A) or (K,T,A)
#         acts_raw = actions_arr[i]
#         acts = (np.asarray(acts_raw, dtype=object)
#                 if isinstance(actions_arr, np.ndarray) and actions_arr.dtype == object
#                 else np.asarray(acts_raw, dtype=float))

#         if acts.ndim == 2:
#             # Single chunk (T,A)
#             T, A = acts.shape
#             if args.cluster_target == "chunks":
#                 # --- CLUSTER CHUNKS (but we only have 1) ---
#                 trajs_chunks = [acts]                  # list of one (T,A)
#                 chunk_labels = np.array([0], dtype=int)  # trivial
#                 # For plotting: expand this single chunk to T points
#                 traj_matrix = acts                     # (T,A)
#                 index_rows  = [(i, 0, t) for t in range(T)]
#                 per_point_labels = np.zeros(T, dtype=int)
#                 # (no spectral call with 1 item; keep it trivial)
#             else:
#                 # --- CLUSTER ACTIONS (points) ---
#                 traj_matrix = acts                     # (T,A)
#                 index_rows  = [(i, 0, t) for t in range(T)]
#                 trajs_points = [traj_matrix[n:n+1, :] for n in range(traj_matrix.shape[0])]
#                 if args.method == "minkowski":
#                     point_labels, D, A_mat = spectral_cluster_minkowski(
#                         trajs_points, n_clusters=args.n_clusters,
#                         sigma=args.sigma, random_state=args.random_state)
#                 else:
#                     point_labels, D, A_mat = spectral_cluster_dtw(
#                         trajs_points, n_clusters=args.n_clusters,
#                         sigma=args.sigma, window=args.dtw_window,
#                         random_state=args.random_state,
#                         parallelize=args.parallel_dtw, max_workers=args.max_workers)
#                 per_point_labels = np.asarray(point_labels, dtype=int)

#         elif acts.ndim == 3:
#             # Multiple samples per state (K,T,A)
#             K, T, A = acts.shape
#             if args.cluster_target == "chunks":
#                 # --- CLUSTER CHUNKS (trajectories) ---
#                 trajs_chunks = [acts[k, :, :] for k in range(K)]     # K items, each (T,A)
#                 if args.method == "minkowski":
#                     chunk_labels, D, A_mat = spectral_cluster_minkowski(
#                         trajs_chunks, n_clusters=args.n_clusters,
#                         sigma=args.sigma, random_state=args.random_state)
#                 else:
#                     chunk_labels, D, A_mat = spectral_cluster_dtw(
#                         trajs_chunks, n_clusters=args.n_clusters,
#                         sigma=args.sigma, window=args.dtw_window,
#                         random_state=args.random_state,
#                         parallelize=args.parallel_dtw, max_workers=args.max_workers)
#                 # For plotting/saving: expand chunk labels to all their T points
#                 traj_matrix = acts.reshape(K*T, A)                    # (K*T, A)
#                 index_rows  = [(i, k, t) for k in range(K) for t in range(T)]
#                 per_point_labels = np.repeat(np.asarray(chunk_labels, dtype=int), T)

#             else:
#                 # --- CLUSTER ACTIONS (points) ---
#                 traj_matrix = acts.reshape(K*T, A)
#                 index_rows  = [(i, k, t) for k in range(K) for t in range(T)]
#                 trajs_points = [traj_matrix[n:n+1, :] for n in range(traj_matrix.shape[0])]
#                 if args.method == "minkowski":
#                     point_labels, D, A_mat = spectral_cluster_minkowski(
#                         trajs_points, n_clusters=args.n_clusters,
#                         sigma=args.sigma, random_state=args.random_state)
#                 else:
#                     point_labels, D, A_mat = spectral_cluster_dtw(
#                         trajs_points, n_clusters=args.n_clusters,
#                         sigma=args.sigma, window=args.dtw_window,
#                         random_state=args.random_state,
#                         parallelize=args.parallel_dtw, max_workers=args.max_workers)
#                 per_point_labels = np.asarray(point_labels, dtype=int)

#         else:
#             raise ValueError(f"Unsupported actions shape for state {i}: {acts.shape}")

#         # Project for visualization
#         xyz = to_xyz(traj_matrix, mode=args.ee_mode)

#         # Sanity checks before plotting
#         if xyz.shape[0] != len(index_rows):
#             raise RuntimeError(f"xyz rows {xyz.shape[0]} != index_rows {len(index_rows)}")
#         if xyz.shape[0] != per_point_labels.shape[0]:
#             raise RuntimeError(f"xyz rows {xyz.shape[0]} != labels {per_point_labels.shape[0]}")

#         # Plot (points small; lines emphasized)
#         title = (f"State {i}: target={args.cluster_target}, N_points={traj_matrix.shape[0]}, "
#                 f"method={args.method}, n_clusters={args.n_clusters}")
#         png_path = os.path.join(args.outdir, f"state_{i:05d}_{args.cluster_target}_{args.method}.png")
#         print("DEBUG shapes:",
#             "traj_matrix", traj_matrix.shape,
#             "xyz", xyz.shape,
#             "labels", per_point_labels.shape,
#             "index_rows", len(index_rows))
#         plot_actions_xyz(
#             xyz, per_point_labels, index_rows,
#             png_path=png_path, title=title,
#             point_size=1, line_width=1.0, line_alpha=0.4,
#             line_color_mode="by_dominant_cluster"
#         )

#         # Per-cluster panel figure (one subplot per cluster)
#         n_cls = len(np.unique(per_point_labels))
#         panel_path = os.path.join(
#             args.outdir, f"state_{i:05d}_{args.cluster_target}_{args.method}_percluster.png"
#         )
#         plot_per_cluster_panels(
#             xyz, per_point_labels, index_rows,
#             n_clusters=n_cls,
#             png_path=panel_path,
#             title_prefix=f"State {i}",
#             point_size=1,
#             line_width=1.2,
#             line_alpha=0.55,
#         )

#         # Per-state cluster CSV (now includes sample/action indices)
#         if args.cluster_target == "chunks" and acts.ndim == 3:
#             # per-chunk table
#             per_chunk = pd.DataFrame({
#                 "state_index": i,
#                 "sample_index": np.arange(K, dtype=int),
#                 "cluster": chunk_labels.astype(int),
#             })
#             per_chunk.to_csv(os.path.join(args.outdir, f"state_{i:05d}_chunk_labels.csv"), index=False)

#         # per-point table (always) with the expanded labels:
#         per_state = pd.DataFrame({
#             "state_index": [r[0] for r in index_rows],
#             "sample_index": [r[1] for r in index_rows],
#             "action_in_chunk": [r[2] for r in index_rows],
#             **{f"a{j}": traj_matrix[:, j] for j in range(traj_matrix.shape[1])},
#             "cluster": per_point_labels.astype(int),
#         })
#         per_state.to_csv(os.path.join(args.outdir, f"state_{i:05d}_{args.method}.csv"), index=False)

#         # Meta row
#         rows_meta.append({
#             "state_index": i,
#             "num_actions": T,
#             "action_dim": acts.shape[1],
#             "plot_path": os.path.abspath(png_path),
#             "percluster_plot_path": os.path.abspath(panel_path),
#             "clusters_csv": os.path.abspath(os.path.join(args.outdir, f"state_{i:05d}_{args.method}.csv")),
#         })

#     # Write summary per state
#     pd.DataFrame(rows_meta).to_csv(os.path.join(args.outdir, "summary_per_state.csv"), index=False)
#     print(f"Done. Wrote per-state plots & clusters to: {os.path.abspath(args.outdir)}")

# if __name__ == "__main__":
#     main()