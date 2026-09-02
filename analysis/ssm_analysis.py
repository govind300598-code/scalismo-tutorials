#!/usr/bin/env python3
"""
Scapula SSM analysis and visualisation.

Usage:
    python ssm_analysis.py [--out-dir PATH] [--mesh-dir PATH]

Reads the CSV outputs produced by FullPipeline / SSMEval (Scala) and the
registered STL meshes, then generates publication-quality figures plus an
additional Python-side metric verification.

Dependencies:
    numpy scipy matplotlib pandas pyvista (or trimesh as fallback)

Install:
    pip install numpy scipy matplotlib pandas pyvista trimesh tqdm
"""

import argparse
import os
import sys
import glob
import warnings
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401

try:
    import pyvista as pv
    HAS_PYVISTA = True
except ImportError:
    HAS_PYVISTA = False

try:
    import trimesh
    HAS_TRIMESH = True
except ImportError:
    HAS_TRIMESH = False

try:
    from tqdm import tqdm
    TQDM = tqdm
except ImportError:
    TQDM = lambda x, **kw: x  # noqa: E731

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

DEFAULT_OUT_DIR  = "/home/g25upadh/Documents/database_v1.11/scapula_ssm_out"
DEFAULT_MESH_DIR = None   # defaults to OUT_DIR/pass2

# ---------------------------------------------------------------------------
# Mesh I/O
# ---------------------------------------------------------------------------

def load_stl(path: str) -> np.ndarray:
    """Load STL and return (N, 3) vertex array."""
    path = str(path)
    if HAS_PYVISTA:
        mesh = pv.read(path)
        return np.array(mesh.points, dtype=np.float64)
    elif HAS_TRIMESH:
        mesh = trimesh.load(path, force="mesh")
        return np.array(mesh.vertices, dtype=np.float64)
    else:
        raise ImportError("Install pyvista or trimesh: pip install pyvista trimesh")


def load_registered_meshes(mesh_dir: str):
    """
    Load all registered STL files.

    Priority:
      1. reg_*.stl  (FullPipeline output naming convention)
      2. *.stl      (fallback: any STL in the directory)

    Returns (dict[id -> vertices], int vertex_count).
    Only meshes whose vertex count matches the majority are kept.
    """
    reg_files = sorted(glob.glob(os.path.join(mesh_dir, "reg_*.stl")))
    if not reg_files:
        warnings.warn(f"No reg_*.stl found in {mesh_dir}; falling back to *.stl")
        reg_files = sorted(glob.glob(os.path.join(mesh_dir, "*.stl")))
    if not reg_files:
        raise FileNotFoundError(f"No STL files found in {mesh_dir}")

    print(f"  Found {len(reg_files)} STL files")

    raw: dict = {}
    for f in TQDM(reg_files, desc="Loading STLs"):
        stem = Path(f).stem
        # Strip leading "reg_" prefix if present
        mesh_id = stem[4:] if stem.startswith("reg_") else stem
        try:
            raw[mesh_id] = load_stl(f)
        except Exception as e:
            warnings.warn(f"Could not load {f}: {e}")

    # Group by vertex count; keep the majority group
    counts: dict = defaultdict(list)
    for mid, verts in raw.items():
        counts[verts.shape[0]].append(mid)

    majority_n = max(counts, key=lambda n: len(counts[n]))
    excluded = {mid for n, mids in counts.items() for mid in mids if n != majority_n}
    if excluded:
        warnings.warn(
            f"Excluded {len(excluded)} mesh(es) with non-matching vertex count: "
            + ", ".join(sorted(excluded))
        )

    meshes = {mid: v for mid, v in raw.items() if mid not in excluded}
    print(f"  Loaded {len(meshes)} meshes with {majority_n} vertices each")
    return meshes, majority_n


# ---------------------------------------------------------------------------
# Metrics (Python-side verification)
# ---------------------------------------------------------------------------

def pairwise_surface_dist(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Min L2 distance from each row of `a` to the nearest row of `b` (brute-force)."""
    # For large meshes, consider using scipy.spatial.cKDTree
    from scipy.spatial import cKDTree
    tree = cKDTree(b)
    dists, _ = tree.query(a, k=1)
    return dists.astype(np.float64)


def symmetric_surface_stats(a: np.ndarray, b: np.ndarray) -> dict:
    fwd = pairwise_surface_dist(a, b)
    bwd = pairwise_surface_dist(b, a)
    d   = np.concatenate([fwd, bwd])
    return {
        "mean":    float(d.mean()),
        "rms":     float(np.sqrt((d**2).mean())),
        "hd95":    float(np.percentile(d, 95)),
        "hd":      float(d.max()),
        "chamfer_l1":  float(fwd.mean() + bwd.mean()),
        "chamfer_sq":  float((fwd**2).mean() + (bwd**2).mean()),
    }


def ptpt_stats(a: np.ndarray, b: np.ndarray) -> dict:
    """Point-to-point stats for meshes already in correspondence."""
    assert a.shape == b.shape, "Meshes must be in correspondence"
    d = np.linalg.norm(a - b, axis=1)
    return {
        "ptpt_mean": float(d.mean()),
        "ptpt_rmse": float(np.sqrt((d**2).mean())),
        "ptpt_max":  float(d.max()),
    }


# ---------------------------------------------------------------------------
# PCA / SSM
# ---------------------------------------------------------------------------

def build_pca(meshes: dict) -> dict:
    """Build a simple PCA from a dict of vertex arrays (all same shape)."""
    ids    = sorted(meshes.keys())
    verts  = np.stack([meshes[i].flatten() for i in ids])   # (N_shapes, N_verts*3)
    mean_v = verts.mean(axis=0)
    centered = verts - mean_v

    # Economy SVD (shapes < dims expected in shape analysis)
    U, S, Vt = np.linalg.svd(centered, full_matrices=False)
    eigenvalues = (S**2) / (len(ids) - 1)
    modes = Vt   # (k, D)

    return {
        "ids":        ids,
        "mean":       mean_v,
        "eigenvalues": eigenvalues,
        "modes":      modes,
        "scores":     U * S,   # (N, k) – shape coefficients
    }


def loo_generalization(meshes: dict, n_dims: int | None = None) -> pd.DataFrame:
    """Leave-one-out reconstruction error using PCA projection."""
    ids  = sorted(meshes.keys())
    rows = []
    for hold_id in TQDM(ids, desc="LOO"):
        train_ids = [i for i in ids if i != hold_id]
        train_v   = np.stack([meshes[i].flatten() for i in train_ids])
        mean_v    = train_v.mean(axis=0)
        centered  = train_v - mean_v
        _, S, Vt  = np.linalg.svd(centered, full_matrices=False)

        k = n_dims or len(train_ids) - 1
        Vk = Vt[:k]   # (k, D)

        hold_v    = meshes[hold_id].flatten()
        proj      = Vk @ (hold_v - mean_v)
        recon_v   = mean_v + Vk.T @ proj
        recon_pts = recon_v.reshape(-1, 3)
        hold_pts  = hold_v.reshape(-1, 3)

        s = ptpt_stats(recon_pts, hold_pts)
        s["id"] = hold_id
        rows.append(s)

    return pd.DataFrame(rows).set_index("id")


# ---------------------------------------------------------------------------
# Figures
# ---------------------------------------------------------------------------

FIG_W, FIG_H = 10, 6
PALETTE = ["#2196F3", "#E91E63", "#4CAF50", "#FF9800", "#9C27B0", "#00BCD4"]


def _savefig(fig, path: str):
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {path}")


def plot_compactness(csv_path: str, out_dir: str):
    if not os.path.exists(csv_path):
        print(f"  Skipping compactness plot (file not found: {csv_path})")
        return
    df = pd.read_csv(csv_path)
    fig, ax = plt.subplots(figsize=(FIG_W, FIG_H))
    ax.plot(df["mode"], df["cumulative_pct"], "o-", color=PALETTE[0], ms=4, lw=2)
    for pct, ls in [(90, "--"), (95, ":"), (99, "-.")]:
        ax.axhline(pct, color="grey", ls=ls, lw=0.8, label=f"{pct}%")
    ax.set_xlabel("Number of modes")
    ax.set_ylabel("Cumulative variance explained (%)")
    ax.set_title("SSM Compactness")
    ax.legend()
    ax.grid(alpha=0.3)
    _savefig(fig, os.path.join(out_dir, "fig_compactness.png"))


def plot_loo_generalization(csv_path: str, out_dir: str):
    if not os.path.exists(csv_path):
        print(f"  Skipping LOO plot (file not found: {csv_path})")
        return
    df = pd.read_csv(csv_path)
    metrics = ["surf_mean", "surf_hd95", "ptpt_rmse"]
    labels  = ["Mean surface dist (mm)", "HD95 (mm)", "Pt-pt RMSE (mm)"]
    fig, axes = plt.subplots(1, 3, figsize=(FIG_W * 1.5, FIG_H))
    for ax, col, lbl in zip(axes, metrics, labels):
        if col not in df.columns:
            ax.set_visible(False); continue
        vals = df[col].values
        ax.bar(range(len(vals)), sorted(vals), color=PALETTE[1], alpha=0.8)
        ax.axhline(vals.mean(), color="k", ls="--", lw=1, label=f"mean={vals.mean():.2f}")
        ax.set_title(lbl)
        ax.set_xlabel("Specimen (sorted)")
        ax.legend(fontsize=8)
        ax.grid(axis="y", alpha=0.3)
    fig.suptitle("Leave-One-Out Generalization Error")
    fig.tight_layout()
    _savefig(fig, os.path.join(out_dir, "fig_loo_generalization.png"))


def plot_specificity(csv_path: str, out_dir: str):
    if not os.path.exists(csv_path):
        print(f"  Skipping specificity plot (file not found: {csv_path})")
        return
    df = pd.read_csv(csv_path)
    # Drop summary rows
    df = df[df["sample_idx"].apply(lambda x: str(x).isdigit())]
    df["dist_to_nearest_training_mm"] = pd.to_numeric(
        df["dist_to_nearest_training_mm"], errors="coerce"
    )
    vals = df["dist_to_nearest_training_mm"].dropna().values

    fig, ax = plt.subplots(figsize=(FIG_W, FIG_H))
    ax.hist(vals, bins=25, color=PALETTE[2], edgecolor="white", alpha=0.85)
    ax.axvline(vals.mean(), color="k", ls="--", lw=1.5, label=f"mean={vals.mean():.2f} mm")
    ax.set_xlabel("Distance to nearest training shape (mm)")
    ax.set_ylabel("Count")
    ax.set_title("SSM Specificity (random samples vs. training set)")
    ax.legend()
    ax.grid(axis="y", alpha=0.3)
    _savefig(fig, os.path.join(out_dir, "fig_specificity.png"))


def plot_per_specimen(csv_path: str, out_dir: str):
    if not os.path.exists(csv_path):
        print(f"  Skipping per-specimen plot (file not found: {csv_path})")
        return
    df = pd.read_csv(csv_path).set_index("id")
    metric_pairs = [
        ("surf_mean",  "surf_hd95",   "Surf mean (mm)",  "HD95 (mm)"),
        ("ptpt_rmse",  "chamfer_l1",  "Pt-pt RMSE (mm)", "Chamfer L1 (mm)"),
    ]
    ids   = list(df.index)
    x     = np.arange(len(ids))
    width = 0.4

    for (c1, c2, l1, l2) in metric_pairs:
        if c1 not in df.columns or c2 not in df.columns:
            continue
        fig, ax = plt.subplots(figsize=(max(FIG_W, len(ids) * 0.5), FIG_H))
        ax.bar(x - width / 2, df[c1].values, width, label=l1, color=PALETTE[0], alpha=0.85)
        ax.bar(x + width / 2, df[c2].values, width, label=l2, color=PALETTE[1], alpha=0.85)
        ax.set_xticks(x)
        ax.set_xticklabels([i.replace("paired_scapula_", "") for i in ids],
                           rotation=45, ha="right", fontsize=7)
        ax.set_ylabel("mm")
        ax.set_title("Per-specimen registration quality")
        ax.legend()
        ax.grid(axis="y", alpha=0.3)
        fig.tight_layout()
        fname = f"fig_per_specimen_{c1}_vs_{c2}.png"
        _savefig(fig, os.path.join(out_dir, fname))


def plot_bilateral(csv_path: str, out_dir: str):
    if not os.path.exists(csv_path):
        return
    df = pd.read_csv(csv_path).set_index("subject")
    fig, ax = plt.subplots(figsize=(max(FIG_W, len(df) * 0.6), FIG_H))
    x = np.arange(len(df))
    ax.bar(x, df["surf_mean"].values, color=PALETTE[3], alpha=0.85, label="surf mean (mm)")
    ax.set_xticks(x)
    ax.set_xticklabels(list(df.index), rotation=45, ha="right", fontsize=7)
    ax.set_ylabel("Surface distance (mm)")
    ax.set_title("Bilateral consistency: within-subject L vs R (Pass-2)")
    ax.legend()
    ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    _savefig(fig, os.path.join(out_dir, "fig_bilateral_consistency.png"))


def plot_pca_variance(pca: dict, out_dir: str):
    evs   = pca["eigenvalues"]
    total = evs.sum()
    cum   = np.cumsum(evs) / total * 100
    fig, ax = plt.subplots(figsize=(FIG_W, FIG_H))
    ax.plot(np.arange(1, len(cum) + 1), cum, "s-", color=PALETTE[4], ms=4, lw=2)
    for pct, ls in [(90, "--"), (95, ":"), (99, "-.")]:
        ax.axhline(pct, color="grey", ls=ls, lw=0.8)
    ax.set_xlabel("Number of principal components")
    ax.set_ylabel("Cumulative variance (%)")
    ax.set_title("Python PCA compactness (cross-check)")
    ax.grid(alpha=0.3)
    _savefig(fig, os.path.join(out_dir, "fig_pca_variance_python.png"))


def plot_mode_shapes(pca: dict, n_modes: int, out_dir: str, n_stds: float = 3.0):
    """Scatter plot ±n_std along each of the first n_modes. Projected to mean shape."""
    fig, axes = plt.subplots(1, n_modes, figsize=(n_modes * 4, 4), subplot_kw={"projection": "3d"})
    if n_modes == 1:
        axes = [axes]
    mean_pts = pca["mean"].reshape(-1, 3)
    evs      = pca["eigenvalues"]
    modes    = pca["modes"]
    for i, ax in enumerate(axes):
        std   = np.sqrt(evs[i])
        delta = (modes[i] * n_stds * std).reshape(-1, 3)
        pos   = mean_pts + delta
        neg   = mean_pts - delta
        ax.scatter(mean_pts[:, 0], mean_pts[:, 1], mean_pts[:, 2],
                   s=0.3, c="grey", alpha=0.4, label="mean")
        ax.scatter(pos[:, 0], pos[:, 1], pos[:, 2],
                   s=0.3, c=PALETTE[0], alpha=0.5, label=f"+{n_stds}σ")
        ax.scatter(neg[:, 0], neg[:, 1], neg[:, 2],
                   s=0.3, c=PALETTE[1], alpha=0.5, label=f"−{n_stds}σ")
        ax.set_title(f"Mode {i + 1}  ({evs[i] / evs.sum() * 100:.1f}%)", fontsize=9)
        ax.axis("off")
    fig.suptitle(f"First {n_modes} PCA modes (±{n_stds}σ)")
    fig.tight_layout()
    _savefig(fig, os.path.join(out_dir, "fig_mode_shapes.png"))


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def parse_args():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--out-dir",  default=DEFAULT_OUT_DIR,
                   help="Directory containing Scala pipeline outputs")
    p.add_argument("--mesh-dir", default=None,
                   help="Directory containing reg_*.stl files (default: OUT_DIR/pass2)")
    p.add_argument("--no-mesh", action="store_true",
                   help="Skip STL loading and Python-side metric recomputation")
    return p.parse_args()


def main():
    args       = parse_args()
    out_dir    = args.out_dir
    mesh_dir   = args.mesh_dir or os.path.join(out_dir, "pass2")
    fig_dir    = os.path.join(out_dir, "figures")
    os.makedirs(fig_dir, exist_ok=True)

    print("=" * 70)
    print("SSM Analysis + Visualisation")
    print("=" * 70)
    print(f"  Output dir:  {out_dir}")
    print(f"  Mesh dir:    {mesh_dir}")
    print(f"  Figures dir: {fig_dir}")
    print()

    # ------------------------------------------------------------------
    # 1.  Plot from Scala CSV outputs
    # ------------------------------------------------------------------
    print("[1] Plotting Scala evaluation outputs...")
    plot_compactness(os.path.join(out_dir, "compactness.csv"),           fig_dir)
    plot_loo_generalization(os.path.join(out_dir, "loo_generalization.csv"), fig_dir)
    plot_specificity(os.path.join(out_dir, "specificity.csv"),           fig_dir)
    plot_per_specimen(os.path.join(out_dir, "per_specimen_metrics.csv"), fig_dir)
    plot_bilateral(os.path.join(out_dir, "bilateral_consistency.csv"),   fig_dir)

    # ------------------------------------------------------------------
    # 2.  Load STL meshes and run Python-side PCA / metrics
    # ------------------------------------------------------------------
    if not args.no_mesh:
        print("\n[2] Loading registered STL meshes...")
        try:
            meshes, n_verts = load_registered_meshes(mesh_dir)
        except FileNotFoundError as e:
            print(f"  Warning: {e}")
            print("  Skipping Python-side analysis (run with --no-mesh to suppress)")
            meshes = None

        if meshes and len(meshes) >= 3:
            print(f"\n[3] Python-side PCA ({len(meshes)} shapes, {n_verts} vertices each)...")
            pca = build_pca(meshes)
            plot_pca_variance(pca, fig_dir)
            plot_mode_shapes(pca, min(5, len(meshes) - 1), fig_dir)

            evs   = pca["eigenvalues"]
            total = evs.sum()
            cum   = np.cumsum(evs) / total * 100
            m90   = int(np.searchsorted(cum, 90)) + 1
            m95   = int(np.searchsorted(cum, 95)) + 1
            m99   = int(np.searchsorted(cum, 99)) + 1
            print(f"  Compactness: 90%={m90} modes  95%={m95}  99%={m99}")

            print(f"\n[4] Python-side LOO generalization ({len(meshes)} specimens)...")
            loo_df = loo_generalization(meshes)
            loo_out = os.path.join(out_dir, "loo_generalization_python.csv")
            loo_df.to_csv(loo_out)
            print(f"  LOO avg ptpt_RMSE: {loo_df['ptpt_rmse'].mean():.3f} mm")
            print(f"  Saved: {loo_out}")

            print("\n[5] Summary statistics...")
            vlist = list(meshes.values())
            mean_shape = np.stack(vlist).mean(axis=0)  # (N, 3)
            print(f"  Mean shape bounding box: "
                  f"X=[{mean_shape[:,0].min():.1f},{mean_shape[:,0].max():.1f}] "
                  f"Y=[{mean_shape[:,1].min():.1f},{mean_shape[:,1].max():.1f}] "
                  f"Z=[{mean_shape[:,2].min():.1f},{mean_shape[:,2].max():.1f}]")

    print("\nDone.  Figures saved to:", fig_dir)


if __name__ == "__main__":
    main()
