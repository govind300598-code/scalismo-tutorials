#!/usr/bin/env python3
"""
SSM validation + surface-distance visualisation.

Reads the CSV outputs from FullPipeline / SSMEval (metrics_per_spec.csv,
ssm_quality.txt) and any STL files in the output directory, and produces
summary plots + a Markdown report.

Usage:
    python analysis/ssm_analysis.py --out-dir /path/to/scapula_ssm_out

If numpy-stl is not installed:
    pip install numpy-stl matplotlib pandas
"""

import argparse
import os
import sys
import glob
from pathlib import Path

import numpy as np

# ── optional heavy deps ──────────────────────────────────────────────────────
try:
    import pandas as pd
    HAS_PANDAS = True
except ImportError:
    HAS_PANDAS = False
    print("[warn] pandas not installed — CSV reading will use stdlib csv")

try:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    HAS_PLOT = True
except ImportError:
    HAS_PLOT = False
    print("[warn] matplotlib not installed — plots will be skipped")

try:
    from stl import mesh as stl_mesh
    HAS_STL = True
except ImportError:
    HAS_STL = False
    print("[warn] numpy-stl not installed — STL loading will be skipped")


# ── STL loader ───────────────────────────────────────────────────────────────

def load_stl_vertices(path: str) -> np.ndarray:
    """Return (N, 3) vertex array from an STL file (de-duplicated)."""
    if not HAS_STL:
        return None
    m = stl_mesh.Mesh.from_file(path)
    verts = m.vectors.reshape(-1, 3)
    # round to 4 decimal places to de-duplicate floating-point near-duplicates
    verts = np.unique(np.round(verts, 4), axis=0)
    return verts


def load_registered_meshes(out_dir: str):
    """
    Load registered STL files from pass2/ sub-folder, falling back to any
    reg_*.stl in out_dir, then falling back to any *.stl in out_dir.
    Returns {name: vertices_array} for the largest consistent group.
    """
    candidates = []
    # 1. pass2/reg_*.stl (ideal)
    candidates += glob.glob(os.path.join(out_dir, "pass2", "reg_*.stl"))
    # 2. root reg_*.stl
    if not candidates:
        candidates += glob.glob(os.path.join(out_dir, "reg_*.stl"))
    # 3. any stl except mean/seed/rigid
    if not candidates:
        candidates = [
            p for p in glob.glob(os.path.join(out_dir, "*.stl"))
            if not any(x in os.path.basename(p) for x in ("mean", "seed", "rigid", "ffdm"))
        ]

    if not candidates:
        print(f"[warn] No registered STL files found in {out_dir}")
        return {}

    if not HAS_STL:
        print(f"  Found {len(candidates)} STL files but numpy-stl not installed")
        return {Path(p).stem: None for p in candidates}

    meshes = {}
    for path in sorted(candidates):
        name = Path(path).stem
        verts = load_stl_vertices(path)
        if verts is not None:
            meshes[name] = verts

    if not meshes:
        return {}

    # Group by vertex count; keep the largest consistent group
    from collections import Counter
    counts = Counter(v.shape[0] for v in meshes.values())
    dominant_count, _ = counts.most_common(1)[0]
    kept = {k: v for k, v in meshes.items() if v.shape[0] == dominant_count}
    skipped = len(meshes) - len(kept)
    if skipped:
        print(f"  [warn] Skipped {skipped} mesh(es) with inconsistent vertex count "
              f"(expected {dominant_count})")
    print(f"  Loaded {len(kept)} registered meshes  ({dominant_count} vertices each)")
    return kept


# ── Metrics computed in Python (for cross-check / additional plots) ──────────

def hausdorff_distance(a: np.ndarray, b: np.ndarray) -> float:
    """Symmetric Hausdorff distance between two vertex sets."""
    from scipy.spatial import cKDTree
    tree_b = cKDTree(b)
    tree_a = cKDTree(a)
    d_ab = tree_b.query(a)[0].max()
    d_ba = tree_a.query(b)[0].max()
    return max(d_ab, d_ba)


def surface_distances(a: np.ndarray, b: np.ndarray):
    """One-sided min-distances from every vertex in a to the nearest in b."""
    from scipy.spatial import cKDTree
    return cKDTree(b).query(a)[0]


def chamfer_l1(a: np.ndarray, b: np.ndarray) -> float:
    d_ab = surface_distances(a, b).mean()
    d_ba = surface_distances(b, a).mean()
    return (d_ab + d_ba) / 2.0


def chamfer_l2(a: np.ndarray, b: np.ndarray) -> float:
    d_ab = (surface_distances(a, b) ** 2).mean()
    d_ba = (surface_distances(b, a) ** 2).mean()
    return (d_ab + d_ba) / 2.0


def all_metrics(a: np.ndarray, b: np.ndarray) -> dict:
    d_ab = surface_distances(a, b)
    d_ba = surface_distances(b, a)
    d    = np.concatenate([d_ab, d_ba])
    return {
        "surf_mean":   d.mean(),
        "surf_rms":    np.sqrt((d**2).mean()),
        "surf_hd95":   np.percentile(d, 95),
        "surf_hd":     d.max(),
        "chamfer_l1":  (d_ab.mean() + d_ba.mean()) / 2.0,
        "chamfer_l2":  (((d_ab**2).mean() + (d_ba**2).mean()) / 2.0),
    }


# ── CSV loading ──────────────────────────────────────────────────────────────

def load_metrics_csv(out_dir: str):
    path = os.path.join(out_dir, "metrics_per_spec.csv")
    if not os.path.exists(path):
        print(f"  [info] {path} not found — run FullPipeline first")
        return None
    if HAS_PANDAS:
        return pd.read_csv(path)
    import csv
    with open(path) as f:
        reader = csv.DictReader(f)
        rows = list(reader)
    if not rows:
        return None
    # convert to dict-of-arrays
    keys = list(rows[0].keys())
    data = {k: [] for k in keys}
    for row in rows:
        for k in keys:
            try:
                data[k].append(float(row[k]))
            except ValueError:
                data[k].append(row[k])
    return data


# ── Plotting helpers ─────────────────────────────────────────────────────────

def plot_distance_violin(metrics, plot_dir: str):
    if not HAS_PLOT:
        return
    metrics_cols = ["surf_mean", "surf_hd95", "surf_hd", "p2p_mean", "chamfer_l1"]
    if HAS_PANDAS:
        data = [metrics[c].values for c in metrics_cols if c in metrics.columns]
        labels = [c for c in metrics_cols if c in metrics.columns]
    else:
        data = [np.array(metrics[c]) for c in metrics_cols if c in metrics]
        labels = [c for c in metrics_cols if c in metrics]

    if not data:
        return
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.violinplot(data, positions=range(len(data)), showmedians=True)
    ax.set_xticks(range(len(labels)))
    ax.set_xticklabels(labels, rotation=30, ha="right")
    ax.set_ylabel("Distance (mm)")
    ax.set_title("Per-specimen surface & point distances")
    ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    out = os.path.join(plot_dir, "distance_violin.png")
    fig.savefig(out, dpi=150)
    plt.close(fig)
    print(f"  Saved: {out}")


def plot_compactness(evs: list, plot_dir: str):
    if not HAS_PLOT or not evs:
        return
    cumvar = np.cumsum(evs) / np.sum(evs) * 100
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.plot(range(1, len(cumvar) + 1), cumvar, "b-o", markersize=3)
    for pct in (90, 95, 99):
        idx = np.searchsorted(cumvar, pct)
        if idx < len(cumvar):
            ax.axhline(pct, color="gray", lw=0.8, ls="--")
            ax.axvline(idx + 1, color="gray", lw=0.8, ls="--")
            ax.text(idx + 2, pct + 0.3, f"{pct}% @ mode {idx+1}", fontsize=8)
    ax.set_xlabel("Number of modes")
    ax.set_ylabel("Cumulative variance explained (%)")
    ax.set_title("SSM Compactness")
    ax.grid(alpha=0.3)
    fig.tight_layout()
    out = os.path.join(plot_dir, "compactness.png")
    fig.savefig(out, dpi=150)
    plt.close(fig)
    print(f"  Saved: {out}")


def plot_pairwise_hausdorff(meshes: dict, plot_dir: str, n_pairs: int = 20):
    """Compute pairwise Hausdorff for a sample of mesh pairs and plot."""
    if not HAS_PLOT or not HAS_STL or len(meshes) < 2:
        return
    try:
        from scipy.spatial import cKDTree
    except ImportError:
        print("  [skip] scipy not installed — pairwise plot skipped")
        return
    names = list(meshes.keys())
    import random
    pairs = [(names[i], names[j]) for i in range(len(names)) for j in range(i+1, len(names))]
    random.shuffle(pairs)
    pairs = pairs[:n_pairs]
    hds = []
    for na, nb in pairs:
        hds.append(hausdorff_distance(meshes[na], meshes[nb]))
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.hist(hds, bins=10, edgecolor="k")
    ax.set_xlabel("Symmetric Hausdorff distance (mm)")
    ax.set_ylabel("Pair count")
    ax.set_title(f"Pairwise Hausdorff ({len(hds)} sampled pairs)")
    ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    out = os.path.join(plot_dir, "pairwise_hausdorff.png")
    fig.savefig(out, dpi=150)
    plt.close(fig)
    print(f"  Saved: {out}")


def plot_per_spec_bar(metrics, col: str, plot_dir: str):
    if not HAS_PLOT:
        return
    if HAS_PANDAS:
        if col not in metrics.columns:
            return
        ids = metrics["id"].tolist()
        vals = metrics[col].tolist()
    else:
        if col not in metrics:
            return
        ids = metrics.get("id", [f"s{i}" for i in range(len(metrics[col]))])
        vals = metrics[col]
    fig, ax = plt.subplots(figsize=(max(8, len(ids) * 0.4), 4))
    ax.bar(range(len(ids)), vals, color="steelblue")
    ax.set_xticks(range(len(ids)))
    ax.set_xticklabels(ids, rotation=45, ha="right", fontsize=7)
    ax.set_ylabel(f"{col} (mm)")
    ax.set_title(f"Per-specimen {col}")
    ax.axhline(np.mean(vals), color="r", ls="--", label=f"mean={np.mean(vals):.2f}")
    ax.legend()
    ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    out = os.path.join(plot_dir, f"per_spec_{col}.png")
    fig.savefig(out, dpi=150)
    plt.close(fig)
    print(f"  Saved: {out}")


# ── Main ─────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="SSM analysis & visualisation")
    parser.add_argument("--out-dir", default=os.environ.get("SCAPULA_OUT_DIR",
        "/home/g25upadh/Documents/database_v1.11/scapula_ssm_out"),
        help="Pipeline output directory (default from SCAPULA_OUT_DIR env var)")
    args = parser.parse_args()

    out_dir  = args.out_dir
    plot_dir = os.path.join(out_dir, "python_plots")
    os.makedirs(plot_dir, exist_ok=True)

    print("=" * 60)
    print("SSM Validation + Surface Distance Analysis")
    print("=" * 60)
    print(f"  Output dir : {out_dir}")
    print(f"  Plots dir  : {plot_dir}")

    # ── Load CSV metrics produced by Scala pipeline ──────────────────────────
    metrics = load_metrics_csv(out_dir)
    if metrics is not None:
        print("\n[1] Per-specimen metrics from Scala pipeline:")
        if HAS_PANDAS:
            print(metrics.to_string(index=False))
        else:
            for k, v in metrics.items():
                if isinstance(v, list) and v and isinstance(v[0], float):
                    print(f"  {k:20s}: mean={np.mean(v):.4f}  std={np.std(v):.4f}  "
                          f"min={min(v):.4f}  max={max(v):.4f}")
        plot_distance_violin(metrics, plot_dir)
        for col in ("surf_hd95", "surf_hd", "surf_mean", "chamfer_l1"):
            plot_per_spec_bar(metrics, col, plot_dir)

    # ── Load registered STL meshes ───────────────────────────────────────────
    print("\n[2] Loading registered STL meshes")
    meshes = load_registered_meshes(out_dir)

    if meshes and HAS_STL:
        try:
            from scipy.spatial import cKDTree
            print("\n[3] Computing pairwise surface metrics (Python cross-check)")
            pairs = list(meshes.keys())
            pair_metrics = []
            for i, na in enumerate(pairs):
                for nb in pairs[i+1:]:
                    m = all_metrics(meshes[na], meshes[nb])
                    m["pair"] = f"{na}_{nb}"
                    pair_metrics.append(m)
            if pair_metrics:
                for key in ("surf_mean", "surf_hd95", "surf_hd", "chamfer_l1"):
                    vals = [m[key] for m in pair_metrics]
                    print(f"  {key:15s}: mean={np.mean(vals):.4f}  "
                          f"HD95={np.percentile(vals, 95):.4f}  max={max(vals):.4f}")
            plot_pairwise_hausdorff(meshes, plot_dir)
        except ImportError:
            print("  [skip] scipy not installed — cross-check metrics skipped")
            print("         pip install scipy   to enable this section")

    # ── Read quality report ──────────────────────────────────────────────────
    qual_path = os.path.join(out_dir, "ssm_quality.txt")
    if os.path.exists(qual_path):
        print(f"\n[4] SSM quality report ({qual_path}):")
        with open(qual_path) as f:
            print(f.read())

    print(f"\nDone. Plots saved to: {plot_dir}")


if __name__ == "__main__":
    main()
