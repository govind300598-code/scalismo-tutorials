#!/usr/bin/env python3
"""
SSM Validation + Surface Distance Analysis
==========================================
Loads registered STL meshes produced by RegistrationComparisonViewer
and computes:

  Surface distance metrics (per specimen pair and per specimen vs mean):
    - Point-to-surface  A->B  and  B->A  (directed)
    - Surface-to-surface symmetric:
        Mean Surface Distance (MSD)
        Root Mean Square Error (RMSE)
        Chamfer Distance
        Hausdorff Distance (HD)
        HD at 95th percentile (HD95)

  SSM validation metrics:
    - Compactness   (cumulative variance per mode)
    - Generalization (leave-one-out reconstruction error, mm)
    - Specificity   (mean distance from random samples to nearest training shape, mm)

Outputs (all in ANALYSIS_DIR):
    ssm_compactness.csv
    ssm_generalization.csv
    ssm_specificity.csv
    ssm_summary.csv
    per_specimen_surface_distances.csv
    pairwise_surface_distances.csv
    ssm_compactness.png
    ssm_generalization_specificity.png
    surface_distance_distributions.png

Usage:
    pip install trimesh numpy scipy scikit-learn matplotlib pandas tqdm
    python ssm_analysis.py

Environment variables (override defaults):
    SCAPULA_OUT_DIR   - directory where reg_*.stl files live
    SCAPULA_ANAL_DIR  - output directory for analysis results
"""

import os
import sys
import numpy as np
import pandas as pd
import trimesh
from pathlib import Path
from sklearn.decomposition import PCA
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from tqdm import tqdm

# ─────────────────────────────────────────────────────────────────────────────
# Configuration
# ─────────────────────────────────────────────────────────────────────────────
OUT_DIR = Path(os.environ.get(
    "SCAPULA_OUT_DIR",
    "/home/g25upadh/Documents/database_v1.11/scapula_ssm_out"
))
ANALYSIS_DIR = Path(os.environ.get(
    "SCAPULA_ANAL_DIR",
    str(OUT_DIR / "python_analysis")
))
ANALYSIS_DIR.mkdir(parents=True, exist_ok=True)


# ─────────────────────────────────────────────────────────────────────────────
# Mesh loading
# ─────────────────────────────────────────────────────────────────────────────
def load_registered_meshes(out_dir: Path) -> dict:
    stl_files = sorted(out_dir.glob("reg_*.stl"))
    if not stl_files:
        sys.exit(f"ERROR: No reg_*.stl files found in {out_dir}\n"
                 f"Run RegistrationComparisonViewer first.")
    meshes = {}
    for f in tqdm(stl_files, desc="Loading meshes"):
        sid = f.stem[4:]  # strip "reg_" prefix
        meshes[sid] = trimesh.load(str(f), process=False)
    print(f"  Loaded {len(meshes)} registered meshes")
    nv = next(iter(meshes.values())).vertices.shape[0]
    for sid, m in meshes.items():
        assert m.vertices.shape[0] == nv, \
            f"Mesh {sid} has {m.vertices.shape[0]} vertices, expected {nv}"
    print(f"  Vertices per mesh: {nv}")
    return meshes


# ─────────────────────────────────────────────────────────────────────────────
# Surface distance helpers
# ─────────────────────────────────────────────────────────────────────────────
def p2s(pts: np.ndarray, target: trimesh.Trimesh) -> np.ndarray:
    """Point-to-surface distances: each point to nearest point on target surface."""
    _, dists, _ = trimesh.proximity.closest_point(target, pts)
    return dists


def surface_metrics(mesh_a: trimesh.Trimesh, mesh_b: trimesh.Trimesh) -> dict:
    """All surface distance metrics between two meshes."""
    va = np.array(mesh_a.vertices)
    vb = np.array(mesh_b.vertices)

    d_a2b = p2s(va, mesh_b)   # directed A -> B
    d_b2a = p2s(vb, mesh_a)   # directed B -> A
    d_sym = np.concatenate([d_a2b, d_b2a])  # symmetric pool

    # point-to-point (valid only when meshes are in correspondence)
    p2p = np.linalg.norm(va - vb, axis=1) if va.shape == vb.shape else None

    result = {
        # ── directed A -> B ────────────────────────────────────────────────
        "p2s_mean_a2b":  float(d_a2b.mean()),
        "p2s_rms_a2b":   float(np.sqrt((d_a2b ** 2).mean())),
        "p2s_hd_a2b":    float(d_a2b.max()),
        "p2s_hd95_a2b":  float(np.percentile(d_a2b, 95)),
        # ── directed B -> A ────────────────────────────────────────────────
        "p2s_mean_b2a":  float(d_b2a.mean()),
        "p2s_rms_b2a":   float(np.sqrt((d_b2a ** 2).mean())),
        "p2s_hd_b2a":    float(d_b2a.max()),
        "p2s_hd95_b2a":  float(np.percentile(d_b2a, 95)),
        # ── symmetric surface-to-surface ───────────────────────────────────
        "s2s_msd":       float(d_sym.mean()),
        "s2s_rms":       float(np.sqrt((d_sym ** 2).mean())),
        "s2s_hd":        float(d_sym.max()),
        "s2s_hd95":      float(np.percentile(d_sym, 95)),
        "s2s_hd50":      float(np.percentile(d_sym, 50)),
        # ── Chamfer = mean of directed means ──────────────────────────────
        "chamfer":       float((d_a2b.mean() + d_b2a.mean()) / 2),
        # ── Hausdorff = max of directed maxes ─────────────────────────────
        "hausdorff":     float(max(d_a2b.max(), d_b2a.max())),
    }
    if p2p is not None:
        result["p2p_mean"] = float(p2p.mean())
        result["p2p_rms"]  = float(np.sqrt((p2p ** 2).mean()))
        result["p2p_hd"]   = float(p2p.max())
        result["p2p_hd95"] = float(np.percentile(p2p, 95))
    return result


# ─────────────────────────────────────────────────────────────────────────────
# Per-specimen vs mean mesh distances
# ─────────────────────────────────────────────────────────────────────────────
def per_specimen_vs_mean(meshes: dict) -> pd.DataFrame:
    ids = list(meshes.keys())
    verts = np.stack([meshes[sid].vertices for sid in ids])   # (N, V, 3)
    mean_v = verts.mean(axis=0)
    ref_mesh = meshes[ids[0]]
    mean_mesh = trimesh.Trimesh(vertices=mean_v,
                                faces=ref_mesh.faces,
                                process=False)
    rows = []
    for sid in tqdm(ids, desc="Per-specimen vs mean"):
        m = surface_metrics(meshes[sid], mean_mesh)
        m["specimen"] = sid
        rows.append(m)
    return pd.DataFrame(rows)


# ─────────────────────────────────────────────────────────────────────────────
# Pairwise surface distances
# ─────────────────────────────────────────────────────────────────────────────
def pairwise_metrics(meshes: dict) -> pd.DataFrame:
    ids = list(meshes.keys())
    n = len(ids)
    rows = []
    total = n * (n - 1) // 2
    with tqdm(total=total, desc="Pairwise surface distances") as pbar:
        for i in range(n):
            for j in range(i + 1, n):
                m = surface_metrics(meshes[ids[i]], meshes[ids[j]])
                m["specimen_a"] = ids[i]
                m["specimen_b"] = ids[j]
                rows.append(m)
                pbar.update(1)
    return pd.DataFrame(rows)


# ─────────────────────────────────────────────────────────────────────────────
# SSM validation
# ─────────────────────────────────────────────────────────────────────────────
def build_shape_matrix(meshes: dict):
    ids = list(meshes.keys())
    X = np.stack([meshes[sid].vertices.flatten() for sid in ids])
    return X, ids


def ssm_compactness(X: np.ndarray) -> pd.DataFrame:
    pca = PCA()
    pca.fit(X)
    ev  = pca.explained_variance_
    cum = np.cumsum(ev) / ev.sum() * 100
    df  = pd.DataFrame({
        "mode":                    np.arange(1, len(ev) + 1),
        "eigenvalue":              ev,
        "variance_pct":            ev / ev.sum() * 100,
        "cumulative_variance_pct": cum,
    })
    m90 = int(np.searchsorted(cum, 90.0) + 1)
    m95 = int(np.searchsorted(cum, 95.0) + 1)
    print(f"  90%% variance -> {m90} modes   |   95%% variance -> {m95} modes")
    return df


def ssm_generalization(X: np.ndarray, mode_list: list) -> pd.DataFrame:
    n = X.shape[0]
    rows = []
    for nM in tqdm(mode_list, desc="Generalization (LOO)"):
        errors = []
        for i in range(n):
            train = np.delete(X, i, axis=0)
            test  = X[i]
            k = min(nM, train.shape[0] - 1)
            pca = PCA(n_components=k)
            pca.fit(train)
            recon = pca.inverse_transform(pca.transform(test.reshape(1, -1))).flatten()
            diff  = (recon - test).reshape(-1, 3)
            errors.append(float(np.linalg.norm(diff, axis=1).mean()))
        mu  = float(np.mean(errors))
        std = float(np.std(errors))
        rows.append({"modes": nM, "mean_mm": mu, "std_mm": std})
    return pd.DataFrame(rows)


def ssm_specificity(X: np.ndarray, mode_list: list, n_samples: int = 50) -> pd.DataFrame:
    rng  = np.random.default_rng(42)
    rows = []
    for nM in tqdm(mode_list, desc="Specificity"):
        k = min(nM, X.shape[0] - 1)
        pca = PCA(n_components=k)
        pca.fit(X)
        dists = []
        for _ in range(n_samples):
            z      = rng.standard_normal(k) * np.sqrt(pca.explained_variance_)
            sample = pca.inverse_transform(z.reshape(1, -1)).flatten()
            nn_dist = np.linalg.norm(
                (X - sample).reshape(X.shape[0], -1, 3), axis=2
            ).mean(axis=1)
            dists.append(float(nn_dist.min()))
        mu  = float(np.mean(dists))
        std = float(np.std(dists))
        rows.append({"modes": nM, "mean_mm": mu, "std_mm": std})
    return pd.DataFrame(rows)


# ─────────────────────────────────────────────────────────────────────────────
# Plots
# ─────────────────────────────────────────────────────────────────────────────
def _style():
    plt.rcParams.update({
        "figure.facecolor": "white", "axes.facecolor": "white",
        "axes.grid": True, "grid.alpha": 0.3,
    })


def plot_compactness(df: pd.DataFrame):
    _style()
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(df["mode"], df["cumulative_variance_pct"], marker="o", ms=3, color="steelblue")
    ax.axhline(90, color="orange", ls="--", lw=1.5, label="90%")
    ax.axhline(95, color="red",    ls="--", lw=1.5, label="95%")
    ax.set_xlabel("Number of modes")
    ax.set_ylabel("Cumulative variance (%)")
    ax.set_title("SSM Compactness")
    ax.legend()
    fig.tight_layout()
    fig.savefig(ANALYSIS_DIR / "ssm_compactness.png", dpi=150)
    plt.close(fig)


def plot_gen_spec(gen_df: pd.DataFrame, spec_df: pd.DataFrame):
    _style()
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    for ax, df, title, color in zip(
        axes,
        [gen_df, spec_df],
        ["Generalization (LOO reconstruction error)", "Specificity (random sample error)"],
        ["steelblue", "darkorange"]
    ):
        ax.errorbar(df["modes"], df["mean_mm"], yerr=df["std_mm"],
                    marker="o", color=color, capsize=4, lw=1.5)
        ax.set_xlabel("Number of modes")
        ax.set_ylabel("Mean distance (mm)")
        ax.set_title(title)
    fig.suptitle("SSM Validation Metrics", fontweight="bold")
    fig.tight_layout()
    fig.savefig(ANALYSIS_DIR / "ssm_generalization_specificity.png", dpi=150)
    plt.close(fig)


def plot_surface_distributions(pw_df: pd.DataFrame):
    _style()
    cols   = ["s2s_msd", "s2s_rms", "s2s_hd95", "hausdorff", "chamfer",
              "p2s_mean_a2b", "p2s_hd95_a2b"]
    labels = ["MSD (mm)", "RMSE (mm)", "HD95 (mm)", "Hausdorff (mm)", "Chamfer (mm)",
              "P2S mean A→B (mm)", "P2S HD95 A→B (mm)"]
    cols   = [c for c in cols if c in pw_df.columns]
    labels = labels[:len(cols)]
    ncols  = len(cols)
    fig, axes = plt.subplots(1, ncols, figsize=(3.5 * ncols, 5))
    if ncols == 1:
        axes = [axes]
    for ax, col, label in zip(axes, cols, labels):
        v = pw_df[col].values
        ax.hist(v, bins=20, color="steelblue", edgecolor="white")
        ax.axvline(v.mean(), color="red", ls="--", lw=1.5, label=f"μ={v.mean():.2f}")
        ax.set_xlabel(label)
        ax.set_title(col, fontsize=9)
        ax.legend(fontsize=8)
    fig.suptitle("Pairwise Surface Distance Distributions (registered meshes)", fontweight="bold")
    fig.tight_layout()
    fig.savefig(ANALYSIS_DIR / "surface_distance_distributions.png", dpi=150)
    plt.close(fig)


def plot_per_specimen(df: pd.DataFrame):
    _style()
    df_s = df.sort_values("s2s_msd")
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    for ax, col, label in zip(
        axes.flat,
        ["s2s_msd", "s2s_rms", "s2s_hd95", "hausdorff"],
        ["MSD (mm)", "RMSE (mm)", "HD95 (mm)", "Hausdorff (mm)"]
    ):
        ax.barh(df_s["specimen"], df_s[col], color="steelblue")
        ax.set_xlabel(label)
        ax.set_title(f"Per-specimen vs mean shape: {label}")
        ax.tick_params(axis="y", labelsize=7)
    fig.suptitle("Per-Specimen Surface Distances vs Mean Shape", fontweight="bold")
    fig.tight_layout()
    fig.savefig(ANALYSIS_DIR / "per_specimen_vs_mean.png", dpi=150)
    plt.close(fig)


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────
def main():
    print("=" * 65)
    print("  SSM Validation + Surface Distance Analysis")
    print("=" * 65)
    print(f"  Registered meshes : {OUT_DIR}")
    print(f"  Analysis output   : {ANALYSIS_DIR}\n")

    # ── load ──────────────────────────────────────────────────────────────────
    meshes = load_registered_meshes(OUT_DIR)
    n = len(meshes)

    # ── shape matrix for SSM ──────────────────────────────────────────────────
    X, ids = build_shape_matrix(meshes)

    mode_list = sorted(set(
        [1, 2, 3, 5, 8, 10, 15, 20, n - 1]
    ))
    mode_list = [m for m in mode_list if m <= n - 1]

    # ── compactness ───────────────────────────────────────────────────────────
    print("\n── Compactness ─────────────────────────────────────────────")
    comp_df = ssm_compactness(X)
    comp_df.to_csv(ANALYSIS_DIR / "ssm_compactness.csv", index=False)

    # ── generalization ────────────────────────────────────────────────────────
    print("\n── Generalization (LOO) ────────────────────────────────────")
    gen_df = ssm_generalization(X, mode_list)
    gen_df.to_csv(ANALYSIS_DIR / "ssm_generalization.csv", index=False)
    print(gen_df.to_string(index=False))

    # ── specificity ───────────────────────────────────────────────────────────
    print("\n── Specificity ─────────────────────────────────────────────")
    spec_df = ssm_specificity(X, mode_list, n_samples=50)
    spec_df.to_csv(ANALYSIS_DIR / "ssm_specificity.csv", index=False)
    print(spec_df.to_string(index=False))

    # ── per-specimen vs mean ──────────────────────────────────────────────────
    print("\n── Per-specimen vs mean shape ──────────────────────────────")
    ps_df = per_specimen_vs_mean(meshes)
    ps_df.to_csv(ANALYSIS_DIR / "per_specimen_surface_distances.csv", index=False)
    print(ps_df[["specimen", "s2s_msd", "s2s_rms", "s2s_hd95", "hausdorff", "chamfer"]].to_string(index=False))

    # ── pairwise ──────────────────────────────────────────────────────────────
    print("\n── Pairwise surface distances ──────────────────────────────")
    pw_df = pairwise_metrics(meshes)
    pw_df.to_csv(ANALYSIS_DIR / "pairwise_surface_distances.csv", index=False)

    print("\n── Pairwise summary (all pairs, mm) ────────────────────────")
    for col, label in [
        ("s2s_msd",      "Mean Surface Distance (MSD)"),
        ("s2s_rms",      "RMSE                      "),
        ("s2s_hd95",     "HD95                      "),
        ("hausdorff",    "Hausdorff                 "),
        ("chamfer",      "Chamfer                   "),
        ("p2s_mean_a2b", "P2S mean A->B             "),
        ("p2s_hd95_a2b", "P2S HD95 A->B             "),
    ]:
        if col not in pw_df.columns:
            continue
        v = pw_df[col]
        print(f"  {label}: mean={v.mean():.3f}  std={v.std():.3f}  "
              f"min={v.min():.3f}  max={v.max():.3f}")

    # ── summary CSV ───────────────────────────────────────────────────────────
    cum = comp_df["cumulative_variance_pct"].values
    summary = pd.DataFrame({
        "metric": [
            "n_shapes",
            "n_vertices",
            "compactness_90pct_modes",
            "compactness_95pct_modes",
            "gen_full_mean_mm",
            "gen_full_std_mm",
            "spec_full_mean_mm",
            "spec_full_std_mm",
            "pairwise_msd_mean_mm",
            "pairwise_rms_mean_mm",
            "pairwise_hd95_mean_mm",
            "pairwise_hausdorff_mean_mm",
            "pairwise_chamfer_mean_mm",
        ],
        "value": [
            n,
            X.shape[1] // 3,
            int(np.searchsorted(cum, 90.0) + 1),
            int(np.searchsorted(cum, 95.0) + 1),
            gen_df["mean_mm"].iloc[-1],
            gen_df["std_mm"].iloc[-1],
            spec_df["mean_mm"].iloc[-1],
            spec_df["std_mm"].iloc[-1],
            pw_df["s2s_msd"].mean(),
            pw_df["s2s_rms"].mean(),
            pw_df["s2s_hd95"].mean(),
            pw_df["hausdorff"].mean(),
            pw_df["chamfer"].mean(),
        ]
    })
    summary.to_csv(ANALYSIS_DIR / "ssm_summary.csv", index=False)

    print("\n── Summary ─────────────────────────────────────────────────")
    print(summary.to_string(index=False))

    # ── plots ─────────────────────────────────────────────────────────────────
    print("\nGenerating plots...")
    plot_compactness(comp_df)
    plot_gen_spec(gen_df, spec_df)
    plot_surface_distributions(pw_df)
    plot_per_specimen(ps_df)

    print(f"\nAll outputs saved to: {ANALYSIS_DIR}")
    print("  ssm_compactness.csv / .png")
    print("  ssm_generalization.csv")
    print("  ssm_specificity.csv")
    print("  ssm_summary.csv")
    print("  per_specimen_surface_distances.csv / _vs_mean.png")
    print("  pairwise_surface_distances.csv")
    print("  surface_distance_distributions.png")
    print("  ssm_generalization_specificity.png")


if __name__ == "__main__":
    main()
