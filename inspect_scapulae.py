#!/usr/bin/env python3
"""
Scapula Visual Inspector
========================
Loads every STL in the dataset directory, computes shape statistics,
identifies statistical outliers, renders a comparison grid, and writes
an HTML report you can open in any browser.

Requirements (install once):
    pip install trimesh numpy scipy matplotlib pillow open3d

Usage:
    python inspect_scapulae.py
    # or override the directory:
    SCAPULA_DATA_DIR=/path/to/STLs python inspect_scapulae.py
"""

import os
import sys
import json
import base64
import io
from pathlib import Path

import numpy as np
import trimesh
from scipy import stats
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.patches import Patch

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
DATA_DIR = Path(
    os.environ.get(
        "SCAPULA_DATA_DIR",
        "/home/g25upadh/Documents/database_v1.11/paired_scapulae_STLs",
    )
)
OUT_DIR = DATA_DIR.parent / "scapula_inspection"
OUTLIER_ZSCORE = 2.0          # |z| > this → flagged as outlier
N_PCA_SAMPLES = 2000          # points sampled per mesh for PCA


# ---------------------------------------------------------------------------
# STL loading
# ---------------------------------------------------------------------------
def load_stls(data_dir: Path) -> dict:
    files = sorted(data_dir.glob("*.stl"))
    meshes = {}
    for f in files:
        try:
            m = trimesh.load(str(f), force="mesh")
            if hasattr(m, "vertices") and len(m.vertices) > 0:
                meshes[f.stem] = m
                print(f"  OK  {f.stem}  ({len(m.vertices)} verts)")
            else:
                print(f"  EMPTY {f.stem}")
        except Exception as e:
            print(f"  ERROR {f.name}: {e}")
    return meshes


# ---------------------------------------------------------------------------
# Shape statistics
# ---------------------------------------------------------------------------
def compute_stats(meshes: dict) -> list[dict]:
    records = []
    for name, m in meshes.items():
        # Bounding box extents
        ext = m.extents  # [dx, dy, dz]
        vol = abs(m.volume) if m.is_volume else abs(m.convex_hull.volume)
        rec = dict(
            name=name,
            volume=vol,
            surface_area=m.area,
            bbox_x=float(ext[0]),
            bbox_y=float(ext[1]),
            bbox_z=float(ext[2]),
            bbox_max=float(ext.max()),
            n_vertices=int(len(m.vertices)),
            n_faces=int(len(m.faces)),
        )
        # Parse metadata from file name:  paired_scapula_NNN_SEX_AGE_SIDE
        parts = name.split("_")
        try:
            rec["subject"] = parts[2]
            rec["sex"] = parts[3]
            rec["age"] = int(parts[4])
            rec["side"] = parts[5]   # L or R
        except (IndexError, ValueError):
            rec["subject"] = rec["sex"] = rec["side"] = "?"
            rec["age"] = 0
        records.append(rec)
    return records


# ---------------------------------------------------------------------------
# Outlier detection
# ---------------------------------------------------------------------------
METRICS_FOR_OUTLIER = ["volume", "surface_area", "bbox_x", "bbox_y", "bbox_z"]

def flag_outliers(records: list[dict], threshold: float = OUTLIER_ZSCORE) -> set:
    outliers = set()
    for k in METRICS_FOR_OUTLIER:
        vals = np.array([r[k] for r in records], dtype=float)
        zs = np.abs(stats.zscore(vals))
        for i, z in enumerate(zs):
            if z > threshold:
                outliers.add(records[i]["name"])
    return outliers


def per_metric_z(records: list[dict]) -> dict:
    """Return {metric: [z-score per record]} for labelling."""
    result = {}
    for k in METRICS_FOR_OUTLIER:
        vals = np.array([r[k] for r in records], dtype=float)
        result[k] = stats.zscore(vals).tolist()
    return result


# ---------------------------------------------------------------------------
# PCA on sampled point clouds  (coarse shape signature)
# ---------------------------------------------------------------------------
def pca_scatter(meshes: dict, records: list[dict], outliers: set, out_path: Path):
    """Sample N points from each mesh, stack, PCA → 2-D scatter."""
    print("  Running PCA on sampled point clouds...")
    names = [r["name"] for r in records]
    clouds = []
    for name in names:
        m = meshes[name]
        pts = trimesh.sample.sample_surface(m, N_PCA_SAMPLES)[0]
        # Translate to centroid (remove position)
        pts -= pts.mean(axis=0)
        clouds.append(pts.flatten())

    X = np.array(clouds)
    # Standardise each feature
    X -= X.mean(axis=0)
    U, S, Vt = np.linalg.svd(X, full_matrices=False)
    coords = U[:, :2] * S[:2]           # project onto first 2 PCs

    fig, ax = plt.subplots(figsize=(12, 8))
    for i, name in enumerate(names):
        color = "#e74c3c" if name in outliers else "#3498db"
        ax.scatter(coords[i, 0], coords[i, 1], c=color, s=90, zorder=3)
        short = name.replace("paired_scapula_", "")
        ax.annotate(
            short, (coords[i, 0], coords[i, 1]),
            fontsize=6, ha="left", va="bottom",
            color="#c0392b" if name in outliers else "#2c3e50",
        )

    ax.set_xlabel(f"PC1  ({100*S[0]**2/np.sum(S**2):.1f}% variance)")
    ax.set_ylabel(f"PC2  ({100*S[1]**2/np.sum(S**2):.1f}% variance)")
    ax.set_title("Shape PCA — point-cloud signature (red = statistical outlier in metrics)")
    legend_elements = [
        Patch(facecolor="#e74c3c", label="Outlier (|z|>2 in ≥1 metric)"),
        Patch(facecolor="#3498db", label="Normal"),
    ]
    ax.legend(handles=legend_elements)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(out_path, dpi=120, bbox_inches="tight")
    plt.close()
    print(f"  PCA scatter saved → {out_path}")


# ---------------------------------------------------------------------------
# Statistics bar charts
# ---------------------------------------------------------------------------
METRIC_LABELS = {
    "volume":       "Volume (mm³)",
    "surface_area": "Surface Area (mm²)",
    "bbox_x":       "Bounding Box X (mm)",
    "bbox_y":       "Bounding Box Y (mm)",
    "bbox_z":       "Bounding Box Z (mm)",
}

def plot_stats(records: list[dict], outliers: set, out_path: Path):
    names = [r["name"] for r in records]
    short = [n.replace("paired_scapula_", "") for n in names]
    colors = ["#e74c3c" if n in outliers else "#3498db" for n in names]

    metrics = list(METRIC_LABELS.keys())
    fig, axes = plt.subplots(len(metrics), 1, figsize=(22, 4.5 * len(metrics)))
    fig.suptitle(
        "Scapula Shape Statistics\n"
        "Red bars = statistical outlier (|z-score| > 2.0 in that metric)\n"
        "Orange dashed = ±2σ from mean",
        fontsize=13, fontweight="bold", y=1.002,
    )

    for ax, metric in zip(axes, metrics):
        vals = np.array([r[metric] for r in records])
        mean, std = vals.mean(), vals.std()
        ax.bar(range(len(names)), vals, color=colors, edgecolor="white", linewidth=0.5)
        ax.axhline(mean, color="green", linewidth=1.4, linestyle="--",
                   label=f"mean = {mean:,.1f}")
        ax.axhline(mean + 2 * std, color="darkorange", linewidth=1, linestyle=":")
        ax.axhline(mean - 2 * std, color="darkorange", linewidth=1, linestyle=":")
        ax.set_xticks(range(len(names)))
        ax.set_xticklabels(short, rotation=55, ha="right", fontsize=7)
        ax.set_ylabel(METRIC_LABELS[metric])
        ax.set_title(METRIC_LABELS[metric], fontsize=10)
        ax.legend(fontsize=8)
        # Annotate outlier bars
        for i, (n, v) in enumerate(zip(names, vals)):
            if n in outliers:
                ax.text(i, v + 0.01 * vals.max(), f"↑{short[i]}",
                        ha="center", fontsize=5.5, color="#c0392b", rotation=90)
        ax.grid(axis="y", alpha=0.25)

    plt.tight_layout()
    plt.savefig(out_path, dpi=110, bbox_inches="tight")
    plt.close()
    print(f"  Stats chart saved → {out_path}")


# ---------------------------------------------------------------------------
# Thumbnail rendering (one PNG per mesh)
# ---------------------------------------------------------------------------
def render_thumbnails(meshes: dict, records: list[dict], outliers: set,
                      thumb_dir: Path, size=(300, 300)):
    thumb_dir.mkdir(exist_ok=True)
    done = []
    for rec in records:
        name = rec["name"]
        m = meshes[name]
        out = thumb_dir / f"{name}.png"
        try:
            scene = trimesh.Scene(m)
            # Consistent camera: look from front-upper-right
            scene.set_camera(angles=[0.4, 0.0, 0.6], distance=m.extents.max() * 2.5)
            data = scene.save_image(resolution=size, visible=False)
            if data:
                with open(out, "wb") as f:
                    f.write(data)
                done.append(name)
        except Exception as e:
            # Create a placeholder image with matplotlib
            fig, ax = plt.subplots(figsize=(3, 3))
            ax.text(0.5, 0.5, f"{name}\n(render failed)\n{e}",
                    ha="center", va="center", fontsize=6, transform=ax.transAxes)
            ax.axis("off")
            plt.tight_layout()
            plt.savefig(out, dpi=70)
            plt.close()
    return done


# ---------------------------------------------------------------------------
# HTML report
# ---------------------------------------------------------------------------
def img_to_b64(path: Path) -> str:
    with open(path, "rb") as f:
        return base64.b64encode(f.read()).decode()


def make_html(records: list[dict], outliers: set, z_scores: dict,
              thumb_dir: Path, stats_png: Path, pca_png: Path, out_html: Path):
    names = [r["name"] for r in records]

    # Thumbnail grid
    thumb_html = ""
    for rec in sorted(records, key=lambda r: r["name"]):
        name = rec["name"]
        tp = thumb_dir / f"{name}.png"
        flag = name in outliers
        border = "border:3px solid #e74c3c;" if flag else "border:1px solid #ccc;"
        badge = '<span class="badge">OUTLIER</span>' if flag else ""
        short = name.replace("paired_scapula_", "")
        meta = f"Subject {rec['subject']} · {rec['sex']} · Age {rec['age']} · {rec['side']} side"
        zscore_rows = "".join(
            f"<tr><td>{METRIC_LABELS.get(k, k)}</td>"
            f"<td class=\"{'zout' if abs(z_scores[k][names.index(name)]) > OUTLIER_ZSCORE else ''}\"> "
            f"{z_scores[k][names.index(name)]:+.2f}</td></tr>"
            for k in METRIC_LABELS
        )
        vol = f"{rec['volume']:,.0f} mm³"
        sa = f"{rec['surface_area']:,.0f} mm²"
        bb = f"({rec['bbox_x']:.1f} × {rec['bbox_y']:.1f} × {rec['bbox_z']:.1f}) mm"
        img_src = ""
        if tp.exists():
            img_src = f"data:image/png;base64,{img_to_b64(tp)}"

        thumb_html += f"""
        <div class="card {'outlier-card' if flag else ''}">
          {badge}
          <div class="label">{short}</div>
          <div class="meta">{meta}</div>
          {'<img src="' + img_src + '" alt="' + short + '">' if img_src else '<div class="no-img">No render</div>'}
          <table class="stats-tbl">
            <tr><td>Volume</td><td>{vol}</td></tr>
            <tr><td>Surface</td><td>{sa}</td></tr>
            <tr><td>BBox</td><td>{bb}</td></tr>
          </table>
          <details>
            <summary>Z-scores</summary>
            <table class="stats-tbl">
              <tr><th>Metric</th><th>z</th></tr>
              {zscore_rows}
            </table>
          </details>
        </div>"""

    # Encode chart images
    stats_b64 = img_to_b64(stats_png) if stats_png.exists() else ""
    pca_b64   = img_to_b64(pca_png)   if pca_png.exists()   else ""

    outlier_summary = ""
    if outliers:
        items = []
        for name in sorted(outliers):
            rec = next(r for r in records if r["name"] == name)
            items.append(
                f"<li><strong>{name}</strong> — "
                f"Subject {rec['subject']}, {rec['sex']}, age {rec['age']}, {rec['side']} side | "
                f"volume {rec['volume']:,.0f} mm³ | surface {rec['surface_area']:,.0f} mm²</li>"
            )
        outlier_summary = (
            f'<div class="outlier-box"><h2>⚠ Outlier Scapulae Found ({len(outliers)})</h2>'
            f'<ul>{"".join(items)}</ul>'
            f"<p>These specimens have at least one shape metric (volume, surface area, "
            f"or a bounding-box dimension) more than {OUTLIER_ZSCORE}σ away from the group mean. "
            f"They are highlighted in red throughout the report.</p></div>"
        )
    else:
        outlier_summary = '<div class="ok-box">✓ No statistical outliers detected (all within 2σ of mean).</div>'

    html = f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8">
<title>Scapula Inspection Report</title>
<style>
  * {{ box-sizing: border-box; margin: 0; padding: 0; }}
  body {{ font-family: Arial, sans-serif; background: #f4f6f8; color: #2c3e50; }}
  header {{ background: #2c3e50; color: white; padding: 1.2rem 2rem; }}
  header h1 {{ font-size: 1.5rem; }}
  header p  {{ font-size: 0.85rem; opacity: 0.75; margin-top: 4px; }}
  main  {{ padding: 1.5rem 2rem; }}
  section {{ margin-bottom: 2.5rem; }}
  h2 {{ font-size: 1.15rem; margin-bottom: 1rem; border-bottom: 2px solid #ddd; padding-bottom: 6px; }}

  .outlier-box {{ background: #fdecea; border: 2px solid #e74c3c; border-radius: 8px;
                  padding: 1rem 1.4rem; margin-bottom: 1.5rem; }}
  .outlier-box h2 {{ color: #c0392b; border-color: #e74c3c; }}
  .outlier-box ul {{ margin-left: 1.2rem; margin-top: 0.5rem; }}
  .outlier-box li {{ margin-bottom: 4px; font-size: 0.88rem; }}
  .ok-box {{ background: #eafaf1; border: 2px solid #27ae60; border-radius: 8px;
             padding: 0.9rem 1.4rem; margin-bottom: 1.5rem; color: #1e8449; font-weight: bold; }}

  .grid {{ display: flex; flex-wrap: wrap; gap: 14px; }}
  .card {{ background: white; border-radius: 8px; padding: 10px; width: 210px;
           position: relative; box-shadow: 0 1px 4px rgba(0,0,0,0.12); }}
  .outlier-card {{ background: #fff5f5; box-shadow: 0 0 0 3px #e74c3c; }}
  .card img {{ width: 100%; border-radius: 4px; display: block; margin: 6px 0; }}
  .no-img {{ height: 120px; background: #ecf0f1; border-radius: 4px; display:flex;
             align-items:center; justify-content:center; color:#7f8c8d;
             font-size:0.75rem; margin: 6px 0; }}
  .label {{ font-weight: bold; font-size: 0.8rem; color: #2c3e50; word-break: break-all; }}
  .meta  {{ font-size: 0.72rem; color: #7f8c8d; margin: 2px 0 4px; }}
  .badge {{ position: absolute; top: 8px; right: 8px; background: #e74c3c; color: white;
            font-size: 0.65rem; font-weight: bold; padding: 2px 6px; border-radius: 4px; }}

  .stats-tbl {{ width: 100%; border-collapse: collapse; font-size: 0.72rem; margin-top: 4px; }}
  .stats-tbl td, .stats-tbl th {{ padding: 2px 4px; border-bottom: 1px solid #eee; }}
  .stats-tbl th {{ background: #f0f0f0; font-weight: bold; }}
  .zout {{ color: #e74c3c; font-weight: bold; }}
  details {{ font-size: 0.72rem; margin-top: 5px; cursor: pointer; }}
  summary {{ color: #2980b9; }}

  .chart-img {{ width: 100%; max-width: 1400px; border-radius: 8px;
                box-shadow: 0 2px 8px rgba(0,0,0,0.1); }}
</style>
</head>
<body>
<header>
  <h1>Scapula Inspection Report</h1>
  <p>Dataset: {DATA_DIR} &nbsp;|&nbsp; {len(records)} specimens loaded &nbsp;|&nbsp; Outlier threshold: |z| &gt; {OUTLIER_ZSCORE}</p>
</header>
<main>

{outlier_summary}

<section>
  <h2>Shape Statistics — All Metrics</h2>
  {'<img class="chart-img" src="data:image/png;base64,' + stats_b64 + '" alt="statistics">' if stats_b64 else '<p>Chart not available.</p>'}
</section>

<section>
  <h2>PCA Shape Scatter (sampled point clouds)</h2>
  {'<img class="chart-img" src="data:image/png;base64,' + pca_b64 + '" alt="PCA">' if pca_b64 else '<p>PCA not available.</p>'}
</section>

<section>
  <h2>Individual Specimen Cards (red = outlier)</h2>
  <div class="grid">{thumb_html}</div>
</section>

</main>
</body>
</html>"""

    out_html.write_text(html, encoding="utf-8")
    print(f"  HTML report saved → {out_html}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    print(f"\n{'='*70}")
    print("  SCAPULA VISUAL INSPECTOR")
    print(f"{'='*70}")
    print(f"  Data dir : {DATA_DIR}")

    if not DATA_DIR.exists():
        print(f"\nERROR: '{DATA_DIR}' does not exist.")
        print("  Set the SCAPULA_DATA_DIR environment variable to your STL folder.")
        sys.exit(1)

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    thumb_dir = OUT_DIR / "thumbnails"

    print("\n[1/6] Loading STL files...")
    meshes = load_stls(DATA_DIR)
    if not meshes:
        print("No STL files loaded. Exiting.")
        sys.exit(1)
    print(f"  {len(meshes)} meshes loaded.")

    print("\n[2/6] Computing shape statistics...")
    records = compute_stats(meshes)

    print("\n[3/6] Detecting outliers...")
    outliers = flag_outliers(records)
    z_scores = per_metric_z(records)

    print("\n" + "="*70)
    print("  SHAPE STATISTICS TABLE")
    print("="*70)
    hdr = f"{'Name':<40} {'Volume':>10} {'Surface':>12} {'BBoxX':>7} {'BBoxY':>7} {'BBoxZ':>7}"
    print(hdr)
    print("-" * len(hdr))
    for r in sorted(records, key=lambda x: x["name"]):
        flag = "  ← OUTLIER" if r["name"] in outliers else ""
        print(f"{r['name']:<40} {r['volume']:>10,.0f} {r['surface_area']:>12,.0f} "
              f"{r['bbox_x']:>7.1f} {r['bbox_y']:>7.1f} {r['bbox_z']:>7.1f}{flag}")

    print()
    if outliers:
        print(f"  OUTLIER SCAPULAE ({len(outliers)}):")
        for name in sorted(outliers):
            r = next(x for x in records if x["name"] == name)
            print(f"    ★  {name}")
            print(f"       Subject {r['subject']}, Sex={r['sex']}, Age={r['age']}, Side={r['side']}")
            print(f"       Volume={r['volume']:,.0f} mm³  |  Surface={r['surface_area']:,.0f} mm²  |  "
                  f"BBox=({r['bbox_x']:.1f} × {r['bbox_y']:.1f} × {r['bbox_z']:.1f}) mm")
    else:
        print("  No outliers found (all specimens within 2σ of mean).")

    print("\n[4/6] Plotting statistics bar charts...")
    stats_png = OUT_DIR / "stats_bars.png"
    plot_stats(records, outliers, stats_png)

    print("\n[5/6] PCA scatter plot...")
    pca_png = OUT_DIR / "pca_scatter.png"
    pca_scatter(meshes, records, outliers, pca_png)

    print("\n[6/6] Rendering thumbnails and writing HTML report...")
    render_thumbnails(meshes, records, outliers, thumb_dir)
    html_path = OUT_DIR / "scapula_report.html"
    make_html(records, outliers, z_scores, thumb_dir, stats_png, pca_png, html_path)

    print(f"\n{'='*70}")
    print("  DONE")
    print(f"  Open this file in your browser:")
    print(f"  {html_path}")
    print(f"{'='*70}\n")


if __name__ == "__main__":
    main()
