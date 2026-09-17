#!/usr/bin/env python3
"""
Render 5 scapulae × 4 anatomical views for publication.

After landmark-based canonical alignment:
  Anterior  — costal / subscapular fossa (concave side)
  Posterior — dorsal surface with spine (convex ridge side)
  Lateral   — glenoid face-on
  Medial    — medial border

Output:
  <OUT_DIR>/paper_views.png        5 × 4 publication grid  (one per specimen × view)
  <OUT_DIR>/individual/<name>.png  One 4-panel figure per specimen

Usage:
    python render_paper_views.py
    SCAPULA_DATA_DIR=/path/to/STLs python render_paper_views.py

Requirements:  pip install trimesh numpy scipy matplotlib pillow
"""

import os, sys, io, csv
from pathlib import Path

import numpy as np
import trimesh
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
from PIL import Image

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
DATA_DIR = Path(os.environ.get(
    "SCAPULA_DATA_DIR",
    "/home/g25upadh/Documents/database_v1.11/paired_scapulae_STLs"))
OUT_DIR = DATA_DIR.parent / "paper_figures"

SPECIMENS = [
    ("paired_scapula_002_M_56_L", "Subj 002  M·56  Left\nNarrowest (BBoxX=75 mm)"),
    ("paired_scapula_006_F_60_R", "Subj 006  F·60  Right\nTall & Narrow variant"),
    ("paired_scapula_001_M_64_L", "Subj 001  M·64  Left\nWidest / Largest"),
    ("paired_scapula_008_F_73_L", "Subj 008  F·73  Left\nSmallest overall"),
    ("paired_scapula_007_M_26_L", "Subj 007  M·26  Left\nYoungest (age 26)"),
]

VIEWS = [
    ("Anterior\n(Costal)",   "ant"),
    ("Posterior\n(Dorsal)",  "post"),
    ("Lateral\n(Glenoid)",   "lat"),
    ("Medial\n(Border)",     "med"),
]

# Bone colour and light for rendering
BONE_BASE  = np.array([0.88, 0.75, 0.60])
LIGHT_DIR  = np.array([0.6, 0.8, 1.0])
LIGHT_DIR /= np.linalg.norm(LIGHT_DIR)
AMBIENT    = 0.35
BG_COLOR   = (0.96, 0.96, 0.96)

# Max triangles passed to matplotlib (decimation target to keep it fast)
MAX_FACES = 12_000


# ---------------------------------------------------------------------------
# Landmark loading
# ---------------------------------------------------------------------------
LM_NAMES = ["GC", "TS", "IA", "PLA", "AC"]

def _norm_hdr(h):
    return h.strip().lower().replace(" ", "").replace("_", "")

def load_landmarks(csv_path: Path) -> dict:
    """Return {modelId: {lm_name: np.array([x,y,z])}}."""
    result = {}
    with open(csv_path, newline="") as f:
        reader = csv.reader(f)
        raw_hdr = next(reader)
        hdr = [_norm_hdr(h) for h in raw_hdr]

        def col(lm, ax):
            target = f"{lm.lower()}{ax}"
            for i, h in enumerate(hdr):
                if h == target or (h.startswith(lm.lower()) and h.endswith(ax)):
                    return i
            raise KeyError(f"Column for {lm}{ax} not found in header {raw_hdr[:26]}")

        cols = {lm: (col(lm, "x"), col(lm, "y"), col(lm, "z")) for lm in LM_NAMES}

        for row in reader:
            if not any(row):
                continue
            mid = row[0].strip()
            pts = {}
            for lm, (xi, yi, zi) in cols.items():
                try:
                    pts[lm] = np.array([float(row[xi]), float(row[yi]), float(row[zi])])
                except (ValueError, IndexError):
                    pass
            if len(pts) == len(LM_NAMES):
                result[mid] = pts
    return result


# ---------------------------------------------------------------------------
# Canonical alignment from landmarks
#   After alignment:
#     +Y  → superior  (IA → TS direction along medial border)
#     +X  → lateral   (toward glenoid / lateral angle)
#     +Z  → anterior  (costal / subscapular surface faces +Z)
# ---------------------------------------------------------------------------
def canonical_frame(lms: dict):
    """
    Returns (R, origin) such that  v_aligned = R.T @ (v - origin)
    """
    IA = lms["IA"]
    TS = lms["TS"]
    GC = lms["GC"]

    # Superior axis along medial border
    y = TS - IA
    y /= np.linalg.norm(y)

    # Raw lateral direction (medial → glenoid)
    lat = GC - IA
    lat /= np.linalg.norm(lat)

    # Anterior normal = cross(lat, y)  [for a LEFT scapula, costal surface faces +Z]
    z = np.cross(lat, y)
    if np.linalg.norm(z) < 1e-6:
        z = np.array([0.0, 0.0, 1.0])
    else:
        z /= np.linalg.norm(z)

    # Recompute lateral to be truly orthogonal
    x = np.cross(y, z)
    x /= np.linalg.norm(x)

    R = np.column_stack([x, y, z])   # columns = new basis in old coords
    return R, IA


def align_mesh(mesh, lms):
    R, origin = canonical_frame(lms)
    verts = mesh.vertices - origin
    verts = (R.T @ verts.T).T
    lms_aligned = {k: R.T @ (v - origin) for k, v in lms.items()}
    m2 = trimesh.Trimesh(vertices=verts, faces=mesh.faces, process=False)
    return m2, lms_aligned


# ---------------------------------------------------------------------------
# Right-scapula handling: mirror so all become "left" orientation
# ---------------------------------------------------------------------------
def mirror_if_right(mesh, lms, name):
    if name.endswith("_R"):
        verts = mesh.vertices.copy()
        verts[:, 0] = -verts[:, 0]
        # flip winding
        faces = mesh.faces[:, [0, 2, 1]]
        m2 = trimesh.Trimesh(vertices=verts, faces=faces, process=False)
        lms2 = {k: np.array([-v[0], v[1], v[2]]) for k, v in lms.items()}
        return m2, lms2
    return mesh, lms


# ---------------------------------------------------------------------------
# Decimation helper
# ---------------------------------------------------------------------------
def decimate(mesh, target=MAX_FACES):
    if len(mesh.faces) <= target:
        return mesh
    try:
        m2 = mesh.simplify_quadric_decimation(target)
        if m2 is not None and len(m2.faces) > 10:
            return m2
    except Exception:
        pass
    # Fallback: uniform random face subset
    idx = np.random.choice(len(mesh.faces), target, replace=False)
    return trimesh.Trimesh(vertices=mesh.vertices,
                           faces=mesh.faces[idx], process=False)


# ---------------------------------------------------------------------------
# matplotlib 3-D renderer (headless, no GPU needed)
# ---------------------------------------------------------------------------
VIEW_PARAMS = {
    # (elev°, azim°)  — after canonical alignment
    "ant":  (  5, -90),   # looking from +Z (anterior / costal)
    "post": (  5,  90),   # looking from -Z (posterior / dorsal)
    "lat":  (  5,   0),   # looking from +X (lateral / glenoid)
    "med":  (  5, 180),   # looking from -X (medial border)
}


def render_view_mpl(ax, mesh, view_key, lms_aligned=None):
    verts = mesh.vertices
    faces = mesh.faces
    tris  = verts[faces]

    # Per-face normals → Phong-ish shading
    e1 = tris[:, 1] - tris[:, 0]
    e2 = tris[:, 2] - tris[:, 0]
    normals = np.cross(e1, e2)
    norms   = np.linalg.norm(normals, axis=1, keepdims=True)
    norms[norms < 1e-12] = 1.0
    normals /= norms

    diffuse   = np.clip(normals @ LIGHT_DIR, 0.0, 1.0)
    intensity = AMBIENT + (1.0 - AMBIENT) * diffuse        # [AMBIENT … 1]
    colors    = np.clip(intensity[:, None] * BONE_BASE, 0, 1)

    poly = Poly3DCollection(tris,
                            facecolor=colors,
                            edgecolor="none",
                            shade=False,
                            zsort="average")
    ax.add_collection3d(poly)

    # Equal-aspect bounding
    mn, mx = verts.min(0), verts.max(0)
    ctr    = (mn + mx) * 0.5
    r      = (mx - mn).max() * 0.55
    ax.set_xlim(ctr[0]-r, ctr[0]+r)
    ax.set_ylim(ctr[1]-r, ctr[1]+r)
    ax.set_zlim(ctr[2]-r, ctr[2]+r)

    elev, azim = VIEW_PARAMS[view_key]
    ax.view_init(elev=elev, azim=azim)
    ax.set_proj_type("ortho")

    ax.set_facecolor(BG_COLOR)
    ax.grid(False)
    ax.set_axis_off()

    # Overlay landmark dots
    if lms_aligned:
        LM_COLORS = dict(GC="#e74c3c", TS="#2980b9", IA="#27ae60",
                         PLA="#8e44ad", AC="#e67e22")
        for nm, pt in lms_aligned.items():
            ax.scatter(*pt, s=18, c=LM_COLORS.get(nm, "k"),
                       depthshade=True, zorder=5)


# ---------------------------------------------------------------------------
# Build individual 4-panel figure for one specimen
# ---------------------------------------------------------------------------
def make_specimen_figure(name, label, mesh, lms_aligned, out_path):
    fig = plt.figure(figsize=(16, 4.5), facecolor="white")
    fig.suptitle(label.replace("\n", "   "), fontsize=13, fontweight="bold", y=1.02)

    for col_idx, (view_label, view_key) in enumerate(VIEWS):
        ax = fig.add_subplot(1, 4, col_idx + 1, projection="3d",
                             facecolor=BG_COLOR)
        render_view_mpl(ax, mesh, view_key, lms_aligned)
        ax.set_title(view_label, fontsize=10, pad=4)

    # Landmark legend
    LM_COLORS = {"GC": "#e74c3c", "TS": "#2980b9", "IA": "#27ae60",
                 "PLA": "#8e44ad", "AC": "#e67e22"}
    LM_LABELS = {"GC": "Glenoid Centre", "TS": "Trigonum Scapulae",
                 "IA": "Inferior Angle", "PLA": "Post-Lat Angle",
                 "AC": "Acromion Corner"}
    patches = [mpatches.Patch(color=c, label=f"{k}: {LM_LABELS[k]}")
               for k, c in LM_COLORS.items()]
    fig.legend(handles=patches, loc="lower center", ncol=5,
               fontsize=7, frameon=False, bbox_to_anchor=(0.5, -0.04))

    plt.tight_layout(rect=[0, 0.04, 1, 1])
    plt.savefig(out_path, dpi=150, bbox_inches="tight",
                facecolor="white", edgecolor="none")
    plt.close()
    print(f"    Saved → {out_path.name}")


# ---------------------------------------------------------------------------
# Build combined 5 × 4 publication grid
# ---------------------------------------------------------------------------
def make_combined_figure(specimen_panels: list, out_path: Path):
    """
    specimen_panels: list of PIL Images, length = 5 (one per specimen, 4-panel each)
    Stacks them vertically into one publication-ready PNG.
    """
    widths  = [img.width  for img in specimen_panels]
    heights = [img.height for img in specimen_panels]
    max_w   = max(widths)
    total_h = sum(heights)

    combined = Image.new("RGB", (max_w, total_h), (255, 255, 255))
    y_off = 0
    for img in specimen_panels:
        combined.paste(img, (0, y_off))
        y_off += img.height

    combined.save(out_path, dpi=(150, 150))
    print(f"\n  Combined grid → {out_path}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    print(f"\n{'='*70}")
    print("  SCAPULA PAPER VIEW RENDERER")
    print(f"{'='*70}")

    if not DATA_DIR.exists():
        print(f"ERROR: {DATA_DIR} not found. Set SCAPULA_DATA_DIR.")
        sys.exit(1)

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    ind_dir = OUT_DIR / "individual"
    ind_dir.mkdir(exist_ok=True)

    # Load landmark CSV
    try:
        csv_files = [f for f in DATA_DIR.iterdir()
                     if f.suffix.lower() == ".csv"
                     and "scapula" in f.name.lower()
                     and "model_data" in f.name.lower()
                     and not f.name.lower().startswith("single")]
        csv_files.sort()
        csv_path = csv_files[0]
        print(f"  Landmark CSV: {csv_path.name}")
        all_lms = load_landmarks(csv_path)
        print(f"  Loaded landmarks for {len(all_lms)} specimens")
    except Exception as e:
        print(f"  WARNING: Could not load landmarks ({e}) — rendering without alignment")
        all_lms = {}

    panels = []
    for name, label in SPECIMENS:
        stl_path = DATA_DIR / f"{name}.stl"
        if not stl_path.exists():
            print(f"\n  SKIP {name}: STL not found at {stl_path}")
            continue

        print(f"\n  Processing {name} ...")

        # Load mesh
        mesh = trimesh.load(str(stl_path), force="mesh")
        print(f"    {len(mesh.vertices):,} verts  {len(mesh.faces):,} faces")

        # Mirror right scapulae → left orientation
        lms_raw = all_lms.get(name, {})
        mesh, lms_raw = mirror_if_right(mesh, lms_raw, name)

        # Canonical alignment from landmarks (if available)
        if lms_raw:
            mesh, lms_aligned = align_mesh(mesh, lms_raw)
            print(f"    Landmark-aligned: IA→origin, medial border→+Y, costal→+Z")
        else:
            # Fallback: just centre the mesh
            mesh.vertices -= mesh.centroid
            lms_aligned = {}
            print(f"    No landmarks — centred only")

        # Decimate for fast rendering
        mesh_draw = decimate(mesh, MAX_FACES)
        print(f"    After decimation: {len(mesh_draw.faces):,} faces")

        # Individual 4-panel figure
        out_ind = ind_dir / f"{name}.png"
        make_specimen_figure(name, label, mesh_draw, lms_aligned, out_ind)

        # Read back for combined grid
        panels.append(Image.open(out_ind))

    if panels:
        combined_path = OUT_DIR / "paper_views.png"
        make_combined_figure(panels, combined_path)
        print(f"\n{'='*70}")
        print("  DONE")
        print(f"  Individual panels : {ind_dir}")
        print(f"  Combined grid     : {combined_path}")
        print(f"{'='*70}\n")
    else:
        print("No specimens rendered.")


if __name__ == "__main__":
    main()
