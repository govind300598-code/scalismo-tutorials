#!/usr/bin/env python3
"""
Convert all *_volume.nrrd files in a directory to *_volume.nii (uncompressed NIfTI).
Scalismo 0.92.x ImageIO supports .nii but NOT .nii.gz or .nrrd directly.

Usage:
    python3 convert_nrrd_to_nii.py "/home/g25upadh/Documents/all ct from hoel 3d"

Requires SimpleITK:
    pip install SimpleITK

IMPORTANT — always delete existing .nii files before re-running so they are fully reconverted:
    rm "/home/g25upadh/Documents/all ct from hoel 3d/"*_volume.nii
"""
import sys
import os
import glob
import struct

try:
    import SimpleITK as sitk
except ImportError:
    print("ERROR: SimpleITK not found. Install it with:")
    print("    pip install SimpleITK")
    sys.exit(1)

nrrd_dir = sys.argv[1] if len(sys.argv) > 1 else "."
nrrd_dir = os.path.abspath(nrrd_dir)

pattern = os.path.join(nrrd_dir, "*_volume.nrrd")
files = sorted(glob.glob(pattern))

if not files:
    print(f"No *_volume.nrrd files found in:\n  {nrrd_dir}")
    sys.exit(1)

print(f"Found {len(files)} volume files in:\n  {nrrd_dir}\n")

ok = 0
skip = 0
fail = 0

def already_converted(path):
    """True if `path` is a NIfTI-1 file already cast to Int16 with scl_slope/scl_inter
    zeroed out — i.e. produced by a previous run of this exact script. Anything else
    (missing file, wrong dtype, non-zero scaling) must be reconverted."""
    try:
        with open(path, "rb") as f:
            header = f.read(348)
        if len(header) < 348:
            return False
        magic = header[344:348]
        if magic not in (b"n+1\x00", b"ni1\x00"):
            return False
        datatype = struct.unpack("<h", header[70:72])[0]
        scl_slope, scl_inter = struct.unpack("<ff", header[112:120])
        DT_INT16 = 4
        return datatype == DT_INT16 and scl_slope == 0.0 and scl_inter == 0.0
    except OSError:
        return False

for src in files:
    dst = src[:-5] + ".nii"   # replace .nrrd with .nii (uncompressed; Scalismo 0.92.x needs this)
    basename = os.path.basename(src)
    if already_converted(dst):
        print(f"  SKIP (already Int16, scl_slope=0): {os.path.basename(dst)}")
        skip += 1
        continue
    try:
        img = sitk.ReadImage(src)
        img = sitk.Cast(img, sitk.sitkInt16)  # HU fits in Int16; avoids 32-bit allocations in Scalismo
        sitk.WriteImage(img, dst)

        # Patch scl_slope and scl_inter to 0.0 in the NIfTI-1 header.
        # VTK (used internally by Scalismo) applies the slope/intercept rescaling when
        # scl_slope != 0, upcasting the data to Float32.  Setting both fields to 0.0
        # tells every NIfTI reader "raw values, no rescaling", so the Int16 voxels
        # arrive in Scalismo as Short without type conversion.
        with open(dst, "r+b") as f:
            f.seek(344)
            magic = f.read(4)
            if magic in (b"n+1\x00", b"ni1\x00"):
                f.seek(112)                          # scl_slope at byte 112, scl_inter at 116
                f.write(struct.pack("<ff", 0.0, 0.0))

        size = os.path.getsize(dst) // (1024 * 1024)
        print(f"  OK  {os.path.basename(dst)}  ({size} MB)")
        ok += 1
    except Exception as e:
        print(f"  FAIL {basename}: {e}")
        fail += 1

print(f"\nDone: {ok} converted, {skip} skipped, {fail} failed.")
if fail:
    print("Failed files will need to be inspected manually.")
