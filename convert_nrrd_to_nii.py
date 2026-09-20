#!/usr/bin/env python3
"""
Convert all *_volume.nrrd files in a directory to *_volume.nii (uncompressed NIfTI).
Scalismo 0.92.x ImageIO supports .nii but NOT .nii.gz or .nrrd directly.

Usage:
    python3 convert_nrrd_to_nii.py "/home/g25upadh/Documents/all ct from hoel 3d"

Requires SimpleITK:
    pip install SimpleITK
"""
import sys
import os
import glob

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

for src in files:
    dst = src[:-5] + ".nii"   # replace .nrrd with .nii (uncompressed; Scalismo 0.92.x needs this)
    basename = os.path.basename(src)
    if os.path.exists(dst):
        print(f"  SKIP (already exists): {os.path.basename(dst)}")
        skip += 1
        continue
    try:
        img = sitk.ReadImage(src)
        img = sitk.Cast(img, sitk.sitkInt16)  # HU fits in Int16; avoids 32-bit allocations in Scalismo
        sitk.WriteImage(img, dst)
        size = os.path.getsize(dst) // (1024 * 1024)
        print(f"  OK  {os.path.basename(dst)}  ({size} MB)")
        ok += 1
    except Exception as e:
        print(f"  FAIL {basename}: {e}")
        fail += 1

print(f"\nDone: {ok} converted, {skip} skipped, {fail} failed.")
if fail:
    print("Failed files will need to be inspected manually.")
