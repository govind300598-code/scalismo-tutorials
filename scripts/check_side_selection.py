#!/usr/bin/env python3
"""
Fast, dependency-free audit: for one or more dataset folders, shows EXACTLY which side (left/right) of each
subject ReferenceSelection.loadPool will pick as that subject's pool representative under the default
SCAPULA_INDEPENDENT_MODEL=true, and whether that pick will be mirrored.

This mirrors the exact subject/side logic in ScapulaData.scala (specimens: isRight, subjectKey) and
ReferenceSelection.scala (loadPool's per-subject selection: left if present, otherwise the mirrored right --
never both, since a person's left and mirrored-right scapulae are not independent samples). Nothing here
touches meshes or runs any registration; it only reads filenames, so it answers "did left/right get combined
correctly" in seconds, before spending time on the real (expensive) Scala run.

Usage:
    python3 scripts/check_side_selection.py <dir1> [dir2] [dir3 ...]

Example -- the same three "100 plus scapula data" folders, combined exactly as SCAPULA_DATA_DIRS would:
    python3 scripts/check_side_selection.py \\
        "/path/to/100 plus scapula data/hill_sachs_STLs_scapula" \\
        "/path/to/100 plus scapula data/paired_shoulder_STLs_scapula" \\
        "/path/to/100 plus scapula data/paired_scapulae_STLs"

Read-only: never writes, moves, or renames anything.
"""
import os
import re
import sys


def specimen_id(stl_filename):
    """Mirrors ScapulaData.specimens: strip '.stl', then a trailing '_Scapula' (case-insensitive)."""
    base = re.sub(r"\.stl$", "", stl_filename, flags=re.IGNORECASE)
    return re.sub(r"_scapula$", "", base, flags=re.IGNORECASE)


def subject_key(model_id):
    """Mirrors ScapulaData.subjectKey: strip a trailing _L or _R."""
    if model_id.endswith("_L") or model_id.endswith("_R"):
        return model_id[:-2]
    return model_id


def list_specimens(directory):
    """Returns [(model_id, is_right, source_filename, source_dir), ...] sorted by filename, same order
    ScapulaData.specimens/ReferenceSelection.loadPool would see them in."""
    try:
        stl_files = sorted(f for f in os.listdir(directory) if f.lower().endswith(".stl"))
    except OSError as e:
        print(f"  !! cannot list {directory}: {e}")
        return []
    out = []
    for f in stl_files:
        mid = specimen_id(f)
        out.append((mid, mid.endswith("_R"), f, directory))
    return out


def main():
    if len(sys.argv) < 2:
        print(__doc__)
        sys.exit(1)
    dirs = sys.argv[1:]

    all_specimens = []
    for d in dirs:
        all_specimens.extend(list_specimens(d))

    if not all_specimens:
        print("No STL files found in any given directory.")
        sys.exit(1)

    by_subject = {}
    for mid, is_right, fname, d in all_specimens:
        by_subject.setdefault(subject_key(mid), []).append((mid, is_right, fname, d))

    print("=" * 100)
    print(f"{len(dirs)} folder(s), {len(all_specimens)} total STL files, {len(by_subject)} distinct subjects")
    print("=" * 100)
    print()

    both = left_only = right_only = 0
    for subject in sorted(by_subject):
        entries = by_subject[subject]
        left = next((e for e in entries if not e[1]), None)
        right = next((e for e in entries if e[1]), None)

        # Mirrors ReferenceSelection.loadPool: left if present, else the (mirrored) right -- never both.
        picked = left if left is not None else (entries[0] if entries else None)
        mirrored = picked is not None and picked[1]

        sides = []
        if left:
            sides.append(f"L={left[2]}")
        if right:
            sides.append(f"R={right[2]}")
        if left and right:
            both += 1
        elif left:
            left_only += 1
        elif right:
            right_only += 1

        pick_desc = "NONE (no specimen for this subject?!)" if picked is None else (
            f"{picked[2]}" + ("  [will be MIRRORED into the left-scapula frame]" if mirrored else "  [used as-is, no mirroring]")
        )
        print(f"  {subject:<32} sides present: {', '.join(sides) if sides else '(none)':<70} -> pool picks: {pick_desc}")

    print()
    print(f"Subjects with BOTH sides   : {both}  (left used, right dropped -- by design, not a bug)")
    print(f"Subjects with LEFT only    : {left_only}  (used as-is)")
    print(f"Subjects with RIGHT only   : {right_only}  (mirrored into the left-scapula frame before use)")
    print()
    print(f"Under the default SCAPULA_INDEPENDENT_MODEL=true, the model-building pool will contain exactly "
          f"{len(by_subject)} entries (one per subject) -- not {len(all_specimens)} (one per STL file).")
    print("Set SCAPULA_INDEPENDENT_MODEL=false to use BOTH sides of every subject instead (doubles sample size,")
    print("but left+mirrored-right from the same person are then treated as if independent, which they are not).")


if __name__ == "__main__":
    main()
