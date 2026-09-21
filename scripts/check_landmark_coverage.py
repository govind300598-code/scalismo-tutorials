#!/usr/bin/env python3
"""
Fast, dependency-free cross-check: for one or more dataset folders, does every STL file have a matching
landmark CSV row with all 5 landmarks (GC, TS, IA, PLA, AC) present and numeric?

This is meant to be run FIRST, before firing up sbt/scalismo at all -- it mirrors the exact CSV-selection and
landmark-parsing logic of ScapulaData.scala (scala/scapula/ScapulaData.scala: csvFile, resolveColumns,
readLandmarkCsv, specimens) in plain Python, so a "yes, this folder is clean" or "no, N specimens are missing
landmarks" answer here is the same answer the Scala pipeline would silently print and skip past in
Stage1Diagnostics -- just surfaced up front, per folder, in seconds, with no JVM startup and no changes made
to any file.

Usage:
    python3 scripts/check_landmark_coverage.py <dir1> [dir2] [dir3 ...]

Example (checking all three "100 plus scapula data" folders in one run):
    python3 scripts/check_landmark_coverage.py \\
        "/path/to/100 plus scapula data/hill_sachs_STLs_scapula" \\
        "/path/to/100 plus scapula data/paired_shoulder_STLs_scapula" \\
        "/path/to/100 plus scapula data/paired_scapulae_STLs"

Read-only: never writes, moves, or renames anything in the folders it checks.
"""
import csv
import os
import re
import sys

LANDMARK_NAMES = ["GC", "TS", "IA", "PLA", "AC"]
FALLBACK_START_IDX = {"GC": 11, "TS": 14, "IA": 17, "PLA": 20, "AC": 23}


def normalise_header(h):
    return re.sub(r"[^a-z0-9]", "", h.strip().lower())


def find_axis(norm_header, lm, axis):
    l = lm.lower()
    for i, h in enumerate(norm_header):
        if h == f"{l}{axis}" or (h.startswith(l) and h.endswith(axis) and len(h) <= len(l) + 2):
            return i
    return None


def resolve_columns(header):
    """Mirrors ScapulaData.resolveColumns: match by header name first, fall back to hardcoded offsets."""
    norm = [normalise_header(h) for h in header]
    by_name = {}
    for lm in LANDMARK_NAMES:
        xi, yi, zi = find_axis(norm, lm, "x"), find_axis(norm, lm, "y"), find_axis(norm, lm, "z")
        if xi is not None and yi is not None and zi is not None:
            by_name[lm] = (xi, yi, zi)
    if len(by_name) == len(LANDMARK_NAMES):
        return by_name, True
    return {lm: (i, i + 1, i + 2) for lm, i in FALLBACK_START_IDX.items()}, False


def tokens(s):
    return {t for t in re.split(r"[^a-z0-9]+", s.lower()) if len(t) > 2}


def pick_csv_file(directory):
    """Mirrors ScapulaData.csvFile: non-'single_*'-prefixed '*model_data*.csv' files, disambiguated (if more
    than one) by how many of the directory's own name tokens each candidate filename contains."""
    try:
        entries = os.listdir(directory)
    except OSError as e:
        return None, [], f"cannot list directory: {e}"

    candidates = [
        f for f in entries
        if f.lower().endswith(".csv") and "model_data" in f.lower() and not f.lower().startswith("single")
    ]
    if not candidates:
        return None, entries, "no non-'single_*' '*model_data*.csv' file found"
    if len(candidates) == 1:
        return candidates[0], candidates, None

    dir_tokens = tokens(os.path.basename(directory.rstrip("/")))
    ranked = sorted(candidates, key=lambda f: (-len(dir_tokens & tokens(f)), f))
    return ranked[0], candidates, None


def parse_coord(raw):
    try:
        return float(raw.strip())
    except (ValueError, AttributeError):
        return None


def read_landmark_csv(path):
    """Returns (fully_valid_rows: {model_id: {lm: (x,y,z)}}, skipped: {model_id: [reason, ...]}, from_header)."""
    with open(path, newline="", encoding="utf-8-sig") as f:
        rows = list(csv.reader(f))
    if not rows:
        return {}, {}, False
    header = [c.strip() for c in rows[0]]
    cols, from_header = resolve_columns(header)

    valid = {}
    skipped = {}
    for row in rows[1:]:
        if not any(c.strip() for c in row):
            continue
        model_id = row[0].strip()
        lms = {}
        reasons = []
        for lm in LANDMARK_NAMES:
            xi, yi, zi = cols[lm]
            if len(row) <= max(xi, yi, zi):
                reasons.append(f"{lm}: row only has {len(row)} columns")
                continue
            x, y, z = parse_coord(row[xi]), parse_coord(row[yi]), parse_coord(row[zi])
            if x is None or y is None or z is None:
                reasons.append(f"{lm}: non-numeric value ('{row[xi]}', '{row[yi]}', '{row[zi]}')")
                continue
            lms[lm] = (x, y, z)
        if len(lms) == len(LANDMARK_NAMES):
            valid[model_id] = lms
        else:
            skipped[model_id] = reasons
    return valid, skipped, from_header


def specimen_id(stl_filename):
    """Mirrors ScapulaData.specimens: strip '.stl', then a trailing '_Scapula' (case-insensitive) some
    datasets' filenames carry but the CSV's own subject-id column omits."""
    base = re.sub(r"\.stl$", "", stl_filename, flags=re.IGNORECASE)
    return re.sub(r"_scapula$", "", base, flags=re.IGNORECASE)


def check_folder(directory):
    print("=" * 100)
    print(directory)
    print("=" * 100)

    if not os.path.isdir(directory):
        print("  !! not a directory / does not exist -- skipped")
        return

    csv_name, all_csv_candidates, err = pick_csv_file(directory)
    if csv_name is None:
        print(f"  !! {err}")
        if all_csv_candidates:
            print(f"     files present: {', '.join(sorted(all_csv_candidates))}")
        return
    if len(all_csv_candidates) > 1:
        print(f"  multiple candidate CSVs found ({', '.join(sorted(all_csv_candidates))}) -- picked '{csv_name}'")
    csv_path = os.path.join(directory, csv_name)
    print(f"  landmark CSV: {csv_name}")

    valid_rows, skipped_rows, from_header = read_landmark_csv(csv_path)
    if from_header:
        print("  landmark columns resolved BY HEADER NAME (safe against column shifts)")
    else:
        print("  !! header names could not be matched -- fell back to hardcoded column offsets; verify manually")

    stl_files = sorted(f for f in os.listdir(directory) if f.lower().endswith(".stl"))
    specimens = [(f, specimen_id(f)) for f in stl_files]

    matched = [(f, sid) for f, sid in specimens if sid in valid_rows]
    incomplete = [(f, sid) for f, sid in specimens if sid in skipped_rows]
    unmatched = [(f, sid) for f, sid in specimens if sid not in valid_rows and sid not in skipped_rows]
    specimen_ids = {sid for _, sid in specimens}
    orphan_rows = sorted(mid for mid in set(valid_rows) | set(skipped_rows) if mid not in specimen_ids)

    print(f"\n  {len(stl_files)} STL files, {len(valid_rows) + len(skipped_rows)} CSV rows")
    print(f"  matched, all 5 landmarks OK : {len(matched)}")
    print(f"  matched, but landmark(s) bad: {len(incomplete)}")
    print(f"  STL with NO matching CSV row: {len(unmatched)}")
    print(f"  CSV rows with no matching STL: {len(orphan_rows)}")

    if incomplete:
        print("\n  -- STL files whose CSV row is missing/invalid landmark(s) --")
        for f, sid in incomplete:
            print(f"    {f}  (id='{sid}'):")
            for reason in skipped_rows[sid]:
                print(f"        {reason}")

    if unmatched:
        print("\n  -- STL files with NO CSV row at all (id not found in first column) --")
        for f, sid in unmatched:
            print(f"    {f}  (looked for id='{sid}')")

    if orphan_rows:
        print("\n  -- CSV rows with no matching STL file (informational; not necessarily a problem) --")
        for mid in orphan_rows:
            print(f"    {mid}")

    if not incomplete and not unmatched:
        print(f"\n  OK: every one of the {len(stl_files)} STL files has a complete, valid landmark row.")
    else:
        print(f"\n  !! {len(incomplete) + len(unmatched)} of {len(stl_files)} STL files do NOT have a usable "
              "landmark row and will be silently dropped from the pool by ScapulaData/ReferenceSelection.")
    print()


def main():
    if len(sys.argv) < 2:
        print(__doc__)
        sys.exit(1)
    for directory in sys.argv[1:]:
        check_folder(directory)


if __name__ == "__main__":
    main()
