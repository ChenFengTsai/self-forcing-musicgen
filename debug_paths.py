#!/usr/bin/env python3
"""
Debug helper for windowed_passt_kld.py path resolution.

Runs through the metadata, constructs the expected reference and generated
paths, and reports which ones exist / don't exist. This tells you if your
'no pair' error is because:
  (a) Reference root is wrong
  (b) Gen root is wrong
  (c) Filename stems don't match what's on disk
  (d) File extensions differ (.wav vs .mp3 vs .flac etc)

Usage:
    python debug_paths.py \
        --metadata /mnt/user-data/uploads/MusicBench_test_A.json \
        --reference_root /storage/ssd1/richtsai1103/MusicBench/datashare \
        --gen_root /storage/ssd3/richtsai1103/MusicBench \
        --models MusicGen_SASS MusicGen_TF \
        --n_sample 10
"""

import argparse
import json
import os
from pathlib import Path


def load_metadata(path):
    entries = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if line:
                entries.append(json.loads(line))
    return entries


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--metadata", required=True)
    ap.add_argument("--reference_root", required=True)
    ap.add_argument("--gen_root", required=True)
    ap.add_argument("--models", nargs="+", required=True)
    ap.add_argument("--segments", nargs="+", type=int, default=[0, 1, 2])
    ap.add_argument("--n_sample", type=int, default=10,
                    help="How many clips to print in detail.")
    args = ap.parse_args()

    metadata = load_metadata(args.metadata)
    print(f"Loaded {len(metadata)} metadata entries.\n")

    # ---- 1. Check if reference_root exists ----
    print("=" * 70)
    print("STEP 1: Check reference_root")
    print("=" * 70)
    print(f"reference_root: {args.reference_root}")
    print(f"  exists: {os.path.isdir(args.reference_root)}")
    if os.path.isdir(args.reference_root):
        # Sample what's actually in the data/ folder
        data_dir = os.path.join(args.reference_root, "data")
        print(f"  data subdir exists: {os.path.isdir(data_dir)}")
        if os.path.isdir(data_dir):
            files = sorted(os.listdir(data_dir))[:5]
            print(f"  first 5 files in data/: {files}")
    print()

    # ---- 2. Check gen_root per model ----
    print("=" * 70)
    print("STEP 2: Check gen_root per model")
    print("=" * 70)
    for model in args.models:
        model_dir = os.path.join(args.gen_root, model, "generation_10s_split")
        print(f"\n{model}:")
        print(f"  {model_dir}")
        print(f"  exists: {os.path.isdir(model_dir)}")
        if os.path.isdir(model_dir):
            for seg in args.segments:
                seg_dir = os.path.join(model_dir, f"seg{seg}")
                if os.path.isdir(seg_dir):
                    files = sorted(os.listdir(seg_dir))
                    print(f"  seg{seg}: {len(files)} files. First 3: {files[:3]}")
                else:
                    print(f"  seg{seg}: MISSING")
    print()

    # ---- 3. For each metadata entry, check expected paths ----
    print("=" * 70)
    print(f"STEP 3: Path resolution for first {args.n_sample} clips")
    print("=" * 70)

    stats = {m: {"ref_exists": 0, "ref_missing": 0,
                 "seg_exists": {s: 0 for s in args.segments},
                 "seg_missing": {s: 0 for s in args.segments}}
             for m in args.models}
    global_ref = {"exists": 0, "missing": 0}

    for i, entry in enumerate(metadata):
        location = entry["location"]
        stem = Path(location).stem

        # Reference path
        ref_path = os.path.join(args.reference_root, location)
        ref_exists = os.path.isfile(ref_path)
        global_ref["exists" if ref_exists else "missing"] += 1

        if i < args.n_sample:
            print(f"\n[{i}] location={location!r}  stem={stem!r}")
            print(f"  REF: {ref_path}")
            print(f"    -> {'FOUND' if ref_exists else 'MISSING'}")

        for model in args.models:
            for seg in args.segments:
                gen_path = os.path.join(
                    args.gen_root, model, "generation_10s_split",
                    f"seg{seg}", f"{stem}_seg{seg}.wav"
                )
                exists = os.path.isfile(gen_path)
                key = "seg_exists" if exists else "seg_missing"
                stats[model][key][seg] += 1

                if i < args.n_sample:
                    print(f"  {model} seg{seg}: {gen_path}")
                    print(f"    -> {'FOUND' if exists else 'MISSING'}")

    # ---- 4. Summary ----
    print("\n" + "=" * 70)
    print("STEP 4: Global summary")
    print("=" * 70)
    print(f"\nReferences: {global_ref['exists']}/{len(metadata)} found, "
          f"{global_ref['missing']} missing")
    for model in args.models:
        print(f"\n{model}:")
        for seg in args.segments:
            e = stats[model]["seg_exists"][seg]
            m = stats[model]["seg_missing"][seg]
            print(f"  seg{seg}: {e}/{e+m} found, {m} missing")

    # ---- 5. If refs missing, try to find where they actually are ----
    if global_ref["missing"] > 0 and len(metadata) > 0:
        print("\n" + "=" * 70)
        print("STEP 5: Ref files missing — trying to locate them")
        print("=" * 70)
        sample_location = metadata[0]["location"]
        sample_stem = Path(sample_location).stem
        print(f"Looking for files named like: {sample_stem}.*")
        print(f"(Searching under {args.reference_root}, max depth 4)")

        # Shallow search
        found = []
        for dirpath, dirnames, filenames in os.walk(args.reference_root):
            depth = dirpath[len(args.reference_root):].count(os.sep)
            if depth > 4:
                dirnames[:] = []  # don't recurse further
                continue
            for f in filenames:
                if sample_stem in f:
                    found.append(os.path.join(dirpath, f))
                    if len(found) >= 5:
                        break
            if len(found) >= 5:
                break
        if found:
            print(f"Found {len(found)} candidate(s):")
            for p in found:
                print(f"  {p}")
        else:
            print(f"Did not find any file containing '{sample_stem}' in name.")


if __name__ == "__main__":
    main()