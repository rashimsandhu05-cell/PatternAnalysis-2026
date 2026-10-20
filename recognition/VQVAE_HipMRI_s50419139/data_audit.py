#!/usr/bin/env python3
"""
data_audit.py
=============
Pre-training audit of the HipMRI 2D prostate MRI slice dataset (keras_slices_data).

Run this BEFORE writing the data loader. It answers the questions the loader
depends on:

1. Which split folders exist and how many slices does each contain?
2. Which slice shapes occur? (Mixed shapes mean we must resize or crop.)
3. What do raw intensities look like? (Decides the normalisation scheme.)
4. Which label values occur in the segmentation folders, and how imbalanced?
5. Do image and segmentation folders pair up slice-for-slice?
6. PATIENT LEAKAGE: does any patient (case ID) appear in more than one split?

Outputs:
    - a printed report
    - <out>/audit_summary.json
    - <out>/sample_slices.png

This is an analysis script, not part of the model, so NumPy is allowed here
(the NumPy restriction in the spec applies to modules.py only).

Usage (on a Rangpur compute node, not the login node):
    python data_audit.py
    python data_audit.py --root /path/to/keras_slices_data --sample 300
"""

import argparse
import json
import re
from collections import Counter
from itertools import combinations
from pathlib import Path

import matplotlib

matplotlib.use("Agg")  # headless backend: Rangpur nodes have no display
import matplotlib.pyplot as plt
import nibabel as nib
import numpy as np

DEFAULT_ROOT = "/home/groups/comp3710/HipMRI_Study_open/keras_slices_data"

# Filenames look like "case_004_week_0_slice_12.nii.gz" (segmentation files may
# use a different prefix). "case" = patient, "week" = scan session, "slice" = z.
CASE_RE = re.compile(r"case_(\d+)", re.IGNORECASE)
KEY_RE = re.compile(r"(\d+)_week_(\d+).*?slice_(\d+)", re.IGNORECASE)


# --------------------------------------------------------------------------- #
# Filename parsing
# --------------------------------------------------------------------------- #
def parse_case(name: str):
    """Return the patient/case ID as a normalised string (e.g. '4'), or None."""
    m = CASE_RE.search(name) or re.search(r"(\d+)", name)
    return str(int(m.group(1))) if m else None


def parse_key(name: str):
    """Return (case, week, slice) as ints, used to pair images with masks."""
    m = KEY_RE.search(name)
    return tuple(int(g) for g in m.groups()) if m else None


def load_slice(path: Path) -> np.ndarray:
    """Load one Nifti slice as a 2D float32 array (drops a trailing singleton dim)."""
    arr = np.asarray(nib.load(str(path)).dataobj, dtype=np.float32)
    if arr.ndim == 3:
        arr = arr[..., 0]  # some HipMRI slices carry an extra dimension
    return arr


# --------------------------------------------------------------------------- #
# Per-folder audit
# --------------------------------------------------------------------------- #
def audit_folder(folder: Path, root: Path, n_sample: int, rng) -> dict:
    """Collect counts, shapes, patient IDs and sampled intensity/label stats."""
    files = sorted(p for p in folder.iterdir() if p.name.endswith((".nii", ".nii.gz")))
    name = str(folder.relative_to(root)) if folder != root else "."
    is_seg = "seg" in name.lower()

    # Cheap pass over every file: header-only reads for shape and dtype.
    case_counts = Counter()
    shapes, dtypes = Counter(), Counter()
    keys = {}
    unparsed = 0
    for f in files:
        case = parse_case(f.name)
        if case is None:
            unparsed += 1
        else:
            case_counts[case] += 1
        key = parse_key(f.name)
        if key is not None:
            keys[key] = f
        img = nib.load(str(f))
        shapes[str(tuple(img.shape))] += 1
        dtypes[str(img.get_data_dtype())] += 1

    per_case = list(case_counts.values()) or [0]
    report = {
        "name": name,
        "is_seg": is_seg,
        "n_files": len(files),
        "n_cases": len(case_counts),
        "cases": sorted(case_counts, key=int),
        "slices_per_case": {
            "min": int(min(per_case)),
            "max": int(max(per_case)),
            "mean": round(float(np.mean(per_case)), 1),
        },
        "unparsed_filenames": unparsed,
        "shape_counts": dict(shapes.most_common()),
        "dtype_counts": dict(dtypes),
        "_keys": keys,      # internal: used for image/mask pairing, not saved
        "_files": files,    # internal: used for the sample figure, not saved
    }

    # Expensive pass on a random sample: actually read voxel data.
    idx = rng.choice(len(files), size=min(n_sample, len(files)), replace=False)
    if is_seg:
        label_pixels = Counter()
        for i in idx:
            vals, counts = np.unique(np.rint(load_slice(files[i])).astype(int), return_counts=True)
            label_pixels.update(dict(zip(vals.tolist(), counts.tolist())))
        total = sum(label_pixels.values())
        report["label_pixel_fraction"] = {
            str(k): round(v / total, 5) for k, v in sorted(label_pixels.items())
        }
    else:
        mins, maxs, means, stds, blank = [], [], [], [], 0
        for i in idx:
            a = load_slice(files[i])
            mins.append(a.min()); maxs.append(a.max())
            means.append(a.mean()); stds.append(a.std())
            blank += int(a.max() == a.min())
        report["intensity_sampled"] = {
            "n_sampled": int(len(idx)),
            "global_min": float(np.min(mins)),
            "global_max": float(np.max(maxs)),
            "median_slice_max": float(np.median(maxs)),
            "median_slice_mean": float(np.median(means)),
            "median_slice_std": float(np.median(stds)),
            "blank_slices": blank,
        }
    return report


# --------------------------------------------------------------------------- #
# Cross-folder checks
# --------------------------------------------------------------------------- #
def find_seg_partner(image_name: str, seg_names: list):
    """Match e.g. 'keras_slices_train' with 'keras_slices_seg_train'."""
    for s in seg_names:
        if s.replace("seg_", "") == image_name or s == image_name.replace(
            "keras_slices_", "keras_slices_seg_"
        ):
            return s
    return None


def check_leakage(image_reports: dict) -> dict:
    """Patients shared between any two image splits (should be none)."""
    out = {}
    for a, b in combinations(sorted(image_reports), 2):
        shared = sorted(set(image_reports[a]["cases"]) & set(image_reports[b]["cases"]), key=int)
        out[f"{a} <-> {b}"] = shared
    return out


def check_pairing(img_rep: dict, seg_rep: dict) -> dict:
    """Compare (case, week, slice) keys between an image folder and its masks."""
    ik, sk = set(img_rep["_keys"]), set(seg_rep["_keys"])
    return {
        "image_folder": img_rep["name"],
        "seg_folder": seg_rep["name"],
        "parsed_image_keys": len(ik),
        "parsed_seg_keys": len(sk),
        "matched": len(ik & sk),
        "only_in_image": len(ik - sk),
        "only_in_seg": len(sk - ik),
    }


# --------------------------------------------------------------------------- #
# Figure
# --------------------------------------------------------------------------- #
def save_sample_figure(image_reports, seg_reports, pairs, out_path, rng, n_cols=4):
    """Grid of random slices per image split, with mask contours where paired."""
    names = sorted(image_reports)
    fig, axes = plt.subplots(len(names), n_cols, figsize=(3 * n_cols, 3 * len(names)), squeeze=False)
    for r, name in enumerate(names):
        rep = image_reports[name]
        seg_rep = seg_reports.get(pairs.get(name))
        picks = rng.choice(len(rep["_files"]), size=min(n_cols, len(rep["_files"])), replace=False)
        for c, i in enumerate(picks):
            f = rep["_files"][i]
            ax = axes[r, c]
            ax.imshow(load_slice(f).T, cmap="gray", origin="lower")
            key = parse_key(f.name)
            if seg_rep is not None and key in seg_rep["_keys"]:
                ax.contour(load_slice(seg_rep["_keys"][key]).T, origin="lower", linewidths=0.6, cmap="tab10")
            ax.set_title(f"{name}\n{f.name}", fontsize=6)
            ax.axis("off")
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


# --------------------------------------------------------------------------- #
# Main
# --------------------------------------------------------------------------- #
def print_report(rep: dict):
    print(f"\n[{rep['name']}]  ({'segmentation' if rep['is_seg'] else 'image'})")
    print(f"  files: {rep['n_files']}   patients: {rep['n_cases']}   "
          f"slices/patient: {rep['slices_per_case']}")
    print(f"  shapes: {rep['shape_counts']}")
    print(f"  dtypes: {rep['dtype_counts']}")
    if rep["unparsed_filenames"]:
        print(f"  WARNING: {rep['unparsed_filenames']} filenames without a case ID")
    if "intensity_sampled" in rep:
        print(f"  intensity (sampled): {rep['intensity_sampled']}")
    if "label_pixel_fraction" in rep:
        print(f"  label pixel fraction (sampled): {rep['label_pixel_fraction']}")


def parse_args():
    p = argparse.ArgumentParser(description="Audit the HipMRI 2D slice dataset.")
    p.add_argument("--root", default=DEFAULT_ROOT, help="path to keras_slices_data")
    p.add_argument("--out", default="audit_outputs", help="output directory")
    p.add_argument("--sample", type=int, default=200, help="slices to read per folder for stats")
    p.add_argument("--seed", type=int, default=0)
    return p.parse_args()


def main():
    args = parse_args()
    root = Path(args.root)
    if not root.is_dir():
        raise SystemExit(f"Dataset root not found: {root}")
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(args.seed)

    folders = sorted({p.parent for p in root.rglob("*.nii*")})
    if not folders:
        raise SystemExit(f"No .nii/.nii.gz files found under {root}")

    print(f"Auditing {len(folders)} folders under {root}")
    reports = {}
    for folder in folders:
        rep = audit_folder(folder, root, args.sample, rng)
        reports[rep["name"]] = rep
        print_report(rep)

    image_reports = {n: r for n, r in reports.items() if not r["is_seg"]}
    seg_reports = {n: r for n, r in reports.items() if r["is_seg"]}

    # Image/mask pairing
    pairs, pairing = {}, []
    for name in sorted(image_reports):
        partner = find_seg_partner(name, list(seg_reports))
        if partner:
            pairs[name] = partner
            pairing.append(check_pairing(image_reports[name], seg_reports[partner]))
    print("\n=== Image / mask pairing ===")
    for p in pairing:
        print(f"  {p}")

    # Patient leakage
    leakage = check_leakage(image_reports)
    print("\n=== Patient leakage between image splits ===")
    any_leak = False
    for split_pair, shared in leakage.items():
        status = f"LEAK: {len(shared)} shared patients {shared}" if shared else "ok (no shared patients)"
        any_leak |= bool(shared)
        print(f"  {split_pair}: {status}")
    print("\nVERDICT:", "PATIENT LEAKAGE FOUND -> must re-split by patient"
          if any_leak else "No patient overlap between provided splits")

    save_sample_figure(image_reports, seg_reports, pairs, out / "sample_slices.png", rng)

    summary = {
        "root": str(root),
        "folders": {n: {k: v for k, v in r.items() if not k.startswith("_")} for n, r in reports.items()},
        "pairing": pairing,
        "leakage": leakage,
        "leakage_found": any_leak,
    }
    (out / "audit_summary.json").write_text(json.dumps(summary, indent=2))
    print(f"\nSaved {out / 'audit_summary.json'} and {out / 'sample_slices.png'}")


if __name__ == "__main__":
    main()
