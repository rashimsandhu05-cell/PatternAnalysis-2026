"""
dataset.py
==========
Data loading and pre-processing for the HipMRI 2D prostate MRI slices.

Pipeline for every slice (decided from the data audit, see README Section 3):

1. Load the Nifti file and drop the occasional trailing singleton dimension.
2. Centre-crop (or zero-pad) to 256 x 128. 60 training slices are 256 x 144;
   cropping avoids the interpolation blur that resizing would introduce.
3. Per-slice robust normalisation: clip to the 0.5th-99.5th intensity
   percentiles, then min-max scale to [0, 1]. MR intensities have no absolute
   scale and per-slice maxima vary widely (audit: ~370-1080).
4. The segmentation mask is cropped identically. Masks are NOT used to train
   the generative models; they are used at evaluation time to measure quality
   inside each organ (e.g. the prostate, < 1% of pixels).

Splits: the provided train / validate / test folders are used as-is, because
the audit found no patient (case ID) shared between them (31 / 4 / 3 patients).

Caching: decoding ~12,700 gzipped Nifti files is slow, so the pre-processed
tensors are saved once per split to <cache_dir>/hipmri2d_<split>_<version>.pt
(images as float16, masks as uint8). The cache directory is git-ignored.

Tensor layout returned by the Dataset: image (1, 256, 128) float32 in [0, 1];
mask (256, 128) int64 with labels 0-5. Axis -2 (length 256) is the patient's
left-right axis, so horizontal flips act on dim -2.

Usage (sanity check, on a Rangpur compute node):
    python dataset.py
    python dataset.py --root /path/to/keras_slices_data --cache-dir data/cache
"""

import argparse
import re
from pathlib import Path

import nibabel as nib
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

DEFAULT_ROOT = "/home/groups/comp3710/HipMRI_Study_open/keras_slices_data"
DEFAULT_CACHE = "data/cache"

# split name -> (image folder, segmentation folder)
SPLITS = {
    "train": ("keras_slices_train", "keras_slices_seg_train"),
    "validate": ("keras_slices_validate", "keras_slices_seg_validate"),
    "test": ("keras_slices_test", "keras_slices_seg_test"),
}

IMG_H, IMG_W = 256, 128

# Label meanings per the dataset description; verified visually by
# save_label_check() below before being relied on in evaluation.
LABEL_NAMES = {0: "background", 1: "body", 2: "bone", 3: "bladder", 4: "rectum", 5: "prostate"}

# Bump this whenever pre-processing changes, so stale caches are not reused.
CACHE_VERSION = "v1"

# "case_004_week_0_slice_12.nii.gz" -> (4, 0, 12); used to pair image and mask.
KEY_RE = re.compile(r"(\d+)_week_(\d+).*?slice_(\d+)", re.IGNORECASE)


# --------------------------------------------------------------------------- #
# Low-level helpers
# --------------------------------------------------------------------------- #
def parse_key(name: str):
    """Return (case, week, slice) as ints, or None if the name does not match."""
    m = KEY_RE.search(name)
    return tuple(int(g) for g in m.groups()) if m else None


def list_nifti(folder: Path) -> list:
    """Sorted list of .nii / .nii.gz files in a folder."""
    return sorted(p for p in folder.iterdir() if p.name.endswith((".nii", ".nii.gz")))


def load_nifti_slice(path: Path, dtype) -> torch.Tensor:
    """Load one Nifti slice as a 2D tensor of the given NumPy dtype."""
    arr = np.asarray(nib.load(str(path)).dataobj)
    if arr.ndim == 3:
        arr = arr[..., 0]  # some HipMRI slices carry an extra trailing dimension
    return torch.from_numpy(np.ascontiguousarray(arr.astype(dtype)))


def center_crop_or_pad(t: torch.Tensor, h: int = IMG_H, w: int = IMG_W) -> torch.Tensor:
    """Centre-crop and/or zero-pad a 2D tensor to exactly (h, w)."""
    H, W = t.shape
    if H > h:
        top = (H - h) // 2
        t = t[top:top + h]
    if W > w:
        left = (W - w) // 2
        t = t[:, left:left + w]
    ph, pw = h - t.shape[0], w - t.shape[1]
    if ph > 0 or pw > 0:
        # F.pad pads the last dim first: (left, right, top, bottom)
        t = F.pad(t, (pw // 2, pw - pw // 2, ph // 2, ph - ph // 2))
    return t


def normalise_slice(x: torch.Tensor, low_pct: float = 0.5, high_pct: float = 99.5) -> torch.Tensor:
    """Clip to per-slice percentiles, then scale to [0, 1].

    Percentile clipping stops a few very bright voxels (e.g. fat or flow
    artefacts) from compressing the useful intensity range.
    """
    flat = x.flatten().float()
    lo = torch.quantile(flat, low_pct / 100.0)
    hi = torch.quantile(flat, high_pct / 100.0)
    if hi <= lo:  # constant slice: return zeros rather than divide by ~0
        return torch.zeros_like(x, dtype=torch.float32)
    return ((x.float().clamp(lo, hi) - lo) / (hi - lo)).clamp(0.0, 1.0)


# --------------------------------------------------------------------------- #
# Cache building
# --------------------------------------------------------------------------- #
def build_split_cache(root: Path, split: str, cache_dir: Path) -> dict:
    """Load, pre-process and cache one split. Returns the cached dict."""
    cache_path = cache_dir / f"hipmri2d_{split}_{CACHE_VERSION}.pt"
    if cache_path.exists():
        return torch.load(cache_path, weights_only=True)

    img_dir, seg_dir = root / SPLITS[split][0], root / SPLITS[split][1]
    img_files = list_nifti(img_dir)
    seg_by_key = {parse_key(p.name): p for p in list_nifti(seg_dir)}

    n = len(img_files)
    images = torch.empty((n, 1, IMG_H, IMG_W), dtype=torch.float16)
    masks = torch.empty((n, IMG_H, IMG_W), dtype=torch.uint8)
    keys = []
    for i, f in enumerate(tqdm(img_files, desc=f"Caching {split}")):
        key = parse_key(f.name)
        if key is None or key not in seg_by_key:
            raise ValueError(f"No matching segmentation mask for {f.name}")
        img = center_crop_or_pad(load_nifti_slice(f, np.float32))
        images[i, 0] = normalise_slice(img).half()
        masks[i] = center_crop_or_pad(load_nifti_slice(seg_by_key[key], np.uint8))
        keys.append(key)

    data = {
        "images": images,
        "masks": masks,
        "keys": keys,
        "files": [f.name for f in img_files],
        "version": CACHE_VERSION,
    }
    cache_dir.mkdir(parents=True, exist_ok=True)
    torch.save(data, cache_path)
    return data


# --------------------------------------------------------------------------- #
# Dataset and loaders
# --------------------------------------------------------------------------- #
class HipMRISlices(Dataset):
    """HipMRI 2D slices for one split.

    Args:
        split: "train", "validate" or "test".
        root: path to keras_slices_data.
        cache_dir: where pre-processed tensors are cached.
        return_mask: if True, __getitem__ returns (image, mask); else image only.
        augment: if True, random left-right flips (p = 0.5). Off by default:
            for the memorisation audit we want generated images compared
            against the exact training slices the model saw.
    """

    def __init__(self, split: str, root: str = DEFAULT_ROOT, cache_dir: str = DEFAULT_CACHE,
                 return_mask: bool = False, augment: bool = False):
        if split not in SPLITS:
            raise ValueError(f"split must be one of {list(SPLITS)}, got {split!r}")
        data = build_split_cache(Path(root), split, Path(cache_dir))
        self.split = split
        self.images = data["images"]   # (N, 1, 256, 128) float16
        self.masks = data["masks"]     # (N, 256, 128) uint8
        self.keys = data["keys"]       # list of (case, week, slice)
        self.files = data["files"]
        self.case_ids = torch.tensor([k[0] for k in self.keys])  # patient per slice
        self.return_mask = return_mask
        self.augment = augment

    def __len__(self) -> int:
        return self.images.shape[0]

    def __getitem__(self, idx: int):
        img = self.images[idx].float()
        mask = self.masks[idx]
        if self.augment and torch.rand(()) < 0.5:
            img, mask = img.flip(-2), mask.flip(-2)  # dim -2 is left-right
        if self.return_mask:
            return img, mask.long()
        return img


def get_dataloaders(root: str = DEFAULT_ROOT, cache_dir: str = DEFAULT_CACHE, batch_size: int = 64,
                    num_workers: int = 2, return_mask: bool = False, augment_train: bool = False) -> dict:
    """Build train / validate / test DataLoaders. Only train is shuffled."""
    loaders = {}
    for split in SPLITS:
        ds = HipMRISlices(split, root, cache_dir, return_mask=return_mask,
                          augment=augment_train and split == "train")
        loaders[split] = DataLoader(
            ds,
            batch_size=batch_size,
            shuffle=(split == "train"),
            num_workers=num_workers,
            pin_memory=torch.cuda.is_available(),
            drop_last=(split == "train"),
        )
    return loaders


# --------------------------------------------------------------------------- #
# Sanity check (run this file directly)
# --------------------------------------------------------------------------- #
def save_label_check(ds: HipMRISlices, out_path: Path, n: int = 4):
    """Overlay labels 2-5 in fixed colours on slices that contain a prostate.

    Used once to confirm which label number is which organ.
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.colors import ListedColormap
    from matplotlib.patches import Patch

    colours = {2: "tab:blue", 3: "gold", 4: "magenta", 5: "red"}
    cmap = ListedColormap(["none", "none"] + [colours[k] for k in (2, 3, 4, 5)])

    has_prostate = torch.nonzero((ds.masks == 5).flatten(1).any(1)).flatten()
    picks = has_prostate[torch.linspace(0, len(has_prostate) - 1, n).long()]

    fig, axes = plt.subplots(1, n, figsize=(4 * n, 3))
    for ax, i in zip(axes, picks.tolist()):
        img, mask = ds.images[i, 0].float().numpy(), ds.masks[i].numpy()
        ax.imshow(img.T, cmap="gray", origin="lower")
        ax.imshow(np.ma.masked_where(mask < 2, mask).T, cmap=cmap, vmin=0, vmax=5,
                  alpha=0.45, origin="lower", interpolation="nearest")
        ax.set_title(ds.files[i], fontsize=7)
        ax.axis("off")
    handles = [Patch(color=c, label=f"{k}: {LABEL_NAMES[k]}") for k, c in colours.items()]
    fig.legend(handles=handles, loc="lower center", ncol=4, fontsize=8)
    fig.tight_layout(rect=(0, 0.08, 1, 1))
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def split_stats(ds: HipMRISlices, chunk: int = 512) -> dict:
    """Summary statistics computed in chunks.

    The train split is ~11,500 slices x 32,768 pixels; converting it all to
    float32 at once would need ~1.5 GB extra RAM, so we iterate in chunks.
    """
    lo, hi, total, count = float("inf"), float("-inf"), 0.0, 0
    label_counts = torch.zeros(256, dtype=torch.long)
    prostate_slices = 0
    for start in range(0, len(ds), chunk):
        imgs = ds.images[start:start + chunk].float()
        masks = ds.masks[start:start + chunk]
        lo, hi = min(lo, imgs.min().item()), max(hi, imgs.max().item())
        total += imgs.sum(dtype=torch.float64).item()
        count += imgs.numel()
        label_counts += torch.bincount(masks.flatten().long(), minlength=256)
        prostate_slices += (masks == 5).flatten(1).any(1).sum().item()
    return {
        "min": lo, "max": hi, "mean": total / count,
        "labels": torch.nonzero(label_counts).flatten().tolist(),
        "prostate_frac": prostate_slices / len(ds),
    }


def main():
    p = argparse.ArgumentParser(description="Build caches and sanity-check the HipMRI 2D dataset.")
    p.add_argument("--root", default=DEFAULT_ROOT)
    p.add_argument("--cache-dir", default=DEFAULT_CACHE)
    p.add_argument("--out", default="audit_outputs")
    args = p.parse_args()

    for split in SPLITS:
        ds = HipMRISlices(split, args.root, args.cache_dir, return_mask=True)
        img, mask = ds[0]
        s = split_stats(ds)
        print(f"[{split}] slices={len(ds)} patients={len(set(ds.case_ids.tolist()))} "
              f"image={tuple(img.shape)} range=[{s['min']:.3f}, {s['max']:.3f}] mean={s['mean']:.3f} "
              f"mask_labels={s['labels']} slices_with_prostate={s['prostate_frac']:.1%}")

    # Quick check that the DataLoader path works end to end.
    loaders = get_dataloaders(args.root, args.cache_dir, batch_size=8, num_workers=0)
    batch = next(iter(loaders["train"]))
    print(f"train batch: {tuple(batch.shape)} {batch.dtype}")

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    save_label_check(HipMRISlices("train", args.root, args.cache_dir), out / "label_check.png")
    print(f"Saved {out / 'label_check.png'}")


if __name__ == "__main__":
    main()
