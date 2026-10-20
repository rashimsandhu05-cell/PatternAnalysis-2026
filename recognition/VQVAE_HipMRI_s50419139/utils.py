"""
utils.py
========
Shared helpers for training and evaluation: image-quality metrics (SSIM, PSNR),
seeding, device selection, GPU memory reporting and figure saving.

SSIM follows Wang et al. (2004) with an 11x11 Gaussian window (sigma = 1.5),
K1 = 0.01, K2 = 0.03 and population (not sample) covariance. The mean is taken
over the interior of the SSIM map (a 5-pixel border is excluded), which makes
the result match `skimage.metrics.structural_similarity(..., gaussian_weights=True,
sigma=1.5, use_sample_covariance=False, data_range=1)`. Run `python utils.py`
to check this against scikit-image.

The full-size SSIM map (`ssim_map`) is also exposed so that evaluation can
average SSIM inside an organ mask (e.g. the prostate).
"""

import json
import math
import random
from pathlib import Path

import torch
import torch.nn.functional as F

SSIM_WINDOW = 11
SSIM_SIGMA = 1.5
SSIM_BORDER = SSIM_WINDOW // 2


# --------------------------------------------------------------------------- #
# Metrics
# --------------------------------------------------------------------------- #
def _gaussian_window(device, dtype, size: int = SSIM_WINDOW, sigma: float = SSIM_SIGMA):
    coords = torch.arange(size, device=device, dtype=dtype) - size // 2
    g = torch.exp(-coords.pow(2) / (2 * sigma ** 2))
    g = g / g.sum()
    return (g[:, None] * g[None, :])[None, None]  # (1, 1, size, size)


def ssim_map(x, y, data_range: float = 1.0):
    """Per-pixel SSIM map, same size as the input. x, y: (B, 1, H, W)."""
    x, y = x.float(), y.float()
    w = _gaussian_window(x.device, x.dtype)
    pad = SSIM_BORDER

    def blur(t):
        return F.conv2d(F.pad(t, (pad, pad, pad, pad), mode="reflect"), w)

    c1, c2 = (0.01 * data_range) ** 2, (0.03 * data_range) ** 2
    mu_x, mu_y = blur(x), blur(y)
    var_x = blur(x * x) - mu_x.pow(2)
    var_y = blur(y * y) - mu_y.pow(2)
    cov = blur(x * y) - mu_x * mu_y
    num = (2 * mu_x * mu_y + c1) * (2 * cov + c2)
    den = (mu_x.pow(2) + mu_y.pow(2) + c1) * (var_x + var_y + c2)
    return num / den


def ssim(x, y, data_range: float = 1.0):
    """Mean SSIM per image, shape (B,). Border pixels are excluded (see module docstring)."""
    b = SSIM_BORDER
    return ssim_map(x, y, data_range)[..., b:-b, b:-b].flatten(1).mean(1)


def masked_ssim(x, y, mask, data_range: float = 1.0):
    """Mean SSIM inside a boolean mask (B, H, W), per image. NaN where the mask is empty."""
    m = ssim_map(x, y, data_range)[:, 0]
    mask = mask.bool()
    total = (m * mask).flatten(1).sum(1)
    count = mask.flatten(1).sum(1)
    return torch.where(count > 0, total / count.clamp(min=1), torch.full_like(total, float("nan")))


def psnr(x, y, data_range: float = 1.0):
    """Peak signal-to-noise ratio per image in dB, shape (B,)."""
    mse = (x.float() - y.float()).pow(2).flatten(1).mean(1).clamp(min=1e-10)
    return 10 * torch.log10(data_range ** 2 / mse)


# --------------------------------------------------------------------------- #
# Reproducibility, device, memory
# --------------------------------------------------------------------------- #
def seed_everything(seed: int):
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def get_device():
    if torch.cuda.is_available():
        return torch.device("cuda")
    if getattr(torch.backends, "mps", None) is not None and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def reset_peak_memory(device):
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)


def peak_memory_mb(device):
    """Peak GPU memory allocated by tensors since the last reset (MB); 0 on CPU/MPS."""
    if device.type == "cuda":
        return torch.cuda.max_memory_allocated(device) / 2 ** 20
    return 0.0


def save_json(obj, path):
    Path(path).write_text(json.dumps(obj, indent=2))


# --------------------------------------------------------------------------- #
# Figures
# --------------------------------------------------------------------------- #
def _plt():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    return plt


def _show(ax, img):
    """Show a (1, H, W) or (H, W) slice in anatomical orientation (transposed, origin lower)."""
    img = img.detach().float().cpu().squeeze()
    ax.imshow(img.T.numpy(), cmap="gray", origin="lower", vmin=0, vmax=1)
    ax.axis("off")


def save_image_grid(images, path, nrow: int = 8, title: str = None):
    """Save a grid of slices. images: (N, 1, H, W) in [0, 1]."""
    plt = _plt()
    n = images.shape[0]
    rows = math.ceil(n / nrow)
    fig, axes = plt.subplots(rows, nrow, figsize=(2.0 * nrow, 1.1 * rows), squeeze=False)
    for k, ax in enumerate(axes.flat):
        if k < n:
            _show(ax, images[k])
        else:
            ax.axis("off")
    if title:
        fig.suptitle(title, fontsize=9)
    fig.tight_layout()
    fig.savefig(path, dpi=130)
    plt.close(fig)


def save_comparison(originals, reconstructions, path, title: str = None):
    """Top row: originals; bottom row: reconstructions; with per-image SSIM."""
    plt = _plt()
    n = originals.shape[0]
    scores = ssim(reconstructions.float(), originals.float()).tolist()
    fig, axes = plt.subplots(2, n, figsize=(2.0 * n, 2.6), squeeze=False)
    for k in range(n):
        _show(axes[0, k], originals[k])
        _show(axes[1, k], reconstructions[k])
        axes[1, k].set_title(f"SSIM {scores[k]:.3f}", fontsize=7)
    axes[0, 0].set_title("original", fontsize=7, loc="left")
    if title:
        fig.suptitle(title, fontsize=9)
    fig.tight_layout()
    fig.savefig(path, dpi=130)
    plt.close(fig)


def plot_history(history, keys, path, title: str = None):
    """One subplot per metric key, plotted against epoch."""
    plt = _plt()
    keys = [k for k in keys if any(k in h for h in history)]
    if not keys:
        return
    fig, axes = plt.subplots(1, len(keys), figsize=(3.2 * len(keys), 2.6), squeeze=False)
    epochs = [h["epoch"] for h in history]
    for ax, key in zip(axes[0], keys):
        ax.plot(epochs, [h.get(key, float("nan")) for h in history], marker=".")
        ax.set_title(key, fontsize=9)
        ax.set_xlabel("epoch", fontsize=8)
        ax.grid(alpha=0.3)
    if title:
        fig.suptitle(title, fontsize=10)
    fig.tight_layout()
    fig.savefig(path, dpi=130)
    plt.close(fig)


# --------------------------------------------------------------------------- #
# Self-test: SSIM against scikit-image
# --------------------------------------------------------------------------- #
if __name__ == "__main__":
    from skimage.metrics import structural_similarity

    torch.manual_seed(0)
    x = torch.rand(3, 1, 256, 128)
    y = (x + 0.1 * torch.randn_like(x)).clamp(0, 1)
    ours = ssim(y, x)
    ref = torch.tensor([
        structural_similarity(x[i, 0].double().numpy(), y[i, 0].double().numpy(), data_range=1.0,
                              gaussian_weights=True, sigma=1.5, use_sample_covariance=False)
        for i in range(3)
    ])
    diff = (ours.double() - ref).abs().max().item()
    print(f"ours={[round(v, 5) for v in ours.tolist()]} skimage={[round(v, 5) for v in ref.tolist()]} "
          f"max_diff={diff:.2e}")
    assert diff < 1e-4, "SSIM does not match scikit-image"
    assert torch.allclose(ssim(x, x), torch.ones(3)), "SSIM(x, x) should be 1"
    mask = torch.zeros(3, 256, 128, dtype=torch.bool)
    mask[:, 100:140, 50:80] = True
    print(f"masked SSIM: {[round(v, 4) for v in masked_ssim(y, x, mask).tolist()]}  "
          f"PSNR: {[round(v, 2) for v in psnr(y, x).tolist()]} dB")
    print("utils self-test passed.")
