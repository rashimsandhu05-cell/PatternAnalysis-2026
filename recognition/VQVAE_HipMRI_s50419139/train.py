"""
train.py
========
Training, validation and checkpointing for the three networks in modules.py.

    python train.py --model vqvae                 # main model (stage 1)
    python train.py --model vae                   # baseline
    python train.py --model pixelcnn              # prior over VQ-VAE codes (stage 2;
                                                  # needs outputs/vqvae/best.pt)
    python train.py --model vqvae --smoke         # a few batches on CPU, to test the code path

Each run writes to outputs/<model>/ (git-ignored):
    best.pt, last.pt       checkpoints (best = highest validation SSIM for vqvae/vae,
                           lowest validation loss for pixelcnn)
    history.json           per-epoch losses, validation metrics, time and peak GPU memory
    curves.png             training curves
    recon_latest.png       validation originals vs reconstructions (vqvae/vae)
    samples.png            generated slices (pixelcnn: from sampled code grids)
    codes.pt               (pixelcnn) train/validate code grids from the VQ-VAE

The test split is never touched here; it is reserved for the final evaluation
in predict.py so that model selection cannot leak into the reported results.

Losses
------
VQ-VAE : mean squared error (per pixel) + vq_loss (beta * commitment loss)
         [+ ssim_weight * (1 - SSIM) if --ssim-weight > 0].
         The codebook itself is updated by EMA inside the quantiser.
VAE    : sum-over-pixels squared error per image + kl_weight * KL, with the KL
         weight warmed up linearly over the first epochs to avoid posterior
         collapse. (Squared error summed per image corresponds to a Gaussian
         likelihood; with a per-pixel mean the KL term would swamp it.)
PixelCNN: cross-entropy over the 512 codes at every grid position, reported
         as bits per code.
"""

import argparse
import contextlib
import math
import time
from collections import defaultdict
from pathlib import Path

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset

from dataset import DEFAULT_CACHE, DEFAULT_ROOT, IMG_H, IMG_W, HipMRISlices, get_dataloaders
from modules import VAE, VQVAE, GatedPixelCNN, count_parameters
from utils import (get_device, peak_memory_mb, plot_history, psnr, reset_peak_memory, save_comparison,
                   save_image_grid, save_json, seed_everything, ssim)

DEFAULT_EPOCHS = {"vqvae": 40, "vae": 40, "pixelcnn": 60}
SMOKE_TRAIN_BATCHES, SMOKE_VAL_BATCHES = 5, 2


def parse_args():
    p = argparse.ArgumentParser(description="Train VQ-VAE, VAE baseline or PixelCNN prior on HipMRI 2D.")
    p.add_argument("--model", choices=["vqvae", "vae", "pixelcnn"], required=True)
    p.add_argument("--epochs", type=int, default=None, help="default: 40 (vqvae, vae), 60 (pixelcnn)")
    p.add_argument("--batch-size", type=int, default=64)
    p.add_argument("--lr", type=float, default=3e-4)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--root", default=DEFAULT_ROOT)
    p.add_argument("--cache-dir", default=DEFAULT_CACHE)
    p.add_argument("--out-dir", default=None, help="default: outputs/<model> (outputs/smoke_<model> with --smoke)")
    p.add_argument("--num-workers", type=int, default=4)
    p.add_argument("--no-amp", action="store_true", help="disable bfloat16 autocast on GPU")
    p.add_argument("--smoke", action="store_true", help="tiny run to check the code path")
    # Shared autoencoder architecture
    p.add_argument("--n-down", type=int, default=3,
                   help="stride-2 downsamplings: 3 -> 32x16 latent grid, 2 -> 64x32")
    p.add_argument("--hidden", type=int, default=128, help="encoder/decoder width")
    p.add_argument("--ssim-weight", type=float, default=0.0, help="add ssim_weight * (1 - SSIM) to the loss")
    # VQ-VAE
    p.add_argument("--num-embeddings", type=int, default=512)
    p.add_argument("--embedding-dim", type=int, default=64)
    # VAE
    p.add_argument("--latent-channels", type=int, default=8)
    p.add_argument("--kl-weight", type=float, default=1.0)
    p.add_argument("--kl-warmup-epochs", type=float, default=5.0)
    # PixelCNN
    p.add_argument("--vqvae-ckpt", default=None, help="default: outputs/vqvae/best.pt")
    p.add_argument("--prior-dim", type=int, default=128)
    p.add_argument("--prior-layers", type=int, default=15)
    p.add_argument("--prior-kernel", type=int, default=5, help="kernel size of the type-B gated layers")
    p.add_argument("--prior-dropout", type=float, default=0.1)
    p.add_argument("--n-samples", type=int, default=16, help="slices to generate at the end (pixelcnn)")
    args = p.parse_args()

    if args.epochs is None:
        args.epochs = 2 if args.smoke else DEFAULT_EPOCHS[args.model]
    prefix = "smoke_" if args.smoke else ""
    if args.out_dir is None:
        args.out_dir = f"outputs/{prefix}{args.model}"
    if args.vqvae_ckpt is None:
        args.vqvae_ckpt = f"outputs/{prefix}vqvae/best.pt"
    if args.smoke:
        args.batch_size = min(args.batch_size, 8)
        args.num_workers = 0
        args.n_samples = min(args.n_samples, 2)
    return args


def autocast(device, enabled: bool):
    """bfloat16 autocast on CUDA (no GradScaler needed); no-op elsewhere."""
    if enabled and device.type == "cuda":
        return torch.autocast(device_type="cuda", dtype=torch.bfloat16)
    return contextlib.nullcontext()


def limited(loader, max_batches):
    """Iterate over at most max_batches batches (all of them if None)."""
    for i, batch in enumerate(loader):
        if max_batches is not None and i >= max_batches:
            break
        yield batch


def save_checkpoint(path, model, model_type, model_kwargs, epoch, metric, args, history):
    torch.save({
        "model": model.state_dict(),
        "model_type": model_type,
        "model_kwargs": model_kwargs,
        "epoch": epoch,
        "val_metric": metric,
        "args": vars(args),
        "history": history,
    }, path)


def load_checkpoint(path, device):
    """Rebuild a model from a checkpoint written by save_checkpoint."""
    ckpt = torch.load(path, map_location=device, weights_only=False)
    cls = {"vqvae": VQVAE, "vae": VAE, "pixelcnn": GatedPixelCNN}[ckpt["model_type"]]
    model = cls(**ckpt["model_kwargs"]).to(device)
    model.load_state_dict(ckpt["model"])
    model.eval()
    return model, ckpt


# --------------------------------------------------------------------------- #
# VQ-VAE and VAE
# --------------------------------------------------------------------------- #
@torch.no_grad()
def evaluate_reconstruction(model, loader, device, amp, max_batches=None):
    """Mean SSIM, PSNR and MSE of reconstructions over a loader."""
    model.eval()
    ssims, psnrs, mses = [], [], []
    for x in limited(loader, max_batches):
        x = x.to(device, non_blocking=True)
        with autocast(device, amp):
            recon = model(x)[0]
        recon = recon.float().clamp(0, 1)
        ssims.append(ssim(recon, x).cpu())
        psnrs.append(psnr(recon, x).cpu())
        mses.append((recon - x).pow(2).flatten(1).mean(1).cpu())
    return {
        "val_ssim": torch.cat(ssims).mean().item(),
        "val_psnr": torch.cat(psnrs).mean().item(),
        "val_mse": torch.cat(mses).mean().item(),
    }


def train_autoencoder(args, device, out):
    loaders = get_dataloaders(args.root, args.cache_dir, args.batch_size, args.num_workers)
    arch = {"hidden": args.hidden, "n_down": args.n_down}
    if args.model == "vqvae":
        model_kwargs = {"num_embeddings": args.num_embeddings, "embedding_dim": args.embedding_dim, **arch}
        model = VQVAE(**model_kwargs)
    else:
        model_kwargs = {"latent_channels": args.latent_channels, **arch}
        model = VAE(**model_kwargs)
    model.to(device)
    print(f"{args.model}: {count_parameters(model):,} trainable parameters on {device}")

    max_train = SMOKE_TRAIN_BATCHES if args.smoke else None
    max_val = SMOKE_VAL_BATCHES if args.smoke else None
    steps_per_epoch = min(len(loaders["train"]), max_train or len(loaders["train"]))
    total_steps = args.epochs * steps_per_epoch
    warmup_steps = max(1, int(args.kl_warmup_epochs * steps_per_epoch))

    opt = torch.optim.Adam(model.parameters(), lr=args.lr)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=total_steps)
    amp = not args.no_amp
    fixed_val = next(iter(loaders["validate"]))[:8]  # same slices every epoch, for the figure

    history, best, step = [], -math.inf, 0
    reset_peak_memory(device)
    train_start = time.time()
    for epoch in range(1, args.epochs + 1):
        model.train()
        t0, sums, n = time.time(), defaultdict(float), 0
        for x in limited(loaders["train"], max_train):
            x = x.to(device, non_blocking=True)
            with autocast(device, amp):
                if args.model == "vqvae":
                    recon, vq_loss, _, perplexity = model(x)
                    recon = recon.float()
                    rec = F.mse_loss(recon, x)
                    loss = rec + vq_loss
                    if args.ssim_weight > 0:
                        loss = loss + args.ssim_weight * (1 - ssim(recon, x).mean())
                    extras = {"vq_loss": vq_loss.item(), "perplexity": perplexity.item()}
                else:
                    recon, mu, logvar = model(x)
                    recon = recon.float()
                    rec = F.mse_loss(recon, x, reduction="sum") / x.shape[0]
                    kl = VAE.kl_divergence(mu.float(), logvar.float())
                    kl_w = args.kl_weight * min(1.0, (step + 1) / warmup_steps)
                    loss = rec + kl_w * kl
                    if args.ssim_weight > 0:
                        # scaled by pixel count to match the summed reconstruction term
                        loss = loss + args.ssim_weight * x[0].numel() * (1 - ssim(recon, x).mean())
                    extras = {"kl": kl.item(), "kl_weight": kl_w}
            if not torch.isfinite(loss):
                raise RuntimeError(f"Non-finite loss ({loss.item()}) at epoch {epoch}, step {step}; stopping.")
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
            sched.step()
            step += 1

            sums["train_loss"] += loss.item()
            sums["train_mse"] += F.mse_loss(recon.detach(), x).item()  # per-pixel, comparable across models
            for k, v in extras.items():
                sums[k] += v
            n += 1

        record = {"epoch": epoch, **{k: v / n for k, v in sums.items()}}
        record.update(evaluate_reconstruction(model, loaders["validate"], device, amp, max_val))
        record["lr"] = sched.get_last_lr()[0]
        record["epoch_time_s"] = time.time() - t0
        record["peak_gpu_mem_mb"] = peak_memory_mb(device)
        if args.model == "vqvae":
            record["dead_code_restarts_total"] = int(model.quantizer.restarts)
        history.append(record)

        save_checkpoint(out / "last.pt", model, args.model, model_kwargs, epoch, record["val_ssim"], args, history)
        if record["val_ssim"] > best:
            best = record["val_ssim"]
            save_checkpoint(out / "best.pt", model, args.model, model_kwargs, epoch, best, args, history)

        model.eval()
        with torch.no_grad(), autocast(device, amp):
            recon_fixed = model(fixed_val.to(device))[0].float().clamp(0, 1).cpu()
        save_comparison(fixed_val, recon_fixed, out / "recon_latest.png",
                        title=f"{args.model} epoch {epoch} (validation)")
        save_json(history, out / "history.json")
        plot_history(history, ["train_loss", "train_mse", "val_ssim", "val_psnr", "perplexity", "kl"],
                     out / "curves.png", title=args.model)

        extra = (f"ppl={record['perplexity']:.1f}" if args.model == "vqvae"
                 else f"kl={record['kl']:.1f} (w={record['kl_weight']:.2f})")
        print(f"[{args.model}] epoch {epoch}/{args.epochs} loss={record['train_loss']:.4f} "
              f"train_mse={record['train_mse']:.5f} val_ssim={record['val_ssim']:.4f} "
              f"val_psnr={record['val_psnr']:.2f}dB {extra} time={record['epoch_time_s']:.0f}s "
              f"peak_mem={record['peak_gpu_mem_mb']:.0f}MB", flush=True)

    if args.model == "vae":
        model.eval()
        with torch.no_grad():
            samples = model.sample(args.n_samples, IMG_H, IMG_W, device).float().cpu()
        save_image_grid(samples, out / "samples.png", title="VAE samples, z ~ N(0, I)")

    summary = {"model": args.model, "params": count_parameters(model), "best_val_ssim": best,
               "model_kwargs": model_kwargs, "epochs": args.epochs, "ssim_weight": args.ssim_weight,
               "total_train_time_s": time.time() - train_start,
               "peak_gpu_mem_mb": max(h["peak_gpu_mem_mb"] for h in history)}
    save_json(summary, out / "summary.json")
    print(f"Done. Best validation SSIM {best:.4f}. Outputs in {out}/")


# --------------------------------------------------------------------------- #
# PixelCNN prior
# --------------------------------------------------------------------------- #
@torch.no_grad()
def encode_split(vqvae, split, args, device, max_images=None):
    """Encode every slice of a split into a (N, h, w) grid of VQ-VAE code indices."""
    ds = HipMRISlices(split, args.root, args.cache_dir)
    loader = DataLoader(ds, batch_size=128, shuffle=False, num_workers=args.num_workers)
    codes, seen = [], 0
    for x in loader:
        if max_images is not None and seen >= max_images:
            break
        codes.append(vqvae.encode(x.to(device)).cpu())
        seen += x.shape[0]
    codes = torch.cat(codes)
    return codes[:max_images] if max_images is not None else codes


@torch.no_grad()
def pixelcnn_loss(prior, loader, device, amp, max_batches=None):
    prior.eval()
    total, n = 0.0, 0
    for (codes,) in limited(loader, max_batches):
        codes = codes.to(device)
        with autocast(device, amp):
            logits = prior(codes)
        total += F.cross_entropy(logits.float(), codes, reduction="sum").item()
        n += codes.numel()
    return total / n


def train_pixelcnn(args, device, out):
    vqvae, vq_ckpt = load_checkpoint(args.vqvae_ckpt, device)
    K = vq_ckpt["model_kwargs"]["num_embeddings"]
    print(f"Loaded VQ-VAE from {args.vqvae_ckpt} (epoch {vq_ckpt['epoch']}, val SSIM {vq_ckpt['val_metric']:.4f})")

    max_images = 64 if args.smoke else None
    t0 = time.time()
    train_codes = encode_split(vqvae, "train", args, device, max_images)
    val_codes = encode_split(vqvae, "validate", args, device, max_images)
    torch.save({"train": train_codes, "validate": val_codes}, out / "codes.pt")
    used = torch.unique(train_codes).numel()
    print(f"Encoded {len(train_codes)} train / {len(val_codes)} val slices to {tuple(train_codes.shape[1:])} "
          f"code grids in {time.time() - t0:.0f}s; {used}/{K} codes used in train")

    train_loader = DataLoader(TensorDataset(train_codes), batch_size=args.batch_size, shuffle=True, drop_last=True)
    val_loader = DataLoader(TensorDataset(val_codes), batch_size=args.batch_size)

    model_kwargs = {"num_embeddings": K, "dim": args.prior_dim, "n_layers": args.prior_layers,
                    "kernel": args.prior_kernel, "dropout": args.prior_dropout}
    prior = GatedPixelCNN(**model_kwargs).to(device)
    print(f"pixelcnn: {count_parameters(prior):,} trainable parameters on {device}; "
          f"receptive field {prior.receptive_rows()} rows of {train_codes.shape[1]}")

    max_train = SMOKE_TRAIN_BATCHES if args.smoke else None
    max_val = SMOKE_VAL_BATCHES if args.smoke else None
    steps_per_epoch = min(len(train_loader), max_train or len(train_loader))
    opt = torch.optim.Adam(prior.parameters(), lr=args.lr)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=args.epochs * steps_per_epoch)
    amp = not args.no_amp

    history, best = [], math.inf
    reset_peak_memory(device)
    train_start = time.time()
    for epoch in range(1, args.epochs + 1):
        prior.train()
        t0, total, n = time.time(), 0.0, 0
        for (codes,) in limited(train_loader, max_train):
            codes = codes.to(device)
            with autocast(device, amp):
                logits = prior(codes)
            loss = F.cross_entropy(logits.float(), codes)
            if not torch.isfinite(loss):
                raise RuntimeError(f"Non-finite loss ({loss.item()}) at epoch {epoch}; stopping.")
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
            sched.step()
            total += loss.item()
            n += 1

        val_nll = pixelcnn_loss(prior, val_loader, device, amp, max_val)
        record = {
            "epoch": epoch,
            "train_nll": total / n,
            "val_nll": val_nll,
            "val_bits_per_code": val_nll / math.log(2),
            "lr": sched.get_last_lr()[0],
            "epoch_time_s": time.time() - t0,
            "peak_gpu_mem_mb": peak_memory_mb(device),
        }
        history.append(record)
        save_checkpoint(out / "last.pt", prior, "pixelcnn", model_kwargs, epoch, val_nll, args, history)
        if val_nll < best:
            best = val_nll
            save_checkpoint(out / "best.pt", prior, "pixelcnn", model_kwargs, epoch, best, args, history)
        save_json(history, out / "history.json")
        plot_history(history, ["train_nll", "val_nll", "val_bits_per_code"], out / "curves.png", title="pixelcnn")
        print(f"[pixelcnn] epoch {epoch}/{args.epochs} train_nll={record['train_nll']:.4f} "
              f"val_nll={val_nll:.4f} ({record['val_bits_per_code']:.3f} bits/code) "
              f"time={record['epoch_time_s']:.0f}s peak_mem={record['peak_gpu_mem_mb']:.0f}MB", flush=True)

    # Generate new slices with the best prior: sample code grids, then decode.
    prior, _ = load_checkpoint(out / "best.pt", device)
    h, w = train_codes.shape[1:]
    t0 = time.time()
    codes = prior.sample(args.n_samples, h, w, device)
    with torch.no_grad():
        samples = vqvae.decode_indices(codes).float().clamp(0, 1).cpu()
    gen_time = (time.time() - t0) / args.n_samples
    save_image_grid(samples, out / "samples.png", title="VQ-VAE + PixelCNN samples")
    print(f"Generated {args.n_samples} samples ({gen_time:.2f}s per slice) -> {out / 'samples.png'}")

    summary = {"model": "pixelcnn", "params": count_parameters(prior), "best_val_nll": best,
               "best_val_bits_per_code": best / math.log(2), "codes_used_in_train": used,
               "receptive_rows": prior.receptive_rows(),
               "sampling_time_per_slice_s": gen_time, "total_train_time_s": time.time() - train_start,
               "peak_gpu_mem_mb": max(r["peak_gpu_mem_mb"] for r in history)}
    save_json(summary, out / "summary.json")
    print(f"Done. Best validation NLL {best:.4f}. Outputs in {out}/")


def main():
    args = parse_args()
    seed_everything(args.seed)
    device = get_device()
    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    save_json(vars(args), out / "args.json")
    if args.model == "pixelcnn":
        train_pixelcnn(args, device, out)
    else:
        train_autoencoder(args, device, out)


if __name__ == "__main__":
    main()
