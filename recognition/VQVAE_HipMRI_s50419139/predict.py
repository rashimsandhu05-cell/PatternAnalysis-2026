"""
predict.py
==========
Loads the trained models and runs the final evaluation and investigation on
the held-out TEST split (3 patients, never used for training or model selection).

    python predict.py                       # all tasks, default checkpoint paths
    python predict.py --tasks evaluate,failures
    python predict.py --n-samples 256 --temperature 1.0

Tasks (all write to --out, default results/):

  evaluate      Test-set reconstruction quality of VQ-VAE and VAE: SSIM, PSNR,
                MSE per slice, SSIM inside the bladder / rectum / prostate masks,
                and per-patient means. -> test_metrics.csv, test_summary.json,
                test_examples.png
  failures      Worst VQ-VAE reconstructions (whole-slice and prostate) with error
                maps and organ contours, and SSIM against slice position.
                -> failures_whole.png, failures_prostate.png, ssim_vs_slice.png
  generate      New slices from VQ-VAE + PixelCNN and from the VAE.
                -> samples_vqvae.png, samples_vae.png (+ .pt tensors)
  memorisation  Nearest-neighbour audit: how close each generated slice is to its
                nearest TRAINING slice, compared with how close real unseen
                test-patient slices are. Also code-grid copying for the VQ-VAE.
                -> memorisation_hist.png, memorisation_pairs.png, memorisation.json
  interpolate   Latent interpolation between slices of two different test patients.
                -> interpolation.png
  profile       Parameters, inference latency/throughput, generation time and peak
                GPU memory, plus training cost from outputs/*/summary.json.
                -> profile.json

A Markdown summary of all tables is written to results/report.md.

Memorisation criterion
----------------------
For every real test slice we find the RMSE to its nearest training slice. Because
test patients are unseen, this distribution says how close a *genuinely new*
patient's anatomy typically is to the training set. We take its 5th percentile
as a threshold tau. A model that generates novel anatomy should place about 5% of
its samples below tau; a model that copies training patients places many more.
"""

import argparse
import csv
import json
import math
import time
from collections import defaultdict
from pathlib import Path

import torch

from dataset import DEFAULT_CACHE, DEFAULT_ROOT, IMG_H, IMG_W, LABEL_NAMES, HipMRISlices
from modules import count_parameters
from train import load_checkpoint
from utils import (_plt, _show, get_device, masked_ssim, peak_memory_mb, psnr, reset_peak_memory, save_image_grid,
                   save_json, seed_everything, ssim)

ALL_TASKS = ["evaluate", "failures", "generate", "memorisation", "interpolate", "profile"]
ORGANS = {3: "bladder", 4: "rectum", 5: "prostate"}
ORGAN_COLOURS = {3: "gold", 4: "magenta", 5: "red"}
MIN_ORGAN_PIXELS = 50  # ignore slices where an organ is only a few pixels (unstable SSIM)


def parse_args():
    p = argparse.ArgumentParser(description="Evaluate trained models on the HipMRI 2D test split.")
    p.add_argument("--vqvae", default="outputs/vqvae/best.pt")
    p.add_argument("--vae", default="outputs/vae/best.pt")
    p.add_argument("--prior", default="outputs/pixelcnn/best.pt")
    p.add_argument("--root", default=DEFAULT_ROOT)
    p.add_argument("--cache-dir", default=DEFAULT_CACHE)
    p.add_argument("--out", default="results")
    p.add_argument("--tasks", default=",".join(ALL_TASKS), help=f"comma-separated subset of {ALL_TASKS}")
    p.add_argument("--n-samples", type=int, default=256, help="generated slices per model")
    p.add_argument("--temperature", type=float, default=1.0, help="PixelCNN sampling temperature")
    p.add_argument("--n-failures", type=int, default=5)
    p.add_argument("--seed", type=int, default=0)
    args = p.parse_args()
    args.tasks = [t.strip() for t in args.tasks.split(",") if t.strip()]
    unknown = set(args.tasks) - set(ALL_TASKS)
    if unknown:
        p.error(f"unknown tasks {sorted(unknown)}")
    return args


# --------------------------------------------------------------------------- #
# Helpers
# --------------------------------------------------------------------------- #
def sync(device):
    if device.type == "cuda":
        torch.cuda.synchronize(device)


@torch.no_grad()
def batched(fn, x, device, bs=64):
    """Apply fn to x in batches on device; return the concatenated CPU result."""
    return torch.cat([fn(x[i:i + bs].to(device).float()).cpu() for i in range(0, len(x), bs)])


@torch.no_grad()
def reconstruct(model, images, device):
    return batched(lambda b: model(b)[0].float().clamp(0, 1), images, device)


def mean_std(values):
    v = torch.tensor([x for x in values if not math.isnan(x)], dtype=torch.float64)
    if len(v) == 0:
        return float("nan"), float("nan"), 0
    return v.mean().item(), (v.std().item() if len(v) > 1 else 0.0), len(v)


def fmt(m, s, digits=3):
    return "n/a" if math.isnan(m) else f"{m:.{digits}f} ± {s:.{digits}f}"


def write_csv(rows, path):
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


class Report:
    """Collects Markdown sections and writes results/report.md."""

    def __init__(self, path):
        self.path, self.sections = Path(path), []

    def add(self, title, body):
        self.sections.append(f"## {title}\n\n{body.strip()}\n")
        self.path.write_text("# Evaluation report (generated by predict.py)\n\n" + "\n".join(self.sections))


# --------------------------------------------------------------------------- #
# Task: evaluate
# --------------------------------------------------------------------------- #
@torch.no_grad()
def task_evaluate(models, test, device, out, report):
    x, masks = test.images.float(), test.masks
    recons, rows_by_model, summary = {}, {}, {}
    for name, model in models.items():
        recon = reconstruct(model, x, device)
        recons[name] = recon
        s_all, p_all = ssim(recon, x), psnr(recon, x)
        mse = (recon - x).pow(2).flatten(1).mean(1)
        organ_ssim = {}
        for label, organ in ORGANS.items():
            m = masks == label
            vals = masked_ssim(recon, x, m)
            vals[m.flatten(1).sum(1) < MIN_ORGAN_PIXELS] = float("nan")
            organ_ssim[organ] = vals

        rows = []
        for i, (case, week, sl) in enumerate(test.keys):
            row = {"file": test.files[i], "case": case, "week": week, "slice": sl,
                   "ssim": s_all[i].item(), "psnr": p_all[i].item(), "mse": mse[i].item()}
            for organ, vals in organ_ssim.items():
                row[f"ssim_{organ}"] = vals[i].item()
            row["prostate_pixels"] = int((masks[i] == 5).sum())
            rows.append(row)
        rows_by_model[name] = rows

        per_patient = defaultdict(list)
        for r in rows:
            per_patient[r["case"]].append(r["ssim"])
        summary[name] = {
            "ssim": mean_std(s_all.tolist()), "psnr": mean_std(p_all.tolist()), "mse": mean_std(mse.tolist()),
            **{f"ssim_{o}": mean_std(v.tolist()) for o, v in organ_ssim.items()},
            "ssim_per_patient": {int(c): sum(v) / len(v) for c, v in sorted(per_patient.items())},
            "frac_slices_ssim_above_0.6": (s_all > 0.6).float().mean().item(),
        }

    # One CSV with both models side by side
    merged = []
    for i in range(len(x)):
        row = {k: rows_by_model[next(iter(models))][i][k] for k in ("file", "case", "week", "slice", "prostate_pixels")}
        for name in models:
            for k, v in rows_by_model[name][i].items():
                if k not in row:
                    row[f"{name}_{k}"] = v
        merged.append(row)
    write_csv(merged, out / "test_metrics.csv")
    save_json(summary, out / "test_summary.json")

    # Example figure: 6 test slices spread over the split
    picks = torch.linspace(0, len(x) - 1, 6).long()
    plt = _plt()
    fig, axes = plt.subplots(1 + len(models), 6, figsize=(12, 1.4 * (1 + len(models))), squeeze=False)
    for c, i in enumerate(picks.tolist()):
        _show(axes[0, c], x[i])
        axes[0, c].set_title(test.files[i].replace(".nii.gz", ""), fontsize=6)
        for r, name in enumerate(models, start=1):
            _show(axes[r, c], recons[name][i])
            axes[r, c].set_title(f"{name} SSIM {ssim(recons[name][i:i+1], x[i:i+1]).item():.3f}", fontsize=6)
    fig.suptitle("Test-set reconstructions (top: original)", fontsize=9)
    fig.tight_layout()
    fig.savefig(out / "test_examples.png", dpi=150)
    plt.close(fig)

    header = "| Metric (test, mean ± std) | " + " | ".join(models) + " |\n|---|" + "---|" * len(models) + "\n"
    lines = []
    for key, label in [("ssim", "SSIM (whole slice)"), ("psnr", "PSNR (dB)"), ("mse", "MSE"),
                       ("ssim_bladder", "SSIM in bladder"), ("ssim_rectum", "SSIM in rectum"),
                       ("ssim_prostate", "SSIM in prostate")]:
        cells = []
        for name in models:
            m, s, n = summary[name][key]
            cells.append(f"{fmt(m, s, 2 if key == 'psnr' else 4 if key == 'mse' else 3)} (n={n})"
                         if key.startswith("ssim_") else fmt(m, s, 2 if key == "psnr" else 4 if key == "mse" else 3))
        lines.append(f"| {label} | " + " | ".join(cells) + " |")
    lines.append("| Slices with SSIM > 0.6 | " + " | ".join(
        f"{summary[n]['frac_slices_ssim_above_0.6']:.1%}" for n in models) + " |")
    for case in summary[next(iter(models))]["ssim_per_patient"]:
        lines.append(f"| SSIM, patient {case} | " + " | ".join(
            f"{summary[n]['ssim_per_patient'][case]:.3f}" for n in models) + " |")
    report.add("Test-set reconstruction", header + "\n".join(lines) +
               f"\n\nOrgan SSIM is averaged only over slices where the organ covers at least "
               f"{MIN_ORGAN_PIXELS} pixels (n = number of such slices).")
    print(header + "\n".join(lines))
    return recons, rows_by_model


# --------------------------------------------------------------------------- #
# Task: failures
# --------------------------------------------------------------------------- #
def failure_figure(indices, test, recons, rows, path, title):
    plt = _plt()
    x, masks = test.images.float(), test.masks
    names = list(recons)
    cols = 2 + len(names)  # original, recon per model, error map of the first model
    fig, axes = plt.subplots(len(indices), cols, figsize=(2.6 * cols, 1.6 * len(indices)), squeeze=False)
    for r, i in enumerate(indices):
        ax = axes[r, 0]
        _show(ax, x[i])
        for label, colour in ORGAN_COLOURS.items():
            m = (masks[i] == label).float()
            if m.sum() > 0:
                ax.contour(m.T.numpy(), levels=[0.5], colors=[colour], linewidths=0.6, origin="lower")
        row = rows[names[0]][i]
        ax.set_title(f"{test.files[i].replace('.nii.gz', '')}\nprostate px={row['prostate_pixels']}", fontsize=6)
        for c, name in enumerate(names, start=1):
            _show(axes[r, c], recons[name][i])
            rr = rows[name][i]
            ps = rr["ssim_prostate"]
            axes[r, c].set_title(f"{name}: SSIM {rr['ssim']:.3f}" + ("" if math.isnan(ps) else f", prostate {ps:.3f}"),
                                 fontsize=6)
        err = (recons[names[0]][i] - x[i]).abs().squeeze()
        axes[r, -1].imshow(err.T.numpy(), cmap="inferno", origin="lower", vmin=0, vmax=0.5)
        axes[r, -1].set_title(f"|error| {names[0]}", fontsize=6)
        axes[r, -1].axis("off")
    fig.suptitle(title, fontsize=9)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def task_failures(test, recons, rows, n, out, report):
    main = "vqvae"
    s = torch.tensor([r["ssim"] for r in rows[main]])
    worst = torch.argsort(s)[:n].tolist()
    failure_figure(worst, test, recons, rows, out / "failures_whole.png",
                   f"{n} worst {main} test reconstructions (whole-slice SSIM)")

    ps = torch.tensor([r["ssim_prostate"] for r in rows[main]])
    valid = torch.nonzero(~torch.isnan(ps)).flatten()
    worst_p = valid[torch.argsort(ps[valid])[:n]].tolist()
    failure_figure(worst_p, test, recons, rows, out / "failures_prostate.png",
                   f"{n} worst {main} test reconstructions inside the prostate")

    # SSIM against slice position, one colour per patient: reveals whether
    # failures cluster at the ends of the scanned volume.
    plt = _plt()
    fig, axes = plt.subplots(1, 2, figsize=(10, 3))
    for case in sorted({r["case"] for r in rows[main]}):
        sel = [r for r in rows[main] if r["case"] == case]
        axes[0].scatter([r["slice"] for r in sel], [r["ssim"] for r in sel], s=6, label=f"patient {case}")
        sel_p = [r for r in sel if not math.isnan(r["ssim_prostate"])]
        axes[1].scatter([r["prostate_pixels"] for r in sel_p], [r["ssim_prostate"] for r in sel_p], s=6)
    axes[0].set_xlabel("slice index")
    axes[0].set_ylabel("whole-slice SSIM")
    axes[0].legend(fontsize=7)
    axes[1].set_xlabel("prostate area (pixels)")
    axes[1].set_ylabel("SSIM inside prostate")
    for ax in axes:
        ax.grid(alpha=0.3)
    fig.suptitle(f"{main}: where do reconstructions fail?", fontsize=9)
    fig.tight_layout()
    fig.savefig(out / "ssim_vs_slice.png", dpi=150)
    plt.close(fig)

    lines = ["| Rank | Slice | Whole SSIM | Prostate SSIM | Prostate px |", "|---|---|---|---|---|"]
    for k, i in enumerate(worst, 1):
        r = rows[main][i]
        ps_txt = "n/a" if math.isnan(r["ssim_prostate"]) else f"{r['ssim_prostate']:.3f}"
        lines.append(f"| {k} | {r['file']} | {r['ssim']:.3f} | {ps_txt} | {r['prostate_pixels']} |")
    report.add("Failure cases (worst whole-slice SSIM, VQ-VAE)",
               "\n".join(lines) + "\n\nFigures: failures_whole.png, failures_prostate.png, ssim_vs_slice.png")


# --------------------------------------------------------------------------- #
# Task: generate
# --------------------------------------------------------------------------- #
@torch.no_grad()
def generate_vqvae(vqvae, prior, n, temperature, device, bs=64):
    h, w = vqvae.latent_shape(IMG_H, IMG_W)
    imgs, codes = [], []
    for start in range(0, n, bs):
        c = prior.sample(min(bs, n - start), h, w, device, temperature)
        imgs.append(vqvae.decode_indices(c).float().clamp(0, 1).cpu())
        codes.append(c.cpu())
    return torch.cat(imgs), torch.cat(codes)


@torch.no_grad()
def generate_vae(vae, n, device, bs=64):
    return torch.cat([vae.sample(min(bs, n - s), IMG_H, IMG_W, device).float().clamp(0, 1).cpu()
                      for s in range(0, n, bs)])


def task_generate(models, prior, args, device, out):
    samples = {}
    if prior is not None:
        imgs, codes = generate_vqvae(models["vqvae"], prior, args.n_samples, args.temperature, device)
        samples["vqvae"] = imgs
        torch.save({"images": imgs.half(), "codes": codes, "temperature": args.temperature}, out / "samples_vqvae.pt")
        save_image_grid(imgs[:32], out / "samples_vqvae.png",
                        title=f"VQ-VAE + PixelCNN samples (T={args.temperature}), uncurated")
    if "vae" in models:
        imgs = generate_vae(models["vae"], args.n_samples, device)
        samples["vae"] = imgs
        torch.save({"images": imgs.half()}, out / "samples_vae.pt")
        save_image_grid(imgs[:32], out / "samples_vae.png", title="VAE samples, z ~ N(0, I), uncurated")
    print(f"Generated {args.n_samples} slices for {list(samples)}")
    return samples


def load_samples(out):
    samples = {}
    for name in ("vqvae", "vae"):
        f = out / f"samples_{name}.pt"
        if f.exists():
            samples[name] = torch.load(f, weights_only=False)["images"].float()
    return samples


# --------------------------------------------------------------------------- #
# Task: memorisation
# --------------------------------------------------------------------------- #
@torch.no_grad()
def nearest_neighbours(queries, ref, device, bs=128):
    """RMSE from each query image to its nearest reference image, and that image's index."""
    R = ref.flatten(1).to(device).float()
    r2 = R.pow(2).sum(1)
    dists, idxs = [], []
    for i in range(0, len(queries), bs):
        Q = queries[i:i + bs].flatten(1).to(device).float()
        d2 = Q.pow(2).sum(1, keepdim=True) + r2 - 2 * Q @ R.t()
        d, j = d2.clamp(min=0).min(dim=1)
        dists.append((d / Q.shape[1]).sqrt().cpu())
        idxs.append(j.cpu())
    return torch.cat(dists), torch.cat(idxs)


@torch.no_grad()
def max_code_overlap(queries, ref, device, bs=16):
    """Largest fraction of identical code positions between each query grid and any reference grid."""
    R = ref.flatten(1).to(device)
    best = []
    for i in range(0, len(queries), bs):
        Q = queries[i:i + bs].flatten(1).to(device)
        best.append((Q[:, None, :] == R[None]).float().mean(-1).max(dim=1).values.cpu())
    return torch.cat(best)


def task_memorisation(models, samples, test, train, device, out, report):
    # Training images stay float16 on the CPU (a float32 copy would need ~1.5 GB
    # extra RAM); nearest_neighbours converts each block to float32 on the device.
    test_x, train_x = test.images.float(), train.images
    d_test, _ = nearest_neighbours(test_x, train_x, device)
    tau = torch.quantile(d_test, 0.05).item()
    result = {"tau_rmse_5th_pct_test_to_train": tau,
              "test_to_train_rmse": {"median": d_test.median().item(), "min": d_test.min().item()}}
    nn_by_model = {}
    for name, imgs in samples.items():
        d_tr, j_tr = nearest_neighbours(imgs, train_x, device)
        d_te, _ = nearest_neighbours(imgs, test_x, device)
        nn_by_model[name] = (d_tr, j_tr)
        result[name] = {
            "gen_to_train_rmse_median": d_tr.median().item(),
            "gen_to_train_rmse_min": d_tr.min().item(),
            "gen_to_test_rmse_median": d_te.median().item(),
            "frac_below_tau": (d_tr < tau).float().mean().item(),
        }

    # Code-level copying (VQ-VAE only): do sampled code grids reuse whole training grids?
    if "vqvae" in samples and (out / "samples_vqvae.pt").exists():
        vqvae = models["vqvae"]
        gen_codes = torch.load(out / "samples_vqvae.pt", weights_only=False)["codes"]
        train_codes = batched(lambda b: vqvae.encode(b), train.images, device)
        test_codes = batched(lambda b: vqvae.encode(b), test.images, device)
        ov_gen = max_code_overlap(gen_codes, train_codes, device)
        ov_test = max_code_overlap(test_codes, train_codes, device)
        result["vqvae"]["code_overlap_gen_median"] = ov_gen.median().item()
        result["vqvae"]["code_overlap_gen_max"] = ov_gen.max().item()
        result["vqvae"]["code_overlap_test_median"] = ov_test.median().item()
        result["vqvae"]["code_overlap_test_max"] = ov_test.max().item()
    save_json(result, out / "memorisation.json")

    plt = _plt()
    fig, ax = plt.subplots(figsize=(6, 3.2))
    bins = torch.linspace(0, max(d_test.max().item(), *(v[0].max().item() for v in nn_by_model.values())), 50)
    ax.hist(d_test.numpy(), bins=bins.numpy(), alpha=0.5, density=True, label="real test slice → train")
    for name, (d, _) in nn_by_model.items():
        ax.hist(d.numpy(), bins=bins.numpy(), alpha=0.5, density=True, label=f"{name} sample → train")
    ax.axvline(tau, color="k", ls="--", lw=1, label=f"τ (5th pct of test) = {tau:.3f}")
    ax.set_xlabel("RMSE to nearest training slice")
    ax.set_ylabel("density")
    ax.legend(fontsize=7)
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(out / "memorisation_hist.png", dpi=150)
    plt.close(fig)

    if "vqvae" in nn_by_model:
        d, j = nn_by_model["vqvae"]
        order = torch.argsort(d)[:8].tolist()
        gen = samples["vqvae"]
        fig, axes = plt.subplots(2, 8, figsize=(16, 2.6), squeeze=False)
        for c, i in enumerate(order):
            _show(axes[0, c], gen[i])
            _show(axes[1, c], train_x[j[i]])
            s = ssim(gen[i:i + 1], train_x[j[i]:j[i] + 1].float()).item()
            axes[0, c].set_title(f"sample {i}", fontsize=6)
            axes[1, c].set_title(f"{train.files[j[i]].replace('.nii.gz', '')}\nRMSE {d[i]:.3f} SSIM {s:.3f}",
                                 fontsize=6)
        fig.suptitle("Generated slices closest to the training set (top) and their nearest training slice (bottom)",
                     fontsize=9)
        fig.tight_layout()
        fig.savefig(out / "memorisation_pairs.png", dpi=150)
        plt.close(fig)

    lines = ["| | Real test → train | " + " | ".join(f"{n} samples → train" for n in nn_by_model) + " |",
             "|---|---|" + "---|" * len(nn_by_model),
             f"| Median NN RMSE | {d_test.median():.4f} | " +
             " | ".join(f"{result[n]['gen_to_train_rmse_median']:.4f}" for n in nn_by_model) + " |",
             f"| Min NN RMSE | {d_test.min():.4f} | " +
             " | ".join(f"{result[n]['gen_to_train_rmse_min']:.4f}" for n in nn_by_model) + " |",
             f"| Fraction below τ = {tau:.4f} | 5% (by definition) | " +
             " | ".join(f"{result[n]['frac_below_tau']:.1%}" for n in nn_by_model) + " |"]
    if "code_overlap_gen_median" in result.get("vqvae", {}):
        v = result["vqvae"]
        lines.append(f"\nVQ-VAE code-grid copying: the most similar training grid shares a median of "
                     f"{v['code_overlap_gen_median']:.1%} (max {v['code_overlap_gen_max']:.1%}) of code positions "
                     f"with a generated grid, versus {v['code_overlap_test_median']:.1%} "
                     f"(max {v['code_overlap_test_max']:.1%}) for real unseen test slices.")
    report.add("Memorisation audit", "\n".join(lines) +
               "\n\nFigures: memorisation_hist.png, memorisation_pairs.png")
    print("\n".join(lines))


# --------------------------------------------------------------------------- #
# Task: interpolate
# --------------------------------------------------------------------------- #
@torch.no_grad()
def task_interpolate(models, test, device, out, steps=8):
    cases = sorted({k[0] for k in test.keys})
    # Middle slice of the first two test patients (week 0 where available)
    picks = []
    for case in cases[:2]:
        idx = [i for i, k in enumerate(test.keys) if k[0] == case]
        picks.append(idx[len(idx) // 2])
    a, b = (test.images[i:i + 1].float().to(device) for i in picks)
    alphas = torch.linspace(0, 1, steps)
    rows = {}
    if "vqvae" in models:
        m = models["vqvae"]
        za, zb = m.encoder(a), m.encoder(b)
        rows["vqvae"] = [m.decoder(m.quantizer(za * (1 - t) + zb * t)[0]).cpu() for t in alphas]
    if "vae" in models:
        m = models["vae"]
        ma, mb = m.encoder(a).chunk(2, 1)[0], m.encoder(b).chunk(2, 1)[0]
        rows["vae"] = [m.decoder(ma * (1 - t) + mb * t).cpu() for t in alphas]
    plt = _plt()
    fig, axes = plt.subplots(len(rows), steps, figsize=(2 * steps, 1.3 * len(rows)), squeeze=False)
    for r, (name, imgs) in enumerate(rows.items()):
        for c, img in enumerate(imgs):
            _show(axes[r, c], img[0])
            axes[r, c].set_title(f"{name} α={alphas[c]:.2f}", fontsize=6)
    fig.suptitle(f"Latent interpolation: {test.files[picks[0]]} → {test.files[picks[1]]}", fontsize=9)
    fig.tight_layout()
    fig.savefig(out / "interpolation.png", dpi=150)
    plt.close(fig)


# --------------------------------------------------------------------------- #
# Task: profile
# --------------------------------------------------------------------------- #
@torch.no_grad()
def time_per_call(fn, device, repeats=20):
    fn()  # warm-up
    sync(device)
    t0 = time.perf_counter()
    for _ in range(repeats):
        fn()
    sync(device)
    return (time.perf_counter() - t0) / repeats


def task_profile(models, prior, args, test, device, out, report):
    x1 = test.images[:1].float().to(device)
    x64 = test.images[:64].float().to(device)
    prof = {}
    for name, model in models.items():
        reset_peak_memory(device)
        lat1 = time_per_call(lambda: model(x1), device)
        lat64 = time_per_call(lambda: model(x64), device)
        mem = peak_memory_mb(device)
        if name == "vqvae" and prior is not None:
            h, w = model.latent_shape(IMG_H, IMG_W)
            reset_peak_memory(device)
            gen = time_per_call(lambda: model.decode_indices(prior.sample(64, h, w, device)), device, repeats=2) / 64
            gen_mem = peak_memory_mb(device)
        elif name == "vae":
            reset_peak_memory(device)
            gen = time_per_call(lambda: model.sample(64, IMG_H, IMG_W, device), device) / 64
            gen_mem = peak_memory_mb(device)
        else:
            gen, gen_mem = float("nan"), float("nan")
        train_summary = Path(getattr(args, name)).parent / "summary.json"
        ts = json.loads(train_summary.read_text()) if train_summary.exists() else {}
        prof[name] = {
            "params": count_parameters(model) + (count_parameters(prior) if name == "vqvae" and prior else 0),
            "params_autoencoder": count_parameters(model),
            "recon_latency_ms_batch1": lat1 * 1e3,
            "recon_throughput_slices_per_s_batch64": 64 / lat64,
            "recon_peak_mem_mb_batch64": mem,
            "generation_ms_per_slice_batch64": gen * 1e3,
            "generation_peak_mem_mb_batch64": gen_mem,
            "train_time_s": ts.get("total_train_time_s"),
            "train_peak_mem_mb": ts.get("peak_gpu_mem_mb"),
        }
        if name == "vqvae":
            ps = Path(args.prior).parent / "summary.json"
            if ps.exists():
                pj = json.loads(ps.read_text())
                prof[name]["prior_train_time_s"] = pj.get("total_train_time_s")
                prof[name]["prior_train_peak_mem_mb"] = pj.get("peak_gpu_mem_mb")
    prof["device"] = torch.cuda.get_device_name(device) if device.type == "cuda" else str(device)
    save_json(prof, out / "profile.json")

    names = [n for n in models]

    def cell(n, key, f):
        v = prof[n].get(key)
        return "n/a" if v is None or (isinstance(v, float) and math.isnan(v)) else f.format(v)

    lines = [f"| Cost ({prof['device']}) | " + " | ".join(names) + " |", "|---|" + "---|" * len(names),
             "| Parameters (autoencoder) | " + " | ".join(cell(n, "params_autoencoder", "{:,}") for n in names) + " |",
             "| Parameters (incl. prior) | " + " | ".join(cell(n, "params", "{:,}") for n in names) + " |",
             "| Reconstruction latency, batch 1 (ms) | " + " | ".join(
                 cell(n, "recon_latency_ms_batch1", "{:.2f}") for n in names) + " |",
             "| Reconstruction throughput, batch 64 (slices/s) | " + " | ".join(
                 cell(n, "recon_throughput_slices_per_s_batch64", "{:,.0f}") for n in names) + " |",
             "| Peak GPU memory, inference batch 64 (MB) | " + " | ".join(
                 cell(n, "recon_peak_mem_mb_batch64", "{:.0f}") for n in names) + " |",
             "| Generation time per slice (ms) | " + " | ".join(
                 cell(n, "generation_ms_per_slice_batch64", "{:.2f}") for n in names) + " |",
             "| Training time (s) | " + " | ".join(cell(n, "train_time_s", "{:.0f}") for n in names) + " |",
             "| Peak GPU memory, training (MB) | " + " | ".join(
                 cell(n, "train_peak_mem_mb", "{:.0f}") for n in names) + " |"]
    if "prior_train_time_s" in prof.get("vqvae", {}):
        lines.append(f"\nThe PixelCNN prior adds {prof['vqvae']['prior_train_time_s']:.0f} s of training "
                     f"(peak {prof['vqvae']['prior_train_peak_mem_mb']:.0f} MB).")
    report.add("Resource profile", "\n".join(lines))
    print("\n".join(lines))


# --------------------------------------------------------------------------- #
# Main
# --------------------------------------------------------------------------- #
def main():
    args = parse_args()
    seed_everything(args.seed)
    device = get_device()
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    report = Report(out / "report.md")

    models = {}
    for name, path in (("vqvae", args.vqvae), ("vae", args.vae)):
        if Path(path).exists():
            models[name], ckpt = load_checkpoint(path, device)
            print(f"Loaded {name} from {path} (epoch {ckpt['epoch']}, val metric {ckpt['val_metric']:.4f})")
        else:
            print(f"WARNING: {path} not found; skipping {name}")
    if "vqvae" not in models:
        raise SystemExit("The VQ-VAE checkpoint is required.")
    prior = None
    if Path(args.prior).exists():
        prior, _ = load_checkpoint(args.prior, device)
        print(f"Loaded PixelCNN prior from {args.prior}")

    test = HipMRISlices("test", args.root, args.cache_dir)
    print(f"Test split: {len(test)} slices from {len(set(test.case_ids.tolist()))} patients")

    recons = rows = None
    if "evaluate" in args.tasks or "failures" in args.tasks:
        recons, rows = task_evaluate(models, test, device, out, report)
    if "failures" in args.tasks:
        task_failures(test, recons, rows, args.n_failures, out, report)
    samples = task_generate(models, prior, args, device, out) if "generate" in args.tasks else load_samples(out)
    if "memorisation" in args.tasks:
        if not samples:
            print("No samples found; run with --tasks generate first.")
        else:
            train = HipMRISlices("train", args.root, args.cache_dir)
            task_memorisation(models, samples, test, train, device, out, report)
    if "interpolate" in args.tasks:
        task_interpolate(models, test, device, out)
    if "profile" in args.tasks:
        task_profile(models, prior, args, test, device, out, report)
    print(f"Done. Results in {out}/ (summary: {out / 'report.md'})")


if __name__ == "__main__":
    main()
