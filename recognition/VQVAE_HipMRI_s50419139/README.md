# VQ-VAE for Synthetic Prostate MRI Generation (HipMRI 2D)

**Author:** Rashim Kaur (s50419139)
**Project:** COMP3710 Report, Section 2.4 Image Generation, VQ-VAE [Hard Difficulty]

> Work in progress. Sections are filled in as the project develops.

## 1. Problem and Algorithm Overview
<!-- One paragraph on the problem, one on how VQ-VAE works, plus a figure. -->

## 2. Feasibility Review (Midpoint Check-Off)
<!-- One-page review: user need and acceptance criteria, model choice, preliminary evidence, risks and fallback. -->

## 3. Dataset, Pre-processing and Splits
### 3.1 Data audit

Before writing the data loader, the HipMRI 2D slice dataset (`keras_slices_data` on Rangpur) was audited with `data_audit.py`, which checks slice counts, shapes, intensity ranges, label balance, image/mask pairing and patient overlap between splits.

| Split | Slices | Patients | Slice shapes |
|---|---|---|---|
| Train | 11,460 | 31 | 256×128 (11,400), 256×144 (60) |
| Validate | 660 | 4 | 256×128 |
| Test | 540 | 3 | 256×128 |

Key findings:

- **No patient leakage.** No patient (case ID) appears in more than one split, so the provided splits are patient-level and are kept as-is (38 patients in total, matching the dataset description).
- **Image/mask pairing is complete.** Every image slice has a matching segmentation mask in all three splits.
- **Mixed slice shapes.** 60 training slices are 256×144. These are centre-cropped to 256×128 rather than resized, to avoid interpolation blur.
- **Raw MR intensities.** Values are non-negative floats with per-slice maxima varying widely (sampled maxima up to 1080, median around 400), so intensities are normalised per slice (percentile clipping, then scaling to [0, 1]).
- **Extreme class imbalance in the masks.** Sampled pixel fractions are roughly 16% label 0, 68% label 1, 12% label 2, 2% label 3, 0.5% label 4 and 0.4% label 5. Per the dataset description, the labelled regions are background, body outline, bone, bladder, rectum and prostate. The prostate occupies well under 1% of each slice, so whole-image metrics can hide poor detail in the clinically important region. Masks are therefore used during evaluation to measure reconstruction quality inside each organ.
- **Repeated scans per patient.** Patients contribute up to 480 slices across multiple weekly scans, so neighbouring training slices are highly similar. The memorisation audit accounts for this by comparing generated images against both training and held-out test patients.
- **Limitation:** the test split contains only 3 patients, so test metrics carry patient-level variance and are reported with that caveat.

<!-- TODO: normalisation details and split justification once dataset.py is written. -->

## 4. Dependencies and Reproducibility
<!-- Exact versions, seeds, hardware, how to run. -->

## 5. Usage
<!-- Commands for train.py and predict.py, with example inputs and outputs. -->

## 6. Results
### 6.1 Baseline (VAE) vs VQ-VAE
### 6.2 Resource Profiling
### 6.3 Memorisation and Privacy Audit
### 6.4 Failure Case Autopsy
### 6.5 Recommendation to Project Manager

## 7. Artificial Intelligence Usage Disclosure
<!-- Tools used, tasks assisted, how outputs were verified. -->

## 8. References
