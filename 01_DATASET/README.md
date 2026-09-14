# Distributed dataset

This repository distributes the prepared `barley_nir_long.npz` in 20 binary parts. Run `python tools/restore_dataset.py` from the project root to restore the original 20,497,950-byte file. The parts descriptor records their sizes and SHA-256 digests.

The NPZ schema is:

| Key | Shape | Meaning |
| --- | --- | --- |
| `X` | 13,452 × 204 | Original mean pseudo-absorbance spectra stored as float64; the modelling scripts convert them to float32 |
| `day` | 13,452 | Acquisition-stage target, 0–5 |
| `petri_dish` | 13,452 | Dish identifier used for primary outer/inner grouping |
| `grain_id` | 13,452 | Kernel identifier; combine with dish to form a unique kernel key |
| `sample_group` | 13,452 | Four original annotation sample codes |
| `cultivar` | 13,452 | Prospect or Laureate |
| `germination_day` | 13,452 | Retained source annotation; never a predictor |
| `germinated_current` | 13,452 | Derived metadata; never a predictor |

Only `X` is passed to estimators and `day` supplies their target. Metadata supply partitioning and descriptive checks. `DATASET_NOTES.md` preserves notes from the earlier local preparation; the CSV export and compact-original ZIP mentioned there are not distributed in this GitHub archive.

The audit compared all 13,452 × 204 values with the retained compact source subset using the compound dish/kernel key, and found exact equality. Original sample codes and germination annotations also matched. This comparison used retained source bytes; fresh external downloading of the entire source archive was not part of the audit. The compact subset digest and comparison evidence are in `AUDIT_EVIDENCE_2026-09-14.json` at the repository root. The optional `--source-zip` audit argument permits the same comparison when a reader supplies those original files in a compact ZIP.

Original data remain available through the [University of Copenhagen ERDA DOI](https://doi.org/10.17894/ucph.71c22737-8005-4588-bd74-80bf7b5ac6b4). Source notices and the CC BY-NC 4.0 attribution and non-commercial restriction remain included. The approximate wavelength range is 950–1650 nm; the README-based uniform plotting grid is not an exact camera calibration.
