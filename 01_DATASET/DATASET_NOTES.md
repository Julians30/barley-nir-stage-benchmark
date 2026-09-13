# Barley NIR longitudinal dataset — subset for CITI 2027

## Source

- Dataset: *A Time Series Dataset of NIR Spectra and RGB and NIR-HSI Images of the Barley Germination Process*.
- Authors: Ole-Christian Galbo Engstrøm, Erik Schou Dreier, Birthe Møller Jespersen, and Kim Steenstrup Pedersen.
- DOI: https://doi.org/10.17894/ucph.71c22737-8005-4588-bd74-80bf7b5ac6b4
- Public archive: https://erda.ku.dk/archives/f61461850198616c29294963a9b5540d/published-archive.html
- License: CC BY-NC 4.0. The original `LICENSE.txt` is included in the archive subset.

## Compact subset

Only the files needed for spectral modelling were retained:

- `annotations.csv`;
- six `mean_spectra.npy` matrices;
- six `mean_spectra_order.csv` files;
- original README, license, and mono12p utility.

No RGB images, hyperspectral cubes, or segmentation masks are included.

## Verified structure

- 2,242 unique barley kernels.
- 90 Petri dishes.
- Six observations per kernel: pre-moisture and days 1–5.
- 13,452 total spectrum-time observations.
- 204 finite NIR pseudo-absorbance bands per spectrum.
- Approximate usable spectral range: 950–1650 nm, according to the original README.
- All six matrices contain the same kernel identifiers and match `annotations.csv`.

## Prepared modelling files

- `barley_nir_long.csv.gz`: long-format table with metadata, outcomes, and 204 spectral bands.
- `barley_nir_long.npz`: compressed NumPy version for faster modelling.
- `barley_nir_original_subset.zip`: untouched compact subset of the original public files.

Important fields in the long-format file:

- `grain_uid`: unique kernel identifier.
- `petri_dish`: experimental dish and primary grouping variable.
- `sample_group`: original sample code (`prospect_0`, `prospect_1`, `laureate_0`, `laureate_1`).
- `cultivar`: Prospect or Laureate.
- `day_since_moisture`: stored name of the numeric acquisition-stage target, 0–5. Stage 0 is the dry pre-moisture reference; 1–5 are subsequent daily acquisitions. This variable does not validate continuous time from an immediate post-wetting day-zero measurement.
- `germination_day`: observed germination day; `-1` means no germination during follow-up.
- `germinated_current`: whether germination had occurred by the observation day.
- `band_001`–`band_204`: NIR pseudo-absorbance values.

## Data audit note

The released annotations contain 381 kernels germinating during follow-up: 27 on day 1, 93 on day 2, 106 on day 3, 88 on day 4, and 67 on day 5. The accompanying paper reports 28 on day 1 and 382 in total, although its per-sample counts sum to 27 and 381 respectively. Analyses must use the released annotations and disclose this one-kernel discrepancy.

## Validation rule

The primary evaluation must be Petri-dish-disjoint, with internal tuning grouped by dish as well. Random-row and kernel-grouped partitions may be reported as descriptive sensitivity comparators, without assuming an optimistic difference in advance. Population-level preprocessing statistics and hyperparameter selection must be fitted inside training folds. Deterministic within-spectrum SNV and Savitzky–Golay transforms do not learn from other observations.

Dish disjointness does not separate acquisition sessions. Stage-specific session or instrument effects may be confounded with biological exposure changes. No external harvest year, instrument, or independent acquisition sequence is available in this subset.
