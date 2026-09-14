# Barley NIR stage benchmark

Reproducible computational materials for **Grouped Validation of Machine Learning for Moisture Exposure Stage Estimation in Barley Using NIR Spectra**, by Julián Coronel Reyes (corresponding author), Vanessa Vergara-Lozano and Carlota Delgado-Vera.

The benchmark uses 13,452 spectra, 204 channels, 2,242 kernels and 90 Petri dishes from one six-stage acquisition sequence. Five outer and four inner folds keep dishes disjoint; model preprocessing and hyperparameter selection use training data only.

## Reproducibility status

The partial upload has been completed. The scientific archive commit `9eb973f162fabaef3265a55c21085a13a0246c18` contains all 177 intended files. Local verification on 14 September 2026 passed original dataset/result digests, prediction and partition audits, 105 review metric checks, three VIP tests and execution of all ten notebook code cells with fitting disabled. Full model training was not repeated. See [REPRODUCIBILITY.md](REPRODUCIBILITY.md) for Windows/Linux commands, exact-version access and fresh-fitting instructions, and [VERIFICATION_2026-09-14.json](VERIFICATION_2026-09-14.json) for the check record. A GitHub Actions workflow now runs saved-result verification. The repository is public and can be accessed by reviewers and readers without an invitation.

## Main finding and scope

The original out-of-fold RBF SVR predictions give **post-moisture MAE 0.680** (stages 1–5; R² 0.618), versus aggregate MAE 0.577 including the easy dry reference. Random-row splitting changes MAE by only approximately −0.021 relative to dish grouping: an informative negative finding, not evidence of a large leakage effect. Stage coincides with acquisition session, so session fingerprinting cannot be excluded. This is not an independently validated biological clock, moisture assay or germination/viability predictor.

Training-only PLS VIP and signed coefficients describe distributed spectral importance; they do not establish a dominant water band or explain the SVR directly. Frozen C=300 and C=1000 checks do not improve the original selected C=100 predictions and do not replace primary selection. Cultivar/group exclusions are exploratory at the already selected model-family level.

## Files

| Folder | Contents |
| --- | --- |
| `01_DATASET` | Prepared matrix in verified binary parts, source attribution and licensing notices |
| `02_CODIGO_NOTEBOOKS` | Nested-validation pipeline, review diagnostics, audits and executed validation notebook |
| `03_RESULTADOS` | Original OOF predictions, candidate scores, bootstrap draws, diagnostic profiles, figures and compatible checkpoints |
| `04_MANUSCRITO_CCIS` | Corrected Word manuscript with embedded workflow and spectral interpretation figures and this repository URL |
| `tools` | Project packaging and repository integrity verification utilities |

The prepared NPZ is stored in 1 MiB binary parts to accommodate upload transport limits. The restore utility reconstructs the exact original file and verifies its SHA-256; spectra and metadata are unchanged. Original archives, duplicate CSV exports, hyperspectral image cubes, temporary renders and unrelated teaching materials are not included. Review/deposit templates in `03_RESULTADOS/revision_r1` are retained unchanged as historical records; their pending-link text is not the current repository status. `REPOSITORY_STATUS.json` records the current URL and public visibility. No public DOI or Zenodo publication is claimed.

## Data source and integrity

Source: Engstrøm et al., University of Copenhagen ERDA dataset, DOI [10.17894/ucph.71c22737-8005-4588-bd74-80bf7b5ac6b4](https://doi.org/10.17894/ucph.71c22737-8005-4588-bd74-80bf7b5ac6b4). The source spectra and covered adaptations remain subject to attribution and the non-commercial restriction of **CC BY-NC 4.0**. Retain the included source notices. See `03_RESULTADOS/revision_r1/RIGHTS_AND_LICENSES.md` for component-specific rights; no software license or coauthor approval of public release is invented.

Prepared NPZ SHA-256:
`3ee66476e6564e6b20d6b3c87433517e04fb6ce99e8ea1625e3f8cc1a3c70cdd`

Original primary OOF SHA-256:
`42cf6520e0c86cc35e8ced1eeb30a0e05698a7538c7cd8aebe9d0b5bac056a64`

The NPZ contains serialized text metadata. Load it with `allow_pickle=True` only after verifying this digest and trusting this source. `REPOSITORY_MANIFEST.json` lists the SHA-256 and Git blob SHA of every distributed file except the manifest itself. The original `03_RESULTADOS/results_manifest.json` separately preserves the scientific result file digests.

## Verify saved results without training models

Use Python 3.12 (recorded run: 3.12.14). From the repository root:

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
python tools/restore_dataset.py
python tools/verify_repository.py
```

This verifies file digests, recalculates primary and review metrics from saved predictions, checks dish-disjoint memberships and inner selection, checks the paired cluster-bootstrap intervals, and runs the VIP unit tests. It does **not** rerun model fitting. The audits regenerate their small verification reports; original predictions are retained.

## Notebook and explicit recomputation

```bash
python -m pip install jupyterlab ipykernel nbformat nbclient
python -m jupyter lab
```

Reconstruct the NPZ with `python tools/restore_dataset.py` before opening `02_CODIGO_NOTEBOOKS/02_VALIDACION_ANIDADA_CEBADA_NIR.ipynb`. Its default switches read and audit saved results. Run Jupyter from the root or code folder. The Colab-specific setup uses the original Drive layout `/content/drive/MyDrive/CITI2027_CEBADA_NIR`; copy these canonical folders there if using that route.

To force a genuinely new primary fit, first copy the project so that the archived results remain untouched:

```bash
python 02_CODIGO_NOTEBOOKS/run_nested_validation.py --project-dir . --force
python 02_CODIGO_NOTEBOOKS/run_review_additions.py --project-dir .
```

Full nested fitting is substantially more expensive than auditing the saved results. The script otherwise resumes compatible checkpoints or skips an already completed compatible run, and rejects mixed dataset/settings/software versions. Recomputed results may differ on other environments and must not be represented as the archived run without new provenance. The bootstrap is conditional on fixed OOF predictions, not a refit bootstrap.

## Access and disclosure

This repository is a **public reproducibility archive**, released by the repository owner. Reviewers and readers can download or clone it without an invitation. Source-data notices and restrictions still apply; no separate software license is claimed. GitHub is not itself a DOI deposit. AI assistance was used for code development and drafting; the numerical claims were checked against executed, saved external predictions, and no synthetic experimental observations were introduced.
