# Reproducing the barley NIR benchmark

The complete archive was verified on 14 September 2026. All missing files from the earlier partial upload are now included. The primary predictions and scientific result manifest are unchanged.

## Obtain the project

Repository: https://github.com/Julians30/barley-nir-stage-benchmark

The repository is public. Reviewers and readers can download the ZIP through **Code > Download ZIP**, or clone it without authentication:

```bash
git clone https://github.com/Julians30/barley-nir-stage-benchmark.git
cd barley-nir-stage-benchmark
```

For the exact scientific archive before the added verification documentation:

```bash
git checkout 9eb973f162fabaef3265a55c21085a13a0246c18
```

That commit contains 177 files. The current branch additionally includes these instructions, a verification record and a GitHub Actions workflow. Use `git rev-parse HEAD` to identify the version being examined.

## Recalculate and audit the archived results

Use Python 3.12 and run commands from the project root. The recorded local verification used Python 3.12.14 with the exact package versions in `requirements.txt`.

Linux/macOS:

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
python tools/verify_repository.py
```

Windows PowerShell:

```powershell
py -3.12 -m venv .venv
.venv\Scripts\python.exe -m pip install -r requirements.txt
.venv\Scripts\python.exe tools\verify_repository.py
```

The verifier restores the prepared NPZ from its 20 binary parts and checks the original SHA-256 before loading serialized metadata. It then checks every file in `REPOSITORY_MANIFEST.json`, the 134 entries of the original scientific result manifest, primary and review prediction metrics, outer/inner dish separation, training-only candidate ranking, and saved bootstrap quantiles. It finishes with three VIP diagnostic tests and rechecks archive integrity after the audits.

Expected final message:

```text
PASS: archived integrity, prediction audits and VIP tests; no model fitting.
```

Primary results recalculated from saved SVR predictions:

| Subset | Spectra | MAE | RMSE | R² |
| --- | ---: | ---: | ---: | ---: |
| All six acquisition stages | 13,452 | 0.576962 | 0.799941 | 0.780604 |
| Post-moisture stages 1–5 | 11,210 | 0.680003 | 0.874511 | 0.617615 |

Errors are expressed in acquisition-stage day units; the dataset does not independently validate continuous storage time, moisture content or biological viability.

## Inspect the notebook

First restore the dataset with `python tools/restore_dataset.py`. Install Jupyter separately:

```bash
python -m pip install jupyterlab ipykernel
python -m jupyter lab
```

Open `02_CODIGO_NOTEBOOKS/02_VALIDACION_ANIDADA_CEBADA_NIR.ipynb`. Keep `RUN_FULL_ANALYSIS = False` and `RUN_REVIEW_DIAGNOSTICS = False` to read the saved run. All ten code cells were executed locally during verification, with both fitting switches disabled. This is an audit of saved results, not a new model-training run.

## Fit the models again

Make a separate project copy first. A forced run overwrites result files in that copy:

```bash
python 02_CODIGO_NOTEBOOKS/run_nested_validation.py --project-dir . --force
python 02_CODIGO_NOTEBOOKS/run_review_additions.py --project-dir .
```

Primary nested validation uses five dish-disjoint outer folds, four dish-disjoint inner folds and 23 candidates across PLSR, RBF SVR and ExtraTrees (460 inner fits and 15 outer fits). Supplementary splitting and domain checks add fits. Seeds, grids, package versions and dataset digest are recorded in `03_RESULTADOS/nested_run_metadata.json` and the checkpoint context. The scripts can resume compatible runs; `--force` explicitly bypasses saved primary fits.

The distributed integrity manifests describe the archived run. After refitting, they are intentionally no longer an integrity certificate for the new outputs. Preserve the new predictions, settings and environment as a separate run and regenerate its provenance before treating it as a new release. Do not claim a fresh fit is byte-identical to the archive merely because its rounded metrics agree. Full training was not repeated in the September 14 closure check.

## Automated check and submission access

The `Verify archived reproducibility` GitHub Actions workflow runs the saved-result verifier on pushes, pull requests and manual dispatch. It does not fit models. Its actual run status appears in the repository's **Actions** tab; adding the workflow alone is not evidence of a passing remote run.

The corrected manuscript is `04_MANUSCRITO_CCIS/ARTICULO_CEBADA_NIR_GITHUB.docx`. The source dataset DOI, notices and CC BY-NC 4.0 restriction are documented in `01_DATASET` and the rights notice. Public GitHub access was confirmed on 14 September 2026 by checking repository visibility and cloning without authentication. The repository access requirement for submission is resolved. Historical review/deposit templates describe an earlier plan; they are not requirements of this GitHub-only closure. No archival deposit was created.
