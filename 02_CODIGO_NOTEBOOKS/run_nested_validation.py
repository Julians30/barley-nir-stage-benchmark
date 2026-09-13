from __future__ import annotations

import argparse
import hashlib
import json
import platform
import time
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scipy
import sklearn
import seaborn as sns
from scipy.signal import savgol_filter
from sklearn.cross_decomposition import PLSRegression
from sklearn.ensemble import ExtraTreesRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import GroupKFold, KFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVR


SEED = 2026
OUTER_SPLITS = 5
INNER_SPLITS = 4
BOOTSTRAP_REPETITIONS = 2_000
PIPELINE_VERSION = 'dish_nested_v1'


def software_versions() -> dict[str, str]:
    return {'python': platform.python_version(), 'numpy': np.__version__,
            'pandas': pd.__version__, 'scipy': scipy.__version__,
            'scikit_learn': sklearn.__version__}


def markdown_table(frame: pd.DataFrame, digits: int = 4) -> str:
    """Small dependency-free renderer for the numeric result reports."""
    def value_text(value):
        text = f'{value:.{digits}f}' if isinstance(value, (float, np.floating)) else str(value)
        return text.replace('|', '\\|').replace('\n', ' ')
    return '\n'.join(['| ' + ' | '.join(map(str, frame.columns)) + ' |',
                      '| ' + ' | '.join(['---'] * len(frame.columns)) + ' |'] +
                     ['| ' + ' | '.join(value_text(value) for value in row) + ' |'
                      for row in frame.itertuples(index=False, name=None)])


@dataclass(frozen=True)
class Candidate:
    family: str
    preprocessing: str
    parameter: str
    build: Callable[[], object]

    @property
    def key(self) -> str:
        return f"{self.preprocessing}|{self.parameter}"


def snv(x: np.ndarray) -> np.ndarray:
    row_mean = x.mean(axis=1, keepdims=True)
    row_std = x.std(axis=1, keepdims=True)
    return ((x - row_mean) / np.where(row_std == 0, 1, row_std)).astype(np.float32)


def spectral_variants(raw: np.ndarray) -> dict[str, np.ndarray]:
    return {
        "raw": raw.astype(np.float32, copy=False),
        "snv": snv(raw),
        "savgol_d1": savgol_filter(
            raw, window_length=11, polyorder=2, deriv=1, axis=1
        ).astype(np.float32),
    }


def candidate_grid(seed: int = SEED) -> dict[str, list[Candidate]]:
    candidates: dict[str, list[Candidate]] = {"PLSR": [], "SVR_RBF": [], "ExtraTrees": []}

    for preprocessing in ["raw", "snv", "savgol_d1"]:
        for n_components in [10, 20, 30]:
            candidates["PLSR"].append(
                Candidate(
                    "PLSR",
                    preprocessing,
                    f"n_components={n_components}",
                    lambda n=n_components: Pipeline(
                        [
                            ("scale", StandardScaler()),
                            (
                                "model",
                                PLSRegression(
                                    n_components=n, scale=False, max_iter=1_000
                                ),
                            ),
                        ]
                    ),
                )
            )

    for preprocessing in ["snv", "savgol_d1"]:
        for c_value in [10.0, 100.0]:
            for epsilon in [0.05, 0.10]:
                candidates["SVR_RBF"].append(
                    Candidate(
                        "SVR_RBF",
                        preprocessing,
                        f"C={c_value:g};epsilon={epsilon:g};gamma=scale",
                        lambda c=c_value, e=epsilon: Pipeline(
                            [
                                ("scale", StandardScaler()),
                                (
                                    "model",
                                    SVR(
                                        C=c,
                                        epsilon=e,
                                        gamma="scale",
                                        cache_size=4_000,
                                    ),
                                ),
                            ]
                        ),
                    )
                )

    for max_features in [0.5, 1.0]:
        for min_samples_leaf in [1, 2, 5]:
            candidates["ExtraTrees"].append(
                Candidate(
                    "ExtraTrees",
                    "raw",
                    f"max_features={max_features:g};min_samples_leaf={min_samples_leaf}",
                    lambda mf=max_features, leaf=min_samples_leaf: ExtraTreesRegressor(
                        n_estimators=300,
                        max_features=mf,
                        min_samples_leaf=leaf,
                        n_jobs=-1,
                        random_state=seed,
                    ),
                )
            )

    return candidates


def regression_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, float]:
    error = np.asarray(y_pred) - np.asarray(y_true)
    return {
        "mae": float(mean_absolute_error(y_true, y_pred)),
        "rmse": float(mean_squared_error(y_true, y_pred) ** 0.5),
        "r2": float(r2_score(y_true, y_pred)),
        "within_1_day": float(np.mean(np.abs(error) <= 1.0)),
        "bias": float(np.mean(error)),
    }


def split_signature(groups: np.ndarray, split_indices: list[tuple[np.ndarray, np.ndarray]]) -> str:
    parts = []
    for _, test_idx in split_indices:
        parts.append(",".join(map(str, sorted(np.unique(groups[test_idx]).tolist()))))
    return hashlib.sha256("|".join(parts).encode("utf-8")).hexdigest()[:16]


def select_candidate(
    candidates: list[Candidate],
    x_variants: dict[str, np.ndarray],
    y: np.ndarray,
    groups: np.ndarray,
    train_idx: np.ndarray,
    n_splits: int,
    checkpoint_path: Path | None = None,
    resume: bool = True,
) -> tuple[Candidate, list[dict[str, object]]]:
    inner = GroupKFold(n_splits=n_splits)
    local_groups = groups[train_idx]
    local_y = y[train_idx]
    inner_splits = list(inner.split(train_idx, local_y, local_groups))
    rows: list[dict[str, object]] = []
    if checkpoint_path is not None and checkpoint_path.exists() and resume:
        rows = pd.read_csv(checkpoint_path).to_dict("records")

    for candidate in candidates:
        cached = [row for row in rows if row["candidate_key"] == candidate.key]
        if len(cached) == n_splits:
            print(f"    restored {candidate.family}/{candidate.key}", flush=True)
            continue
        fold_scores = []
        for inner_fold, (local_train, local_valid) in enumerate(inner_splits, 1):
            fit_idx = train_idx[local_train]
            valid_idx = train_idx[local_valid]
            assert set(groups[fit_idx]).isdisjoint(set(groups[valid_idx]))
            model = candidate.build()
            started = time.time()
            model.fit(x_variants[candidate.preprocessing][fit_idx], y[fit_idx])
            prediction = np.asarray(
                model.predict(x_variants[candidate.preprocessing][valid_idx])
            ).reshape(-1)
            score = regression_metrics(y[valid_idx], prediction)
            fold_scores.append(score["mae"])
            rows.append(
                {
                    "family": candidate.family,
                    "preprocessing": candidate.preprocessing,
                    "parameter": candidate.parameter,
                    "candidate_key": candidate.key,
                    "inner_fold": inner_fold,
                    **score,
                    "fit_predict_seconds": time.time() - started,
                    "n_train": len(fit_idx),
                    "n_valid": len(valid_idx),
                }
            )

        mean_mae = float(np.mean(fold_scores))
        for row in rows[-len(inner_splits) :]:
            row["candidate_inner_mae_mean"] = mean_mae
        if checkpoint_path is not None:
            pd.DataFrame(rows).to_csv(checkpoint_path, index=False, compression="gzip")
        print(f"    {candidate.family}/{candidate.key}: inner MAE={mean_mae:.4f}", flush=True)

    summary = (
        pd.DataFrame(rows)
        .groupby(["candidate_key"], as_index=False)
        .agg(mae=("mae", "mean"), rmse=("rmse", "mean"))
        .sort_values(["mae", "rmse", "candidate_key"])
    )
    best_key = str(summary.iloc[0]["candidate_key"])
    selected = next(candidate for candidate in candidates if candidate.key == best_key)
    return selected, rows


def run_nested_cv(
    x_variants: dict[str, np.ndarray],
    y: np.ndarray,
    dish: np.ndarray,
    grain_uid: np.ndarray,
    sample_group: np.ndarray,
    cultivar: np.ndarray,
    seed: int,
    checkpoint_dir: Path | None = None,
    resume: bool = True,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    grid = candidate_grid(seed)
    outer_splits = list(GroupKFold(OUTER_SPLITS).split(y, y, dish))
    inner_rows: list[dict[str, object]] = []
    outer_rows: list[dict[str, object]] = []
    prediction_frames: list[pd.DataFrame] = []

    print(
        f"Nested CV: {OUTER_SPLITS} outer folds x {INNER_SPLITS} inner folds; "
        f"outer split signature={split_signature(dish, outer_splits)}",
        flush=True,
    )
    for outer_fold, (train_idx, test_idx) in enumerate(outer_splits, 1):
        train_dishes = np.unique(dish[train_idx])
        test_dishes = np.unique(dish[test_idx])
        assert set(train_dishes).isdisjoint(test_dishes)

        for family, candidates in grid.items():
            prefix = checkpoint_dir / f"outer_{outer_fold:02d}_{family}" if checkpoint_dir else None
            inner_file = Path(f"{prefix}_inner.csv.gz") if prefix else None
            outer_file = Path(f"{prefix}_outer.csv") if prefix else None
            prediction_file = Path(f"{prefix}_predictions.csv.gz") if prefix else None
            if resume and prefix and all(p.exists() for p in [inner_file, outer_file, prediction_file]):
                cached_inner = pd.read_csv(inner_file)
                cached_outer = pd.read_csv(outer_file)
                cached_prediction = pd.read_csv(prediction_file)
                if "selected" in cached_inner and cached_prediction["row_index"].nunique() == len(test_idx):
                    inner_rows.extend(cached_inner.to_dict("records"))
                    outer_rows.extend(cached_outer.to_dict("records"))
                    prediction_frames.append(cached_prediction)
                    print(f"Restored outer {outer_fold}/{family}", flush=True)
                    continue
            print(f"Outer {outer_fold}/{OUTER_SPLITS} - tuning {family}", flush=True)
            selected, family_inner_rows = select_candidate(
                candidates,
                x_variants,
                y,
                dish,
                train_idx,
                INNER_SPLITS,
                inner_file,
                resume,
            )
            for row in family_inner_rows:
                row["outer_fold"] = outer_fold
                row["selected"] = row["candidate_key"] == selected.key
                inner_rows.append(row)

            model = selected.build()
            started = time.time()
            model.fit(x_variants[selected.preprocessing][train_idx], y[train_idx])
            prediction = np.asarray(
                model.predict(x_variants[selected.preprocessing][test_idx])
            ).reshape(-1)
            score = regression_metrics(y[test_idx], prediction)
            outer_rows.append(
                {
                    "family": family,
                    "outer_fold": outer_fold,
                    "preprocessing": selected.preprocessing,
                    "parameter": selected.parameter,
                    "candidate_key": selected.key,
                    **score,
                    "fit_predict_seconds": time.time() - started,
                    "n_train": len(train_idx),
                    "n_test": len(test_idx),
                    "n_train_dishes": len(train_dishes),
                    "n_test_dishes": len(test_dishes),
                }
            )
            prediction_frames.append(
                pd.DataFrame(
                    {
                        "family": family,
                        "outer_fold": outer_fold,
                        "row_index": test_idx,
                        "petri_dish": dish[test_idx],
                        "grain_uid": grain_uid[test_idx],
                        "sample_group": sample_group[test_idx],
                        "cultivar": cultivar[test_idx],
                        "y_true": y[test_idx],
                        "y_pred": prediction,
                        "preprocessing": selected.preprocessing,
                        "parameter": selected.parameter,
                    }
                )
            )
            if prefix:
                pd.DataFrame(family_inner_rows).to_csv(inner_file, index=False, compression="gzip")
                pd.DataFrame([outer_rows[-1]]).to_csv(outer_file, index=False)
                prediction_frames[-1].to_csv(prediction_file, index=False, compression="gzip")
            print(
                f"  selected {selected.key}; outer MAE={score['mae']:.4f}, "
                f"R2={score['r2']:.4f}",
                flush=True,
            )

    return (
        pd.DataFrame(inner_rows),
        pd.DataFrame(outer_rows),
        pd.concat(prediction_frames, ignore_index=True),
    )


def cluster_bootstrap_ci(
    predictions: pd.DataFrame,
    repetitions: int,
    seed: int,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    rng = np.random.default_rng(seed)
    replicate_rows = []

    all_blocks = {}
    for family, frame in predictions.groupby("family", sort=False):
        all_blocks[family] = {
            dish_id: block[["y_true", "y_pred"]].to_numpy(dtype=float)
            for dish_id, block in frame.groupby("petri_dish", sort=False)
        }
    dish_ids = np.array(sorted(next(iter(all_blocks.values()))))
    for repetition in range(1, repetitions + 1):
        sampled = rng.choice(dish_ids, size=len(dish_ids), replace=True)
        for family, blocks in all_blocks.items():
            values = np.vstack([blocks[dish_id] for dish_id in sampled])
            replicate_rows.append(
                {
                    "family": family,
                    "bootstrap_repetition": repetition,
                    **regression_metrics(values[:, 0], values[:, 1]),
                }
            )

    replicates = pd.DataFrame(replicate_rows)
    point_rows = []
    for family, frame in predictions.groupby("family", sort=False):
        point = regression_metrics(frame["y_true"].to_numpy(), frame["y_pred"].to_numpy())
        family_boot = replicates.loc[replicates["family"].eq(family)]
        for metric, estimate in point.items():
            point_rows.append(
                {
                    "family": family,
                    "metric": metric,
                    "estimate": estimate,
                    "ci_low": float(family_boot[metric].quantile(0.025)),
                    "ci_high": float(family_boot[metric].quantile(0.975)),
                    "bootstrap_unit": "petri_dish",
                    "bootstrap_repetitions": repetitions,
                }
            )
    return pd.DataFrame(point_rows), replicates


def choose_consensus_candidate(
    family: str, inner_results: pd.DataFrame, grid: dict[str, list[Candidate]]
) -> Candidate:
    family_results = inner_results.loc[inner_results["family"].eq(family)].copy()
    selected = family_results.loc[family_results["selected"]]
    counts = Counter(selected.groupby("outer_fold")["candidate_key"].first())
    max_count = max(counts.values())
    tied = {key for key, count in counts.items() if count == max_count}
    tie_break = (
        family_results.loc[family_results["candidate_key"].isin(tied)]
        .groupby("candidate_key")["mae"]
        .mean()
        .sort_values()
        .index[0]
    )
    return next(candidate for candidate in grid[family] if candidate.key == tie_break)


def evaluate_fixed_validation_schemes(
    candidate: Candidate,
    x_variants: dict[str, np.ndarray],
    y: np.ndarray,
    dish: np.ndarray,
    grain_uid: np.ndarray,
    seed: int,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    schemes = {
        "random_row": list(KFold(5, shuffle=True, random_state=seed).split(y)),
        "grain_disjoint": list(GroupKFold(5).split(y, y, grain_uid)),
        "dish_disjoint": list(GroupKFold(5).split(y, y, dish)),
    }
    rows = []
    predictions = []
    x_use = x_variants[candidate.preprocessing]
    for scheme, splits in schemes.items():
        for fold, (train_idx, test_idx) in enumerate(splits, 1):
            model = candidate.build()
            model.fit(x_use[train_idx], y[train_idx])
            pred = np.asarray(model.predict(x_use[test_idx])).reshape(-1)
            rows.append(
                {
                    "scheme": scheme,
                    "fold": fold,
                    "family": candidate.family,
                    "preprocessing": candidate.preprocessing,
                    "parameter": candidate.parameter,
                    **regression_metrics(y[test_idx], pred),
                }
            )
            predictions.append(
                pd.DataFrame(
                    {
                        "scheme": scheme,
                        "fold": fold,
                        "row_index": test_idx,
                        "y_true": y[test_idx],
                        "y_pred": pred,
                    }
                )
            )
    return pd.DataFrame(rows), pd.concat(predictions, ignore_index=True)


def run_domain_stress_tests(
    candidates: list[Candidate],
    x_variants: dict[str, np.ndarray],
    y: np.ndarray,
    dish: np.ndarray,
    grain_uid: np.ndarray,
    sample_group: np.ndarray,
    cultivar: np.ndarray,
    checkpoint_dir: Path | None = None,
    resume: bool = True,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    tasks = []
    for held_out in sorted(np.unique(cultivar)):
        tasks.append(("leave_one_cultivar_out", held_out, cultivar != held_out, cultivar == held_out))
    for held_out in sorted(np.unique(sample_group)):
        tasks.append(
            (
                "leave_one_sample_group_out",
                held_out,
                sample_group != held_out,
                sample_group == held_out,
            )
        )

    result_rows = []
    selection_rows = []
    prediction_frames = []
    for scheme, held_out, train_mask, test_mask in tasks:
        train_idx = np.flatnonzero(train_mask)
        test_idx = np.flatnonzero(test_mask)
        assert set(dish[train_idx]).isdisjoint(set(dish[test_idx]))
        n_inner = min(INNER_SPLITS, len(np.unique(dish[train_idx])))
        prefix = checkpoint_dir / f"stress_{scheme}_{held_out}" if checkpoint_dir else None
        inner_file = Path(f"{prefix}_inner.csv.gz") if prefix else None
        result_file = Path(f"{prefix}_result.csv") if prefix else None
        prediction_file = Path(f"{prefix}_predictions.csv.gz") if prefix else None
        if resume and prefix and all(p.exists() for p in [inner_file, result_file, prediction_file]):
            selection_rows.extend(pd.read_csv(inner_file).to_dict("records"))
            result_rows.extend(pd.read_csv(result_file).to_dict("records"))
            prediction_frames.append(pd.read_csv(prediction_file))
            print(f"Restored stress test {scheme}/{held_out}", flush=True)
            continue
        selected, inner_rows = select_candidate(
            candidates, x_variants, y, dish, train_idx, n_inner, inner_file, resume
        )
        for row in inner_rows:
            row["scheme"] = scheme
            row["held_out"] = held_out
            row["selected"] = row["candidate_key"] == selected.key
            selection_rows.append(row)

        model = selected.build()
        started = time.time()
        model.fit(x_variants[selected.preprocessing][train_idx], y[train_idx])
        pred = np.asarray(
            model.predict(x_variants[selected.preprocessing][test_idx])
        ).reshape(-1)
        score = regression_metrics(y[test_idx], pred)
        result_rows.append(
            {
                "scheme": scheme,
                "held_out": held_out,
                "family": selected.family,
                "preprocessing": selected.preprocessing,
                "parameter": selected.parameter,
                **score,
                "n_train": len(train_idx),
                "n_test": len(test_idx),
                "n_train_dishes": len(np.unique(dish[train_idx])),
                "n_test_dishes": len(np.unique(dish[test_idx])),
                "fit_predict_seconds": time.time() - started,
            }
        )
        prediction_frames.append(
            pd.DataFrame(
                {
                    "scheme": scheme,
                    "held_out": held_out,
                    "row_index": test_idx,
                    "petri_dish": dish[test_idx],
                    "grain_uid": grain_uid[test_idx],
                    "sample_group": sample_group[test_idx],
                    "cultivar": cultivar[test_idx],
                    "y_true": y[test_idx],
                    "y_pred": pred,
                    "preprocessing": selected.preprocessing,
                    "parameter": selected.parameter,
                }
            )
        )
        if prefix:
            pd.DataFrame(inner_rows).to_csv(inner_file, index=False, compression="gzip")
            pd.DataFrame([result_rows[-1]]).to_csv(result_file, index=False)
            prediction_frames[-1].to_csv(prediction_file, index=False, compression="gzip")
        print(
            f"Stress test {scheme}/{held_out}: {selected.key}; "
            f"MAE={score['mae']:.4f}, R2={score['r2']:.4f}",
            flush=True,
        )
    return (
        pd.DataFrame(result_rows),
        pd.DataFrame(selection_rows),
        pd.concat(prediction_frames, ignore_index=True),
    )


def save_figures(
    out_dir: Path,
    ci: pd.DataFrame,
    predictions: pd.DataFrame,
    best_family: str,
) -> None:
    sns.set_theme(style="whitegrid", context="talk")
    palette = {"PLSR": "#4472C4", "SVR_RBF": "#ED7D31", "ExtraTrees": "#70AD47"}

    mae = ci.loc[ci["metric"].eq("mae")].sort_values("estimate")
    colors = [palette.get(family, "#777777") for family in mae["family"]]
    fig, ax = plt.subplots(figsize=(8.2, 5.2))
    xerr = np.vstack([mae["estimate"] - mae["ci_low"], mae["ci_high"] - mae["estimate"]])
    ax.bar(mae["family"], mae["estimate"], color=colors, alpha=0.9)
    ax.errorbar(
        mae["family"], mae["estimate"], yerr=xerr, fmt="none", ecolor="black", capsize=5
    )
    ax.set(
        xlabel="Model family",
        ylabel="Nested out-of-fold MAE (stage units)",
        title="Petri-dish-disjoint nested validation",
    )
    ax.set_ylim(bottom=0)
    fig.tight_layout()
    fig.savefig(out_dir / "fig_nested_model_comparison.png", dpi=300, bbox_inches="tight")
    plt.close(fig)

    best = predictions.loc[predictions["family"].eq(best_family)].copy()
    fig, ax = plt.subplots(figsize=(7.2, 6.2))
    hb = ax.hexbin(
        best["y_true"], best["y_pred"], gridsize=34, cmap="viridis", mincnt=1
    )
    ax.plot([0, 5], [0, 5], ls="--", lw=2, color="crimson", label="Ideal")
    ax.set(
        xlabel="Reference acquisition stage (day units)",
        ylabel="Predicted stage index (day units)",
        title=f"Out-of-fold predictions — {best_family}",
        xlim=(-0.15, 5.15),
    )
    ax.legend()
    fig.colorbar(hb, ax=ax, label="Observations")
    fig.tight_layout()
    fig.savefig(out_dir / "fig_best_observed_vs_predicted.png", dpi=300, bbox_inches="tight")
    plt.close(fig)

    best["error"] = best["y_pred"] - best["y_true"]
    per_day = (
        best.groupby("y_true", as_index=False)
        .agg(mae=("error", lambda s: np.mean(np.abs(s))), bias=("error", "mean"))
        .rename(columns={"y_true": "day"})
    )
    fig, ax = plt.subplots(figsize=(8.2, 5.2))
    ax.plot(per_day["day"], per_day["mae"], marker="o", lw=2.2, label="MAE")
    ax.plot(per_day["day"], per_day["bias"], marker="s", lw=2.2, label="Bias")
    ax.axhline(0, color="black", lw=1)
    ax.set(
        xlabel="Reference acquisition stage (day units)",
        ylabel="Error (stage units)",
        title=f"Error profile by acquisition stage — {best_family}",
        xticks=range(6),
    )
    ax.legend()
    fig.tight_layout()
    fig.savefig(out_dir / "fig_best_error_by_day.png", dpi=300, bbox_inches="tight")
    plt.close(fig)


def write_results_note(
    path: Path,
    nested_summary: pd.DataFrame,
    ci: pd.DataFrame,
    best_family: str,
    consensus: Candidate,
    stress_results: pd.DataFrame,
    scheme_summary: pd.DataFrame,
) -> None:
    best_metrics = ci.loc[ci["family"].eq(best_family)].set_index("metric")
    cross_cultivar = stress_results.loc[
        stress_results["scheme"].eq("leave_one_cultivar_out")
    ]
    lines = [
        "# Resultados finales de validación — CITI 2027",
        "",
        "## Diseño principal",
        "",
        f"- Validación cruzada anidada: {OUTER_SPLITS} pliegues externos y {INNER_SPLITS} internos.",
        "- Agrupación externa e interna: caja de Petri.",
        "- Unidad del bootstrap: caja de Petri.",
        f"- Repeticiones bootstrap: {BOOTSTRAP_REPETITIONS:,}.",
        "- Variable objetivo: índice de etapa experimental (0–5), expresado en unidades diarias.",
        "- Etapa 0: referencia seca previa a la humectación; etapas 1–5: adquisiciones diarias posteriores.",
        "",
        "## Resultado principal",
        "",
        f"La mejor familia fue **{best_family}**. La configuración de consenso fue "
        f"**{consensus.preprocessing}; {consensus.parameter}**.",
        "",
        f"- MAE: {best_metrics.loc['mae', 'estimate']:.3f} días "
        f"(IC 95% por bootstrap de placas: {best_metrics.loc['mae', 'ci_low']:.3f}–"
        f"{best_metrics.loc['mae', 'ci_high']:.3f}).",
        f"- RMSE: {best_metrics.loc['rmse', 'estimate']:.3f} días "
        f"(IC 95%: {best_metrics.loc['rmse', 'ci_low']:.3f}–"
        f"{best_metrics.loc['rmse', 'ci_high']:.3f}).",
        f"- R²: {best_metrics.loc['r2', 'estimate']:.3f} "
        f"(IC 95%: {best_metrics.loc['r2', 'ci_low']:.3f}–"
        f"{best_metrics.loc['r2', 'ci_high']:.3f}).",
        f"- Predicciones dentro de ±1 día: "
        f"{100 * best_metrics.loc['within_1_day', 'estimate']:.1f}% "
        f"(IC 95%: {100 * best_metrics.loc['within_1_day', 'ci_low']:.1f}–"
        f"{100 * best_metrics.loc['within_1_day', 'ci_high']:.1f}%).",
        "",
        "## Comparación de familias",
        "",
        markdown_table(nested_summary),
        "",
        "## Auditoría de esquemas de partición",
        "",
        "Esta comparación usa una configuración fija elegida mediante la validación agrupada; "
        "es un análisis de sensibilidad y no reemplaza la estimación anidada principal.",
        "",
        markdown_table(scheme_summary),
        "",
        "## Robustez fuera del dominio",
        "",
        "Las pruebas por cultivar y por grupo muestral son análisis de estrés. Con solo dos "
        "cultivares y cuatro grupos no deben interpretarse como una estimación poblacional amplia.",
        "",
        markdown_table(stress_results),
        "",
        "## Nota de interpretación",
        "",
        "El objetivo es estimar la etapa experimental de una adquisición. No hay observaciones "
        "intermedias que validen una resolución temporal subdiaria. No se predicen viabilidad, "
        "concentración de humedad ni germinación prospectiva.",
        "",
        "Separar placas evita compartir granos o placas entre entrenamiento y test, pero no "
        "separa sesiones de adquisición. La etapa puede estar confundida con efectos de sesión "
        "o instrumento compartidos. No se demuestra transferencia a nuevas sesiones, cosechas "
        "o equipos. Las rejillas se orientaron mediante factibilidad sobre este mismo dataset.",
        "",
        "Los intervalos bootstrap son condicionales a las predicciones externas guardadas: "
        "remuestrean placas, pero no reajustan modelos ni repiten la selección de familia. "
        "Las pruebas de transferencia son exploratorias a nivel de familia.",
        "",
    ]
    path.write_text("\n".join(lines), encoding="utf-8")


def main(project_dir: Path, force: bool = False) -> None:
    started = time.time()
    project_dir = project_dir.resolve()
    data_file = project_dir / "01_DATASET" / "barley_nir_long.npz"
    out_dir = project_dir / "03_RESULTADOS"
    if not data_file.exists():
        fallback = project_dir / "project_artifacts" / "barley_nir_long.npz"
        if fallback.exists():
            data_file = fallback
            out_dir = project_dir / "project_artifacts"
    out_dir.mkdir(parents=True, exist_ok=True)

    expected = out_dir / "nested_oof_predictions.csv.gz"
    metadata_file = out_dir / 'nested_run_metadata.json'
    complete_outputs = ['nested_inner_search.csv.gz', 'nested_outer_fold_metrics.csv',
                        'nested_oof_predictions.csv.gz', 'nested_model_summary.csv',
                        'nested_cluster_bootstrap_ci.csv', 'nested_cluster_bootstrap_replicates.csv.gz',
                        'nested_paired_mae_differences.csv', 'split_sensitivity_fold_metrics.csv',
                        'split_sensitivity_oof_predictions.csv.gz', 'split_sensitivity_summary.csv',
                        'domain_stress_test_summary.csv', 'domain_stress_inner_search.csv.gz',
                        'domain_stress_predictions.csv.gz', 'best_model_metrics_by_day.csv',
                        'nested_selection_frequency.csv', 'RESULTADOS_VALIDACION_FINAL.md',
                        'fig_nested_model_comparison.png', 'fig_best_observed_vs_predicted.png',
                        'fig_best_error_by_day.png']
    if metadata_file.exists() and not force:
        stored = json.loads(metadata_file.read_text())
        current_grid = {family: [candidate.key for candidate in candidates]
                        for family, candidates in candidate_grid(SEED).items()}
        if (stored['seed'] != SEED or stored['outer_splits'] != OUTER_SPLITS
            or stored['inner_splits'] != INNER_SPLITS
            or stored['bootstrap_repetitions'] != BOOTSTRAP_REPETITIONS
            or stored['candidate_grid'] != current_grid
            or stored.get('pipeline_version', PIPELINE_VERSION) != PIPELINE_VERSION):
            raise RuntimeError('Analysis settings differ from the saved run. Use --force explicitly to replace results.')
        current_digest = hashlib.sha256(data_file.read_bytes()).hexdigest()
        if stored['data_sha256'] != current_digest:
            raise RuntimeError('Dataset differs from the saved completed run. Use --force explicitly to replace results.')
        if stored['versions'] != software_versions():
            raise RuntimeError('Software versions differ from the saved run. Read cached results without fitting, or use --force explicitly.')
        if all((out_dir / name).exists() for name in complete_outputs):
            print(f"Completed compatible results already exist: {expected}. Use --force to recompute.")
            return

    z = np.load(data_file, allow_pickle=True)
    raw = z["X"].astype(np.float32)
    y = z["day"].astype(np.float32)
    dish = z["petri_dish"].astype(int)
    grain_id = z["grain_id"].astype(int)
    grain_uid = np.array([f"{d}_{g}" for d, g in zip(dish, grain_id)])
    sample_group = z["sample_group"].astype(str)
    cultivar = z["cultivar"].astype(str)
    assert raw.shape == (13_452, 204)
    assert len(np.unique(dish)) == 90
    assert len(np.unique(grain_uid)) == 2_242
    assert np.isfinite(raw).all()
    structural = pd.DataFrame({"grain_uid": grain_uid, "stage": y, "dish": dish, "sample_group": sample_group})
    assert not structural.duplicated(["grain_uid", "stage"]).any()
    assert structural.groupby("grain_uid")["stage"].nunique().eq(6).all()
    assert structural.groupby("dish")["sample_group"].nunique().eq(1).all()

    digest = hashlib.sha256()
    with data_file.open("rb") as handle:
        for block in iter(lambda: handle.read(1_048_576), b""):
            digest.update(block)
    data_sha256 = digest.hexdigest()
    grid_description = {family: [candidate.key for candidate in candidates] for family, candidates in candidate_grid().items()}
    run_signature = hashlib.sha256(json.dumps({"data": data_sha256, "grid": grid_description, "seed": SEED, "outer": OUTER_SPLITS, "inner": INNER_SPLITS}, sort_keys=True).encode()).hexdigest()[:16]
    checkpoint_dir = out_dir / "checkpoints" / run_signature
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_context = {'pipeline_version': PIPELINE_VERSION, 'versions': software_versions(),
                          'data_sha256': data_sha256, 'candidate_grid': grid_description,
                          'seed': SEED, 'outer_splits': OUTER_SPLITS, 'inner_splits': INNER_SPLITS}
    context_file = checkpoint_dir / 'checkpoint_context.json'
    if context_file.exists() and not force:
        if json.loads(context_file.read_text()) != checkpoint_context:
            raise RuntimeError('Checkpoint environment or pipeline version differs. Use --force explicitly rather than mixing cached fits.')
    context_file.write_text(json.dumps(checkpoint_context, indent=2), encoding='utf-8')

    x_variants = spectral_variants(raw)
    inner, outer, predictions = run_nested_cv(
        x_variants, y, dish, grain_uid, sample_group, cultivar, SEED, checkpoint_dir, not force
    )
    inner.to_csv(out_dir / "nested_inner_search.csv.gz", index=False, compression="gzip")
    outer.to_csv(out_dir / "nested_outer_fold_metrics.csv", index=False)
    predictions.to_csv(expected, index=False, compression="gzip")

    nested_summary = (
        outer.groupby("family", as_index=False)
        .agg(
            mae_mean=("mae", "mean"),
            mae_sd=("mae", "std"),
            rmse_mean=("rmse", "mean"),
            rmse_sd=("rmse", "std"),
            r2_mean=("r2", "mean"),
            r2_sd=("r2", "std"),
            within_1_day_mean=("within_1_day", "mean"),
            bias_mean=("bias", "mean"),
        )
        .sort_values("mae_mean")
    )
    nested_summary.to_csv(out_dir / "nested_model_summary.csv", index=False)

    ci, boot = cluster_bootstrap_ci(predictions, BOOTSTRAP_REPETITIONS, SEED)
    ci.to_csv(out_dir / "nested_cluster_bootstrap_ci.csv", index=False)
    boot.to_csv(out_dir / "nested_cluster_bootstrap_replicates.csv.gz", index=False, compression="gzip")

    best_family = str(nested_summary.iloc[0]["family"])
    paired = boot.pivot(index="bootstrap_repetition", columns="family", values="mae")
    contrast_rows = []
    for family in paired.columns:
        if family == best_family:
            continue
        differences = paired[family] - paired[best_family]
        point = ci.loc[ci["family"].eq(family) & ci["metric"].eq("mae"), "estimate"].iloc[0] - ci.loc[ci["family"].eq(best_family) & ci["metric"].eq("mae"), "estimate"].iloc[0]
        contrast_rows.append({"comparison": f"{family} minus {best_family}", "mae_difference_days": point, "ci_low": differences.quantile(0.025), "ci_high": differences.quantile(0.975)})
    pd.DataFrame(contrast_rows).to_csv(out_dir / "nested_paired_mae_differences.csv", index=False)
    grid = candidate_grid(SEED)
    consensus = choose_consensus_candidate(best_family, inner, grid)
    scheme_fold_file = out_dir / 'split_sensitivity_fold_metrics.csv'
    scheme_prediction_file = out_dir / 'split_sensitivity_oof_predictions.csv.gz'
    reuse_schemes = False
    if not force and scheme_fold_file.exists() and scheme_prediction_file.exists():
        cached_folds = pd.read_csv(scheme_fold_file)
        cached_predictions = pd.read_csv(scheme_prediction_file)
        reuse_schemes = (len(cached_folds) == 15 and len(cached_predictions) == 3 * len(y)
                         and set(cached_folds['family']) == {consensus.family}
                         and set(cached_folds['preprocessing']) == {consensus.preprocessing}
                         and set(cached_folds['parameter']) == {consensus.parameter}
                         and cached_predictions.groupby('scheme')['row_index'].nunique().eq(len(y)).all())
    if reuse_schemes:
        scheme_folds, scheme_predictions = cached_folds, cached_predictions
        print('Restored completed fixed-configuration partition sensitivity.', flush=True)
    else:
        scheme_folds, scheme_predictions = evaluate_fixed_validation_schemes(
            consensus, x_variants, y, dish, grain_uid, SEED
        )
    scheme_folds.to_csv(out_dir / "split_sensitivity_fold_metrics.csv", index=False)
    scheme_predictions.to_csv(
        out_dir / "split_sensitivity_oof_predictions.csv.gz",
        index=False,
        compression="gzip",
    )
    scheme_summary = (
        scheme_folds.groupby("scheme", as_index=False)
        .agg(
            mae=("mae", "mean"),
            rmse=("rmse", "mean"),
            r2=("r2", "mean"),
            within_1_day=("within_1_day", "mean"),
            bias=("bias", "mean"),
        )
        .sort_values("mae")
    )
    scheme_summary.to_csv(out_dir / "split_sensitivity_summary.csv", index=False)

    stress_results, stress_inner, stress_predictions = run_domain_stress_tests(
        grid[best_family],
        x_variants,
        y,
        dish,
        grain_uid,
        sample_group,
        cultivar,
        checkpoint_dir,
        not force,
    )
    stress_results.to_csv(out_dir / "domain_stress_test_summary.csv", index=False)
    stress_inner.to_csv(out_dir / "domain_stress_inner_search.csv.gz", index=False, compression="gzip")
    stress_predictions.to_csv(
        out_dir / "domain_stress_predictions.csv.gz", index=False, compression="gzip"
    )

    per_day = predictions.loc[predictions["family"].eq(best_family)].copy()
    per_day["error"] = per_day["y_pred"] - per_day["y_true"]
    per_day_summary = (
        per_day.groupby("y_true", as_index=False)
        .agg(
            n=("error", "size"),
            mae=("error", lambda values: np.mean(np.abs(values))),
            rmse=("error", lambda values: np.sqrt(np.mean(np.square(values)))),
            bias=("error", "mean"),
            within_1_day=("error", lambda values: np.mean(np.abs(values) <= 1.0)),
        )
        .rename(columns={"y_true": "day_since_moisture"})
    )
    per_day_summary.to_csv(out_dir / "best_model_metrics_by_day.csv", index=False)

    selection_frequency = (
        outer.groupby(["family", "preprocessing", "parameter"], as_index=False)
        .size()
        .rename(columns={"size": "outer_folds_selected"})
        .sort_values(["family", "outer_folds_selected"], ascending=[True, False])
    )
    selection_frequency.to_csv(out_dir / "nested_selection_frequency.csv", index=False)

    save_figures(out_dir, ci, predictions, best_family)
    write_results_note(
        out_dir / "RESULTADOS_VALIDACION_FINAL.md",
        nested_summary,
        ci,
        best_family,
        consensus,
        stress_results,
        scheme_summary,
    )

    metadata = {
        "created_utc": pd.Timestamp.utcnow().isoformat(),
        "seed": SEED,
        "outer_splits": OUTER_SPLITS,
        "inner_splits": INNER_SPLITS,
        "bootstrap_repetitions": BOOTSTRAP_REPETITIONS,
        "grouping_unit": "petri_dish",
        "target": "day_since_moisture",
        "n_rows": int(raw.shape[0]),
        "n_bands": int(raw.shape[1]),
        "n_grains": int(len(np.unique(grain_uid))),
        "n_petri_dishes": int(len(np.unique(dish))),
        "best_family": best_family,
        "data_sha256": data_sha256,
        "run_signature": run_signature,
        "pipeline_version": PIPELINE_VERSION,
        "candidate_grid": grid_description,
        "bootstrap_interpretation": "Paired dish-cluster percentile bootstrap of fixed OOF predictions; conditional uncertainty, without pipeline retraining or acquisition-session resampling.",
        "consensus_preprocessing": consensus.preprocessing,
        "consensus_parameter": consensus.parameter,
        "elapsed_seconds": time.time() - started,
        "elapsed_seconds_note": "Elapsed time for this invocation, which may restore checkpoints; not cumulative historical fitting time.",
        "versions": software_versions(),
    }
    (out_dir / "nested_run_metadata.json").write_text(
        json.dumps(metadata, indent=2), encoding="utf-8"
    )
    print("\nFINAL NESTED SUMMARY\n", nested_summary.to_string(index=False), flush=True)
    print("\nCLUSTER BOOTSTRAP CI\n", ci.to_string(index=False), flush=True)
    print(f"\nCompleted in {metadata['elapsed_seconds'] / 60:.1f} minutes.", flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--project-dir",
        type=Path,
        default=(Path(__file__).resolve().parent.parent
                 if Path(__file__).resolve().parent.name == '02_CODIGO_NOTEBOOKS'
                 else Path(__file__).resolve().parent),
        help="Project root containing 01_DATASET and 03_RESULTADOS, or the local scratch root.",
    )
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--review-additions", action="store_true",
                        help="After the original run, execute separate frozen C and training-only PLS review diagnostics. Does not replace primary predictions.")
    args = parser.parse_args()
    main(args.project_dir, force=args.force)
    if args.review_additions:
        from run_review_additions import main as review_main
        review_main(args.project_dir)
