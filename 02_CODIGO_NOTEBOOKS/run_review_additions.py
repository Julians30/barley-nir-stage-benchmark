"""Post-review diagnostics; never select a model from external diagnostic errors.

The original nested comparison is immutable. C=300/1000 are frozen sensitivity
checks with each fold's original training-selected preprocessing and epsilon.
PLS profiles use outer-training observations only. Post-moisture PLS profiles
reuse the original component count, without claiming a post-only nested model.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import time
from datetime import datetime, timezone
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.model_selection import GroupKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVR
from threadpoolctl import threadpool_limits

import run_nested_validation as primary

VERSION = 'frozen_review_diagnostics_v1'
C_VALUES = (100, 300, 1000)


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def locate(root: Path) -> tuple[Path, Path]:
    if (root / '01_DATASET' / 'barley_nir_long.npz').exists():
        return root / '01_DATASET' / 'barley_nir_long.npz', root / '03_RESULTADOS'
    return root / 'project_artifacts' / 'barley_nir_long.npz', root / 'project_artifacts'


def vip(pls) -> np.ndarray:
    t, w, q = pls.x_scores_, pls.x_weights_, pls.y_loadings_
    explained = np.sum(t * t, axis=0) * np.sum(q * q, axis=0)
    norm = np.sum(w * w, axis=0)
    if explained.sum() <= 0 or np.any(norm <= 0):
        raise ValueError('Degenerate PLS component in VIP calculation.')
    result = np.sqrt(w.shape[0] * ((w * w / norm) @ explained) / explained.sum())
    np.testing.assert_allclose(np.mean(result**2), 1.0, atol=1e-10)
    return result


def draw_profiles(revision: Path, means: pd.DataFrame, profiles: pd.DataFrame):
    colors = ['#353535', '#4477AA', '#66AABB', '#228833', '#CCBB44', '#AA3377']
    fig, axes = plt.subplots(3, 1, figsize=(7.0, 7.0), sharex=True, constrained_layout=True)
    for stage, group in means.groupby('stage'):
        axes[0].plot(group.wavelength_nm_approx, group.mean_pseudo_absorbance,
                     color=colors[int(stage)], label=f'Stage {int(stage)}', linewidth=1.2)
    axes[0].set_ylabel('Mean pseudo-absorbance')
    axes[0].legend(ncol=3, fontsize=8, frameon=False)
    for label, color, style in [('all_six_stages', '#4477AA', '-'),
                                ('post_moisture_1_to_5', '#AA3377', '--')]:
        group = profiles.loc[profiles.training_subset.eq(label)]
        summary = group.groupby('wavelength_nm_approx').agg(
            vip_mean=('vip', 'mean'), vip_min=('vip', 'min'), vip_max=('vip', 'max'),
            coefficient_mean=('coefficient_per_training_sd', 'mean'),
            coefficient_min=('coefficient_per_training_sd', 'min'),
            coefficient_max=('coefficient_per_training_sd', 'max'))
        x = summary.index.to_numpy()
        name = 'All stages' if label == 'all_six_stages' else 'Post-moisture training'
        axes[1].plot(x, summary.vip_mean, color=color, linestyle=style, label=name)
        axes[1].fill_between(x, summary.vip_min, summary.vip_max, color=color, alpha=.12)
        axes[2].plot(x, summary.coefficient_mean, color=color, linestyle=style)
        axes[2].fill_between(x, summary.coefficient_min, summary.coefficient_max, color=color, alpha=.12)
    axes[1].axhline(1, color='#777777', linestyle=':', linewidth=1)
    axes[1].set_ylabel('PLS VIP')
    axes[1].legend(fontsize=8, frameon=False, loc='upper center')
    axes[2].axhline(0, color='#777777', linewidth=.7)
    axes[2].set_ylabel('Coefficient\n(stage / training SD)')
    axes[2].set_xlabel('Approximate wavelength (nm; README-based grid)')
    for index, ax in enumerate(axes):
        ax.axvspan(1400, 1500, color='#CCBB44', alpha=.13)
        ax.spines[['top', 'right']].set_visible(False)
        ax.text(.01, .93, f'({chr(97+index)})', transform=ax.transAxes, fontsize=10)
    fig.savefig(revision / 'fig_spectral_interpretation.png', dpi=240)
    plt.close(fig)


def main(root: Path, fold_subset: list[int] | None = None):
    data_path, results = locate(root.resolve())
    revision = results / 'revision_r1'
    revision.mkdir(exist_ok=True)
    run = json.loads((results / 'nested_run_metadata.json').read_text())
    outer = pd.read_csv(results / 'nested_outer_fold_metrics.csv')
    oof = pd.read_csv(results / 'nested_oof_predictions.csv.gz')
    context = {'version': VERSION, 'data_sha256': sha256(data_path),
               'primary_run_signature': run['run_signature'],
               'primary_oof_sha256': sha256(results / 'nested_oof_predictions.csv.gz'),
               'primary_outer_sha256': sha256(results / 'nested_outer_fold_metrics.csv'),
               'C_values': list(C_VALUES), 'versions': primary.software_versions(),
               'wavelength_grid': 'np.linspace(950,1650,204); approximate uniform range from source README; not exact camera calibration'}
    assert context['data_sha256'] == run['data_sha256']
    assert context['versions'] == run['versions']
    context_path = revision / 'diagnostics_context.json'
    if context_path.exists():
        if json.loads(context_path.read_text()) != context:
            raise RuntimeError('Review checkpoint context differs; use a separate revision directory.')
    else:
        context_path.write_text(json.dumps(context, indent=2) + '\n')
    data = np.load(data_path, allow_pickle=True)
    raw, y, dish = data['X'].astype(np.float32), data['day'].astype(np.float32), data['petri_dish'].astype(int)
    assert raw.shape == (13452, 204) and np.isfinite(raw).all()
    x_variants = primary.spectral_variants(raw)
    waves = np.linspace(950, 1650, raw.shape[1])
    splits = list(GroupKFold(5).split(raw, y, dish))
    pls_candidates = primary.candidate_grid()['PLSR']
    profile_parts, check_rows, c_predictions, c_metrics = [], [], [], []
    for fold, (train, test) in enumerate(splits, 1):
        if fold_subset is not None and fold not in fold_subset:
            continue
        assert set(dish[train]).isdisjoint(set(dish[test]))
        check_rows.append({'outer_fold': fold, 'train_dishes': len(np.unique(dish[train])),
                           'test_dishes': len(np.unique(dish[test])), 'dish_overlap': 0,
                           'profile_test_observations_used': 0})
        pls_row = outer.loc[outer.family.eq('PLSR') & outer.outer_fold.eq(fold)].iloc[0]
        candidate = next(c for c in pls_candidates if c.key == pls_row.candidate_key)
        assert candidate.preprocessing == 'raw', 'Physical raw-band interpretation requires raw-selected PLS.'
        for subset in ['all_six_stages', 'post_moisture_1_to_5']:
            path = revision / f'profiles_fold_{fold:02d}_{subset}.csv'
            fit = train if subset == 'all_six_stages' else train[y[train] >= 1]
            if path.exists():
                profile = pd.read_csv(path)
            else:
                pipeline = candidate.build()
                with threadpool_limits(limits=2):
                    pipeline.fit(raw[fit], y[fit])
                model = pipeline.named_steps['model']
                coefficient = np.asarray(model.coef_).reshape(-1)
                assert coefficient.shape == (204,)
                profile = pd.DataFrame({'outer_fold': fold, 'training_subset': subset,
                                        'channel_index_0based': np.arange(204),
                                        'wavelength_nm_approx': waves, 'vip': vip(model),
                                        'coefficient_per_training_sd': coefficient,
                                        'n_training_observations': len(fit),
                                        'n_components': model.n_components})
                profile.to_csv(path, index=False)
            assert len(profile) == 204 and profile.outer_fold.eq(fold).all()
            np.testing.assert_allclose(np.mean(profile.vip.to_numpy()**2), 1, atol=1e-10)
            profile_parts.append(profile)
        saved = oof.loc[oof.family.eq('SVR_RBF') & oof.outer_fold.eq(fold)].sort_values('row_index').copy()
        assert np.array_equal(saved.row_index.to_numpy(), test)
        selection = outer.loc[outer.family.eq('SVR_RBF') & outer.outer_fold.eq(fold)].iloc[0]
        settings = dict(item.split('=') for item in selection.parameter.split(';'))
        assert float(settings['C']) == 100 and settings['gamma'] == 'scale'
        for c_value in C_VALUES:
            path = revision / f'svr_C_{c_value}_fold_{fold:02d}_predictions.csv.gz'
            if path.exists():
                frame = pd.read_csv(path)
            else:
                frame = saved.copy()
                seconds = 0.0
                if c_value != 100:
                    pipeline = Pipeline([('scale', StandardScaler()),
                                         ('model', SVR(C=c_value, epsilon=float(settings['epsilon']),
                                                       gamma='scale', cache_size=2000))])
                    started = time.monotonic()
                    with threadpool_limits(limits=2):
                        pipeline.fit(x_variants[selection.preprocessing][train], y[train])
                        frame['y_pred'] = pipeline.predict(x_variants[selection.preprocessing][test])
                    seconds = time.monotonic() - started
                frame['C'] = c_value
                frame['diagnostic_fit_predict_seconds'] = seconds
                frame['diagnostic_status'] = 'frozen_check_not_model_selection'
                frame.to_csv(path, index=False, compression='gzip')
            assert np.array_equal(frame.row_index.to_numpy(), test)
            frame['primary_selected_parameter'] = selection.parameter
            frame['parameter'] = f'C={c_value};epsilon={float(settings["epsilon"]):g};gamma=scale'
            frame.to_csv(path, index=False, compression='gzip')
            np.testing.assert_array_equal(frame.y_true, y[test])
            if c_value == 100:
                np.testing.assert_allclose(frame.y_pred, saved.y_pred, atol=1e-12, rtol=0)
            c_predictions.append(frame)
            c_metrics.append({'outer_fold': fold, 'C': c_value, 'preprocessing': selection.preprocessing,
                              'epsilon': float(settings['epsilon']), 'gamma': 'scale',
                              **primary.regression_metrics(frame.y_true, frame.y_pred)})
            print(f'Frozen check fold={fold}, C={c_value}, MAE={c_metrics[-1]["mae"]:.6f}', flush=True)
    if fold_subset is not None and set(fold_subset) != {1, 2, 3, 4, 5}:
        print(f'Completed separate fold preparation: {fold_subset}; aggregate publication deferred.', flush=True)
        return
    profiles = pd.concat(profile_parts, ignore_index=True)
    predictions = pd.concat(c_predictions, ignore_index=True)
    profiles.to_csv(revision / 'pls_training_profiles.csv', index=False)
    predictions.to_csv(revision / 'svr_frozen_C_predictions.csv.gz', index=False, compression='gzip')
    fold_metrics = pd.DataFrame(c_metrics)
    fold_metrics.to_csv(revision / 'svr_frozen_C_fold_metrics.csv', index=False)
    pd.DataFrame(check_rows).to_csv(revision / 'diagnostics_split_audit.csv', index=False)
    summaries = []
    for c_value, frame in predictions.groupby('C'):
        assert len(frame) == len(y) and frame.row_index.nunique() == len(y)
        for subset in ['all_six_stages', 'post_moisture_1_to_5']:
            values = frame if subset == 'all_six_stages' else frame.loc[frame.y_true.ge(1)]
            summaries.append({'C': int(c_value), 'subset': subset, 'n': len(values),
                              **primary.regression_metrics(values.y_true, values.y_pred),
                              'all_stage_fold_mae_sd': fold_metrics.loc[fold_metrics.C.eq(c_value), 'mae'].std(ddof=1)})
    summary = pd.DataFrame(summaries)
    summary.to_csv(revision / 'svr_frozen_C_summary.csv', index=False)
    bands = []
    for (subset, fold), group in profiles.groupby(['training_subset', 'outer_fold']):
        for low, high in [(1400, 1500), (950, 1400), (1500, 1650)]:
            mask = group.wavelength_nm_approx.ge(low) & group.wavelength_nm_approx.le(high)
            band = group.loc[mask]
            bands.append({'training_subset': subset, 'outer_fold': int(fold),
                          'band_nm_approx': f'{low}-{high}', 'n_channels': len(band),
                          'vip_mean': band.vip.mean(), 'vip_max': band.vip.max(),
                          'fraction_vip_gt_1': band.vip.gt(1).mean(),
                          'coefficient_abs_mean': band.coefficient_per_training_sd.abs().mean()})
    band_table = pd.DataFrame(bands)
    band_table.to_csv(revision / 'pls_band_summaries.csv', index=False)
    mean_rows = []
    for stage in range(6):
        values = raw[y == stage]
        for channel, (wave, value, std) in enumerate(zip(waves, values.mean(axis=0), values.std(axis=0))):
            mean_rows.append({'stage': stage, 'channel_index_0based': channel,
                              'wavelength_nm_approx': wave, 'mean_pseudo_absorbance': float(value),
                              'sd_pseudo_absorbance': float(std), 'n': len(values)})
    means = pd.DataFrame(mean_rows)
    means.to_csv(revision / 'mean_spectra_by_stage.csv', index=False)
    draw_profiles(revision, means, profiles)
    main_before = context['primary_oof_sha256']
    assert sha256(results / 'nested_oof_predictions.csv.gz') == main_before
    assert sha256(results / 'nested_outer_fold_metrics.csv') == context['primary_outer_sha256']
    report = {'result': 'passed', 'created_utc': datetime.now(timezone.utc).isoformat(),
              **context, 'n_pls_profiles': 10, 'n_channels_per_profile': 204,
              'n_C_settings': 3, 'n_observations_per_C': len(y),
              'outer_dish_overlap': 0, 'primary_predictions_unchanged': True,
              'external_diagnostic_model_selection': False,
              'post_only_profiles': 'Exploratory; reuse the all-stage training-selected PLS component count; no post-only prediction performance claimed.',
              'interpretation': 'VIP/coefficient profiles explain PLS only, not SVR; approximate band association is not chemical moisture validation or exclusion of session fingerprinting.'}
    (revision / 'review_additions_audit.json').write_text(json.dumps(report, indent=2) + '\n')
    print(summary.to_string(index=False), flush=True)
    print(band_table.loc[band_table.band_nm_approx.eq('1400-1500')].to_string(index=False), flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    script_dir = Path(__file__).resolve().parent
    parser.add_argument('--project-dir', type=Path,
                        default=script_dir.parent if script_dir.name == '02_CODIGO_NOTEBOOKS' else script_dir)
    parser.add_argument('--folds', type=int, nargs='+', choices=[1, 2, 3, 4, 5],
                        help='Prepare only these independent folds; does not write aggregate results. Omit to aggregate all five after preparation.')
    args = parser.parse_args()
    main(args.project_dir, fold_subset=args.folds)
